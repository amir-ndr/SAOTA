import time
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from client import Client
from server import Server
from model import CNNMnist
from model_cifar import CNNCifar10
from dataloader import load_cifar10, partition_cifar10_dirichlet, partition_mnist_dirichlet, load_mnist


def evaluate_model(model, test_loader, device="cpu"):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / max(total, 1)


def main():
    # ---------------- Logging ----------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("fl_system.log"), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting SAOTA simulation")

    # ---------------- Parameters ----------------
    NUM_CLIENTS = 10
    NUM_ROUNDS = 3000
    BATCH_SIZE = 64
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # SAOTA server params (match your server.py signature)
    V = 100.0
    SIGMA_N = 1e-5
    T_MAX = 5000.0
    ETA = 0.02
    BISECTION_TOL = 1e-6

    # Client system params
    MU_K = 1e-27
    C_CYCLES_PER_SAMPLE = 1e6

    logger.info(f"Device: {DEVICE}")
    print(f"Using device: {DEVICE}")

    # ---------------- Data ----------------
    train_dataset, test_dataset = load_cifar10()
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Non-IID split
    client_data_map = partition_cifar10_dirichlet(train_dataset, NUM_CLIENTS, alpha=0.2)

    # ---------------- Clients ----------------
    clients = []
    base_model = CNNCifar10().to(DEVICE)

    for cid in range(NUM_CLIENTS):
        indices = client_data_map.get(cid, [])
        if len(indices) == 0:
            indices = [0]
            logger.warning(f"Client {cid} has no data; injecting one dummy sample index.")

        client = Client(
            client_id=cid,
            data_indices=indices,
            model=base_model,  # will be deep-copied inside Client
            fk=np.random.uniform(1e9, 2e9),       # 1-2 GHz
            mu_k=MU_K,
            P_max=3.0 + np.random.rand(),         # ~[2,3)
            C=C_CYCLES_PER_SAMPLE,
            Ak=BATCH_SIZE,
            train_dataset=train_dataset,
            device=DEVICE,
            seed=cid,  # reproducibility per-client
        )
        clients.append(client)

        # dt_k starts at 0 until computation is started by server
        logger.info(f"Client {cid}: |Dk|={len(indices)} samples, initial dt_k={client.dt_k:.4f}s")

    # Per-client total energy budgets (over NUM_ROUNDS)
    E_max_dict = {cid: float(np.random.uniform(30, 35)) for cid in range(NUM_CLIENTS)}
    print("Client Energy Budgets:")
    for cid, budget in E_max_dict.items():
        print(f"  Client {cid}: {budget:.2f} J")

    # ---------------- Server ----------------
    global_model = CNNCifar10().to(DEVICE)
    server = Server(
        global_model=global_model,
        clients=clients,
        V=V,
        sigma_n=SIGMA_N,
        T_max=T_MAX,
        E_max=E_max_dict,
        T_total_rounds=NUM_ROUNDS,
        eta=ETA,
        device=DEVICE,
        bisection_tol=BISECTION_TOL,
    )

    # ---------------- Training loop ----------------
    accuracies = []
    eval_rounds = []
    round_durations = []
    energy_queues_max = []
    time_queue_hist = []
    avg_staleness_per_round = []
    selected_counts = []
    client_selection_counts = {cid: 0 for cid in range(NUM_CLIENTS)}

    # initial eval
    acc0 = evaluate_model(server.global_model, test_loader, DEVICE)
    accuracies.append(acc0)
    eval_rounds.append(0)
    logger.info(f"Initial accuracy: {acc0:.2f}%")
    print(f"Initial accuracy: {acc0:.2f}%")

    for t in range(NUM_ROUNDS):
        round_start = time.time()
        print(f"\n=== Round {t+1}/{NUM_ROUNDS} ===")

        # Run one SAOTA round (this performs: start compute -> select -> elapse -> power -> OTA -> update -> broadcast -> staleness -> queues)
        selected, power_alloc, D_t = server.run_round(t)

        selected_ids = [c.client_id for c in selected]
        selected_counts.append(len(selected_ids))
        for cid in selected_ids:
            client_selection_counts[cid] += 1

        # Record metrics
        round_durations.append(float(D_t))
        max_energy_q = max(server.Q_e.values()) if server.Q_e else 0.0
        energy_queues_max.append(float(max_energy_q))
        time_queue_hist.append(float(server.Q_time))
        avg_tau = float(np.mean([c.tau_k for c in clients])) if clients else 0.0
        avg_staleness_per_round.append(avg_tau)

        wall_time = time.time() - round_start

        print(f"Selected {len(selected_ids)} clients: {selected_ids}")
        print(f"D_t: {D_t:.4f}s | wall: {wall_time:.2f}s | "
              f"max Q_e: {max_energy_q:.2f} | Q_time: {server.Q_time:.2f} | "
              f"avg staleness: {avg_tau:.2f}")

        # Evaluate every 5 rounds (and last round)
        if ((t + 1) % 5 == 0) or (t == NUM_ROUNDS - 1):
            acc = evaluate_model(server.global_model, test_loader, DEVICE)
            accuracies.append(acc)
            eval_rounds.append(t + 1)
            print(f"Global model accuracy: {acc:.2f}%")
            logger.info(f"Eval @ round {t+1}: acc={acc:.2f}%")

    # ---------------- Summary ----------------
    final_acc = accuracies[-1]
    print("\n=== Training Complete ===")
    print(f"Final accuracy: {final_acc:.2f}%")
    print(f"Average round duration: {np.mean(round_durations):.2f}s")
    print(f"Max energy queue: {max(energy_queues_max) if energy_queues_max else 0.0:.2f}")

    print("\nClient Selection Statistics:")
    sorted_counts = sorted(client_selection_counts.items(), key=lambda x: x[1], reverse=True)
    for cid, count in sorted_counts:
        print(f"Client {cid}: Selected {count} times ({count/NUM_ROUNDS:.1%} of rounds)")

    # Energy stats (from server.per_round_energy)
    print("\nEnergy Consumption Statistics:")
    total_energy = 0.0
    for cid in range(NUM_CLIENTS):
        client_energy = 0.0
        for r in range(len(server.per_round_energy)):
            client_energy += float(server.per_round_energy[r].get(cid, 0.0))
        total_energy += client_energy
        print(f"Client {cid}: {client_energy:.4f} J")
    print(f"Total system energy: {total_energy:.4f} J")
    print(f"Average per-client energy: {total_energy/NUM_CLIENTS:.4f} J")

    # ---------------- Plots ----------------
    plt.figure(figsize=(15, 12))

    # Accuracy
    plt.subplot(321)
    plt.plot(eval_rounds, accuracies, "o-")
    plt.title("Test Accuracy")
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)

    # Selected clients per round
    plt.subplot(322)
    plt.plot(selected_counts)
    plt.title("Selected Clients per Round")
    plt.xlabel("Rounds")
    plt.ylabel("Number of Clients")
    plt.grid(True)

    # Max energy queue
    plt.subplot(323)
    plt.plot(energy_queues_max)
    plt.title("Max Energy Queue Value")
    plt.xlabel("Rounds")
    plt.ylabel("Queue Value")
    plt.grid(True)

    # Per-client cumulative energy
    plt.subplot(324)
    for cid in range(NUM_CLIENTS):
        per_round = [float(server.per_round_energy[r].get(cid, 0.0)) for r in range(len(server.per_round_energy))]
        cum = np.cumsum(per_round)
        plt.plot(cum, label=f"Client {cid}")
    plt.title("Cumulative Energy per Client")
    plt.xlabel("Rounds")
    plt.ylabel("Energy (J)")
    plt.legend(fontsize=7, loc="upper left", ncol=2)
    plt.grid(True)

    # Avg staleness
    plt.subplot(325)
    plt.plot(avg_staleness_per_round)
    plt.title("Average Client Staleness")
    plt.xlabel("Rounds")
    plt.ylabel("Staleness (rounds)")
    plt.grid(True)

    # Selection distribution
    plt.subplot(326)
    plt.bar(range(NUM_CLIENTS), [client_selection_counts[cid] for cid in range(NUM_CLIENTS)])
    plt.title("Client Selection Distribution")
    plt.xlabel("Client ID")
    plt.ylabel("Times Selected")
    plt.xticks(range(NUM_CLIENTS))
    plt.grid(True)

    plt.tight_layout(pad=3.0)
    plt.savefig("semi_async_ota_fl_results.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()

import time
import numpy as np
import torch
import csv
import matplotlib.pyplot as plt

from client import Client
from server import Server
from model import CNNMnist
from dataloader import load_mnist, partition_mnist_dirichlet
from main import evaluate_model


def build_server(V, NUM_CLIENTS=10, NUM_ROUNDS=100, device="cpu", seed=0):
    BATCH_SIZE = 64
    ETA = 0.09
    SIGMA_N = 1e-7
    T_MAX = 21.84
    TAU_MAX = 5
    MU_K = 1e-27
    C_CYCLES_PER_SAMPLE = 1e6

    train_dataset, test_dataset = load_mnist()
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False, pin_memory=(device=="cuda"), num_workers=0
    )

    client_data_map = partition_mnist_dirichlet(train_dataset, NUM_CLIENTS, alpha=0.2)

    base_model = CNNMnist().to(device)
    rng = np.random.RandomState(seed)

    clients = []
    for cid in range(NUM_CLIENTS):
        indices = client_data_map.get(cid, [])
        if len(indices) == 0:
            indices = [0]

        client = Client(
            client_id=cid,
            data_indices=indices,
            model=base_model,
            fk=float(rng.uniform(1e9, 2e9)),
            mu_k=MU_K,
            P_max=30.0 + float(rng.rand()),
            C=C_CYCLES_PER_SAMPLE,
            Ak=BATCH_SIZE,
            train_dataset=train_dataset,
            device=device,
            seed=cid,
        )
        clients.append(client)

    # E_max_dict = {cid: float(rng.uniform(1000, 1200)) for cid in range(NUM_CLIENTS)}
    E_max_dict = {cid: 1000. for cid in range(NUM_CLIENTS)}


    global_model = CNNMnist().to(device)
    server = Server(
        global_model=global_model,
        clients=clients,
        V=V,
        sigma_n=SIGMA_N,
        T_max=T_MAX,
        E_max=E_max_dict,
        T_total_rounds=NUM_ROUNDS,
        eta=ETA,
        tau_max=TAU_MAX,
        device=device,
    )

    return server, test_loader


def sweep_V(V_values, NUM_ROUNDS=100, device="cpu", eval_interval=5):
    results = []

    for V in V_values:
        print(f"Running V={V:.4f}")
        server, test_loader = build_server(V, NUM_CLIENTS=10, NUM_ROUNDS=NUM_ROUNDS, device=device, seed=0)

        accuracies = []
        selected_counts = []
        avg_staleness = []

        # initial eval
        acc0 = evaluate_model(server.global_model, test_loader, device)
        accuracies.append((0, acc0))

        for t in range(NUM_ROUNDS):
            selected, powers, D_t = server.run_round(t)
            selected_counts.append(len(selected))
            avg_tau = float(np.mean([c.tau_k for c in server.clients])) if server.clients else 0.0
            avg_staleness.append(avg_tau)

            if ((t + 1) % eval_interval == 0) or (t == NUM_ROUNDS - 1):
                acc = evaluate_model(server.global_model, test_loader, device)
                accuracies.append((t + 1, acc))

        total_energy = sum(server.total_energy_per_round)
        frac_selection = float(np.mean(selected_counts) / len(server.clients)) if server.clients else 0.0
        mean_staleness = float(np.mean(avg_staleness)) if avg_staleness else 0.0
        final_acc = accuracies[-1][1] if accuracies else 0.0

        results.append({
            "V": float(V),
            "final_accuracy": float(final_acc),
            "total_energy": float(total_energy),
            "avg_selection_fraction": float(frac_selection),
            "mean_staleness": float(mean_staleness),
            "accuracy_curve": accuracies,
        })

    return results


def plot_and_save(results, out_png="sweep_V_results.png", out_csv="sweep_V_results.csv"):
    V = [r["V"] for r in results]
    acc = [r["final_accuracy"] for r in results]
    energy = [r["total_energy"] for r in results]
    frac = [r["avg_selection_fraction"] for r in results]
    stale = [r["mean_staleness"] for r in results]

    plt.figure(figsize=(12, 9))

    plt.subplot(221)
    plt.plot(V, acc, "o-")
    plt.xlabel("V")
    plt.ylabel("Final Accuracy (%)")
    plt.grid(True)

    plt.subplot(222)
    plt.plot(V, energy, "o-")
    plt.xlabel("V")
    plt.ylabel("Total Energy (J)")
    plt.grid(True)

    plt.subplot(223)
    plt.plot(V, frac, "o-")
    plt.xlabel("V")
    plt.ylabel("Avg Selection Fraction")
    plt.grid(True)

    plt.subplot(224)
    plt.plot(V, stale, "o-")
    plt.xlabel("V")
    plt.ylabel("Mean Staleness")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(out_png)
    print(f"Saved plot to {out_png}")

    # Save CSV summary
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["V", "final_accuracy", "total_energy", "avg_selection_fraction", "mean_staleness"])
        for r in results:
            writer.writerow([r["V"], r["final_accuracy"], r["total_energy"], r["avg_selection_fraction"], r["mean_staleness"]])

    print(f"Saved CSV to {out_csv}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_ROUNDS = 500
    V_values = list(np.linspace(100, 6000., 5))

    t0 = time.time()
    results = sweep_V(V_values, NUM_ROUNDS=NUM_ROUNDS, device=device, eval_interval=10)
    t1 = time.time()

    plot_and_save(results)
    print(f"Sweep finished in {t1-t0:.2f}s")

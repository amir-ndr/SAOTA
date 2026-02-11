import os
import csv
import time
import random
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from client import Client
from server import Server
from model_cifar import CNNCifar10
from model import CNNMnist
from dataloader import load_cifar10, partition_cifar10_dirichlet, load_mnist, partition_mnist_dirichlet


# -------------------------
# Configuration
# -------------------------
RESULT_ROOT = "results_v_study2_new"

V_VALUES = [0.01, 0.1, 1, 10, 100, 500, 1000, 5000, 10000, 50000, 100000]

NUM_CLIENTS = 10
NUM_ROUNDS = 780
N_RUNS_PER_V = 1
EVAL_EVERY = 20

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fixed params except V
BATCH_SIZE = 64
SIGMA_N = 1e-7
T_MAX = 21.84
ETA = 0.09944922699073541
BISECTION_TOL = 1e-6

# Client system params
MU_K = 1e-27
C_CYCLES_PER_SAMPLE = 1e6

# Dataset split
DIRICHLET_ALPHA = 0.2

# Energy budget range
E_BUDGET_LOW = 2400.0
E_BUDGET_HIGH = 2600

# Optional: filter runs by final accuracy threshold (only affects Plot 1)
PLOT_FINAL_ACC_THRESHOLD = None  # e.g., 50.0, else None


# -------------------------
# Helpers
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate_model(model, test_loader, device="cpu") -> float:
    model.eval()
    correct, total = 0, 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return 100.0 * correct / max(total, 1)


def make_energy_budgets(num_clients: int, low: float, high: float) -> dict:
    return {cid: float(np.random.uniform(low, high)) for cid in range(num_clients)}


def safe_v_dir_name(v: float) -> str:
    # folder names: V_0.01, V_0.1, V_1, ...
    return f"V_{v}"


def build_clients(train_dataset, client_data_map, base_model, batch_size, seed_base):
    clients = []
    for cid in range(NUM_CLIENTS):
        indices = client_data_map.get(cid, [])
        if len(indices) == 0:
            indices = [0]  # avoid empty data edge-case

        client = Client(
            client_id=cid,
            data_indices=indices,
            model=base_model,
            fk=np.random.uniform(1e9, 2e9),
            mu_k=MU_K,
            P_max=3.0 + np.random.rand(),
            C=C_CYCLES_PER_SAMPLE,
            Ak=batch_size,
            train_dataset=train_dataset,
            device=DEVICE,
            seed=(seed_base + cid) % (2**32 - 1),
        )
        clients.append(client)
    return clients


def qmax_e_from_server(server: Server) -> float:
    """Q_max^e(t) = max_k Q_e^k(t)."""
    if not hasattr(server, "Q_e") or server.Q_e is None or len(server.Q_e) == 0:
        return float("nan")
    return float(max(server.Q_e.values()))


# -------------------------
# One run for a single V
# -------------------------
def run_one(
    v_value: float,
    run_id: int,
    train_dataset,
    test_loader,
    csv_path: str,
    seed: int,
):
    set_seed(seed)

    # Re-partition per run (recommended)
    client_data_map = partition_mnist_dirichlet(
        train_dataset,
        NUM_CLIENTS,
        alpha=float(DIRICHLET_ALPHA),
        seed=int(seed),
        min_size=10,
    )

    # Clients + server
    base_model = CNNMnist().to(DEVICE)
    clients = build_clients(train_dataset, client_data_map, base_model, BATCH_SIZE, seed_base=seed * 1000)

    E_max_dict = make_energy_budgets(NUM_CLIENTS, E_BUDGET_LOW, E_BUDGET_HIGH)

    global_model = CNNMnist().to(DEVICE)
    server = Server(
        global_model=global_model,
        clients=clients,
        V=float(v_value),
        sigma_n=float(SIGMA_N),
        T_max=float(T_MAX),
        E_max=E_max_dict,
        T_total_rounds=int(NUM_ROUNDS),
        eta=float(ETA),
        device=DEVICE,
        bisection_tol=float(BISECTION_TOL),
    )

    # Track cumulative energy per client (system energy accounting)
    cum_energy_per_client = {cid: 0.0 for cid in range(NUM_CLIENTS)}
    selection_counts = {cid: 0 for cid in range(NUM_CLIENTS)}

    # Round 0 metrics
    acc0 = evaluate_model(server.global_model, test_loader, DEVICE)
    cum_system_energy0 = 0.0
    unified_energy0 = 0.0
    selected_count0 = 0
    avg_staleness0 = float(np.mean([float(getattr(c, "tau_k", 0.0)) for c in clients])) if clients else 0.0
    qmax0 = qmax_e_from_server(server)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "round",
                "accuracy",
                "cumulative_energy",
                "selected_count",
                "avg_staleness",
                "qmax_e",
            ],
        )
        writer.writerow(
            {
                "run_id": int(run_id),
                "round": 0,
                "accuracy": f"{acc0:.6f}",
                "cumulative_energy": f"{cum_system_energy0:.6f}",
                "selected_count": str(selected_count0),
                "avg_staleness": f"{avg_staleness0:.6f}",
                "qmax_e": f"{qmax0:.6f}" if not np.isnan(qmax0) else "",
            }
        )

    final_acc = float(acc0)

    # Main rounds 1..NUM_ROUNDS
    cum_system_energy = 0.0
    for t in range(NUM_ROUNDS):
        selected, power_alloc, D_t = server.run_round(t)

        # Selected IDs
        selected_ids = [c.client_id for c in selected]
        selected_count = int(len(selected_ids))
        for cid in selected_ids:
            selection_counts[int(cid)] += 1

        # Per-round energy from server update_queues()
        round_energy_dict = server.per_round_energy[-1] if getattr(server, "per_round_energy", None) else {}
        for cid, e_val in round_energy_dict.items():
            cum_energy_per_client[int(cid)] += float(e_val)

        # Unified cumulative energy: max_k (cum_energy_k / Emax_k)
        # unified_energy = 0.0
        # for cid in range(NUM_CLIENTS):
        #     denom = float(E_max_dict[cid]) if float(E_max_dict[cid]) > 0 else 1.0
        #     unified_energy = max(unified_energy, float(cum_energy_per_client[cid]) / denom)
        E_round = float(server.total_energy_per_round[-1]) if getattr(server, "total_energy_per_round", None) else 0.0
        cum_system_energy += E_round

        # Staleness after the round
        avg_staleness = float(np.mean([float(getattr(c, "tau_k", 0.0)) for c in clients])) if clients else 0.0

        # Q_max^e after the queue update
        qmax_e = qmax_e_from_server(server)

        # Accuracy (sparse logging)
        acc_str = ""
        if ((t + 1) % EVAL_EVERY == 0) or (t == NUM_ROUNDS - 1):
            acc = evaluate_model(server.global_model, test_loader, DEVICE)
            final_acc = float(acc)
            acc_str = f"{final_acc:.6f}"

        # Write row for round index t+1
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "run_id",
                    "round",
                    "accuracy",
                    "cumulative_energy",
                    "selected_count",
                    "avg_staleness",
                    "qmax_e",
                ],
            )
            writer.writerow(
                {
                    "run_id": int(run_id),
                    "round": int(t + 1),
                    "accuracy": acc_str,
                    "cumulative_energy": f"{cum_system_energy:.6f}",
                    "selected_count": str(selected_count),
                    "avg_staleness": f"{avg_staleness:.6f}",
                    "qmax_e": f"{qmax_e:.6f}" if not np.isnan(qmax_e) else "",
                }
            )

    return final_acc, selection_counts


# -------------------------
# Plotting (your style) + Qmax figure
# -------------------------
def plot_results():
    # 2x2 figure (same style as yours)
    plt.figure(figsize=(15, 10))

    # -------------------------
    # Plot 1: Accuracy progression
    # -------------------------
    plt.subplot(2, 2, 1)
    for v in V_VALUES:
        v_dir = os.path.join(RESULT_ROOT, safe_v_dir_name(v))
        csv_path = os.path.join(v_dir, "round_metrics.csv")
        if not os.path.exists(csv_path):
            continue

        keep_run_ids = None
        if PLOT_FINAL_ACC_THRESHOLD is not None:
            final_acc_by_run = {}
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rid = int(row["run_id"])
                    if row["accuracy"]:
                        final_acc_by_run[rid] = float(row["accuracy"])
            keep_run_ids = {rid for rid, acc in final_acc_by_run.items() if acc >= PLOT_FINAL_ACC_THRESHOLD}

        avg_acc = []
        rounds = []
        current_round = None
        sum_acc = 0.0
        count = 0

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row["accuracy"]:
                    continue
                rid = int(row["run_id"])
                if keep_run_ids is not None and rid not in keep_run_ids:
                    continue

                r = int(row["round"])
                a = float(row["accuracy"])

                if current_round is None:
                    current_round = r

                if r != current_round:
                    if count > 0:
                        avg_acc.append(sum_acc / count)
                        rounds.append(current_round)
                    current_round = r
                    sum_acc = 0.0
                    count = 0

                sum_acc += a
                count += 1

        if count > 0 and current_round is not None:
            avg_acc.append(sum_acc / count)
            rounds.append(current_round)

        if len(rounds) > 0:
            plt.plot(rounds, avg_acc, "o-", label=f"V={v}")

    title = "Test Accuracy Progression"
    if PLOT_FINAL_ACC_THRESHOLD is not None:
        title += f" (Only runs with final accuracy >= {PLOT_FINAL_ACC_THRESHOLD}%)"
    plt.title(title)
    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    # -------------------------
    # Plot 2: Unified cumulative energy
    # -------------------------
    plt.subplot(2, 2, 2)
    for v in V_VALUES:
        v_dir = os.path.join(RESULT_ROOT, safe_v_dir_name(v))
        csv_path = os.path.join(v_dir, "round_metrics.csv")
        if not os.path.exists(csv_path):
            continue

        energy_data = {r: [] for r in range(NUM_ROUNDS + 1)}
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                r = int(row["round"])
                energy_data[r].append(float(row["cumulative_energy"]))

        avg_energy = [float(np.mean(energy_data[r])) if energy_data[r] else 0.0 for r in range(NUM_ROUNDS + 1)]
        plt.plot(range(NUM_ROUNDS + 1), avg_energy, label=f"V={v}")

    plt.title("Unified Cumulative Energy (max_k E_k/Emax_k)")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Max Normalized Energy")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.1)

    # -------------------------
    # Plot 3: Average selection fraction (0..1)
    # -------------------------
    plt.subplot(2, 2, 3)
    for v in V_VALUES:
        v_dir = os.path.join(RESULT_ROOT, safe_v_dir_name(v))
        csv_path = os.path.join(v_dir, "round_metrics.csv")
        if not os.path.exists(csv_path):
            continue

        selection_data = {r: [] for r in range(NUM_ROUNDS + 1)}
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                r = int(row["round"])
                frac = float(row["selected_count"]) / float(NUM_CLIENTS)
                selection_data[r].append(frac)

        avg_sel = [float(np.mean(selection_data[r])) if selection_data[r] else 0.0 for r in range(NUM_ROUNDS + 1)]
        plt.plot(range(NUM_ROUNDS + 1), avg_sel, label=f"V={v}")

    plt.title("Average Client Selection Fraction (0-1)")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Selection Fraction")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.1)

    # -------------------------
    # Plot 4: Average staleness
    # -------------------------
    plt.subplot(2, 2, 4)
    for v in V_VALUES:
        v_dir = os.path.join(RESULT_ROOT, safe_v_dir_name(v))
        csv_path = os.path.join(v_dir, "round_metrics.csv")
        if not os.path.exists(csv_path):
            continue

        stale_data = {r: [] for r in range(NUM_ROUNDS + 1)}
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                r = int(row["round"])
                stale_data[r].append(float(row["avg_staleness"]))

        avg_stale = [float(np.mean(stale_data[r])) if stale_data[r] else 0.0 for r in range(NUM_ROUNDS + 1)]
        plt.plot(range(NUM_ROUNDS + 1), avg_stale, label=f"V={v}")

    plt.title("Average Client Staleness")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Staleness (rounds)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    os.makedirs(RESULT_ROOT, exist_ok=True)
    out_path = os.path.join(RESULT_ROOT, "comparison_plots_unified.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Saved] {out_path}")

    # -------------------------
    # EXTRA FIGURE: Qmax^e vs rounds
    # -------------------------
    plt.figure(figsize=(10, 6))
    for v in V_VALUES:
        v_dir = os.path.join(RESULT_ROOT, safe_v_dir_name(v))
        csv_path = os.path.join(v_dir, "round_metrics.csv")
        if not os.path.exists(csv_path):
            continue

        q_data = {r: [] for r in range(NUM_ROUNDS + 1)}
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                r = int(row["round"])
                q_str = row.get("qmax_e", "")
                if q_str is None or q_str == "":
                    continue
                q_data[r].append(float(q_str))

        avg_q = [float(np.mean(q_data[r])) if q_data[r] else 0.0 for r in range(NUM_ROUNDS + 1)]
        plt.plot(range(NUM_ROUNDS + 1), avg_q, label=f"V={v}")

    plt.title("Max Energy Queue $Q_{\max}^e$ vs Rounds")
    plt.xlabel("Communication Rounds")
    plt.ylabel("$Q_{\max}^e$")
    plt.legend()
    plt.grid(True)
    out_q = os.path.join(RESULT_ROOT, "qmax_e_vs_round.png")
    plt.tight_layout()
    plt.savefig(out_q, dpi=300)
    plt.close()
    print(f"[Saved] {out_q}")


# -------------------------
# Main: run all V experiments + save CSVs
# -------------------------
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logger = logging.getLogger("V_comparison")

    os.makedirs(RESULT_ROOT, exist_ok=True)

    # Load data once
    train_dataset, test_dataset = load_mnist()
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    logger.info(f"Device: {DEVICE}")
    logger.info(f"NUM_ROUNDS={NUM_ROUNDS} | N_RUNS_PER_V={N_RUNS_PER_V} | BATCH_SIZE={BATCH_SIZE}")
    logger.info(f"Fixed params: SIGMA_N={SIGMA_N} | T_MAX={T_MAX} | ETA={ETA}")

    for v in V_VALUES:
        v_dir = os.path.join(RESULT_ROOT, safe_v_dir_name(v))
        os.makedirs(v_dir, exist_ok=True)

        round_csv = os.path.join(v_dir, "round_metrics.csv")
        frac_csv = os.path.join(v_dir, "client_selection_fractions.csv")

        # Fresh CSV header (overwrite)
        with open(round_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "run_id",
                    "round",
                    "accuracy",
                    "cumulative_energy",
                    "selected_count",
                    "avg_staleness",
                    "qmax_e",
                ],
            )
            writer.writeheader()

        # Aggregate selection counts across runs
        total_selection_counts = {cid: 0 for cid in range(NUM_CLIENTS)}
        final_accs = []

        logger.info(f"=== V={v} ===")
        for run_id in range(N_RUNS_PER_V):
            seed = 10000 + int(100 * float(v)) + run_id
            t0 = time.time()

            final_acc, sel_counts = run_one(
                v_value=v,
                run_id=run_id,
                train_dataset=train_dataset,
                test_loader=test_loader,
                csv_path=round_csv,
                seed=seed,
            )

            dt = time.time() - t0
            final_accs.append(final_acc)

            for cid in range(NUM_CLIENTS):
                total_selection_counts[cid] += int(sel_counts.get(cid, 0))

            logger.info(
                f"  Run {run_id}/{N_RUNS_PER_V-1} done | final_acc={final_acc:.2f}% | wall={dt/60:.1f} min"
            )

        # Save per-client average selection fraction
        denom = float(N_RUNS_PER_V * NUM_ROUNDS)
        with open(frac_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["client_id", "avg_selection_fraction"])
            writer.writeheader()
            for cid in range(NUM_CLIENTS):
                frac = float(total_selection_counts[cid]) / denom
                writer.writerow({"client_id": cid, "avg_selection_fraction": f"{frac:.6f}"})

        logger.info(f"  Saved: {round_csv}")
        logger.info(f"  Saved: {frac_csv}")
        logger.info(f"  Avg final acc over runs: {np.mean(final_accs):.2f}%")

    print("\nAll V experiments completed.")


if __name__ == "__main__":
    main()
    plot_results()

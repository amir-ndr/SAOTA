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
from dataloader import load_cifar10, partition_cifar10_dirichlet


# -------------------------
# Configuration
# -------------------------
RESULT_ROOT = "results_v_study2"

V_VALUES = [0.01, 0.1, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# V_VALUES = [0.01, 100]

NUM_CLIENTS = 10

# Your plotting template often uses 300; keep consistent
NUM_ROUNDS = 5000

# independent runs (seeds) per V (averaged in plots)
N_RUNS_PER_V = 1

# accuracy evaluation frequency
EVAL_EVERY = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fixed params except V
BATCH_SIZE = 32
SIGMA_N = 6.287625692659408e-06
T_MAX = 2814.156090700375
ETA = 0.053
BISECTION_TOL = 1e-6

# Client system params
MU_K = 1e-27
C_CYCLES_PER_SAMPLE = 1e6

# Dataset split
DIRICHLET_ALPHA = 0.2

# Energy budget range
E_BUDGET_LOW = 26.37
E_BUDGET_HIGH = 33.59

# Optional: filter runs by final accuracy (only affects Plot 1)
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
    # keep folder names consistent with your example: V_0.01, V_0.1, V_1, ...
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
            fk=np.random.uniform(1e9, 2e9),       # 1-2 GHz
            mu_k=MU_K,
            P_max=3.0 + np.random.rand(),
            C=C_CYCLES_PER_SAMPLE,
            Ak=batch_size,
            train_dataset=train_dataset,
            device=DEVICE,
            seed=seed_base + cid,
        )
        clients.append(client)
    return clients


# -------------------------
# One run for a single V
# -------------------------
def run_one(
    v_value: float,
    run_id: int,
    train_dataset,
    test_loader,
    base_client_data_map,
    csv_path: str,
    seed: int,
):
    set_seed(seed)

    # Re-partition per run (recommended).
    client_data_map = partition_cifar10_dirichlet(train_dataset, NUM_CLIENTS, alpha=DIRICHLET_ALPHA)

    # Clients and server
    base_model = CNNCifar10().to(DEVICE)
    clients = build_clients(train_dataset, client_data_map, base_model, BATCH_SIZE, seed_base=seed * 1000)

    E_max_dict = make_energy_budgets(NUM_CLIENTS, E_BUDGET_LOW, E_BUDGET_HIGH)

    global_model = CNNCifar10().to(DEVICE)
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

    # Tracking for unified cumulative energy
    cum_energy_per_client = {cid: 0.0 for cid in range(NUM_CLIENTS)}

    # Selection counts for per-client fractions
    selection_counts = {cid: 0 for cid in range(NUM_CLIENTS)}

    # Initial metrics at round 0
    acc0 = evaluate_model(server.global_model, test_loader, DEVICE)
    unified_energy0 = 0.0
    selected_count0 = 0
    avg_staleness0 = float(np.mean([c.tau_k for c in clients])) if clients else 0.0

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run_id", "round", "accuracy", "cumulative_energy", "selected_count", "avg_staleness"],
        )
        writer.writerow(
            {
                "run_id": run_id,
                "round": 0,
                "accuracy": f"{acc0:.6f}",
                "cumulative_energy": f"{unified_energy0:.6f}",
                "selected_count": str(selected_count0),
                "avg_staleness": f"{avg_staleness0:.6f}",
            }
        )

    final_acc = acc0

    # Main rounds (write rows 1..NUM_ROUNDS)
    for t in range(NUM_ROUNDS):
        selected, power_alloc, D_t = server.run_round(t)
        selected_ids = [c.client_id for c in selected]
        selected_count = len(selected_ids)

        # Update selection counts
        for cid in selected_ids:
            selection_counts[cid] += 1

        # Update cumulative energy per selected client using server.per_round_energy[-1]
        round_energy_dict = server.per_round_energy[-1] if len(server.per_round_energy) > 0 else {}
        for cid, e_val in round_energy_dict.items():
            cum_energy_per_client[int(cid)] += float(e_val)

        # Unified cumulative energy metric: max_k (cum_energy_k / E_max^k)
        unified_energy = 0.0
        for cid in range(NUM_CLIENTS):
            denom = float(E_max_dict[cid]) if float(E_max_dict[cid]) > 0 else 1.0
            unified_energy = max(unified_energy, cum_energy_per_client[cid] / denom)

        # Avg staleness AFTER the round's staleness updates
        avg_staleness = float(np.mean([c.tau_k for c in clients])) if clients else 0.0

        # Accuracy only every EVAL_EVERY rounds (and last)
        acc_str = ""
        if ((t + 1) % EVAL_EVERY == 0) or (t == NUM_ROUNDS - 1):
            acc = evaluate_model(server.global_model, test_loader, DEVICE)
            final_acc = acc
            acc_str = f"{acc:.6f}"

        # Save per-round metrics (round index uses 1..NUM_ROUNDS)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["run_id", "round", "accuracy", "cumulative_energy", "selected_count", "avg_staleness"],
            )
            writer.writerow(
                {
                    "run_id": run_id,
                    "round": t + 1,
                    "accuracy": acc_str,
                    "cumulative_energy": f"{unified_energy:.6f}",
                    "selected_count": str(selected_count),
                    "avg_staleness": f"{avg_staleness:.6f}",
                }
            )

    return final_acc, selection_counts


# -------------------------
# Plotting (your style), with staleness as subplot (2,2,4)
# -------------------------
def plot_results():
    plt.figure(figsize=(15, 10))

    # Plot 1: accuracy progression
    plt.subplot(2, 2, 1)
    for v in V_VALUES:
        v_dir = os.path.join(RESULT_ROOT, safe_v_dir_name(v))
        csv_path = os.path.join(v_dir, "round_metrics.csv")
        if not os.path.exists(csv_path):
            continue

        # Optional: filter runs by final accuracy threshold
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

    # Plot 2: unified cumulative energy (already normalized by E_max^k)
    plt.subplot(2, 2, 2)
    for v in V_VALUES:
        v_dir = os.path.join(RESULT_ROOT, safe_v_dir_name(v))
        csv_path = os.path.join(v_dir, "round_metrics.csv")
        if not os.path.exists(csv_path):
            continue

        energy_data = {r: [] for r in range(NUM_ROUNDS + 1)}  # include round 0
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                r = int(row["round"])
                energy_data[r].append(float(row["cumulative_energy"]))

        avg_energy = []
        for r in range(NUM_ROUNDS + 1):
            avg_energy.append(float(np.mean(energy_data[r])) if energy_data[r] else 0.0)

        plt.plot(range(NUM_ROUNDS + 1), avg_energy, label=f"V={v}")

    plt.title("Unified Cumulative Energy")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Max Normalized Energy")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.1)

    # Plot 3: average selection fraction (0..1)
    plt.subplot(2, 2, 3)
    for v in V_VALUES:
        v_dir = os.path.join(RESULT_ROOT, safe_v_dir_name(v))
        csv_path = os.path.join(v_dir, "round_metrics.csv")
        if not os.path.exists(csv_path):
            continue

        selection_data = {r: [] for r in range(NUM_ROUNDS + 1)}  # include round 0
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                r = int(row["round"])
                frac = float(row["selected_count"]) / float(NUM_CLIENTS)
                selection_data[r].append(frac)

        avg_sel = []
        for r in range(NUM_ROUNDS + 1):
            avg_sel.append(float(np.mean(selection_data[r])) if selection_data[r] else 0.0)

        plt.plot(range(NUM_ROUNDS + 1), avg_sel, label=f"V={v}")

    plt.title("Average Client Selection Fraction (0-1)")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Selection Fraction")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.1)

    # Plot 4: staleness plot (instead of histogram/bar chart)
    plt.subplot(2, 2, 4)
    for v in V_VALUES:
        v_dir = os.path.join(RESULT_ROOT, safe_v_dir_name(v))
        csv_path = os.path.join(v_dir, "round_metrics.csv")
        if not os.path.exists(csv_path):
            continue

        stale_data = {r: [] for r in range(NUM_ROUNDS + 1)}  # include round 0
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                r = int(row["round"])
                stale_data[r].append(float(row["avg_staleness"]))

        avg_stale = []
        for r in range(NUM_ROUNDS + 1):
            avg_stale.append(float(np.mean(stale_data[r])) if stale_data[r] else 0.0)

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
    train_dataset, test_dataset = load_cifar10()
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # base partition (we re-partition inside each run anyway)
    base_client_data_map = partition_cifar10_dirichlet(train_dataset, NUM_CLIENTS, alpha=DIRICHLET_ALPHA)

    logger.info(f"Device: {DEVICE}")
    logger.info(f"NUM_ROUNDS={NUM_ROUNDS} | N_RUNS_PER_V={N_RUNS_PER_V} | BATCH_SIZE={BATCH_SIZE}")
    logger.info(f"Fixed params: SIGMA_N={SIGMA_N} | T_MAX={T_MAX} | ETA={ETA}")

    for v in V_VALUES:
        v_dir = os.path.join(RESULT_ROOT, safe_v_dir_name(v))
        os.makedirs(v_dir, exist_ok=True)

        round_csv = os.path.join(v_dir, "round_metrics.csv")
        frac_csv = os.path.join(v_dir, "client_selection_fractions.csv")

        # Write header fresh each time (overwrite old)
        with open(round_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["run_id", "round", "accuracy", "cumulative_energy", "selected_count", "avg_staleness"],
            )
            writer.writeheader()

        # Aggregate selection counts across runs
        total_selection_counts = {cid: 0 for cid in range(NUM_CLIENTS)}
        final_accs = []

        logger.info(f"=== V={v} ===")
        for run_id in range(N_RUNS_PER_V):
            seed = 10000 + int(100 * float(v)) + run_id  # deterministic but depends on v
            t0 = time.time()
            final_acc, sel_counts = run_one(
                v_value=v,
                run_id=run_id,
                train_dataset=train_dataset,
                test_loader=test_loader,
                base_client_data_map=base_client_data_map,
                csv_path=round_csv,
                seed=seed,
            )
            dt = time.time() - t0
            final_accs.append(final_acc)

            for cid in range(NUM_CLIENTS):
                total_selection_counts[cid] += int(sel_counts.get(cid, 0))

            logger.info(f"  Run {run_id}/{N_RUNS_PER_V-1} done | final_acc={final_acc:.2f}% | wall={dt/60:.1f} min")

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

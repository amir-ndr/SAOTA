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
RESULT_ROOT = "V_time_based"

V_VALUES = [0.01, 0.1, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

NUM_CLIENTS = 10
NUM_ROUNDS = 5000

# independent runs (seeds) per V (averaged in plots)
N_RUNS_PER_V = 1

# Evaluate & log accuracy every N rounds (log blank otherwise)
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
    return f"V_{v}"


def build_clients(train_dataset, client_data_map, base_model, batch_size, seed_base):
    clients = []
    for cid in range(NUM_CLIENTS):
        indices = client_data_map.get(cid, [])
        if len(indices) == 0:
            indices = [0]

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
            seed=seed_base + cid,
        )
        clients.append(client)
    return clients


# -------------------------
# One run for a single V (time-indexed logging)
# -------------------------
def run_one_time_indexed(
    v_value: float,
    run_id: int,
    train_dataset,
    test_loader,
    csv_path: str,
    seed: int,
):
    set_seed(seed)

    # repartition each run
    client_data_map = partition_cifar10_dirichlet(train_dataset, NUM_CLIENTS, alpha=DIRICHLET_ALPHA)

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

    # unified cumulative energy tracking
    cum_energy_per_client = {cid: 0.0 for cid in range(NUM_CLIENTS)}
    selection_counts = {cid: 0 for cid in range(NUM_CLIENTS)}

    run_start = time.time()

    # Round 0 snapshot
    acc0 = evaluate_model(server.global_model, test_loader, DEVICE)
    unified_energy0 = 0.0
    selected_count0 = 0
    avg_staleness0 = float(np.mean([c.tau_k for c in clients])) if clients else 0.0
    t0 = 0.0

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run_id", "time_sec", "round", "accuracy", "cumulative_energy", "selected_count", "avg_staleness"],
        )
        writer.writerow(
            {
                "run_id": run_id,
                "time_sec": f"{t0:.6f}",
                "round": 0,
                "accuracy": f"{acc0:.6f}",
                "cumulative_energy": f"{unified_energy0:.6f}",
                "selected_count": str(selected_count0),
                "avg_staleness": f"{avg_staleness0:.6f}",
            }
        )

    final_acc = acc0

    # Main rounds
    for t in range(NUM_ROUNDS):
        selected, power_alloc, _ = server.run_round(t)

        selected_ids = [c.client_id for c in selected]
        selected_count = len(selected_ids)

        for cid in selected_ids:
            selection_counts[cid] += 1

        # energy accounting from server.per_round_energy[-1]
        round_energy_dict = server.per_round_energy[-1] if len(server.per_round_energy) > 0 else {}
        for cid, e_val in round_energy_dict.items():
            cum_energy_per_client[int(cid)] += float(e_val)

        unified_energy = 0.0
        for cid in range(NUM_CLIENTS):
            denom = float(E_max_dict[cid]) if float(E_max_dict[cid]) > 0 else 1.0
            unified_energy = max(unified_energy, cum_energy_per_client[cid] / denom)

        avg_staleness = float(np.mean([c.tau_k for c in clients])) if clients else 0.0

        # time since run start (seconds)
        t_sec = float(time.time() - run_start)

        # accuracy only at eval points
        acc_str = ""
        if ((t + 1) % EVAL_EVERY == 0) or (t == NUM_ROUNDS - 1):
            acc = evaluate_model(server.global_model, test_loader, DEVICE)
            final_acc = acc
            acc_str = f"{acc:.6f}"

        # write a row EVERY round (time-indexed)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["run_id", "time_sec", "round", "accuracy", "cumulative_energy", "selected_count", "avg_staleness"],
            )
            writer.writerow(
                {
                    "run_id": run_id,
                    "time_sec": f"{t_sec:.6f}",
                    "round": t + 1,
                    "accuracy": acc_str,
                    "cumulative_energy": f"{unified_energy:.6f}",
                    "selected_count": str(selected_count),
                    "avg_staleness": f"{avg_staleness:.6f}",
                }
            )

    return final_acc, selection_counts


# -------------------------
# Plotting vs time
# Strategy:
# - For each V, we aggregate multiple runs by binning in time.
# - We create a common time grid from 0..min(max_time over runs for that V) with N_GRID points.
# - For each run, we "step-hold" the last observed value at or before each grid time.
# - Then average across runs at each grid time.
# -------------------------
def _load_runs_time_series(csv_path: str):
    """
    Returns dict run_id -> list of rows sorted by time:
      row = (time_sec, accuracy_or_None, cumulative_energy, selected_count, avg_staleness)
    accuracy may be None when not logged for that round.
    """
    runs = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = int(row["run_id"])
            t_sec = float(row["time_sec"])
            acc = None
            if row["accuracy"] != "":
                acc = float(row["accuracy"])
            energy = float(row["cumulative_energy"])
            sel = float(row["selected_count"])
            stl = float(row["avg_staleness"])
            runs.setdefault(rid, []).append((t_sec, acc, energy, sel, stl))

    for rid in runs:
        runs[rid].sort(key=lambda x: x[0])
    return runs


def _step_value_at_or_before(times, values, query_t):
    """
    times: increasing list
    values: same length
    returns last value where time <= query_t, else first value.
    """
    if not times:
        return 0.0
    # quick exits
    if query_t <= times[0]:
        return values[0]
    if query_t >= times[-1]:
        return values[-1]
    # binary search
    lo, hi = 0, len(times) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if times[mid] <= query_t:
            lo = mid + 1
        else:
            hi = mid - 1
    return values[hi]


def plot_results_time_based():
    plt.figure(figsize=(15, 10))

    # Build per-V aggregated time grids
    N_GRID = 200  # number of points on time axis per V (adjust if needed)

    # ---- Plot 1: accuracy vs time ----
    plt.subplot(2, 2, 1)
    for v in V_VALUES:
        v_dir = os.path.join(RESULT_ROOT, safe_v_dir_name(v))
        csv_path = os.path.join(v_dir, "time_metrics.csv")
        if not os.path.exists(csv_path):
            continue

        runs = _load_runs_time_series(csv_path)

        # optional: filter runs by final accuracy threshold
        if PLOT_FINAL_ACC_THRESHOLD is not None:
            keep = {}
            for rid, rows in runs.items():
                # last non-empty accuracy in the run
                last_acc = None
                for (_, acc, _, _, _) in rows:
                    if acc is not None:
                        last_acc = acc
                if last_acc is not None and last_acc >= PLOT_FINAL_ACC_THRESHOLD:
                    keep[rid] = rows
            runs = keep

        if not runs:
            continue

        # For accuracy, we want a step series even though it's sparse.
        # We'll forward-fill the last known accuracy (starting from round0 accuracy which exists).
        max_times = [rows[-1][0] for rows in runs.values()]
        t_max = min(max_times)  # common horizon across runs for averaging
        grid = np.linspace(0.0, max(t_max, 1e-9), N_GRID)

        acc_mat = []
        for rid, rows in runs.items():
            times = [r[0] for r in rows]
            # build "filled" accuracy values per row: last known acc
            filled_acc = []
            last = None
            for (_, acc, _, _, _) in rows:
                if acc is not None:
                    last = acc
                if last is None:
                    last = 0.0
                filled_acc.append(last)
            vals = [_step_value_at_or_before(times, filled_acc, t) for t in grid]
            acc_mat.append(vals)

        avg_acc = np.mean(np.array(acc_mat), axis=0)
        plt.plot(grid, avg_acc, label=f"V={v}")

    title = "Test Accuracy vs Wall-Clock Time"
    if PLOT_FINAL_ACC_THRESHOLD is not None:
        title += f" (Only runs with final accuracy >= {PLOT_FINAL_ACC_THRESHOLD}%)"
    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    # ---- Plot 2: unified cumulative energy vs time ----
    plt.subplot(2, 2, 2)
    for v in V_VALUES:
        v_dir = os.path.join(RESULT_ROOT, safe_v_dir_name(v))
        csv_path = os.path.join(v_dir, "time_metrics.csv")
        if not os.path.exists(csv_path):
            continue

        runs = _load_runs_time_series(csv_path)
        if not runs:
            continue

        max_times = [rows[-1][0] for rows in runs.values()]
        t_max = min(max_times)
        grid = np.linspace(0.0, max(t_max, 1e-9), N_GRID)

        mats = []
        for rid, rows in runs.items():
            times = [r[0] for r in rows]
            energies = [r[2] for r in rows]
            vals = [_step_value_at_or_before(times, energies, t) for t in grid]
            mats.append(vals)

        avg_energy = np.mean(np.array(mats), axis=0)
        plt.plot(grid, avg_energy, label=f"V={v}")

    plt.title("Unified Cumulative Energy vs Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Max Normalized Energy")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.1)

    # ---- Plot 3: average selection fraction vs time ----
    plt.subplot(2, 2, 3)
    for v in V_VALUES:
        v_dir = os.path.join(RESULT_ROOT, safe_v_dir_name(v))
        csv_path = os.path.join(v_dir, "time_metrics.csv")
        if not os.path.exists(csv_path):
            continue

        runs = _load_runs_time_series(csv_path)
        if not runs:
            continue

        max_times = [rows[-1][0] for rows in runs.values()]
        t_max = min(max_times)
        grid = np.linspace(0.0, max(t_max, 1e-9), N_GRID)

        mats = []
        for rid, rows in runs.items():
            times = [r[0] for r in rows]
            sel_frac = [float(r[3]) / float(NUM_CLIENTS) for r in rows]
            vals = [_step_value_at_or_before(times, sel_frac, t) for t in grid]
            mats.append(vals)

        avg_sel = np.mean(np.array(mats), axis=0)
        plt.plot(grid, avg_sel, label=f"V={v}")

    plt.title("Average Client Selection Fraction vs Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Selection Fraction")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.1)

    # ---- Plot 4: average staleness vs time ----
    plt.subplot(2, 2, 4)
    for v in V_VALUES:
        v_dir = os.path.join(RESULT_ROOT, safe_v_dir_name(v))
        csv_path = os.path.join(v_dir, "time_metrics.csv")
        if not os.path.exists(csv_path):
            continue

        runs = _load_runs_time_series(csv_path)
        if not runs:
            continue

        max_times = [rows[-1][0] for rows in runs.values()]
        t_max = min(max_times)
        grid = np.linspace(0.0, max(t_max, 1e-9), N_GRID)

        mats = []
        for rid, rows in runs.items():
            times = [r[0] for r in rows]
            stl = [r[4] for r in rows]
            vals = [_step_value_at_or_before(times, stl, t) for t in grid]
            mats.append(vals)

        avg_stl = np.mean(np.array(mats), axis=0)
        plt.plot(grid, avg_stl, label=f"V={v}")

    plt.title("Average Client Staleness vs Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Staleness (rounds)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    os.makedirs(RESULT_ROOT, exist_ok=True)
    out_path = os.path.join(RESULT_ROOT, "comparison_plots_time_based.png")
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
    logger = logging.getLogger("V_time_based")

    os.makedirs(RESULT_ROOT, exist_ok=True)

    # Load data once
    train_dataset, test_dataset = load_cifar10()
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    logger.info(f"Device: {DEVICE}")
    logger.info(f"NUM_ROUNDS={NUM_ROUNDS} | N_RUNS_PER_V={N_RUNS_PER_V} | BATCH_SIZE={BATCH_SIZE}")
    logger.info(f"Fixed params: SIGMA_N={SIGMA_N} | T_MAX={T_MAX} | ETA={ETA}")

    for v in V_VALUES:
        v_dir = os.path.join(RESULT_ROOT, safe_v_dir_name(v))
        os.makedirs(v_dir, exist_ok=True)

        time_csv = os.path.join(v_dir, "time_metrics.csv")
        frac_csv = os.path.join(v_dir, "client_selection_fractions.csv")

        # Write header fresh
        with open(time_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["run_id", "time_sec", "round", "accuracy", "cumulative_energy", "selected_count", "avg_staleness"],
            )
            writer.writeheader()

        total_selection_counts = {cid: 0 for cid in range(NUM_CLIENTS)}
        final_accs = []

        logger.info(f"=== V={v} ===")
        for run_id in range(N_RUNS_PER_V):
            seed = 10000 + int(100 * float(v)) + run_id
            t0 = time.time()

            final_acc, sel_counts = run_one_time_indexed(
                v_value=v,
                run_id=run_id,
                train_dataset=train_dataset,
                test_loader=test_loader,
                csv_path=time_csv,
                seed=seed,
            )

            dt = time.time() - t0
            final_accs.append(final_acc)
            for cid in range(NUM_CLIENTS):
                total_selection_counts[cid] += int(sel_counts.get(cid, 0))

            logger.info(f"  Run {run_id}/{N_RUNS_PER_V-1} done | final_acc={final_acc:.2f}% | wall={dt/60:.1f} min")

        # Save per-client selection fractions (still round-based denominator)
        denom = float(N_RUNS_PER_V * NUM_ROUNDS)
        with open(frac_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["client_id", "avg_selection_fraction"])
            writer.writeheader()
            for cid in range(NUM_CLIENTS):
                frac = float(total_selection_counts[cid]) / denom
                writer.writerow({"client_id": cid, "avg_selection_fraction": f"{frac:.6f}"})

        logger.info(f"  Saved: {time_csv}")
        logger.info(f"  Saved: {frac_csv}")
        logger.info(f"  Avg final acc over runs: {np.mean(final_accs):.2f}%")

    print("\nAll V experiments completed.")


if __name__ == "__main__":
    main()
    plot_results_time_based()

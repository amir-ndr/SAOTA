import os
import csv
import time
import random
import logging
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from client import Client
from server import Server  # your SAOTA server
from server_dynamic import ServerDynamicBaseline  # baseline above
from model_cifar import CNNCifar10
from model import CNNMnist
from dataloader import load_cifar10, partition_cifar10_dirichlet, load_mnist, partition_mnist_dirichlet


# -------------------------
# Config
# -------------------------
OUT_DIR = "Comparison_time_based"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLIENTS = 10
NUM_ROUNDS = 300
BATCH_SIZE = 32
EVAL_EVERY = 5

# Shared system params (your SAOTA best-like values)
SIGMA_N = 6.287625692659408e-06
T_MAX = 2814.156090700375
ETA = 0.053
BISECTION_TOL = 1e-6

# SAOTA-specific
SAOTA_V = 77.23

# Baseline-specific
BASELINE_V = 77.23       # keep same V for fair penalty scaling
BASELINE_P_FIXED = 1.0   # fixed transmit power baseline (can sweep later)

# Client params
MU_K = 1e-27
C_CYCLES_PER_SAMPLE = 1e6
DIRICHLET_ALPHA = 0.2

E_BUDGET_LOW = 26.37
E_BUDGET_HIGH = 33.59


# -------------------------
# Utilities
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


def make_energy_budgets(num_clients: int) -> dict:
    return {cid: float(np.random.uniform(E_BUDGET_LOW, E_BUDGET_HIGH)) for cid in range(num_clients)}


def build_clients(train_dataset, client_data_map, base_model):
    clients = []
    for cid in range(NUM_CLIENTS):
        indices = client_data_map.get(cid, [])
        if len(indices) == 0:
            indices = [0]
        client = Client(
            client_id=cid,
            data_indices=indices,
            model=base_model,  # deep-copied inside client
            fk=np.random.uniform(1e9, 2e9),
            mu_k=MU_K,
            P_max=3.0 + np.random.rand(),
            C=C_CYCLES_PER_SAMPLE,
            Ak=BATCH_SIZE,
            train_dataset=train_dataset,
            device=DEVICE,
            seed=cid,
        )
        clients.append(client)
    return clients


def unified_cum_energy(cum_energy_per_client: dict, E_max_dict: dict) -> float:
    mx = 0.0
    for cid, e in cum_energy_per_client.items():
        denom = float(E_max_dict[cid]) if float(E_max_dict[cid]) > 0 else 1.0
        mx = max(mx, float(e) / denom)
    return mx


# -------------------------
# Run one method and log time-based metrics
# -------------------------
def run_method(method_name: str, server_obj, clients, E_max_dict, test_loader):
    """
    Returns:
      times_eval: list of wall-clock seconds from start (only at eval points)
      acc_eval:   list of accuracies at those times
      times_round: list of wall-clock seconds from start for each round end
      unified_energy_round: list unified energy at each round end (len = NUM_ROUNDS+1)
      avg_staleness_round: list avg staleness at each round end (len = NUM_ROUNDS+1)
      selected_count_round: list selected count at each round end (len = NUM_ROUNDS+1)
      per_client_cum_energy: dict cid -> total energy
    """
    t0 = time.time()

    # round 0 metrics
    acc0 = evaluate_model(server_obj.global_model, test_loader, DEVICE)
    times_eval = [0.0]
    acc_eval = [acc0]

    times_round = [0.0]
    selected_count_round = [0]
    avg_staleness_round = [float(np.mean([c.tau_k for c in clients]))]
    unified_energy_round = [0.0]

    # track cumulative energy from server.per_round_energy (same style for both)
    cum_energy_per_client = {cid: 0.0 for cid in E_max_dict.keys()}

    for t in range(NUM_ROUNDS):
        selected, power_alloc, D_t = server_obj.run_round(t)

        # wall time (seconds) at end of round
        tr = time.time() - t0
        times_round.append(tr)

        selected_ids = [c.client_id for c in selected]
        selected_count_round.append(len(selected_ids))

        # energy update
        round_energy_dict = server_obj.per_round_energy[-1] if len(server_obj.per_round_energy) > 0 else {}
        for cid, e in round_energy_dict.items():
            cum_energy_per_client[int(cid)] += float(e)

        unified_energy_round.append(unified_cum_energy(cum_energy_per_client, E_max_dict))

        avg_staleness_round.append(float(np.mean([c.tau_k for c in clients])))

        # eval
        if ((t + 1) % EVAL_EVERY == 0) or (t == NUM_ROUNDS - 1):
            acc = evaluate_model(server_obj.global_model, test_loader, DEVICE)
            times_eval.append(time.time() - t0)
            acc_eval.append(acc)

    return (
        times_eval,
        acc_eval,
        times_round,
        unified_energy_round,
        avg_staleness_round,
        selected_count_round,
        cum_energy_per_client,
    )


def save_csv(method_name: str, times_round, unified_energy_round, avg_staleness_round, selected_count_round,
             times_eval, acc_eval, cum_energy_per_client):
    method_dir = os.path.join(OUT_DIR, method_name)
    os.makedirs(method_dir, exist_ok=True)

    # per-round time series
    with open(os.path.join(method_dir, "round_metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["round", "wall_time_sec", "unified_cumulative_energy", "avg_staleness", "selected_count"],
        )
        writer.writeheader()
        for r in range(len(times_round)):
            writer.writerow(
                {
                    "round": r,
                    "wall_time_sec": f"{times_round[r]:.6f}",
                    "unified_cumulative_energy": f"{unified_energy_round[r]:.6f}",
                    "avg_staleness": f"{avg_staleness_round[r]:.6f}",
                    "selected_count": int(selected_count_round[r]),
                }
            )

    # eval points
    with open(os.path.join(method_dir, "eval_metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["eval_idx", "wall_time_sec", "accuracy"])
        writer.writeheader()
        for i in range(len(times_eval)):
            writer.writerow({"eval_idx": i, "wall_time_sec": f"{times_eval[i]:.6f}", "accuracy": f"{acc_eval[i]:.6f}"})

    # per-client energy
    with open(os.path.join(method_dir, "client_energy.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["client_id", "total_energy"])
        writer.writeheader()
        for cid, e in cum_energy_per_client.items():
            writer.writerow({"client_id": int(cid), "total_energy": f"{float(e):.6f}"})


def plot_comparison():
    methods = ["SAOTA", "BASELINE"]
    plt.figure(figsize=(15, 10))

    # -------- (1) Accuracy vs time --------
    plt.subplot(2, 2, 1)
    for m in methods:
        eval_path = os.path.join(OUT_DIR, m, "eval_metrics.csv")
        if not os.path.exists(eval_path):
            continue
        tvals, avals = [], []
        with open(eval_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tvals.append(float(row["wall_time_sec"]))
                avals.append(float(row["accuracy"]))
        plt.plot(tvals, avals, "o-", label=m)
    plt.title("Accuracy vs Elapsed Time")
    plt.xlabel("Elapsed Time (s)")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.legend()

    # -------- (2) Unified cumulative energy vs time --------
    plt.subplot(2, 2, 2)
    for m in methods:
        round_path = os.path.join(OUT_DIR, m, "round_metrics.csv")
        if not os.path.exists(round_path):
            continue
        tvals, evals = [], []
        with open(round_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tvals.append(float(row["wall_time_sec"]))
                evals.append(float(row["unified_cumulative_energy"]))
        plt.plot(tvals, evals, label=m)
    plt.title("Unified Cumulative Energy vs Elapsed Time")
    plt.xlabel("Elapsed Time (s)")
    plt.ylabel("Max Normalized Energy")
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.legend()

    # -------- (3) Selection fraction vs time --------
    plt.subplot(2, 2, 3)
    for m in methods:
        round_path = os.path.join(OUT_DIR, m, "round_metrics.csv")
        if not os.path.exists(round_path):
            continue
        tvals, fracs = [], []
        with open(round_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tvals.append(float(row["wall_time_sec"]))
                fracs.append(float(row["selected_count"]) / float(NUM_CLIENTS))
        plt.plot(tvals, fracs, label=m)
    plt.title("Selection Fraction vs Elapsed Time")
    plt.xlabel("Elapsed Time (s)")
    plt.ylabel("Selected/Clients (0-1)")
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.legend()

    # -------- (4) Avg staleness vs time --------
    plt.subplot(2, 2, 4)
    for m in methods:
        round_path = os.path.join(OUT_DIR, m, "round_metrics.csv")
        if not os.path.exists(round_path):
            continue
        tvals, svals = [], []
        with open(round_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tvals.append(float(row["wall_time_sec"]))
                svals.append(float(row["avg_staleness"]))
        plt.plot(tvals, svals, label=m)
    plt.title("Average Staleness vs Elapsed Time")
    plt.xlabel("Elapsed Time (s)")
    plt.ylabel("Staleness (rounds)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "comparison_time_based_2x2.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Saved] {out_path}")



def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("Compare")

    set_seed(123)

    # Data
    train_dataset, test_dataset = load_mnist()
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Non-IID split (same for both)
    client_data_map = partition_mnist_dirichlet(train_dataset, NUM_CLIENTS, alpha=DIRICHLET_ALPHA)

    # Shared budgets (same for both)
    E_max_dict = make_energy_budgets(NUM_CLIENTS)

    # ---------------- SAOTA ----------------
    logger.info("=== Running SAOTA ===")
    base_model_saota = CNNMnist().to(DEVICE)
    clients_saota = build_clients(train_dataset, client_data_map, base_model_saota)

    global_model_saota = CNNMnist().to(DEVICE)
    saota_server = Server(
        global_model=global_model_saota,
        clients=clients_saota,
        V=float(SAOTA_V),
        sigma_n=float(SIGMA_N),
        T_max=float(T_MAX),
        E_max=copy.deepcopy(E_max_dict),
        T_total_rounds=int(NUM_ROUNDS),
        eta=float(ETA),
        device=DEVICE,
        bisection_tol=float(BISECTION_TOL),
    )

    saota_res = run_method("SAOTA", saota_server, clients_saota, E_max_dict, test_loader)
    save_csv("SAOTA", *saota_res)

    # ---------------- Baseline ----------------
    logger.info("=== Running BASELINE ===")
    base_model_base = CNNMnist().to(DEVICE)
    set_seed(123)
    clients_base = build_clients(train_dataset, client_data_map, base_model_base)

    global_model_base = CNNMnist().to(DEVICE)
    baseline_server = ServerDynamicBaseline(
        global_model=global_model_base,
        clients=clients_base,
        V=float(BASELINE_V),
        sigma_n=float(SIGMA_N),
        T_max=float(T_MAX),
        E_max=copy.deepcopy(E_max_dict),
        T_total_rounds=int(NUM_ROUNDS),
        eta=float(ETA),
        device=DEVICE,
        p_fixed=float(BASELINE_P_FIXED),
    )

    base_res = run_method("BASELINE", baseline_server, clients_base, E_max_dict, test_loader)
    save_csv("BASELINE", *base_res)

    # Plot
    plot_comparison()
    logger.info("Done.")


if __name__ == "__main__":
    main()

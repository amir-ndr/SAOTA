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
from server import Server  # SAOTA server
from model import CNNMnist
from dataloader import load_mnist, partition_mnist_dirichlet
from client_safa import ClientSAFA
from server_safa import ServerSAFA
from clientEAD import ClientEAD
from serverEAD import ServerDynamicEAD


# -------------------------
# Config
# -------------------------
OUT_DIR = "Comparison_time_based"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLIENTS = 10
NUM_ROUNDS = 500
BATCH_SIZE = 32
EVAL_EVERY = 5

# Shared system params
SIGMA_N = 6.287625692659408e-06
T_MAX = 100
ETA = 0.1
BISECTION_TOL = 1e-6

# SAOTA-specific
SAOTA_V = 15

# EAD-specific
EAD_V = 15
EAD_P_FIXED = 1.0

# Client params
MU_K = 1e-27
C_CYCLES_PER_SAMPLE = 1e6
DIRICHLET_ALPHA = 0.2

E_BUDGET_LOW = 26.37
E_BUDGET_HIGH = 33.59

SAFA_C = 0.2
SAFA_TAU = 2
SAFA_TLIM = 100.0       # seconds (round deadline)
SAFA_LOCAL_EPOCHS = 1
SAFA_LR = 0.05
CRASH_PROB = 0.1        # example



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

def build_clients_ead(train_dataset, client_data_map, base_model):
    clients = []
    for cid in range(NUM_CLIENTS):
        idxs = client_data_map.get(cid, [])
        if len(idxs) == 0:
            idxs = [0]
        clients.append(
            ClientEAD(
                client_id=cid,
                data_indices=idxs,
                model=base_model,
                train_dataset=train_dataset,
                batch_size=BATCH_SIZE,
                e_per_sample=1e-8,   # set this
                device=DEVICE,
                seed=cid,
            )
        )
    return clients


def build_clients_safa(train_dataset, client_data_map, base_model):
    clients = []
    for cid in range(NUM_CLIENTS):
        indices = client_data_map.get(cid, [])
        if len(indices) == 0:
            indices = [0]

        # paper uses exp distribution for performance; you can mimic it:
        perf = float(np.random.exponential(scale=1.0) + 1e-6)

        clients.append(
            ClientSAFA(
                client_id=cid,
                data_indices=indices,
                model=base_model,
                train_dataset=train_dataset,
                batch_size=BATCH_SIZE,
                local_epochs=SAFA_LOCAL_EPOCHS,
                lr=SAFA_LR,
                device=DEVICE,
                perf_batches_per_sec=perf,
                crash_prob=CRASH_PROB,
                seed=cid,
            )
        )
    return clients



def unified_cum_energy(cum_energy_per_client: dict, E_max_dict: dict) -> float:
    # max_k (E_k_so_far / Emax_k)
    mx = 0.0
    for cid, e in cum_energy_per_client.items():
        denom = float(E_max_dict[cid]) if float(E_max_dict[cid]) > 0 else 1.0
        mx = max(mx, float(e) / denom)
    return mx


# -------------------------
# Run one method and log SIM-time metrics
# -------------------------
def run_method(method_name: str, server_obj, clients, E_max_dict, test_loader):
    """
    Returns:
      eval_sim_t: simulated time (sec) at eval points
      eval_wall_t: wall time (sec) at eval points
      acc_eval: accuracy at eval points

      round_sim_t: simulated time (sec) after each round (len NUM_ROUNDS+1)
      round_wall_t: wall time (sec) after each round (len NUM_ROUNDS+1)

      sys_energy_round: cumulative SYSTEM energy (J) after each round
      unified_energy_round: unified cumulative energy after each round (max normalized)
      avg_staleness_round: avg staleness after each round
      selected_count_round: selected count after each round

      cum_energy_per_client: dict cid -> total energy (J)
    """
    t0_wall = time.time()
    cum_sim_time = 0.0

    # --- Round 0 (before any training) ---
    acc0 = evaluate_model(server_obj.global_model, test_loader, DEVICE)

    eval_sim_t = [0.0]
    eval_wall_t = [0.0]
    acc_eval = [acc0]

    round_sim_t = [0.0]
    round_wall_t = [0.0]
    selected_count_round = [0]
    avg_staleness_round = [float(np.mean([c.tau_k for c in clients])) if clients else 0.0]

    cum_energy_per_client = {cid: 0.0 for cid in E_max_dict.keys()}
    sys_energy_round = [0.0]       # total Joules so far
    unified_energy_round = [0.0]   # max normalized energy so far

    # sanity flags
    warned_no_energy = False
    warned_bad_dt = False

    for t in range(NUM_ROUNDS):
        selected, power_alloc, D_t = server_obj.run_round(t)

        # Update simulated time
        D_t = float(D_t)
        if D_t < 0:
            if not warned_bad_dt:
                print(f"[WARN][{method_name}] D_t < 0 at round {t}: {D_t}. Clipping to 0.")
                warned_bad_dt = True
            D_t = 0.0
        cum_sim_time += D_t

        # Update wall time
        wall_elapsed = time.time() - t0_wall

        # Round time series
        round_sim_t.append(cum_sim_time)
        round_wall_t.append(wall_elapsed)

        selected_ids = [c.client_id for c in selected]
        selected_count_round.append(len(selected_ids))

        # Energy update from server.per_round_energy (must be present for BOTH methods)
        if not getattr(server_obj, "per_round_energy", None):
            if not warned_no_energy:
                print(
                    f"[WARN][{method_name}] server_obj.per_round_energy is missing/empty. "
                    f"Energy curves will be zero. Fix ServerDynamicEAD.update_queues() to append per-round dict."
                )
                warned_no_energy = True
            round_energy_dict = {}
        else:
            round_energy_dict = server_obj.per_round_energy[-1] if server_obj.per_round_energy else {}

        for cid, e in round_energy_dict.items():
            cum_energy_per_client[int(cid)] += float(e)

        # cumulative system energy (J)
        total_energy_so_far = float(sum(cum_energy_per_client.values()))
        sys_energy_round.append(total_energy_so_far)

        # unified energy (max normalized)
        unified_energy_round.append(unified_cum_energy(cum_energy_per_client, E_max_dict))

        # staleness
        avg_staleness_round.append(float(np.mean([c.tau_k for c in clients])) if clients else 0.0)

        # Eval every EVAL_EVERY rounds (and last)
        if ((t + 1) % EVAL_EVERY == 0) or (t == NUM_ROUNDS - 1):
            acc = evaluate_model(server_obj.global_model, test_loader, DEVICE)
            acc_eval.append(acc)
            eval_sim_t.append(cum_sim_time)
            eval_wall_t.append(wall_elapsed)

    # Also produce energy-at-eval aligned to eval points (optional, but nice for analysis)
    # We align by using the evaluation *round index* (every EVAL_EVERY rounds):
    eval_round_indices = [0] + [min(r, NUM_ROUNDS) for r in range(EVAL_EVERY, NUM_ROUNDS + 1, EVAL_EVERY)]
    if eval_round_indices[-1] != NUM_ROUNDS:
        eval_round_indices.append(NUM_ROUNDS)

    # ensure same length as acc_eval
    eval_round_indices = eval_round_indices[: len(acc_eval)]

    sys_energy_at_eval = [sys_energy_round[r] for r in eval_round_indices]
    unified_energy_at_eval = [unified_energy_round[r] for r in eval_round_indices]

    return (
        eval_sim_t, eval_wall_t, acc_eval,
        round_sim_t, round_wall_t,
        sys_energy_round, unified_energy_round,
        avg_staleness_round, selected_count_round,
        cum_energy_per_client,
        sys_energy_at_eval, unified_energy_at_eval,
    )


def save_csv(
    method_name: str,
    eval_sim_t, eval_wall_t, acc_eval,
    round_sim_t, round_wall_t,
    sys_energy_round, unified_energy_round,
    avg_staleness_round, selected_count_round,
    cum_energy_per_client,
    sys_energy_at_eval, unified_energy_at_eval,
):
    method_dir = os.path.join(OUT_DIR, method_name)
    os.makedirs(method_dir, exist_ok=True)

    # Per-round time series
    with open(os.path.join(method_dir, "round_metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "round",
                "sim_time_sec",
                "wall_time_sec",
                "cumulative_system_energy_j",
                "unified_cumulative_energy",
                "avg_staleness",
                "selected_count",
            ],
        )
        writer.writeheader()
        for r in range(len(round_sim_t)):
            writer.writerow(
                {
                    "round": r,
                    "sim_time_sec": f"{round_sim_t[r]:.6f}",
                    "wall_time_sec": f"{round_wall_t[r]:.6f}",
                    "cumulative_system_energy_j": f"{sys_energy_round[r]:.6f}",
                    "unified_cumulative_energy": f"{unified_energy_round[r]:.6f}",
                    "avg_staleness": f"{avg_staleness_round[r]:.6f}",
                    "selected_count": int(selected_count_round[r]),
                }
            )

    # Eval points
    with open(os.path.join(method_dir, "eval_metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "eval_idx",
                "sim_time_sec",
                "wall_time_sec",
                "accuracy",
                "cumulative_system_energy_j",
                "unified_cumulative_energy",
            ],
        )
        writer.writeheader()
        for i in range(len(acc_eval)):
            writer.writerow(
                {
                    "eval_idx": i,
                    "sim_time_sec": f"{eval_sim_t[i]:.6f}",
                    "wall_time_sec": f"{eval_wall_t[i]:.6f}",
                    "accuracy": f"{acc_eval[i]:.6f}",
                    "cumulative_system_energy_j": f"{sys_energy_at_eval[i]:.6f}",
                    "unified_cumulative_energy": f"{unified_energy_at_eval[i]:.6f}",
                }
            )

    # Per-client energy
    with open(os.path.join(method_dir, "client_energy.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["client_id", "total_energy_j", "energy_budget_j", "ratio"])
        writer.writeheader()
        for cid, e in cum_energy_per_client.items():
            budget = float(E_BUDGET_HIGH)  # fallback
            if cid in cum_energy_per_client:
                budget = float(cum_energy_per_client.keys().__len__())  # not used
            budget = float(e)  # overwritten below correctly
        for cid, e in cum_energy_per_client.items():
            budget = float(0.0)
            # budgets are passed in E_max_dict; we don't have it here, so store energy only.
            # (If you want budget+ratio here, pass E_max_dict into save_csv.)
            writer.writerow({"client_id": int(cid), "total_energy_j": f"{float(e):.6f}", "energy_budget_j": "", "ratio": ""})


def plot_comparison():
    methods = ["SAOTA", "EAD", "SAFA"]
    plt.figure(figsize=(18, 10))

    # -------- (1) Accuracy vs simulated time --------
    plt.subplot(2, 3, 1)
    for m in methods:
        eval_path = os.path.join(OUT_DIR, m, "eval_metrics.csv")
        if not os.path.exists(eval_path):
            continue
        tvals, avals = [], []
        with open(eval_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tvals.append(float(row["sim_time_sec"]))
                avals.append(float(row["accuracy"]))
        plt.plot(tvals, avals, "o-", label=m)
    plt.title("Accuracy vs Simulated Time")
    plt.xlabel("Simulated Time (s) [sum D_t]")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.legend()

    # -------- (2) System cumulative energy (J) vs simulated time --------
    plt.subplot(2, 3, 2)
    for m in methods:
        round_path = os.path.join(OUT_DIR, m, "round_metrics.csv")
        if not os.path.exists(round_path):
            continue
        tvals, eJ = [], []
        with open(round_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tvals.append(float(row["sim_time_sec"]))
                eJ.append(float(row["cumulative_system_energy_j"]))
        plt.plot(tvals, eJ, label=m)
    plt.title("System Cumulative Energy vs Simulated Time")
    plt.xlabel("Simulated Time (s) [sum D_t]")
    plt.ylabel("Energy (J)")
    plt.grid(True)
    plt.legend()

    # -------- (3) Unified energy vs simulated time --------
    plt.subplot(2, 3, 3)
    for m in methods:
        round_path = os.path.join(OUT_DIR, m, "round_metrics.csv")
        if not os.path.exists(round_path):
            continue
        tvals, evals = [], []
        with open(round_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tvals.append(float(row["sim_time_sec"]))
                evals.append(float(row["unified_cumulative_energy"]))
        plt.plot(tvals, evals, label=m)
    plt.title("Unified Cumulative Energy vs Simulated Time")
    plt.xlabel("Simulated Time (s) [sum D_t]")
    plt.ylabel("max_k (E_k/Emax_k)")
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.legend()

    # -------- (4) Selection fraction vs simulated time --------
    plt.subplot(2, 3, 4)
    for m in methods:
        round_path = os.path.join(OUT_DIR, m, "round_metrics.csv")
        if not os.path.exists(round_path):
            continue
        tvals, fracs = [], []
        with open(round_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tvals.append(float(row["sim_time_sec"]))
                fracs.append(float(row["selected_count"]) / float(NUM_CLIENTS))
        plt.plot(tvals, fracs, label=m)
    plt.title("Selection Fraction vs Simulated Time")
    plt.xlabel("Simulated Time (s) [sum D_t]")
    plt.ylabel("Selected/Clients")
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.legend()

    # -------- (5) Avg staleness vs simulated time --------
    plt.subplot(2, 3, 5)
    for m in methods:
        round_path = os.path.join(OUT_DIR, m, "round_metrics.csv")
        if not os.path.exists(round_path):
            continue
        tvals, svals = [], []
        with open(round_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tvals.append(float(row["sim_time_sec"]))
                svals.append(float(row["avg_staleness"]))
        plt.plot(tvals, svals, label=m)
    plt.title("Avg Staleness vs Simulated Time")
    plt.xlabel("Simulated Time (s) [sum D_t]")
    plt.ylabel("Staleness (rounds)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "comparison_sim_time_2x3.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Saved] {out_path}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("Compare")

    # Use identical seeds for fair comparison
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
    set_seed(123)
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

    # ---------------- EAD ----------------
    logger.info("=== Running EAD ===")
    set_seed(123)
    base_model_base = CNNMnist().to(DEVICE)
    clients_base = build_clients_ead(train_dataset, client_data_map, base_model_base)

    global_model_base = CNNMnist().to(DEVICE)
    EAD_server = ServerDynamicEAD(
        global_model=global_model_base,
        clients=clients_base,
        V=float(EAD_V),
        eta=float(ETA),
        T_total_rounds=int(NUM_ROUNDS),
        sigma0=float(SIGMA_N),
        gamma0=float(10.0),
        G2=float(1.0),
        E_bar=copy.deepcopy(E_max_dict),  # your dict is total budget per client
        device=DEVICE,
    )


    base_res = run_method("EAD", EAD_server, clients_base, E_max_dict, test_loader)
    save_csv("EAD", *base_res)

    logger.info("=== Running SAFA ===")
    set_seed(123)
    base_model_safa = CNNMnist().to(DEVICE)
    clients_safa = build_clients_safa(train_dataset, client_data_map, base_model_safa)

    global_model_safa = CNNMnist().to(DEVICE)
    safa_server = ServerSAFA(
        global_model=global_model_safa,
        clients=clients_safa,
        C=SAFA_C,
        tau_lag=SAFA_TAU,
        T_lim=SAFA_TLIM,
        device=DEVICE,
    )

    safa_res = run_method("SAFA", safa_server, clients_safa, E_max_dict, test_loader)
    save_csv("SAFA", *safa_res)

    # Plot
    plot_comparison()
    logger.info("Done.")


if __name__ == "__main__":
    main()
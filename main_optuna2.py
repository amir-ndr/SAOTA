import os
import time
import logging
import random
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

import optuna

# from client_saota_multiple import Client
from client import Client
from server import Server
from model_cifar import CNNCifar10
from model import CNNMnist
from dataloader import load_cifar10, partition_cifar10_dirichlet, load_mnist, partition_mnist_dirichlet


# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic (slower but reproducible)
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


def build_clients(
    num_clients: int,
    train_dataset,
    client_data_map,
    base_model,
    batch_size: int,
    device: str,
    mu_k: float,
    c_cycles_per_sample: float,
    seed_base: int,
    # local_steps: int,
    # lr_local: float,
):
    clients = []
    for cid in range(num_clients):
        indices = client_data_map.get(cid, [])
        if len(indices) == 0:
            indices = [0]  # avoid empty data edge-case

        client = Client(
            client_id=cid,
            data_indices=indices,
            model=base_model,
            fk=float(np.random.uniform(1e9, 2e9)),          # 1-2 GHz
            mu_k=float(mu_k),
            P_max=float(3.0 + np.random.rand()),            # ~[3,4)
            C=float(c_cycles_per_sample),
            Ak=int(batch_size),
            train_dataset=train_dataset,
            device=device,
            seed=int(seed_base + cid),
            # local_steps=int(local_steps),
            # lr_local=float(lr_local),
        )
        clients.append(client)
    return clients


# -------------------------
# Metrics for constraints
# -------------------------
def selection_fraction(server: Server, num_clients: int) -> float:
    """Average over rounds: |S_t|/K."""
    hist = getattr(server, "selected_history", None)
    if not hist:
        return 0.0
    counts = [len(s) for s in hist]
    return float(np.mean(counts) / max(int(num_clients), 1))


def qmax_e_overall(server: Server) -> float:
    """Max Q_e across all clients and all rounds."""
    qhist = getattr(server, "queue_history", None)
    if not qhist:
        qcur = getattr(server, "Q_e", None)
        return float(max(qcur.values())) if qcur else 0.0
    mx = 0.0
    for qdict in qhist:
        if qdict:
            mx = max(mx, float(max(qdict.values())))
    return float(mx)


def avg_energy_per_selected(server: Server) -> float:
    """
    Average energy per selected client (averaged over all selected clients across all rounds).
    Uses server.per_round_energy + server.selected_history.
    """
    sel_hist = getattr(server, "selected_history", None)
    e_hist = getattr(server, "per_round_energy", None)
    if not sel_hist or not e_hist:
        return 0.0

    total_e = 0.0
    total_sel = 0
    T = min(len(sel_hist), len(e_hist))

    for t in range(T):
        S = sel_hist[t]
        if not S:
            continue
        ed = e_hist[t]  # dict {cid: energy}
        for cid in S:
            total_e += float(ed.get(int(cid), 0.0))
        total_sel += len(S)

    if total_sel == 0:
        return 0.0
    return float(total_e / total_sel)


# -------------------------
# Optuna objective
# -------------------------
def objective(trial: optuna.Trial) -> float:
    # -------------------------
    # Search space tuned for CIFAR-10
    # -------------------------
    NUM_ROUNDS = trial.suggest_int("NUM_ROUNDS", 300, 1000)

    # BATCH_SIZE = trial.suggest_categorical("BATCH_SIZE", [64, 128])
    # LOCAL_STEPS = trial.suggest_categorical("LOCAL_STEPS", [3, 5, 8, 10])

    # Global update step (server-side)
    ETA = trial.suggest_float("ETA", 0.01, 0.1, log=True)

    # Local SGD lr (client-side)
    # LR_LOCAL = trial.suggest_float("LR_LOCAL", 0.005, 0.05, log=True)

    # OTA noise should be small for CIFAR-10
    # SIGMA_N = trial.suggest_float("SIGMA_N", 1e-9, 1e-7, log=True)

    # V too large can encourage tiny sets depending on your cost scaling
    V = trial.suggest_float("V", 1e2, 1e6, log=True)

    # Per-round time allowance => total T_MAX
    T_allow = trial.suggest_float("T_allow_per_round", 0.05, 0.5, log=True)
    T_MAX = float(T_allow * NUM_ROUNDS)

    # Per-round energy allowance => total E_max^k = E_allow * NUM_ROUNDS (with spread)
    E_allow = trial.suggest_float("E_allow_per_round", 0.015, 4.0, log=True)
    E_spread = trial.suggest_float("E_spread_frac", 0.0, 0.3)

    # -------------------------
    # Fixed experiment settings
    # -------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_CLIENTS = 10
    DIRICHLET_ALPHA = 0.2

    # Client system params
    MU_K = 1e-27
    C_CYCLES_PER_SAMPLE = 1e6

    # Constraints
    ACC_TARGET = 90.0
    SEL_TARGET = 0.5

    # Seed per-trial
    seed = 1234 + trial.number
    set_seed(seed)

    # -------------------------
    # Data
    # -------------------------
    train_dataset, test_dataset = load_mnist()
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    client_data_map = partition_mnist_dirichlet(train_dataset, NUM_CLIENTS, alpha=DIRICHLET_ALPHA)

    # -------------------------
    # Build clients + server
    # -------------------------
    base_model = CNNMnist().to(DEVICE)
    clients = build_clients(
        num_clients=NUM_CLIENTS,
        train_dataset=train_dataset,
        client_data_map=client_data_map,
        base_model=base_model,
        batch_size=64,
        device=DEVICE,
        mu_k=MU_K,
        c_cycles_per_sample=C_CYCLES_PER_SAMPLE,
        seed_base=seed * 10,
        # local_steps=LOCAL_STEPS,
        # lr_local=LR_LOCAL,
    )

    # Total per-client budgets consistent with queue definition E_max/T
    E_mean_total = float(E_allow * NUM_ROUNDS)
    E_max_dict: Dict[int, float] = {}
    for cid in range(NUM_CLIENTS):
        jitter = float(np.random.uniform(-E_spread, E_spread))
        E_max_dict[cid] = float(E_mean_total * (1.0 + jitter))

    global_model = CNNMnist().to(DEVICE)
    server = Server(
        global_model=global_model,
        clients=clients,
        V=float(V),
        sigma_n=float(1e-7),
        T_max=float(T_MAX),
        E_max=E_max_dict,
        T_total_rounds=int(NUM_ROUNDS),
        eta=float(ETA),
        device=DEVICE,
        bisection_tol=1e-6,
    )

    # -------------------------
    # Training + pruning
    # -------------------------
    eval_every = max(10, NUM_ROUNDS // 10)

    best_acc = evaluate_model(server.global_model, test_loader, DEVICE)
    trial.report(best_acc, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    for t in range(NUM_ROUNDS):
        server.run_round(t)

        if ((t + 1) % eval_every == 0) or (t == NUM_ROUNDS - 1):
            acc = evaluate_model(server.global_model, test_loader, DEVICE)
            best_acc = max(best_acc, acc)

            trial.report(best_acc, step=t + 1)
            if trial.should_prune():
                raise optuna.TrialPruned()

    # -------------------------
    # Constraint metrics
    # -------------------------
    f_sel = selection_fraction(server, NUM_CLIENTS)
    qmax_obs = qmax_e_overall(server)
    Ebar = avg_energy_per_selected(server)
    Emax_mean = float(np.mean(list(E_max_dict.values())))

    # Your approximation:
    # Qmax_e ≈ max(0, f * Ebar * T - Emax_mean)
    qmax_pred = max(0.0, float(f_sel * Ebar * NUM_ROUNDS - Emax_mean))

    # Store in Optuna attributes for debugging
    trial.set_user_attr("best_acc", float(best_acc))
    trial.set_user_attr("avg_selection_fraction", float(f_sel))
    trial.set_user_attr("qmax_obs", float(qmax_obs))
    trial.set_user_attr("qmax_pred", float(qmax_pred))
    trial.set_user_attr("Ebar_selected", float(Ebar))
    trial.set_user_attr("Emax_mean_total", float(Emax_mean))

    # -------------------------
    # Composite objective
    # -------------------------
    score = float(best_acc)

    # Hard-ish constraint: accuracy >= 65
    if best_acc < ACC_TARGET:
        score -= 2000.0 * (ACC_TARGET - best_acc)

    # Hard-ish constraint: avg selection fraction >= 0.30
    if f_sel < SEL_TARGET:
        score -= 2000.0 * (SEL_TARGET - f_sel) * 100.0

    # Soft constraint: observed Qmax shouldn't wildly exceed predicted
    tol = 50.0
    if qmax_obs > (qmax_pred + tol):
        score -= 0.01 * (qmax_obs - (qmax_pred + tol))

    return float(score)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("optuna_saota")

    # Make Optuna quieter
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Pruner for long runs
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=300)

    study = optuna.create_study(direction="maximize", pruner=pruner)

    N_TRIALS = 30
    logger.info(f"Starting Optuna study with {N_TRIALS} trials...")
    start = time.time()

    study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True)

    elapsed = time.time() - start
    best = study.best_trial

    logger.info(f"Done. Elapsed: {elapsed/60:.1f} min")
    print("\n===== BEST TRIAL =====")
    print(f"Best score: {best.value:.6f}")
    print("Best params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    # Print constraint-related user attrs if present
    print("\nBest trial extra metrics (user attrs):")
    for k in ["best_acc", "avg_selection_fraction", "qmax_obs", "qmax_pred", "Ebar_selected", "Emax_mean_total"]:
        if k in best.user_attrs:
            print(f"  {k}: {best.user_attrs[k]}")

    # Save
    os.makedirs("optuna_results", exist_ok=True)
    out_path = os.path.join("optuna_results", "best_params.txt")
    with open(out_path, "w") as f:
        f.write(f"Best score: {best.value:.6f}\n\n")
        f.write("Best params:\n")
        for k, v in best.params.items():
            f.write(f"{k}: {v}\n")
        f.write("\nUser attrs:\n")
        for k, v in best.user_attrs.items():
            f.write(f"{k}: {v}\n")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

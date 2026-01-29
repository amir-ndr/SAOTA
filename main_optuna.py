# main.py
import os
import time
import logging
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

import optuna

from client import Client
from server import Server
from model_cifar import CNNCifar10
from dataloader import load_cifar10, partition_cifar10_dirichlet


# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic (slower but reproducible). Comment out if you want faster.
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
            fk=np.random.uniform(1e9, 2e9),       # 1-2 GHz
            mu_k=mu_k,
            P_max=3.0 + np.random.rand(),         # ~[3,4)
            C=c_cycles_per_sample,
            Ak=batch_size,
            train_dataset=train_dataset,
            device=device,
            seed=seed_base + cid,
        )
        clients.append(client)
    return clients


def build_energy_budgets(num_clients: int, e_min: float, e_max: float) -> dict:
    # Per-client total budgets over NUM_ROUNDS
    return {cid: float(np.random.uniform(e_min, e_max)) for cid in range(num_clients)}


# -------------------------
# Optuna objective
# -------------------------
def objective(trial: optuna.Trial) -> float:
    # ----- search space (as you requested) -----
    NUM_ROUNDS = trial.suggest_int("NUM_ROUNDS", 1000, 10000)

    BATCH_SIZE = trial.suggest_categorical("BATCH_SIZE", [32, 64])

    V = trial.suggest_float("V", 0.1, 1000.0, log=True)
    SIGMA_N = trial.suggest_float("SIGMA_N", 1e-6, 1e-3, log=True)
    T_MAX = trial.suggest_float("T_MAX", 500.0, 10000.0)
    ETA = trial.suggest_float("ETA", 0.005, 0.1, log=True)

    E_min = trial.suggest_float("E_min", 5.0, 30.0)
    E_max = trial.suggest_float("E_max", E_min + 3.0, 50.0)

    # ----- fixed experiment settings -----
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_CLIENTS = 10
    DIRICHLET_ALPHA = 0.2

    # Client system params (keep consistent with your earlier setup)
    MU_K = 1e-27
    C_CYCLES_PER_SAMPLE = 1e6

    # Seed for reproducibility per-trial
    seed = 1234 + trial.number
    set_seed(seed)

    # Data
    train_dataset, test_dataset = load_cifar10()
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Non-IID split
    client_data_map = partition_cifar10_dirichlet(train_dataset, NUM_CLIENTS, alpha=DIRICHLET_ALPHA)

    # Build clients & server
    base_model = CNNCifar10().to(DEVICE)
    clients = build_clients(
        num_clients=NUM_CLIENTS,
        train_dataset=train_dataset,
        client_data_map=client_data_map,
        base_model=base_model,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        mu_k=MU_K,
        c_cycles_per_sample=C_CYCLES_PER_SAMPLE,
        seed_base=seed * 10,
    )

    E_max_dict = build_energy_budgets(NUM_CLIENTS, E_min, E_max)

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
        bisection_tol=1e-6,
    )

    # Training loop with intermediate evaluation for pruning
    # Evaluate ~20 times per run (plus initial), so optuna can prune.
    eval_every = max(10, NUM_ROUNDS // 20)

    best_acc = evaluate_model(server.global_model, test_loader, DEVICE)
    trial.report(best_acc, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    for t in range(NUM_ROUNDS):
        server.run_round(t)

        # intermediate eval
        if ((t + 1) % eval_every == 0) or (t == NUM_ROUNDS - 1):
            acc = evaluate_model(server.global_model, test_loader, DEVICE)
            if acc > best_acc:
                best_acc = acc

            # Report for pruning (step uses iteration index)
            trial.report(best_acc, step=t + 1)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return float(best_acc)  # maximize best achieved accuracy


def main():
    # Logging (quiet-ish)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("optuna_saota")

    # Reduce Optuna verbosity if you want:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Pruner helps a lot because NUM_ROUNDS can be large
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=200)

    study = optuna.create_study(direction="maximize", pruner=pruner)

    # You can change n_trials to what you want.
    N_TRIALS = 30

    logger.info(f"Starting Optuna study with {N_TRIALS} trials...")
    start = time.time()
    study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True)
    elapsed = time.time() - start

    best = study.best_trial
    logger.info(f"Done. Elapsed: {elapsed/60:.1f} min")
    print("\n===== BEST TRIAL =====")
    print(f"Best accuracy: {best.value:.4f}")
    print("Best params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    # Optional: save results
    os.makedirs("optuna_results", exist_ok=True)
    with open("optuna_results/best_params.txt", "w") as f:
        f.write(f"Best accuracy: {best.value:.6f}\n")
        for k, v in best.params.items():
            f.write(f"{k}: {v}\n")
    print("\nSaved: optuna_results/best_params.txt")


if __name__ == "__main__":
    main()

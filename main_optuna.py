import optuna
import torch
import numpy as np
from torch.utils.data import DataLoader
from client import Client
from server import Server
from model import CNNMnist
from dataloader import load_mnist, partition_mnist_noniid
import time

def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def objective(trial):
    # Fixed parameters
    NUM_CLIENTS = 10
    NUM_ROUNDS = trial.suggest_int("NUM_ROUNDS", 100, 1000)
    BATCH_SIZE = trial.suggest_categorical("BATCH_SIZE", [16, 32, 64])
    LOCAL_EPOCHS = 1
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {DEVICE}")
    
    # Load and partition dataset
    train_dataset, test_dataset = load_mnist()
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    client_data_map = partition_mnist_noniid(train_dataset, NUM_CLIENTS)
    
    # Hyperparameters to tune
    V = trial.suggest_float("V", 5.0, 50.0, log=True)
    sigma_n = trial.suggest_float("sigma_n", 0.001, 0.1, log=True)
    tau_cm = trial.suggest_float("tau_cm", 0.001, 0.05)
    T_max = trial.suggest_int("T_max", 100, 1500)
    # learning_rate = trial.suggest_float("learning_rate", 0.01, 0.12, log=True)
    
    # Energy budget range
    E_min = trial.suggest_float("E_min", 5.0, 30.0)
    E_max = trial.suggest_float("E_max", E_min + 3.0, 50.0)
    
    # Initialize clients with heterogeneous parameters
    clients = []
    for cid in range(NUM_CLIENTS):
        indices = client_data_map[cid]
        if len(indices) == 0:
            indices = [0]  # Ensure at least one sample
        
        # Per-client heterogeneity
        client_fk = np.random.uniform(1e9, 2e9)
        client_mu_k = np.random.uniform(1e-28, 1e-26)
        client_P_max = np.random.uniform(1.0, 2.0)
        
        client = Client(
            client_id=cid,
            data_indices=indices,
            model=CNNMnist(),
            fk=client_fk,
            mu_k=client_mu_k,
            P_max=client_P_max,
            C=1e6,
            Ak=BATCH_SIZE,
            train_dataset=train_dataset,
            device=DEVICE,
            local_epochs=LOCAL_EPOCHS
        )
        clients.append(client)
    
    # Energy budgets
    E_max_dict = {cid: np.random.uniform(E_min, E_max) for cid in range(NUM_CLIENTS)}
    
    # Initialize global model and server
    global_model = CNNMnist().to(DEVICE)
    server = Server(
        global_model=global_model,
        clients=clients,
        V=V,
        sigma_n=sigma_n,
        tau_cm=tau_cm,
        T_max=T_max,
        E_max=E_max_dict,
        T_total_rounds=NUM_ROUNDS,
        device=DEVICE
    )
    
    # Training loop
    for round_idx in range(NUM_ROUNDS):
        round_start = time.time()
        
        # 1. Select clients and broadcast current model
        selected, power_alloc = server.select_clients()
        selected_ids = [c.client_id for c in selected]
        
        # Broadcast model to selected clients
        server.broadcast_model(selected)
        
        # 2. Compute gradients on selected clients
        comp_times = []
        for client in selected:
            start_comp = time.time()
            client.compute_gradient()
            comp_time = time.time() - start_comp
            comp_times.append(comp_time)
        
        # 3. Reset staleness AFTER computation (as in your previous version)
        for client in selected:
            client.reset_staleness()
        
        # 4. Calculate round duration and aggregate
        max_comp_time = max(comp_times) if selected else 0
        D_t = max_comp_time + server.tau_cm
        
        if selected:
            aggregated = server.aggregate(selected, power_alloc)
            server.update_model(aggregated, round_idx)  # Pass round index for LR decay
        
        # 5. Update queues
        server.update_queues(selected, power_alloc, D_t)
        
        # 6. Update computation time for ALL clients
        for client in clients:
            if client in selected:
                # Reset for next round (new model)
                client.reset_computation()
            else:
                # Progress computation
                client.dt_k = max(0, client.dt_k - D_t)
                client.increment_staleness()
            
        # Report intermediate accuracy for pruning
        if (round_idx + 1) % 10 == 0:
            acc = evaluate_model(server.global_model, test_loader, DEVICE)
            trial.report(acc, step=round_idx)
            
            # Prune if accuracy is poor
            if acc < 10.0 and round_idx > 50:
                raise optuna.TrialPruned()
    
    # Final accuracy evaluation
    final_acc = evaluate_model(server.global_model, test_loader, DEVICE)
    return final_acc

if __name__ == "__main__":
    # Create study with pruning support
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=20)
    )
    
    # Run optimization
    study.optimize(objective, n_trials=300, timeout=3600)
    
    # Print best results
    print("\nBest trial:")
    print(f"  Accuracy: {study.best_value:.2f}%")
    print("  Params:", study.best_params)
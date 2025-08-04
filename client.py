import torch
import numpy as np
import copy
import time
import logging
import random

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Client:
    def __init__(self, client_id, data_indices, model, fk, mu_k, P_max, C, Ak, 
                 train_dataset, device='cpu', local_epochs=1):
        self.client_id = client_id
        self.data_indices = list(data_indices)  # Ensure list type
        self.stale_model = copy.deepcopy(model).to(device)
        self.local_model = copy.deepcopy(model).to(device)
        self.fk = fk
        self.mu_k = mu_k
        self.P_max = P_max
        self.C = C
        self.Ak = Ak
        self.train_dataset = train_dataset
        self.device = device
        self.local_epochs = local_epochs
        
        # State variables
        self.dt_k = self._full_computation_time()
        self.tau_k = 0
        self.last_gradient = None
        self.gradient_norm = 0.0
        self.h_t_k = None
        self.actual_comp_time = 0.0
        self.energy_this_round= 0.0
        self.ready = (self.dt_k <= 0)

        logger.info(f"Client {client_id} initialized | "
                    f"CPU: {fk/1e9:.2f} GHz | "
                    f"Batch: {Ak} samples | "
                    f"Comp Time: {self.dt_k:.4f}s | "
                    f"Data samples: {len(data_indices)}")

    def _full_computation_time(self):
        return (self.C * self.Ak * self.local_epochs) / self.fk

    def update_stale_model(self, model_state_dict):
        self.stale_model.load_state_dict(model_state_dict)
        self.local_model.load_state_dict(model_state_dict)
        logger.debug(f"Client {self.client_id}: Model updated")

    def compute_gradient(self):
        start_time = time.time()
        self.local_model.load_state_dict(self.stale_model.state_dict())
        
        # Handle insufficient data samples
        n_available = len(self.data_indices)
        batch_size = min(self.Ak, n_available)
        
        if n_available == 0:
            logger.warning(f"Client {self.client_id} has no data! Returning zero gradient")
            # Return zero gradient with correct dimension
            self.last_gradient = torch.zeros(self._model_dimension())
            self.gradient_norm = 0.0
            self.actual_comp_time = 0.0
            return True
        
        # Randomly select a mini-batch
        indices = random.sample(self.data_indices, batch_size)
        batch = [self.train_dataset[i] for i in indices]
        images = torch.stack([x[0] for x in batch]).to(self.device)
        labels = torch.tensor([x[1] for x in batch]).to(self.device)
        
        # Forward pass
        self.local_model.zero_grad()
        outputs = self.local_model(images)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        
        # Backward pass with gradient clipping
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=5.0)
        
        # Extract and flatten gradients
        gradients = []
        for param in self.local_model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.detach().view(-1))
        flat_gradient = torch.cat(gradients)
        
        # Store results
        self.last_gradient = flat_gradient
        self.gradient_norm = torch.norm(flat_gradient).item()
        self.actual_comp_time = time.time() - start_time
        
        logger.info(f"Client {self.client_id}: Gradient computed | "
                    f"Samples: {batch_size}/{n_available} | "
                    f"Norm: {self.gradient_norm:.4f} | "
                    f"Comp time: {self.actual_comp_time:.4f}s")
        return True

    def _model_dimension(self):
        """Get model parameter dimension"""
        return sum(p.numel() for p in self.local_model.parameters())

    def reset_computation(self):
        self.dt_k = self._full_computation_time()
        logger.debug(f"Client {self.client_id}: Comp time reset")

    def increment_staleness(self):
        self.tau_k += 1
        logger.debug(f"Client {self.client_id}: Staleness → {self.tau_k}")

    def set_channel_gain(self):
        # More realistic Rayleigh fading with scale=1/sqrt(2)
        magnitude = np.random.rayleigh(scale=1/np.sqrt(2))
        phase = np.random.uniform(0, 2*np.pi)
        self.h_t_k = magnitude * np.exp(1j * phase)
        logger.debug(f"Client {self.client_id}: Channel set | "
                     f"|h|: {abs(self.h_t_k):.4f}")
        return abs(self.h_t_k)
    
    def reset_staleness(self):
        self.tau_k = 0
        logger.debug(f"Client {self.client_id}: Staleness reset")
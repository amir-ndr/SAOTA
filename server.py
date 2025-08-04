import torch
import numpy as np
import copy
import logging
import math

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Server:
    def __init__(self, global_model, clients, V=1.0, sigma_n=0.1, 
                 tau_cm=0.1, T_max=100, E_max=1.0, T_total_rounds=50, 
                 device='cpu'):
        self.global_model = global_model
        self.clients = clients
        self.device = device
        self.V = V
        self.sigma_n = sigma_n
        self.tau_cm = tau_cm
        self.T_max = T_max
        self.energy_this_round = 0.0
        self.T_total_rounds = T_total_rounds
        self.d = self._get_model_dimension()

        self.total_energy_per_round = []  # Total energy per round
        self.cumulative_energy_per_client = {  # Cumulative energy per client
            client.client_id: 0.0 for client in clients
        }
        self.per_round_energy = []
        
        # Virtual queues
        self.Q_e = {client.client_id: 0.0 for client in clients}
        self.Q_time = 0.0
        
        # Energy budgets
        self.E_max = E_max if isinstance(E_max, dict) else \
                    {c.client_id: E_max for c in clients}
        
        # History
        self.selected_history = []
        self.queue_history = []

        logger.info(f"Server initialized | "
                    f"Model dim: {self.d} | "
                    f"V: {V} | "
                    f"Noise: {sigma_n} | "
                    f"Rounds: {T_total_rounds}")

    def _get_model_dimension(self):
        return sum(p.numel() for p in self.global_model.parameters())
    
    def broadcast_model(self, selected_clients):
        global_state = self.global_model.state_dict()
        for client in selected_clients:
            client.update_stale_model(global_state)
        logger.info(f"Broadcast model to {len(selected_clients)} clients")
    
    def select_clients(self):
        epsilon = 1e-8
        
        # Log queue status before selection
        queue_status = ", ".join([f"Client {cid}: {q:.2f}" 
                                 for cid, q in self.Q_e.items()])
        logger.info(f"Pre-selection queues | "
                    f"Time Q: {self.Q_time:.2f} | "
                    f"Energy Qs: {queue_status}")
        
        # Compute scores
        for client in self.clients:
            client.set_channel_gain()
            numerator = abs(client.h_t_k)**2
            # denominator = (self.Q_e[client.client_id] + epsilon) * \
            #              (client.gradient_norm**2 + epsilon) * \
            #              (client.dt_k + epsilon)
            denominator = np.sqrt(self.Q_e[client.client_id] + epsilon) * \
              (client.gradient_norm + epsilon) * \
              (np.log1p(client.dt_k) + epsilon)
            client.score = numerator / denominator
        
        # Log client status
        for client in self.clients:
            logger.info(f"Client {client.client_id} status | "
                         f"Score: {client.score:.4e} | "
                         f"dt_k: {client.dt_k:.4f}s | "
                         f"Q_e: {self.Q_e[client.client_id]:.2f} | "
                         f"|h|: {abs(client.h_t_k):.4f} | "
                         f"Grad norm: {client.gradient_norm:.4f}")
        
        # Sort and select
        sorted_clients = sorted(self.clients, key=lambda c: c.score, reverse=True)
        selected = []
        best_cost = float('inf')
        
        for client in sorted_clients:
            candidate_set = selected + [client]
            cost_k = self._exact_cost(candidate_set)
            if cost_k < best_cost:
                selected.append(client)
                best_cost = cost_k
                logger.debug(f"  Added client {client.client_id} | "
                             f"New cost: {cost_k:.4e}")
            else:
                logger.debug(f"  Stopping selection | "
                             f"Client {client.client_id} would increase cost to {cost_k:.4e}")
                break
        
        # Compute power allocation
        power_alloc = self._compute_power(selected)
        
        logger.info(f"Selected {len(selected)} clients: {[c.client_id for c in selected]}")
        for client in selected:
            logger.info(f"  Client {client.client_id} | "
                        f"Power: {power_alloc.get(client.client_id, 0):.4f} | "
                        f"Score: {client.score:.4e}")
        
        return selected, power_alloc

    def random_selection(self):
        """Select random clients with proper attribute initialization"""
        # Ensure all clients have required attributes set
        for client in self.clients:
            client.set_channel_gain()
            # Use last gradient norm or default to 1.0 if never computed
            if not hasattr(client, 'gradient_norm') or client.gradient_norm == 0:
                client.gradient_norm = 1.0
        
        # Determine random number of clients to select (between 2 and 8)
        n_selected = np.random.randint(2, 9)
        
        # Randomly select clients
        selected = np.random.choice(
            self.clients, 
            size=min(n_selected, len(self.clients)),
            replace=False
        ).tolist()
        
        # Compute power allocation
        power_alloc = self._compute_power(selected)
        
        logger.info(f"Randomly selected {len(selected)} clients: {[c.client_id for c in selected]}")
        for client in selected:
            logger.info(f"  Client {client.client_id} | "
                        f"Power: {power_alloc.get(client.client_id, 0):.4f} | "
                        f"Channel: {abs(client.h_t_k):.4f} | "
                        f"Grad Norm: {client.gradient_norm:.4f}")
        
        return selected, power_alloc

    def _exact_cost(self, candidate_set):
        """Calculate exact drift-plus-penalty cost for candidate set"""
        if not candidate_set:
            return float('inf')
            
        n = len(candidate_set)
        power_alloc = self._compute_power(candidate_set)
        total_power = sum(power_alloc.values())
        
        # Convergence penalty
        conv_penalty = 0
        if total_power > 1e-8:
            # Calculate actual α values
            alphas = {cid: p/total_power for cid, p in power_alloc.items()}
            conv_penalty = self.V * sum(alpha**2 for alpha in alphas.values()) 
            conv_penalty += self.V * self.d * self.sigma_n**2 / total_power**2
        
        # Energy cost
        energy_cost = 0
        for client in candidate_set:
            cid = client.client_id
            # Transmission energy based on allocated power
            E_comm = (power_alloc[cid] * client.gradient_norm / abs(client.h_t_k))**2
            E_comp = client.mu_k * client.fk**2 * client.C * client.Ak * client.local_epochs
            energy_cost += self.Q_e[cid] * (E_comp + E_comm)
        
        # Time penalty
        # D_temp = max(c.dt_k for c in candidate_set) + self.tau_cm
        D_temp = max(c.actual_comp_time for c in candidate_set)
        
        return conv_penalty + energy_cost + self.Q_time * D_temp

    def _compute_power(self, selected):
        """Compute optimal power allocation for selected clients"""
        if not selected:
            return {}
        
        n = len(selected)
        c_values = {}
        
        # Calculate c_k = Q_e[k] * ||g_k||^2 / |h_k|^2
        for client in selected:
            cid = client.client_id
            ck = self.Q_e[cid] * client.gradient_norm**2 / (abs(client.h_t_k)**2 + 1e-8)
            c_values[cid] = max(ck, 1e-8)  # Avoid zero
        
        weights = [1/np.sqrt(c) for c in c_values.values()]
        total_weight = sum(weights)
        
        # Handle near-zero weights
        if total_weight < 1e-8:
            # Equal power allocation fallback
            return {client.client_id: min(0.1, client.P_max) for client in selected}
        
        # Optimize total power
        base = (self.V * self.d * self.sigma_n**2 * total_weight**2) / n
        S_t = base ** 0.25
        
        # Allocate power proportionally
        power_alloc = {}
        for client in selected:
            cid = client.client_id
            pk = (1/np.sqrt(c_values[cid]) / total_weight * S_t)
            power_alloc[cid] = min(pk, client.P_max)
        
        return power_alloc

    def aggregate(self, selected, power_alloc):
        total_power = sum(power_alloc.values())
        if total_power < 1e-8:
            logger.warning("Aggregation failed: total power near zero!")
            return torch.zeros(self.d, device=self.device)
        
        aggregated = torch.zeros(self.d, device=self.device)
        for client in selected:
            cid = client.client_id
            # Apply staleness discount
            staleness_factor = 0.9 ** client.tau_k
            # Simple aggregation without complex precoding
            aggregated += client.last_gradient * power_alloc[cid] * staleness_factor
        
        # Add and scale noise
        noise = torch.randn(self.d, device=self.device) * self.sigma_n
        result = (aggregated + noise) / total_power
        
        logger.info(f"Aggregation complete | "
                    f"Total power: {total_power:.4f} | "
                    f"Noise std: {self.sigma_n} | "
                    f"Update norm: {torch.norm(result).item():.4f}")
        return result

    def update_model(self, update, round_idx):
        """Update global model with decaying learning rate"""
        # Decaying learning rate
        lr = 0.1 * (0.95 ** (round_idx // 10))
        
        with torch.no_grad():
            params = torch.nn.utils.parameters_to_vector(self.global_model.parameters())
            prev_norm = torch.norm(params).item()
            params -= lr * update
            new_norm = torch.norm(params).item()
            torch.nn.utils.vector_to_parameters(params, self.global_model.parameters())
            
            logger.info(f"Model updated | "
                        f"LR: {lr:.4f} | "
                        f"Param change: {prev_norm - new_norm:.4e} | "
                        f"New norm: {new_norm:.4f}")

    def update_queues(self, selected, power_alloc, D_t):
        logger.info(f"Updating queues | "
                    f"Round duration: {D_t:.4f}s | "
                    f"Time Q before: {self.Q_time:.2f}")

        round_energy = 0
        round_client_energy = {}
        
        # Update energy queues
        for client in selected:
            cid = client.client_id
            # Compute actual transmission energy
            E_comm = (power_alloc[cid])**2 * client.gradient_norm**2 / (abs(client.h_t_k)**2 + 1e-8)
            # Use actual computation time
            E_comp = client.mu_k * client.fk**2 * client.C * client.Ak * (client.actual_comp_time * client.fk / (client.C * client.Ak))

            client_energy = E_comm #+ E_comp
            round_energy += client_energy
            self.cumulative_energy_per_client[cid] += client_energy
            round_client_energy[cid] = client_energy

            # Per-round energy budget
            per_round_budget = self.E_max[cid] / self.T_total_rounds
            energy_increment = client_energy - per_round_budget
            
            prev_q = self.Q_e[cid]
            self.Q_e[cid] = max(0, self.Q_e[cid] + energy_increment)
            
            logger.info(f"  Client {cid} energy update | "
                         f"Comp: {E_comp:.4e} J | "
                         f"Comm: {E_comm:.4e} J | "
                         f"ΔQ: {energy_increment:.4e} | "
                         f"Q_e: {prev_q:.2f} → {self.Q_e[cid]:.2f}")

        self.total_energy_per_round.append(round_energy)
        self.energy_this_round = round_energy
        self.per_round_energy.append(round_client_energy)
        
        # Update time queue
        per_round_time_budget = self.T_max / self.T_total_rounds
        time_increment = D_t - per_round_time_budget
        prev_time_q = self.Q_time
        self.Q_time = max(0, self.Q_time + time_increment)
        
        logger.info(f"Time queue update | "
                    f"Δ: {time_increment:.4e} | "
                    f"Q_time: {prev_time_q:.2f} → {self.Q_time:.2f}")
        
        # Record history
        self.selected_history.append([c.client_id for c in selected])
        self.queue_history.append(copy.deepcopy(self.Q_e))
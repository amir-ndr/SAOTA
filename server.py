import torch
import numpy as np
import copy
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Server:
    def __init__(
        self,
        global_model,
        clients,
        V=1.0,
        sigma_n=0.1,
        T_max=100.0,
        E_max=1.0,
        T_total_rounds=50,
        eta=0.1,
        tau_max=None,
        device="cpu",
        bisection_tol=1e-6,
    ):
        self.global_model = global_model.to(device)
        self.clients = clients
        self.device = device

        self.V = float(V)
        self.sigma_n = float(sigma_n)
        self.T_max = float(T_max)
        self.T_total_rounds = int(T_total_rounds)
        self.eta = float(eta)
        self.bisection_tol = float(bisection_tol)
        self.tau_max = int(tau_max) if tau_max is not None else self.T_total_rounds

        self.d = self._get_model_dimension()

        # Virtual queues (Lyapunov framework)
        self.Q_e = {c.client_id: 0.0 for c in clients}
        self.Q_time = 0.0

        # Per-client energy budgets
        self.E_max = E_max if isinstance(E_max, dict) else {c.client_id: float(E_max) for c in clients}

        # History
        self.selected_history = []
        self.queue_history = []
        self.total_energy_per_round = []
        self.per_round_energy = []
        self.cumulative_energy_per_client = {c.client_id: 0.0 for c in clients}

        logger.info(
            f"Server initialized | d={self.d} | V={self.V} | sigma_n={self.sigma_n} | "
            f"T={self.T_total_rounds} | eta={self.eta} | tau_max={self.tau_max}"
        )

    def _get_model_dimension(self):
        return sum(p.numel() for p in self.global_model.parameters())

    # -------------------------
    # Channel model (Rayleigh fading)
    # -------------------------
    def _rayleigh_channel(self):
        """Complex Rayleigh fading: h = (x + j y)/sqrt(2), x,y ~ N(0,1)"""
        x = np.random.randn()
        y = np.random.randn()
        return (x + 1j * y) / np.sqrt(2.0)



    # -------------------------
    # Power allocation via KKT + bisection (P3 from paper)
    # Solve: min V/(Σp)² + Σ Qe_k * (‖g_k‖²/|h_k|²) * p_k²
    # -------------------------
    def _compute_power(self, selected_clients, use_actual_grad=True):
        """
        Solve power allocation subproblem P3 using KKT + bisection.
        
        Args:
            selected_clients: list of selected clients
            use_actual_grad (bool): 
                - True: use grad_sqnorm (actual buffered gradient) for final power allocation
                - False: use last_grad_sqnorm (estimate) for temporary selection evaluation
        
        NOTE: During selection (temporary), grad_sqnorm is 0 because gradients aren't buffered yet.
              After selection, actual buffered gradients are available, so use_actual_grad=True.
        """
        if len(selected_clients) == 0:
            return {}

        eps = 1e-12

        # Compute coefficients: a_k = |h|²/(Qe * ‖g‖²)
        coefficients = {}
        pmax_values = {}

        for client in selected_clients:
            cid = client.client_id
            pmax_values[cid] = float(client.P_max)

            h_abs2 = float(np.abs(client.h_t_k) ** 2) if client.h_t_k is not None else eps

            # Gradient norm selection:
            # - use_actual_grad=True: use current buffered ||g_k^t||^2 (final power, after selection)
            # - use_actual_grad=False: use last_grad_sqnorm (temporary, during selection)
            if use_actual_grad:
                G = max(getattr(client, "grad_sqnorm", 0.0), eps)
            else:
                G = max(getattr(client, "last_grad_sqnorm", 0.0), eps)

            Qe = float(self.Q_e[cid])
            if Qe <= 0.0:
                Qe = eps

            # Coefficient for KKT solution (Eq. pk_interior from paper)
            coefficients[cid] = h_abs2 / (Qe * G + eps)

        # Fixed-point bisection (Eq. S_fixed_point from paper)
        S_high = sum(pmax_values.values())
        if S_high <= eps:
            return {c.client_id: 0.0 for c in selected_clients}

        def rhs(S_val):
            """Right-hand side of fixed-point equation"""
            S3 = max(S_val ** 3, eps)
            total = 0.0
            for cid, a_k in coefficients.items():
                term = self.V * a_k / S3
                total += min(pmax_values[cid], term)
            return total

        # Bisection search
        S_low = eps
        S_star = S_high

        for iteration in range(100):
            S_mid = (S_low + S_high) / 2.0
            if S_mid <= eps:
                S_mid = eps

            r_val = rhs(S_mid)

            if abs(r_val - S_mid) < self.bisection_tol:
                S_star = S_mid
                break
            elif r_val > S_mid:
                S_low = S_mid
            else:
                S_high = S_mid

        S_star = max((S_low + S_high) / 2.0, eps)

        # Compute final powers (Eq. pk_star from paper)
        S3 = S_star ** 3
        power_alloc = {}

        for cid, a_k in coefficients.items():
            p = self.V * a_k / S3
            power_alloc[cid] = min(pmax_values[cid], p)

        logger.info(f"Power allocation: S*={S_star:.6f} | powers={list(power_alloc.values())}")
        return power_alloc

    # -------------------------
    # Drift-plus-penalty cost (Algorithm 1 evaluation)
    # -------------------------
    def _evaluate_dpp_cost(self, candidate_set, power_alloc):
        """
        Evaluate predicted cost for candidate set (Algorithm 1).
        Uses LAST gradient norms as ESTIMATES (unavailable until selection is fixed).
        
        Cost = V/(Σp)² + Σ Qe[E_cmp + p²‖g‖²/|h|²] + Q_time * D_temp
        
        NOTE: This is an estimate because we don't know actual g_k^t until after
        client selection is finalized. For final power allocation, actual buffered
        gradient norms are used in _compute_power().
        """
        if len(candidate_set) == 0:
            return float("inf")

        eps = 1e-12

        # 1. Convergence penalty: V/(Σp)²
        total_power = sum(power_alloc.get(c.client_id, 0.0) for c in candidate_set)
        if total_power <= eps:
            return float("inf")

        conv_penalty = self.V / (total_power ** 2)

        # 2. Energy cost: Σ Qe[E_cmp + E_tx]
        energy_cost = 0.0
        for client in candidate_set:
            cid = client.client_id
            p = float(power_alloc.get(cid, 0.0))
            h_abs2 = float(np.abs(client.h_t_k) ** 2) if client.h_t_k is not None else eps

            # Use LAST gradient norm as ESTIMATE (actual gradient unavailable during selection)
            G_last = max(getattr(client, "last_grad_sqnorm", 0.0), eps)

            # Computation energy (only for selected clients)
            E_cmp = float(client.mu_k * (client.fk ** 2) * client.C * client.Ak)

            # Transmission energy (estimated): p² * ‖g‖² / |h|²
            E_tx = (p ** 2) * (G_last / h_abs2)

            energy_cost += self.Q_e[cid] * (E_cmp + E_tx)

        # 3. Time cost: Q_time * max(d)
        D_temp = max(client.dt_k for client in candidate_set) if candidate_set else 0.0
        time_cost = self.Q_time * D_temp

        return conv_penalty + energy_cost + time_cost

    # -------------------------
    # Greedy selection (Algorithm 1 from paper)
    # -------------------------
    def select_clients(self):
        """
        Algorithm 1: Time-Sorted Greedy Client Selection.
        - Sort by remaining computation time d_k^t (ascending)
        - Greedily add clients if DPP cost decreases
        - Uses LAST gradient norms (no EMA), not priority scores
        
        Steps:
        1. Reset stale clients at tau_max
        2. Assign channels for this round
        3. Sort clients by remaining time d_k^t
        4. Greedy loop: add client if DPP cost improves
        5. Final power allocation with actual gradients
        """
        eps = 1e-12

        # 1. Reset stale clients at tau_max
        for c in self.clients:
            if c.tau_k >= self.tau_max:
                logger.info(f"Client {c.client_id} reached tau_max={self.tau_max}; broadcasting model")
                c.receive_model_and_reset(self.global_model.state_dict())

        # 2. Set channels for this round (Rayleigh fading)
        for c in self.clients:
            h = self._rayleigh_channel()
            c.set_channel(h)

        # 3. Sort clients by ASCENDING remaining computation time d_k^t
        #    Clients with d_k^t = 0 (already finished) come first
        sorted_clients = sorted(self.clients, key=lambda c: c.dt_k)

        # 4. Greedy selection loop (Algorithm 1, Lines 6-14)
        selected = []
        best_cost = float("inf")
        best_power = {}

        for candidate in sorted_clients:
            # Form candidate set S_temp = S_t ∪ {k}
            candidate_set = selected + [candidate]

            # Temporary power allocation using ESTIMATED gradient (last_grad_sqnorm)
            # Because actual gradients not available until after selection is fixed
            temp_powers = self._compute_power(candidate_set, use_actual_grad=False)

            if temp_powers is None or len(temp_powers) == 0:
                continue

            # Evaluate DPP cost (Eq. temp_dpp_cost_time from paper, Algorithm 1 Line 10)
            cost = self._evaluate_dpp_cost(candidate_set, temp_powers)

            logger.info(
                f"Candidate client {candidate.client_id} | "
                f"set size: {len(candidate_set)} | cost: {cost:.6f} | best: {best_cost:.6f}"
            )

            # Accept if cost decreases (Algorithm 1, Line 12-15)
            if cost < best_cost:
                selected = candidate_set
                best_cost = cost
                best_power = temp_powers
                logger.info(f"  → ACCEPTED | new best cost: {cost:.6f}")
            else:
                logger.info(f"  → REJECTED | cost increased")
                break  # Greedy termination (Algorithm 1, Line 16)

        # Safety: ensure at least one client selected
        if len(selected) == 0:
            logger.warning("No clients selected; forcing selection of one client.")
            selected = [self.clients[np.random.randint(len(self.clients))]]
            best_power = self._compute_power(selected, use_actual_grad=False)

        logger.info(
            f"Final selection: {[c.client_id for c in selected]} | "
            f"count: {len(selected)} | final_cost: {best_cost:.6f}"
        )
        return selected, best_power

    # -------------------------
    # OTA aggregation (paper Eq)
    # -------------------------
    def aggregate(self, selected, power_alloc):
        """
        OTA aggregation (simplified model):
        y = Σ (p_k * g_k) where p_k are power allocations from KKT optimization
        update = y / sum(p_k) (normalized by total power)
        
        Note: In true OTA with complex pre-equalization, channel effects cancel out
        in the receive signal, so we use power-weighted averaging here.
        """
        eps = 1e-12
        if len(selected) == 0:
            return torch.zeros(self.d, device=self.device, dtype=torch.float32)

        # Sum weighted gradients
        y = torch.zeros(self.d, device=self.device, dtype=torch.float32)
        total_power = 0.0
        num_gradients = 0
        
        for c in selected:
            cid = c.client_id
            p = float(power_alloc.get(cid, 0.0))
            if p <= eps:
                continue

            g = getattr(c, "gradient_buffer", None)
            if g is None:
                continue

            num_gradients += 1
            total_power += p
            
            # Ensure tensor is on GPU
            if not torch.is_tensor(g):
                g = torch.tensor(g, device=self.device, dtype=torch.float32)
            else:
                g = g.to(device=self.device, dtype=torch.float32, non_blocking=True)

            # Weighted accumulation: alpha = p_k (power coefficient)
            y.add_(g, alpha=p)

        if num_gradients == 0 or total_power <= eps:
            logger.warning(f"Aggregation: no valid gradients ({num_gradients} clients)")
            return torch.zeros(self.d, device=self.device, dtype=torch.float32)

        # Add OTA noise (paper Eq: n_t ~ CN(0, sigma_n^2 * I))
        noise = torch.randn(self.d, device=self.device, dtype=torch.float32) * self.sigma_n
        y.add_(noise)

        # Normalize by total power (y_tilde = y_t / varsigma_t)
        update = y / total_power
        
        norm_val = float(torch.norm(update).item())
        logger.info(
            f"OTA aggregation | clients={num_gradients} | total_power={total_power:.6f} | "
            f"update_norm={norm_val:.6f}"
        )
        return update

    # -------------------------
    # Global model update
    # -------------------------
    def update_model(self, update):
        """Global model update: w_t = w_{t-1} - eta * y_tilde (gradient descent)"""
        with torch.no_grad():
            # Ensure update is on GPU with correct dtype
            update = update.to(device=self.device, non_blocking=True, dtype=torch.float32)
            
            eta_update = self.eta * update
            offset = 0
            
            for param in self.global_model.parameters():
                param_size = param.numel()
                param.sub_(eta_update[offset:offset + param_size].view_as(param))
                offset += param_size

    # -------------------------
    # Virtual queue updates (Lyapunov framework)
    # -------------------------
    def update_queues(self, selected, power_alloc, D_t):
        """
        Update virtual queues (Eq. energyqu and timequ from paper).
        E_k^t = u_k^t * E_cmp + z_k^t * E_tx (only z_k^t=1 transmit, u_k^t computed)
        """
        eps = 1e-12
        round_energy = 0.0
        round_client_energy = {}

        for c in self.clients:
            cid = c.client_id
            p = float(power_alloc.get(cid, 0.0))
            h_abs2 = float(np.abs(c.h_t_k) ** 2) if c.h_t_k is not None else eps

            # Use current buffered gradient norm
            G = max(getattr(c, "grad_sqnorm", 0.0), 0.0)

            # u_k^t: indicator of whether client performs computation in this round
            u_k = 1 if c.dt_k > 0 else 0

            # Computation energy: E_cmp = mu_k * f_k^2 * C * A_k (only if computing)
            E_cmp = u_k * float(c.mu_k * (c.fk ** 2) * c.C * c.Ak)

            # Transmission energy: E_tx = p² * ‖g‖² / |h|²
            # Only for selected clients (z_k^t = 1)
            is_selected = cid in [sc.client_id for sc in selected]
            E_tx = 0.0
            if is_selected and G > 0.0:
                E_tx = (p ** 2) * (G / (h_abs2 + eps))

            # Total energy for this client this round includes both computation and transmission
            E_k = E_tx + E_cmp
            round_energy += E_k
            # Store total per-client energy for this round (kept as float for compatibility)
            round_client_energy[cid] = E_k
            self.cumulative_energy_per_client[cid] += E_k

            # Update energy queue (Eq. energyqu from paper)
            per_round_budget = self.E_max[cid] / self.T_total_rounds
            self.Q_e[cid] = max(0.0, self.Q_e[cid] + ((E_tx + E_cmp) - per_round_budget))

        self.total_energy_per_round.append(round_energy)
        self.per_round_energy.append(round_client_energy)
        logger.info(f"Round energy breakdown: {round_client_energy}")

        # Update time queue (Eq. timequ from paper)
        per_round_time_budget = self.T_max / self.T_total_rounds
        self.Q_time = max(0.0, self.Q_time + (D_t - per_round_time_budget))

        self.selected_history.append([c.client_id for c in selected])
        self.queue_history.append(copy.deepcopy(self.Q_e))

    # -------------------------
    # One full round
    # -------------------------
    def run_round(self, t):
        """
        Execute one global round per paper workflow.
        """
        logger.info(f"\n========== Round {t} ==========")

        # 1) Start computation for idle clients
        for c in self.clients:
            c.maybe_start_local_computation()

        # 2) Select clients (Algorithm 1)
        selected, power_alloc = self.select_clients()

        # 3) Round duration = slowest selected client's remaining time
        D_t = max((c.dt_k for c in selected), default=0.0)
        logger.info(f"Round duration D_t: {D_t:.6f}s")

        # 4) Time elapses for ALL clients
        for c in self.clients:
            c.advance_time(D_t)

        # 5) Wait for selected clients to be ready (complete computation)
        not_ready = [c for c in selected if not c.is_ready()]
        if len(not_ready) > 0:
            extra_wait = max((c.dt_k for c in not_ready), default=0.0)
            if extra_wait > 0.0:
                logger.info(f"Waiting extra {extra_wait:.6f}s for clients to finish")
                D_t += extra_wait
                for c in self.clients:
                    c.advance_time(extra_wait)

        # 6) Final power allocation (with actual buffered gradients)
        final_powers = self._compute_power(selected, use_actual_grad=True)

        # 7) OTA aggregation + global model update
        update = self.aggregate(selected, final_powers)
        self.update_model(update)

        # 8) Update virtual queues
        self.update_queues(selected, final_powers, D_t)

        # 9) Broadcast to selected clients + reset
        global_state = self.global_model.state_dict()
        for c in selected:
            c.receive_model_and_reset(global_state)

        # 10) Increment staleness for non-selected clients
        selected_ids = {c.client_id for c in selected}
        for c in self.clients:
            if c.client_id not in selected_ids:
                c.increment_staleness_and_keep_buffer()

        logger.info(
            f"Staleness: {[c.tau_k for c in self.clients]} | "
            f"Buffers ready: {[c.is_ready() for c in self.clients]}"
        )

        return selected, final_powers, D_t
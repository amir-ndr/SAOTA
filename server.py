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
        device="cpu",
        bisection_tol=1e-6,
    ):
        self.global_model = global_model.to(device)
        self.clients = clients
        self.device = device

        self.V = float(V)
        self.sigma_n = float(sigma_n)  # noise std per dimension (real)
        self.T_max = float(T_max)
        self.T_total_rounds = int(T_total_rounds)
        self.eta = float(eta)
        self.bisection_tol = float(bisection_tol)

        self.d = self._get_model_dimension()

        # Virtual queues
        self.Q_e = {c.client_id: 0.0 for c in clients}
        self.Q_time = 0.0

        # Per-client energy budgets (either scalar or dict)
        self.E_max = E_max if isinstance(E_max, dict) else {c.client_id: float(E_max) for c in clients}

        # History
        self.selected_history = []
        self.queue_history = []
        self.total_energy_per_round = []
        self.per_round_energy = []
        self.cumulative_energy_per_client = {c.client_id: 0.0 for c in clients}

        logger.info(
            f"Server initialized | d={self.d} | V={self.V} | sigma_n={self.sigma_n} | "
            f"T={self.T_total_rounds} | eta={self.eta}"
        )

    def _get_model_dimension(self):
        return sum(p.numel() for p in self.global_model.parameters())

    # -------------------------
    # Channel model
    # -------------------------
    def _rayleigh_channel(self):
        # Complex Rayleigh fading: h = (x + j y)/sqrt(2), x,y ~ N(0,1)
        x = np.random.randn()
        y = np.random.randn()
        return (x + 1j * y) / np.sqrt(2.0)

    # -------------------------
    # Priority score (Equation 12) WITHOUT EMA
    # ρ_k^t = [(τ_k^t/τ_max)*(|h|²/h_ref)] / [ Qe*‖g_last‖²/(E_max/T) + Q_time*d/(T_max/T) ]
    # -------------------------
    def _priority_score(self, client, h_ref_abs2, eps=1e-12):
        cid = client.client_id

        # Get max staleness for normalization
        tau_max = max(c.tau_k for c in self.clients) if self.clients else 1.0
        tau_max = max(tau_max, 1.0)  # Ensure at least 1
        
        h_abs2 = float(np.abs(client.h_t_k) ** 2) if client.h_t_k is not None else eps

        # Use last gradient norm squared (no EMA)
        G_last = getattr(client, "grad_sqnorm", 0.0)
        if G_last <= 0.0:
            G_last = 1.0  # Default value if no gradient yet

        # Per-round allowances for normalization
        E_allow = self.E_max[cid] / self.T_total_rounds
        T_allow = self.T_max / self.T_total_rounds

        # Avoid division by zero
        if E_allow < eps:
            E_allow = eps
        if T_allow < eps:
            T_allow = eps

        # Numerator: (τ_k^t/τ_max) * (|h|²/h_ref)
        # Paper uses h_ref^t = max_i |h_i^t|^2 for normalization
        numerator = (client.tau_k / tau_max) * (h_abs2 / (h_ref_abs2 + eps))

        # Denominator: Qe*‖g_last‖²/(E_max/T) + Q_time*d/(T_max/T)
        # Each term normalized by per-round allowance
        denom_term1 = (self.Q_e[cid] * G_last) / E_allow
        denom_term2 = (self.Q_time * client.dt_k) / T_allow
        denominator = denom_term1 + denom_term2
        
        # Avoid division by zero
        denominator = max(denominator, eps)

        return numerator / denominator

    # -------------------------
    # Power allocation via KKT + bisection (CORRECTED)
    # Solve: min V/(Σp)² + Σ Qe_k * (‖g_k‖²/|h_k|²) * p_k²
    # Solution: p_k^t = min(P_max, V/S³ * |h|²/(Qe*‖g‖²))
    # -------------------------
    def _compute_power(self, selected_clients, use_predicted=True):
        """Compute optimal powers for selected clients."""
        if len(selected_clients) == 0:
            return {}

        eps = 1e-12

        # For each selected client, compute coefficient: a_k = |h|²/(Qe * ‖g‖²)
        # This matches equation (13): p_k^t = min(P_max, V/S³ * a_k)
        coefficients = {}
        pmax_values = {}
        
        for client in selected_clients:
            cid = client.client_id
            pmax_values[cid] = float(client.P_max)
            
            h_abs2 = float(np.abs(client.h_t_k) ** 2) if client.h_t_k is not None else eps
            
            # Use last gradient norm (predicted or actual)
            if use_predicted:
                # For selection: use last gradient (grad_sqnorm)
                G = getattr(client, "grad_sqnorm", 0.0)
            else:
                # For final allocation: use actual buffered gradient
                G = getattr(client, "grad_sqnorm", 0.0)  # Same in no-EMA case
            
            if G <= 0.0:
                G = 1.0  # Default value
                
            Qe = float(self.Q_e[cid])
            if Qe <= 0.0:
                Qe = eps  # Avoid division by zero
            
            # Coefficient a_k = |h|²/(Qe * ‖g‖²)
            coefficients[cid] = h_abs2 / (Qe * G + eps)

        # Fixed-point equation: S = Σ min(P_max, V/S³ * a_k)
        # Solve for S using bisection
        S_high = sum(pmax_values.values())
        if S_high <= eps:
            return {c.client_id: 0.0 for c in selected_clients}

        def rhs(S_val):
            S3 = S_val ** 3
            total = 0.0
            for cid, a_k in coefficients.items():
                # p_k = min(P_max, V/S³ * a_k)
                term = self.V * a_k / (S3 + eps)
                total += min(pmax_values[cid], term)
            return total

        # Bisection search
        S_low = 0.0
        S_star = S_high
        
        for _ in range(100):  # Max iterations
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
        
        # Compute final powers
        S3 = S_star ** 3
        power_alloc = {}
        
        for cid, a_k in coefficients.items():
            p = self.V * a_k / (S3 + eps)
            power_alloc[cid] = min(pmax_values[cid], p)
        
        return power_alloc

    # -------------------------
    # Temporary DPP cost (Equation temp_dpp_cost) CORRECTED
    # C = V/(Σp)² + Σ Qe[μf²CA + p²‖g‖²/|h|²] + Q_time * max(d)
    # -------------------------
    def _exact_cost(self, candidate_set, power_alloc):
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

            # Use last gradient norm (no EMA)
            G_last = getattr(client, "grad_sqnorm", 0.0)
            if G_last <= 0.0:
                G_last = 1.0

            # Computation energy (always for selected clients)
            E_cmp = float(client.mu_k * (client.fk ** 2) * client.C * client.Ak)
            
            # Transmission energy: p² * ‖g‖² / |h|²
            E_tx = (p ** 2) * (G_last / h_abs2)
            
            energy_cost += self.Q_e[cid] * (E_cmp + E_tx)

        # 3. Time cost: Q_time * max(d)
        D_temp = max(client.dt_k for client in candidate_set) if candidate_set else 0.0
        time_cost = self.Q_time * D_temp

        return conv_penalty + energy_cost + time_cost

    # -------------------------
    # Greedy selection (Algorithm 1) - CORRECTED
    # -------------------------
    def select_clients(self):
        eps = 1e-12

        # Set channels at round start
        for c in self.clients:
            h = self._rayleigh_channel()
            if hasattr(c, "set_channel"):
                c.set_channel(h)
            else:
                c.h_t_k = h

        # Get reference channel for normalization
        h_ref_abs2 = max(float(np.abs(c.h_t_k) ** 2) for c in self.clients) + eps

        # Compute priority scores
        scores = []
        for client in self.clients:
            score = self._priority_score(client, h_ref_abs2)
            scores.append((client, score))

        # Sort by descending priority score
        scores.sort(key=lambda x: x[1], reverse=True)
        sorted_clients = [client for client, _ in scores]

        # Greedy selection
        selected = []
        best_cost = float("inf")
        best_power = {}

        for candidate in sorted_clients:
            # Form candidate set
            candidate_set = selected + [candidate]
            
            # Compute temporary powers using last gradient norms
            temp_powers = self._compute_power(candidate_set, use_predicted=True)
            
            # Compute DPP cost
            cost = self._exact_cost(candidate_set, temp_powers)
            
            # Accept if cost decreases
            if cost < best_cost:
                selected = candidate_set
                best_cost = cost
                best_power = temp_powers
            else:
                break  # Stop when adding more clients doesn't help

        logger.info(f"Selected {len(selected)} clients: {[c.client_id for c in selected]} | cost={best_cost:.4e}")
        return selected, best_power

    # -------------------------
    # OTA aggregation (paper) - CORRECT
    # y = sum p_k g_k + n,  y_tilde = y / sum p_k
    # -------------------------
    def aggregate(self, selected, power_alloc):
        eps = 1e-12
        if len(selected) == 0:
            return torch.zeros(self.d, device=self.device)

        varsigma = sum(float(power_alloc.get(c.client_id, 0.0)) for c in selected)
        if varsigma <= eps:
            logger.warning("Aggregation skipped: varsigma near zero.")
            return torch.zeros(self.d, device=self.device)

        # Sum p_k * g_k
        y = torch.zeros(self.d, device=self.device)
        for c in selected:
            cid = c.client_id
            p = float(power_alloc.get(cid, 0.0))

            # Get gradient from buffer
            g = getattr(c, "gradient_buffer", None)
            if g is None:
                continue

            if not torch.is_tensor(g):
                g = torch.tensor(g, device=self.device, dtype=torch.float32)
            else:
                g = g.to(self.device)

            y += p * g

        # Add AWGN noise (real)
        n = torch.randn(self.d, device=self.device) * self.sigma_n
        y = y + n

        update = y / varsigma
        logger.info(f"OTA aggregation complete | varsigma={varsigma:.4f} | update_norm={torch.norm(update).item():.4f}")
        return update

    # -------------------------
    # Global update - CORRECT
    # w_t = w_{t-1} - eta * y_tilde
    # -------------------------
    def update_model(self, update):
        with torch.no_grad():
            params = torch.nn.utils.parameters_to_vector(self.global_model.parameters())
            params -= self.eta * update
            torch.nn.utils.vector_to_parameters(params, self.global_model.parameters())

    # -------------------------
    # Queue updates (Equation 10) - CORRECTED
    # Qe_k(t+1)= [Qe_k(t)+E_k^t - Emax_k/T]^+
    # Q_time(t+1)= [Q_time(t)+D_t - Tmax/T]^+
    # E_k^t = E_cmp + p²‖g‖²/|h|²
    # -------------------------
    def update_queues(self, selected, power_alloc, D_t):
        eps = 1e-12

        round_energy = 0.0
        round_client_energy = {}

        for c in selected:
            cid = c.client_id
            p = float(power_alloc.get(cid, 0.0))
            h_abs2 = float(np.abs(c.h_t_k) ** 2) + eps

            # Use actual gradient norm from buffer
            G = getattr(c, "grad_sqnorm", 0.0)
            if G <= 0.0:
                G = 0.0

            # Computation energy
            E_cmp = float(c.mu_k * (c.fk ** 2) * c.C * c.Ak)
            
            # Transmission energy: p² * ‖g‖² / |h|²
            E_tx = (p ** 2) * (G / h_abs2)

            E_k = E_cmp + E_tx
            round_energy += E_k

            self.cumulative_energy_per_client[cid] += E_k
            round_client_energy[cid] = E_k

            # Update energy queue
            per_round_budget = self.E_max[cid] / self.T_total_rounds
            self.Q_e[cid] = max(0.0, self.Q_e[cid] + (E_k - per_round_budget))

        self.total_energy_per_round.append(round_energy)
        self.per_round_energy.append(round_client_energy)

        # Update time queue
        per_round_time_budget = self.T_max / self.T_total_rounds
        self.Q_time = max(0.0, self.Q_time + (D_t - per_round_time_budget))

        self.selected_history.append([c.client_id for c in selected])
        self.queue_history.append(copy.deepcopy(self.Q_e))

    # -------------------------
    # One full round - CORRECTED for no-EMA
    # -------------------------
    def run_round(self, t):
        # 1) Start computation for idle clients (no buffered gradient, not already computing)
        for c in self.clients:
            if hasattr(c, "maybe_start_local_computation"):
                c.maybe_start_local_computation()

        # 2) Select clients + temporary power (selection-time)
        selected, _ = self.select_clients()

        # Safety: ensure at least one selected client (avoids varsigma=0 / inf objective)
        if len(selected) == 0:
            logger.warning("No clients selected; forcing selection of one client.")
            selected = [self.clients[np.random.randint(len(self.clients))]]

        # 3) Round duration determined by slowest selected client
        D_t = max((c.dt_k for c in selected), default=0.0)

        # 4) Elapse time for ALL clients (some may finish and buffer gradients here)
        for c in self.clients:
            if hasattr(c, "advance_time"):
                c.advance_time(D_t)

        # 5) Verify selected clients are ready; if not, wait extra until they are (optional but safer)
        not_ready = [c for c in selected if hasattr(c, "is_ready") and not c.is_ready()]
        if len(not_ready) > 0:
            extra_wait = max((c.dt_k for c in not_ready), default=0.0)
            if extra_wait > 0.0:
                D_t += extra_wait
                for c in self.clients:
                    c.advance_time(extra_wait)

            still_not_ready = [c.client_id for c in selected if hasattr(c, "is_ready") and not c.is_ready()]
            if len(still_not_ready) > 0:
                logger.warning(f"Selected clients still not ready after extra wait: {still_not_ready}")

        # 6) Final power allocation using actual buffered gradients (no-EMA case)
        final_powers = self._compute_power(selected, use_predicted=False)

        # --- IMPORTANT: snapshot norms BEFORE any reset, for correct energy accounting ---
        grad_sqnorm_snapshot = {c.client_id: float(getattr(c, "grad_sqnorm", 0.0)) for c in selected}

        # 7) OTA aggregation + global update
        update = self.aggregate(selected, final_powers)
        self.update_model(update)

        # 8) Queue updates MUST happen before resetting selected clients (otherwise grad_sqnorm becomes 0)
        # Pass snapshot if you update update_queues to accept it; otherwise, it will still work because
        # we haven't reset yet. Snapshot is kept here as a safety / future-proofing.
        self.update_queues(selected, final_powers, D_t)

        # 9) Broadcast only to selected clients (this resets tau_k and clears buffers)
        global_state = self.global_model.state_dict()
        for c in selected:
            c.receive_model_and_reset(global_state)

        # 10) Staleness update for non-selected
        selected_ids = {c.client_id for c in selected}
        for c in self.clients:
            if c.client_id not in selected_ids:
                c.increment_staleness_and_keep_buffer()

        return selected, final_powers, D_t

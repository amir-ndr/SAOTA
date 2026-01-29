# server_dynamic_baseline.py
import numpy as np
import torch
import copy
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ServerDynamicBaseline:
    """
    Baseline inspired by the paper:
      - Virtual energy queues q_k(t)
      - Estimated transmission energy uses last observed ||g||^2 (EST-P)
      - Dynamic choice of subset size k* by minimizing:
            V * (1/k + 1/k^2) + sum_{i=1..k} ( q_i * E_i_est )
        where i are clients sorted by increasing (q_k * E_k_est)

    Notes for compatibility with your SAOTA code:
      - We reuse your client energy model:
            E_cmp = mu_k * f_k^2 * C * A_k
            E_tx  = p^2 * ||g||^2 / |h|^2
      - Here we use a fixed transmit power p_fixed for selected clients
        (paper uses a fixed sigma_t notion; this is the closest match in your framework).
      - We update queues for ALL clients each round:
            q_k <- max(q_k + z_k*E_k - (Emax_k/T), 0)
        where z_k is 1 if selected else 0.
    """

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
        p_fixed=1.0,
    ):
        self.global_model = global_model.to(device)
        self.clients = clients
        self.device = device

        self.V = float(V)
        self.sigma_n = float(sigma_n)
        self.T_max = float(T_max)
        self.T_total_rounds = int(T_total_rounds)
        self.eta = float(eta)

        self.p_fixed = float(p_fixed)

        # Per-client energy budgets
        self.E_max = E_max if isinstance(E_max, dict) else {c.client_id: float(E_max) for c in clients}

        # Model dimension
        self.d = sum(p.numel() for p in self.global_model.parameters())

        # Energy virtual queues q_k
        self.q_energy = {c.client_id: 0.0 for c in clients}

        # Energy accounting (same format as SAOTA)
        self.per_round_energy = []  # list of dict {cid: E_k^t}
        self.cumulative_energy_per_client = {c.client_id: 0.0 for c in clients}

        logger.info(
            f"[Baseline] Server initialized | d={self.d} | V={self.V} | sigma_n={self.sigma_n} | "
            f"T={self.T_total_rounds} | eta={self.eta} | p_fixed={self.p_fixed}"
        )

    # -------------------------
    # Channel model (same as SAOTA)
    # -------------------------
    def _rayleigh_channel(self):
        x = np.random.randn()
        y = np.random.randn()
        return (x + 1j * y) / np.sqrt(2.0)

    # -------------------------
    # Estimated energy for scheduling (EST-P)
    # -------------------------
    def _estimated_energy(self, client, eps=1e-12):
        """
        E_est = E_cmp + E_tx_est
        E_tx_est uses client's last known grad_sqnorm (may be 0 before first finished compute).
        """
        h_abs2 = float(np.abs(client.h_t_k) ** 2) if client.h_t_k is not None else eps
        G_hat = float(getattr(client, "grad_sqnorm", 0.0))
        if G_hat <= 0.0:
            # if no gradient yet, give a mild default (prevents all zeros)
            G_hat = 1.0

        E_cmp = float(client.calculate_computation_energy())

        # fixed power baseline
        p = float(np.clip(self.p_fixed, 0.0, client.P_max))
        E_tx = (p ** 2) * (G_hat / (h_abs2 + eps))

        return E_cmp + E_tx

    # -------------------------
    # Choose subset size k* and the k clients (paper-style)
    # -------------------------
    def select_clients(self):
        eps = 1e-12

        # set channels at round start
        for c in self.clients:
            h = self._rayleigh_channel()
            c.set_channel(h)

        # compute cost metric: C_k = q_k * E_k_est
        costs = []
        for c in self.clients:
            cid = c.client_id
            E_est = self._estimated_energy(c)
            Ck = float(self.q_energy[cid]) * float(E_est)
            costs.append((c, Ck))

        # sort by increasing cost
        costs.sort(key=lambda x: x[1])
        sorted_clients = [c for c, _ in costs]
        sorted_costs = [ck for _, ck in costs]

        # prefix sums for fast sum_{i<=k} C_(i)
        prefix = np.cumsum(sorted_costs) if len(sorted_costs) > 0 else np.array([0.0])

        # pick k* that minimizes: V*(1/k + 1/k^2) + prefix[k-1]
        best_k = 1
        best_val = float("inf")

        for k in range(1, len(sorted_clients) + 1):
            penalty = (1.0 / k) + (1.0 / (k * k))
            val = self.V * penalty + float(prefix[k - 1])
            if val < best_val:
                best_val = val
                best_k = k

        selected = sorted_clients[:best_k]
        power_alloc = {c.client_id: float(np.clip(self.p_fixed, 0.0, c.P_max)) for c in selected}

        logger.info(f"[Baseline] Selected k={best_k} clients: {[c.client_id for c in selected]} | obj={best_val:.4e}")
        return selected, power_alloc

    # -------------------------
    # OTA aggregation (same structure as your SAOTA)
    # -------------------------
    def aggregate(self, selected, power_alloc):
        eps = 1e-12
        if len(selected) == 0:
            return torch.zeros(self.d, device=self.device)

        varsigma = sum(float(power_alloc.get(c.client_id, 0.0)) for c in selected)
        if varsigma <= eps:
            return torch.zeros(self.d, device=self.device)

        y = torch.zeros(self.d, device=self.device)

        for c in selected:
            cid = c.client_id
            p = float(power_alloc.get(cid, 0.0))
            g = getattr(c, "gradient_buffer", None)
            if g is None:
                continue
            g = g.to(self.device) if torch.is_tensor(g) else torch.tensor(g, device=self.device, dtype=torch.float32)
            y += p * g

        # AWGN noise
        n = torch.randn(self.d, device=self.device) * self.sigma_n
        y = y + n
        return y / varsigma

    def update_model(self, update):
        with torch.no_grad():
            params = torch.nn.utils.parameters_to_vector(self.global_model.parameters())
            params -= self.eta * update
            torch.nn.utils.vector_to_parameters(params, self.global_model.parameters())

    # -------------------------
    # Queue + energy updates (paper style)
    # q_k <- [q_k + z_k*E_k - Emax_k/T]^+
    # -------------------------
    def update_queues_and_energy(self, selected, power_alloc):
        eps = 1e-12
        selected_ids = {c.client_id for c in selected}

        # compute actual round energy for selected clients
        round_energy_dict = {}
        for c in selected:
            cid = c.client_id
            p = float(power_alloc.get(cid, 0.0))
            h_abs2 = float(np.abs(c.h_t_k) ** 2) + eps
            G = float(getattr(c, "grad_sqnorm", 0.0))
            if G < 0.0:
                G = 0.0

            E_cmp = float(c.calculate_computation_energy())
            E_tx = (p ** 2) * (G / h_abs2)
            E_k = E_cmp + E_tx

            round_energy_dict[cid] = E_k
            self.cumulative_energy_per_client[cid] += E_k

        self.per_round_energy.append(round_energy_dict)

        # update queues for ALL clients (paper-style)
        for c in self.clients:
            cid = c.client_id
            budget = float(self.E_max[cid]) / float(self.T_total_rounds)
            z = 1.0 if cid in selected_ids else 0.0
            E_k = float(round_energy_dict.get(cid, 0.0))
            self.q_energy[cid] = max(0.0, self.q_energy[cid] + (z * E_k - budget))

    # -------------------------
    # One full round (aligned with your semi-async client buffers)
    # -------------------------
    def run_round(self, t):
        # 1) start computation for idle clients
        for c in self.clients:
            c.maybe_start_local_computation()

        # 2) select clients
        selected, power_alloc = self.select_clients()
        if len(selected) == 0:
            selected = [self.clients[np.random.randint(len(self.clients))]]
            power_alloc = {selected[0].client_id: float(np.clip(self.p_fixed, 0.0, selected[0].P_max))}

        # 3) make them ready using YOUR mechanism (advance time until they finish)
        # define round duration as the slowest selected's remaining dt
        D_t = max((c.dt_k for c in selected), default=0.0)

        # elapse time for all clients
        for c in self.clients:
            c.advance_time(D_t)

        # ensure selected ready (optional extra wait)
        not_ready = [c for c in selected if not c.is_ready()]
        if len(not_ready) > 0:
            extra_wait = max((c.dt_k for c in not_ready), default=0.0)
            if extra_wait > 0.0:
                D_t += extra_wait
                for c in self.clients:
                    c.advance_time(extra_wait)

        # 4) OTA aggregate + update model
        update = self.aggregate(selected, power_alloc)
        self.update_model(update)

        # 5) energy queues and accounting
        self.update_queues_and_energy(selected, power_alloc)

        # 6) broadcast to selected (resets tau, clears buffer)
        global_state = self.global_model.state_dict()
        for c in selected:
            c.receive_model_and_reset(global_state)

        # 7) staleness for non-selected
        selected_ids = {c.client_id for c in selected}
        for c in self.clients:
            if c.client_id not in selected_ids:
                c.increment_staleness_and_keep_buffer()

        return selected, power_alloc, D_t

import torch
import numpy as np
import copy
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ServerDynamicEAD:
    """
    Implements EAD paper:
      - sigma_t from SNR-threshold rule (Eq. (27))
      - scheduling via Algorithm 1 (sort q*Etilde, compute v_t(k), pick k*)
      - OTA aggregation: y = sigma_t * sum g + z, update uses y/(sigma_t*k)
      - virtual queue: q_{n,t+1} = [q_{n,t} + beta*E_{n,t} - Ebar/T]^+
    """

    def __init__(
        self,
        global_model,
        clients,
        V: float,
        eta: float,
        T_total_rounds: int,
        # noise std per dimension (z ~ N(0, sigma0^2 I))
        sigma0: float = 0.1,
        # SNR threshold gamma0 (paper)
        gamma0: float = 10.0,
        # bound G^2 used in v_t(k) (paper constant)
        G2: float = 1.0,
        # total energy budget per client over all rounds: E_bar_n (dict cid->budget)
        E_bar=None,
        device: str = "cpu",
    ):
        self.global_model = global_model.to(device)
        self.clients = clients
        self.device = device

        self.V = float(V)
        self.eta = float(eta)
        self.T = int(T_total_rounds)

        self.sigma0 = float(sigma0)
        self.gamma0 = float(gamma0)
        self.G2 = float(G2)

        # model dimension s in paper (they denote s)
        self.s = sum(p.numel() for p in self.global_model.parameters())

        # budgets
        if E_bar is None:
            # default 1.0 J each (you should pass dict)
            self.E_bar = {c.client_id: 1.0 for c in clients}
        elif isinstance(E_bar, dict):
            self.E_bar = {int(k): float(v) for k, v in E_bar.items()}
        else:
            self.E_bar = {c.client_id: float(E_bar) for c in clients}

        # virtual queues q_{n,t} (paper uses q_{n,1}=0)
        self.q = {c.client_id: 0.0 for c in clients}

        # logging compatible with your main.py
        self.per_round_energy = []           # list of dict cid->E_n,t (ONLY scheduled have >0)
        self.selected_history = []
        self.sigma_history = []

    def _rayleigh_channel(self):
        x = np.random.randn()
        y = np.random.randn()
        return (x + 1j * y) / np.sqrt(2.0)

    def _set_channels(self):
        for c in self.clients:
            c.set_channel(self._rayleigh_channel())

    def _compute_sigma_t(self):
        """
        Eq. (27): sigma_t = gamma0 * sigma0^2 * sqrt(s) / min_n ||g_n,t||  (approx with EST-P).
        """
        min_norm = min(c.estimated_grad_norm() for c in self.clients)
        min_norm = max(min_norm, 1e-12)
        sigma_t = self.gamma0 * (self.sigma0 ** 2) * np.sqrt(self.s) / min_norm
        return float(sigma_t)

    def _estimated_energy(self, sigma_t: float):
        """
        Etilde_n,t = E_cmp + (sigma_t^2/|h|^2) * ||g_est||^2
        """
        Etilde = {}
        for c in self.clients:
            h_abs2 = float(np.abs(c.h_t) ** 2)
            e_cmp = c.computation_energy()
            e_tx = (sigma_t ** 2) * c.estimated_grad_sqnorm() / (h_abs2 + 1e-12)
            Etilde[c.client_id] = float(e_cmp + e_tx)
        return Etilde

    def _select_by_algorithm1(self, sigma_t: float, Etilde: dict):
        """
        Algorithm 1:
          Ct = q_n,t * Etilde_n,t
          sort ascending => C[1],...,C[N]
          for k=1..N: v_t(k) = V * U_t(k) + sum_{m=1}^k C[m]
          U_t(k) = (eta^2/2) * ( G^2/(Lb*k) + (sigma0^2 * s)/(sigma_t^2 * k^2) )
          choose k* that minimizes v_t(k)
          schedule k* devices with smallest Ct
        """
        N = len(self.clients)
        if N == 0:
            return []

        # batch size Lb (paper uses L_b): assume all clients share batch_size
        Lb = max(int(self.clients[0].batch_size), 1)

        Ct_list = []
        for c in self.clients:
            Ct_list.append((c, float(self.q[c.client_id]) * float(Etilde[c.client_id])))

        Ct_list.sort(key=lambda x: x[1])  # ascending
        C_sorted = [v for _, v in Ct_list]

        # prefix sums of C_sorted
        prefix = np.cumsum(C_sorted)

        best_k = 1
        best_v = float("inf")

        for k in range(1, N + 1):
            Ut_k = (self.eta ** 2) / 2.0 * (
                (self.G2 / (Lb * k)) + ((self.sigma0 ** 2) * self.s) / ((sigma_t ** 2) * (k ** 2) + 1e-12)
            )
            vt_k = self.V * Ut_k + float(prefix[k - 1])
            if vt_k < best_v:
                best_v = vt_k
                best_k = k

        selected = [c for (c, _) in Ct_list[:best_k]]
        return selected

    def _ota_aggregate_and_update(self, selected, sigma_t: float):
        """
        OTA model:
          y = sigma_t * sum_{n in B_t} g_n,t + z
          update uses y / (sigma_t * |B_t|)
          w_t = w_{t-1} - eta * ( y / (sigma_t * k) )
        """
        k = len(selected)
        if k == 0:
            return {}

        global_state = self.global_model.state_dict()

        # each scheduled device computes gradient
        grads = []
        grad_sq_map = {}
        for c in selected:
            g_vec, g_sq = c.compute_gradient_vector(global_state)
            grads.append(g_vec)
            grad_sq_map[c.client_id] = float(g_sq)

        sum_g = torch.stack(grads, dim=0).sum(dim=0)  # sum of gradients

        # noise z ~ N(0, sigma0^2 I)
        z = torch.randn_like(sum_g) * self.sigma0

        y = float(sigma_t) * sum_g + z
        update = y / (float(sigma_t) * k + 1e-12)

        with torch.no_grad():
            w = torch.nn.utils.parameters_to_vector(self.global_model.parameters())
            w -= self.eta * update
            torch.nn.utils.vector_to_parameters(w, self.global_model.parameters())

        return grad_sq_map

    def _update_queues(self, selected, sigma_t: float, grad_sq_map: dict):
        """
        q_{n,t+1} = [ q_{n,t} + beta_{n,t} * E_{n,t} - Ebar_n/T ]^+
        Paper applies the -Ebar/T every round (even if not scheduled).
        """
        per_round_budget = {cid: (self.E_bar[cid] / self.T) for cid in self.q.keys()}
        selected_ids = {c.client_id for c in selected}

        round_energy = {}

        for c in self.clients:
            cid = c.client_id
            if cid in selected_ids:
                E = c.actual_energy(sigma_t, grad_sq_map[cid])
                round_energy[cid] = float(E)
                self.q[cid] = max(0.0, float(self.q[cid]) + float(E) - float(per_round_budget[cid]))
            else:
                # beta=0 => only subtract budget share
                round_energy[cid] = 0.0
                self.q[cid] = max(0.0, float(self.q[cid]) - float(per_round_budget[cid]))

        # store ONLY scheduled energy (matches your plotting logic better)
        scheduled_energy = {cid: e for cid, e in round_energy.items() if e > 0.0}
        self.per_round_energy.append(scheduled_energy)

    def run_round(self, t: int):
        # Step 1: channel acquisition
        self._set_channels()

        # Step 2: set sigma_t (Eq. 27)
        sigma_t = self._compute_sigma_t()
        self.sigma_history.append(sigma_t)

        # Step 3: estimate energy for all devices (Etilde)
        Etilde = self._estimated_energy(sigma_t)

        # Step 4: schedule by Algorithm 1
        selected = self._select_by_algorithm1(sigma_t, Etilde)
        if len(selected) == 0:
            # safety (should not happen because best_k>=1)
            selected = [self.clients[0]]

        # Step 5-7: broadcast + local gradients + OTA aggregation + global update
        grad_sq_map = self._ota_aggregate_and_update(selected, sigma_t)

        # Step 8: queue update (needs actual energy)
        self._update_queues(selected, sigma_t, grad_sq_map)

        self.selected_history.append([c.client_id for c in selected])

        # For compatibility with your main.py signature:
        # power_alloc is not per-client here; we return sigma_t as a scalar in a dict.
        power_alloc = {"sigma_t": sigma_t}
        D_t = 0.0  # EAD paper doesn’t model round time as max(d_k); keep 0 unless you add a time model.
        return selected, power_alloc, D_t

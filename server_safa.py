# server_safa.py
import copy
import math
import torch


class ServerSAFA:
    """
    SAFA server:
      - distributes global model in a lag-tolerant way:contentReference[oaicite:12]{index=12}
      - starts client training, listens for uploads, selects via CFCFM:contentReference[oaicite:13]{index=13}
      - does 3-step discriminative aggregation with cache:contentReference[oaicite:14]{index=14}
    """

    def __init__(
        self,
        global_model,
        clients,
        C: float = 0.2,          # selection fraction
        tau_lag: int = 2,        # lag tolerance τ
        T_lim: float = 1e9,      # round deadline (seconds)
        device="cpu",
    ):
        self.global_model = global_model.to(device)
        self.clients = clients
        self.device = device

        self.C = float(C)
        self.tau_lag = int(tau_lag)
        self.T_lim = float(T_lim)

        # weights nk/n in Eq.(7):contentReference[oaicite:15]{index=15}
        self.nk = {c.client_id: max(len(getattr(c, "data_indices", [])), 1) for c in clients}
        self.n_total = float(sum(self.nk.values()))

        # cache w*_k (vector) initialized to initial global for all k
        init_vec = torch.nn.utils.parameters_to_vector(self.global_model.parameters()).detach().clone()
        self.cache = {c.client_id: init_vec.clone() for c in clients}

        # histories for cmp.py compatibility
        self.selected_history = []
        self.per_round_energy = []   # dict {cid: energy}; SAFA paper focuses on efficiency; keep zeros
        self.round_time_history = []

        # last-round picked P(t-1)
        self.last_picked = set()

    def _set_global_from_vec(self, vec: torch.Tensor):
        torch.nn.utils.vector_to_parameters(vec, self.global_model.parameters())

    def _aggregate_from_cache(self) -> torch.Tensor:
        # Eq.(7): w(t) = sum_k (n_k/n) w*_k(t):contentReference[oaicite:16]{index=16}
        out = None
        for cid, wvec in self.cache.items():
            w = (self.nk[cid] / self.n_total) * wvec.to(self.device)
            out = w if out is None else (out + w)
        return out

    def _cfcfm_select(self, arrived_clients_in_order):
        """
        Implements Algorithm 1 CFCFM: prioritize clients not in last-round picked:contentReference[oaicite:17]{index=17}.
        Returns:
          P(t) picked set, Q(t) undrafted set (arrived but not picked), in arrival order
        """
        quota = int(math.ceil(self.C * len(self.clients)))
        P, Q = [], []
        for c in arrived_clients_in_order:
            if len(P) >= quota:
                Q.append(c)
                continue
            if c.client_id not in self.last_picked:
                P.append(c)
            else:
                Q.append(c)

        # If still not enough picked, fill from earliest Q
        if len(P) < quota and len(Q) > 0:
            need = quota - len(P)
            P_extra = Q[:need]
            Q = Q[need:]
            P.extend(P_extra)

        return P, Q

    def run_round(self, t: int):
        """
        Returns (picked_clients, {}, D_t) to match your cmp.py.
        D_t here is the simulated round length: time until quota reached or deadline.
        """

        # ---------- 1) lag-tolerant distribution ----------
        global_state = copy.deepcopy(self.global_model.state_dict())
        for c in self.clients:
            c.maybe_refresh_from_global(global_state, tau_lag=self.tau_lag)

        # ---------- 2) start local training ----------
        for c in self.clients:
            c.start_round()

        # ---------- 3) event-driven "listen until quota or deadline" ----------
        # Sort by predicted finish time (dt). This approximates arrival order used by CFCFM.
        active = [c for c in self.clients if not c.crashed_this_round]
        active_sorted = sorted(active, key=lambda x: x.dt)

        quota = int(math.ceil(self.C * len(self.clients)))
        arrived = []
        t_now = 0.0

        # Advance time client-by-client until quota uploads OR deadline
        for c_fast in active_sorted:
            if len(arrived) >= quota:
                break
            next_finish = float(c_fast.dt)
            if t_now + next_finish > self.T_lim:
                break

            # advance all clients by delta = next_finish
            delta = next_finish
            t_now += delta
            for c in self.clients:
                c.advance_time(delta_t=delta, t_now=t_now)

            # collect arrivals that just became ready at this time
            newly_arrived = [c for c in self.clients if c.has_upload() and (c not in arrived) and c.arrival_time == t_now]
            # keep stable order
            newly_arrived.sort(key=lambda x: x.client_id)
            arrived.extend(newly_arrived)

        D_t = min(t_now, self.T_lim)

        # Any client that did not upload before selection end is treated as "crashed/uncommitted" this round
        arrived_ids = {c.client_id for c in arrived}

        # ---------- 4) CFCFM selection (post-training selection) ----------
        # Use actual arrival ordering by arrival_time then id
        arrived_sorted = sorted(arrived, key=lambda c: (c.arrival_time, c.client_id))
        P, Q = self._cfcfm_select(arrived_sorted)

        P_ids = {c.client_id for c in P}
        Q_ids = {c.client_id for c in Q}

        # ---------- 5) 3-step discriminative aggregation ----------
        # (1) Pre-aggregation cache update Eq.(6):contentReference[oaicite:18]{index=18}
        # - picked: cache[k] = w0_k(t)
        # - deprecated clients: cache[k] = w(t-1)
        # - otherwise keep cache
        w_global_prev = torch.nn.utils.parameters_to_vector(self.global_model.parameters()).detach().clone()

        for c in self.clients:
            cid = c.client_id
            if cid in P_ids:
                self.cache[cid] = c.w0_vec.detach().clone().to(self.device)
            else:
                # "deprecated": lag > tau -> replace entry by global to avoid heavy staleness:contentReference[oaicite:19]{index=19}
                if c.lag > self.tau_lag:
                    self.cache[cid] = w_global_prev.clone()

        # (2) Aggregate Eq.(7):contentReference[oaicite:20]{index=20}
        new_global_vec = self._aggregate_from_cache()
        self._set_global_from_vec(new_global_vec)

        # (3) Post-aggregation cache update Eq.(8):contentReference[oaicite:21]{index=21}
        # undrafted Q(t) get written into cache for next round
        for c in self.clients:
            cid = c.client_id
            if cid in Q_ids:
                self.cache[cid] = c.w0_vec.detach().clone().to(self.device)

        # ---------- 6) update lag + last_picked ----------
        # If uploaded (picked or undrafted), lag resets to 0 and they are "missed_last_round = False".
        # Otherwise lag++ (crashed/uncommitted) and missed_last_round = True.
        for c in self.clients:
            cid = c.client_id
            if cid in arrived_ids:
                c.lag = 0
                c.missed_last_round = False
                c.tau_k = 0
            else:
                c.lag += 1
                c.missed_last_round = True
                c.tau_k += 1

        self.last_picked = set(P_ids)

        # ---------- 7) histories for cmp.py ----------
        self.selected_history.append([c.client_id for c in P])
        self.round_time_history.append(D_t)
        self.per_round_energy.append({})  # keep empty; cmp.py will warn and treat as zeros

        return P, {}, D_t

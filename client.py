import torch
import numpy as np
import copy
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Client:
    """
    Client state machine aligned with SAOTA server orchestration:

    - Client may be computing across rounds (dt_k decreases each round by D_t).
    - When dt_k reaches 0, client computes one gradient on its stale model and buffers it.
    - Buffered gradient is kept until client is selected + receives global model broadcast.
    - Staleness tau_k increments when NOT selected, resets when selected and receives model.
    - Channel h_t_k is set by server at the beginning of each round.

    NOTE: Using LAST buffered gradient's squared norm as G_hat (no EMA).
    """

    def __init__(
        self,
        client_id,
        data_indices,
        model,
        fk,
        mu_k,
        P_max,
        C,
        Ak,
        train_dataset,
        device="cpu",
        seed=None,
    ):
        self.client_id = int(client_id)
        self.data_indices = list(data_indices)

        # Model used for computing gradient on current stale/global snapshot
        self.stale_model = copy.deepcopy(model).to(device)
        self.device = device

        # System parameters
        self.fk = float(fk)        # cycles/sec
        self.mu_k = float(mu_k)    # energy coefficient
        self.P_max = float(P_max)  # max transmit power
        self.C = float(C)          # cycles per sample
        self.Ak = int(Ak)          # batch size
        self.train_dataset = train_dataset

        # RNG
        self.rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        if seed is not None:
            torch.manual_seed(seed)

        # State variables (paper)
        self.dt_k = 0.0            # remaining computation time d_k^t
        self.tau_k = 0             # staleness
        self.h_t_k = None          # complex channel h_k^t

        # Gradient buffer (at most one outstanding update)
        self.gradient_buffer = None    # torch tensor shape [d]
        self.grad_sqnorm = 0.0         # ||g||^2 of LAST buffered gradient
        self.gradient_norm = 0.0       # ||g||

        # Computation state
        self.computing = False

        logger.info(
            f"Client {self.client_id} initialized | "
            f"CPU: {self.fk/1e9:.2f} GHz | Batch: {self.Ak} | |Dk|: {len(self.data_indices)}"
        )

    # -------------------- Helpers --------------------
    def _gradient_dimension(self):
        return sum(p.numel() for p in self.stale_model.parameters())

    def _full_compute_time(self):
        # tau_cp,k = C * A_k / f_k
        return (self.C * self.Ak) / self.fk

    # -------------------- Model & staleness --------------------
    def update_stale_model(self, model_state_dict):
        """
        If you want to keep your old naming, server can call this,
        but in the corrected server it's typically `receive_model_and_reset`.
        """
        self.receive_model_and_reset(model_state_dict)

    def receive_model_and_reset(self, model_state_dict):
        """
        Called by server ONLY for selected clients after global update broadcast.
        This:
          - refreshes stale_model to latest global model,
          - resets staleness,
          - clears buffered gradient (paper: at most one outstanding per latest model),
          - resets computation state to idle.
        """
        self.stale_model.load_state_dict(model_state_dict)
        self.tau_k = 0
        self.reset_computation_state()

    def increment_staleness_and_keep_buffer(self):
        """
        Called by server for NON-selected clients.
        IMPORTANT: we KEEP buffered gradient and ongoing computation.
        """
        self.tau_k += 1

    # -------------------- Channel --------------------
    def set_channel(self, h_complex):
        """Server sets h_k^t (complex scalar)."""
        self.h_t_k = complex(h_complex)

    # -------------------- Semi-async computation --------------------
    def maybe_start_local_computation(self):
        """
        Start computing ONLY if:
          - no buffered gradient exists (otherwise we already have an outstanding update),
          - not already computing.
        """
        if self.gradient_buffer is not None:
            return False
        if self.computing:
            return False

        self.dt_k = float(self._full_compute_time())
        self.computing = True
        return True

    def advance_time(self, D_t):
        """
        Called by server to elapse time by D_t for this round.
        If computing, dt_k decreases; when it hits 0, compute gradient ONCE and buffer it.
        """
        if not self.computing:
            return

        prev_dt = self.dt_k
        self.dt_k = max(0.0, self.dt_k - float(D_t))

        # Finished during this elapsed time
        if prev_dt > 0.0 and self.dt_k == 0.0:
            self._compute_gradient()
            self.computing = False

    def _compute_gradient(self):
        """Compute g_k on current stale_model and buffer it."""
        d = self._gradient_dimension()

        # No data edge case
        if len(self.data_indices) == 0:
            self.gradient_buffer = torch.zeros(d, device=self.device)
            self.grad_sqnorm = 0.0
            self.gradient_norm = 0.0
            return

        # Sample batch indices
        if len(self.data_indices) < self.Ak:
            batch_indices = np.array(self.data_indices, dtype=int)
        else:
            batch_indices = self.rng.choice(self.data_indices, self.Ak, replace=False).astype(int)

        # Build batch
        batch = [self.train_dataset[int(i)] for i in batch_indices]
        images = torch.stack([x[0] for x in batch]).to(self.device)
        labels = torch.tensor([x[1] for x in batch], device=self.device, dtype=torch.long)

        self.stale_model.train()
        self.stale_model.zero_grad(set_to_none=True)

        outputs = self.stale_model(images)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        loss.backward()

        grads = []
        for p in self.stale_model.parameters():
            if p.grad is not None:
                grads.append(p.grad.detach().view(-1))

        g = torch.cat(grads) if grads else torch.zeros(d, device=self.device)

        # Buffer update
        self.gradient_buffer = g
        self.grad_sqnorm = float((g * g).sum().item())
        self.gradient_norm = float(np.sqrt(self.grad_sqnorm))

    def is_ready(self):
        """Ready to transmit when a buffered gradient exists."""
        return self.gradient_buffer is not None

    def reset_computation_state(self):
        """Clear buffered gradient and reset compute state to idle."""
        self.gradient_buffer = None
        self.grad_sqnorm = 0.0
        self.gradient_norm = 0.0
        self.dt_k = 0.0
        self.computing = False

    # -------------------- OTA signal/energy --------------------
    def calculate_transmit_signal(self, p_k_t):
        """
        AirComp pre-equalized transmit:
          s_k^t = p_k^t * (h_k^H / |h_k|^2) * g_k^t
        For scalar complex h: (h* / |h|^2) = 1/h
        Returns numpy array for OTA summation.
        """
        if self.gradient_buffer is None or self.h_t_k is None:
            return None

        p = float(np.clip(p_k_t, 0.0, self.P_max))
        h = self.h_t_k
        scale = p / (h + 1e-12)  # 1/h
        return (scale * self.gradient_buffer.detach().cpu().numpy())

    def calculate_transmission_energy(self, p_k_t):
        """
        ||s_k^t||^2 = p^2 * ||g||^2 / |h|^2
        """
        if self.gradient_buffer is None or self.h_t_k is None:
            return 0.0
        p = float(np.clip(p_k_t, 0.0, self.P_max))
        h_abs2 = float(np.abs(self.h_t_k) ** 2)
        return (p ** 2) * self.grad_sqnorm / (h_abs2 + 1e-12)

    def calculate_computation_energy(self):
        """E_cmp^k = mu_k * f_k^2 * C * A_k (per your paper’s model)."""
        return float(self.mu_k * (self.fk ** 2) * self.C * self.Ak)

    # -------------------- For server selection --------------------
    def get_priority_score_components(self):
        """
        Server-side priority/score uses last gradient (no EMA):
          G_hat := ||g_last||^2 = self.grad_sqnorm
        """
        return {
            "tau_k": int(self.tau_k),
            "h_abs2": float(np.abs(self.h_t_k) ** 2) if self.h_t_k is not None else 0.0,
            "G_hat": float(self.grad_sqnorm),
            "d_k": float(self.dt_k),
        }

# client_safa.py
import copy
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


class ClientSAFA:
    """
    SAFA client:
      - maintains a local model (may be stale)
      - trains for E local epochs, then uploads the trained *model vector* w0_k(t)
      - may crash each round (then it uploads nothing)
      - 'missed_last_round' flag is used by CFCFM priority:contentReference[oaicite:4]{index=4}
      - 'lag' (in rounds) is used to mark deprecated if lag > tau (tau is lag tolerance):contentReference[oaicite:5]{index=5}
    """

    def __init__(
        self,
        client_id: int,
        data_indices,
        model,
        train_dataset,
        batch_size: int,
        local_epochs: int,
        lr: float,
        device="cpu",
        perf_batches_per_sec: float = 1.0,   # s_k in paper (batches/sec):contentReference[oaicite:6]{index=6}
        crash_prob: float = 0.0,             # cr_k in paper
        seed: int = 0,
    ):
        self.client_id = int(client_id)
        self.data_indices = list(data_indices)
        self.train_dataset = train_dataset
        self.tau_k = 0        # staleness in rounds (for plotting)
        self.dt_k = 0.0
        self.device = device
        self.max_batches_per_round = 1

        self.batch_size = int(batch_size)
        self.local_epochs = int(local_epochs)
        self.lr = float(lr)

        self.perf = float(perf_batches_per_sec)
        self.crash_prob = float(crash_prob)

        self.rng = np.random.RandomState(seed)

        # local model
        self.local_model = copy.deepcopy(model).to(device)

        # SAFA state
        self.lag = 0                 # number of consecutive rounds without successful upload
        self.missed_last_round = True

        # round execution state
        self.computing = False
        self.dt = 0.0                # remaining time in seconds
        self.w0_vec = None           # uploaded local model vector (torch tensor)
        self.arrival_time = None     # time (sec) since round start when upload arrived
        self.crashed_this_round = False

    # ---------- timing ----------
    def _num_batches(self) -> int:
        n = len(self.data_indices)
        return int(math.ceil(n / max(self.batch_size, 1)))

    def _full_train_time(self) -> float:
        # Eq.(18) style: T_train = |B_k| * E / s_k  (batches * epochs / batches_per_sec):contentReference[oaicite:7]{index=7}
        return float(self._num_batches() * self.local_epochs / max(self.perf, 1e-12))

    # ---------- lag-tolerant distribution ----------
    def maybe_refresh_from_global(self, global_state_dict, tau_lag: int):
        """
        If up-to-date (lag==0) OR deprecated (lag > tau), refresh to latest global:contentReference[oaicite:8]{index=8}.
        Otherwise keep local model (tolerable).
        """
        if self.lag == 0 or self.lag > int(tau_lag):
            self.local_model.load_state_dict(global_state_dict)

    # ---------- start/advance training ----------
    def start_round(self):
        self.computing = True
        self.dt = self._full_train_time()
        self.dt_k = self.dt
        self.w0_vec = None
        self.arrival_time = None
        self.crashed_this_round = (self.rng.rand() < self.crash_prob)

    def advance_time(self, delta_t: float, t_now: float):
        if not self.computing:
            return
        if self.crashed_this_round:
            # crashed clients never upload this round
            self.dt = max(0.0, self.dt - float(delta_t))
            self.dt_k = self.dt
            if self.dt == 0.0:
                self.computing = False
            return

        prev = self.dt
        self.dt = max(0.0, self.dt - float(delta_t))
        if prev > 0.0 and self.dt == 0.0:
            # finished: train + upload w0_k(t)
            self._local_train()
            self.w0_vec = torch.nn.utils.parameters_to_vector(self.local_model.parameters()).detach().clone()
            self.arrival_time = float(t_now)
            self.computing = False

    def _local_train(self):
        if len(self.data_indices) == 0:
            return

        subset = Subset(self.train_dataset, self.data_indices)
        loader = DataLoader(subset, batch_size=self.batch_size, shuffle=True)

        opt = torch.optim.SGD(self.local_model.parameters(), lr=self.lr)

        self.local_model.train()
        for _ in range(self.local_epochs):
            for b, (x, y) in enumerate(loader):
                if b >= self.max_batches_per_round:   # <<< EARLY STOP
                    break
                x = x.to(self.device)
                y = y.to(self.device)
                opt.zero_grad(set_to_none=True)
                out = self.local_model(x)
                loss = torch.nn.functional.cross_entropy(out, y)
                loss.backward()
                opt.step()


    def has_upload(self) -> bool:
        return self.w0_vec is not None and (not self.crashed_this_round)

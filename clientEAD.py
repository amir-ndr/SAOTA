import torch
import numpy as np
import copy


class ClientEAD:
    """
    EAD / OTA-FEEL client (synchronous rounds):
      - Keeps last reported gradient norm^2 as EST-P proxy (paper uses ||g_est||).
      - When scheduled, computes ONE mini-batch gradient on broadcast global model.
      - OTA uses channel inversion with scalar sigma_t: transmit x = (sigma_t / h) * g.
      - Reports actual energy E_n,t = E_cmp + (sigma_t^2/|h|^2) * ||g||^2.
    """

    def __init__(
        self,
        client_id: int,
        data_indices,
        model,
        train_dataset,
        batch_size: int,
        # computation energy model (paper uses e_n * L_b; you can map yours into e_n)
        e_per_sample: float = 1e-9,
        device: str = "cpu",
        seed: int | None = None,
    ):
        self.client_id = int(client_id)
        self.data_indices = list(data_indices)
        self.train_dataset = train_dataset
        self.batch_size = int(batch_size)
        self.device = device

        self.model = copy.deepcopy(model).to(device)

        self.rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        if seed is not None:
            torch.manual_seed(seed)

        # channel for current round (complex)
        self.h_t = 1.0 + 0j

        # EST-P memory: last observed ||g||^2 (initialize >0 to avoid sigma blow-up)
        self.last_grad_sqnorm = 1.0

        # computation energy coefficient (J per sample)
        self.e_per_sample = float(e_per_sample)

    def set_channel(self, h_complex: complex):
        self.h_t = complex(h_complex)

    def _sample_minibatch(self):
        if len(self.data_indices) == 0:
            # fallback
            idx = 0
            x, y = self.train_dataset[idx]
            return x.unsqueeze(0).to(self.device), torch.tensor([y], device=self.device)

        if len(self.data_indices) <= self.batch_size:
            batch_indices = np.array(self.data_indices, dtype=int)
        else:
            batch_indices = self.rng.choice(self.data_indices, self.batch_size, replace=False).astype(int)

        batch = [self.train_dataset[int(i)] for i in batch_indices]
        xs = torch.stack([b[0] for b in batch]).to(self.device)
        ys = torch.tensor([b[1] for b in batch], device=self.device, dtype=torch.long)
        return xs, ys

    def compute_gradient_vector(self, global_state_dict):
        """Compute g_n,t on the latest broadcast global model (one mini-batch)."""
        self.model.load_state_dict(global_state_dict)
        self.model.train()
        self.model.zero_grad(set_to_none=True)

        xs, ys = self._sample_minibatch()
        out = self.model(xs)
        loss = torch.nn.functional.cross_entropy(out, ys)
        loss.backward()

        grads = []
        for p in self.model.parameters():
            if p.grad is not None:
                grads.append(p.grad.detach().view(-1))
        if len(grads) == 0:
            g = torch.zeros(sum(p.numel() for p in self.model.parameters()), device=self.device)
        else:
            g = torch.cat(grads)

        grad_sq = float((g * g).sum().item())
        self.last_grad_sqnorm = max(grad_sq, 1e-12)
        return g, grad_sq

    def estimated_grad_norm(self):
        """Return ||g_est|| (not squared) for sigma_t rule."""
        return float(np.sqrt(max(self.last_grad_sqnorm, 1e-12)))

    def estimated_grad_sqnorm(self):
        """Return ||g_est||^2 for energy estimation."""
        return float(max(self.last_grad_sqnorm, 1e-12))

    def computation_energy(self):
        """E_cmp = e_n * L_b (paper form)."""
        return float(self.e_per_sample * self.batch_size)

    def actual_energy(self, sigma_t: float, grad_sqnorm: float):
        """E = E_cmp + (sigma_t^2/|h|^2) * ||g||^2 (channel inversion OTA)."""
        h_abs2 = float(np.abs(self.h_t) ** 2)
        e_cmp = self.computation_energy()
        e_tx = (float(sigma_t) ** 2) * float(grad_sqnorm) / (h_abs2 + 1e-12)
        return e_cmp + e_tx

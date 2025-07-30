import torch
import torch.nn as nn
from nflows.transforms import CompositeTransform, ReversePermutation
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.normalization import ActNorm
from nflows.flows.base import Flow


__all__ = ("INNV4E4A6")


class INNV4E4A6(nn.Module):
    def __init__(self, **kwargs):
        input_dim = kwargs.pop("input_dim", 1)
        hidden_dim= kwargs.pop("hidden_dim", 256)
        num_layers = kwargs.pop("num_layers", 64)
        dropout = kwargs.pop("dropout", 0.0)
        #output_gain = kwargs.pop("output_gain", 3.0) # not used in this version
        init_z_scale = kwargs.pop("init_z_scale", 1.0)
        warp_gain_init = kwargs.pop("warp_gain_init", 2.5)
        range_edge = kwargs.pop("clamp_range", 2.5 )
        clamp_range = (-range_edge, range_edge)
        train_mean = kwargs.pop("train_mean", 0.0)
        train_std = kwargs.pop("train_std", 1.0)
        super(INNV4E4A6, self).__init__(**kwargs)

        transforms = []
        for _ in range(num_layers):
            transforms.append(ActNorm(features=input_dim))
            transforms.append(ReversePermutation(features=input_dim))
            transforms.append(
                MaskedAffineAutoregressiveTransform(
                    features=input_dim,
                    hidden_features=hidden_dim,
                    context_features=None,
                    dropout_probability=dropout
                )
            )
        transforms.append(MaskedAffineAutoregressiveTransform(features=input_dim, hidden_features=hidden_dim * 2))
        flow_transform = CompositeTransform(transforms)

        # base_distribution = StandardNormal(shape=[input_dim])
        base_distribution = MixtureOfGaussiansBase(dim=input_dim)
        flow = Flow(flow_transform, base_distribution)
        self.flow = flow
        self._distribution = base_distribution

        # ── learnable global affine on raw latent 
        self.z_center = nn.Parameter(torch.tensor(0.0))
        self.z_scale  = nn.Parameter(torch.tensor(init_z_scale))
        self.z_shift  = nn.Parameter(torch.tensor(0.0))    # kept for optimiser compat

        # ── latent-shift MLP (conditioned on observed mass x) 
        self.latent_shift = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Tanh())

        # ── monotone warp (Integral-of-Exp) 
        self.monotonic_warp = IntegralOfExpWarp()
        with torch.no_grad():
            self.monotonic_warp.output_gain.fill_(warp_gain_init)

        self.enable_z_bent_clamp = True
        self.z_bent_clamp_range  = clamp_range

        # ── compression stats (σ fixed after first forward) 
        self._sigma_initialized = False    # register buffer lazily

        # ?~\~E Clamp z_scale
        with torch.no_grad():
            self.z_scale.data.clamp_(min=0.5, max=6.0)
            self.z_scale.data.fill_(1.0)

        initialize_actnorm(self, train_mean, train_std)

    # ------------------------------------------------------------------
    def _compress(self, z_scaled: torch.Tensor) -> torch.Tensor:
        """Batch-centred, fixed-σ compression (σ captured once)."""
        if not self._sigma_initialized:
            with torch.no_grad():
                sigma = z_scaled.std().detach().clone()
                self.register_buffer("z_sigma_fixed", sigma)
            self._sigma_initialized = True
        return (z_scaled - z_scaled.mean()) / (self.z_sigma_fixed + 1e-6)

   # ------------------------------------------------------------------
    def inverse(self,
                observed_batch: torch.Tensor,
                z_raw: torch.Tensor | None = None,
                alpha: float = 1.0,
                epoch: int | None = None):
        """Map observed mass → latent and back to predicted mass."""
        if z_raw is None:
            #  Forward through the flow's transform only (no base log prob)
            z_raw = self.flow._transform.forward(observed_batch)[0]

        # ---- latent shift --------------------------------------------------
        z_shift  = torch.tanh(self.latent_shift(observed_batch) / 6.0)
        z_shifted = z_raw + alpha * z_shift
        self.z_shifted = z_shifted                    # diagnostics for loss

        # ---- global centre / scale ----------------------------------------
        z_centered = z_shifted - self.z_center
        z_scaled   = z_centered / (self.z_scale + 1e-6)

        # ── in inverse() right after you compute z_scaled ─────────────
        if getattr(self, "_stats_frozen", False):
            z_compressed = (z_scaled - self.z_mu_fixed) / (self.z_sigma_fixed + 1e-6)
        else:   # fallback (first few batches before stats frozen)
            z_compressed = (z_scaled - z_scaled.mean()) / (z_scaled.std() + 1e-6)

        # ---- monotone warp -------------------------------------------------
        z_bent = self.monotonic_warp(z_compressed)
        if self.enable_z_bent_clamp and epoch is not None and epoch >= 1000:
            z_bent = torch.clamp(z_bent, *self.z_bent_clamp_range)

        # ---- invert residual flow -----------------------------------------
        pred_mass = z_bent.view(-1, 1)
        logdet = None
        for t in reversed(self.flow._transform._transforms):
            pred_mass, ld = t.inverse(pred_mass)
            logdet = ld if logdet is None else logdet + ld

        # ---- optional final affine to physical units ----------------------

        return pred_mass, logdet, z_bent

    # wrappers -------------------------------------------------------------
    def forward (self, x): return self.flow._transform.forward(x)
    def log_prob(self, x): return self.flow.log_prob(x)
    def sample  (self, n): return self.flow.sample(n)




# -------------------------------
# Fix: ActNorm Initialization
# -------------------------------
# this should go to initialization part of the model
def initialize_actnorm(model, train_mean, train_std):
    """
    Initializes ActNorm layers using multiple stable batches.
    Avoids collapse due to unstable or low-variance initialization.
    """
    with torch.no_grad():
        train_std_tensor = torch.as_tensor(train_std, dtype=torch.float32)
        train_mean_tensor = torch.as_tensor(train_mean, dtype=torch.float32)
        inv_std = 1.0 / (train_std_tensor + 1e-2)
        clamped_inv_std = torch.clamp(inv_std, min=0.5, max=1.5)

        for module in model.modules():
#           print(type(module))
            if isinstance(module, ActNorm):
                # Safe shape copying
                module.scale.data.copy_(clamped_inv_std.view_as(module.scale))
                module.scale.data += torch.randn_like(module.scale.data) * 5e-2
                module.shift.data.copy_(-train_mean_tensor.view_as(module.shift))
#               print(f"✅ ActNorm initialized: shift={module.shift.mean().item():.3f}, scale={module.scale.mean().item():.3f}")

        print("✅ ActNorm initialized using stable batch statistics.")



class IntegralOfExpWarp(nn.Module):
    """
    f(z) = a * ∫ exp(g(t)) dt + b
    with bounded slopes and internal normalization of F(z)
    to stabilize early training and prevent scale drift.
    """

    def __init__(self,
                 n_points: int = 400,
                 hidden: int = 64,
                 z_range: float = 3.0,
                 max_log_slope: float = 1.0  # g(t) ∈ [-C, +C]
                 ):
        super().__init__()
        self.n_points = n_points
        self.z_min, self.z_max = -z_range, +z_range
        self.C = max_log_slope
        self.output_gain = nn.Parameter(torch.tensor(1.0))  # optional final gain

        # Define MLP g(t)
        self.body = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

        # global gain & bias (learnable)
        self.a_raw = nn.Parameter(torch.tensor(0.0))  # softplus(a_raw) + 1e-3
        self.b     = nn.Parameter(torch.tensor(0.0))

        # fixed integration grid
        t = torch.linspace(self.z_min, self.z_max, n_points)
        self.register_buffer("t_grid", t)
        dt = (self.z_max - self.z_min) / (n_points - 1)
        self.register_buffer("dt", torch.tensor(dt))
        self.a_raw = nn.Parameter(torch.tensor(1.0))  # not 0.0!

        # ➊ one-time frozen stats for the warp output
        self.register_buffer("warp_mu_fixed",    torch.tensor(0.0))
        self.register_buffer("warp_sigma_fixed", torch.tensor(1.0))
        self._warp_stats_frozen = False          # ➋ flag used in forward

# -----------------------------------------------------------
# IntegralOfExpWarp.forward
# -----------------------------------------------------------
    def forward(self, z, *, return_logdet: bool = False):
        """
        Parameters
        ----------
        z :  tensor [B, 1]        – latent input (already centred / scaled)
        return_logdet : bool      – if True, also return log-|dF/dz|
    
        Returns
        -------
        Fz_out            : [B, 1]  warped latent
        log_det_optional  : [B, 1]  (only if return_logdet=True)
        """
        # 1) evaluate g(t) on the fixed grid and clamp slopes
        g_grid = self.body(self.t_grid.unsqueeze(-1)).view(-1)          # [n_pts]
        g_clamped = g_grid.clamp(-self.C, self.C)
    
        # slope w(t) = exp(g_clamped)
        w_grid = torch.exp(g_clamped)
    
        # 2) cumulative trapezoid integral   (∫ w dt)
        F_grid = torch.cumsum(0.5 * (w_grid[:-1] + w_grid[1:]) * self.dt, dim=0)
        F_grid = torch.cat([F_grid.new_zeros(1), F_grid], dim=0)        # [n_pts]
    
        # 3) linear interpolation (with extrapolation)  → F(z)
        zf  = z.view(-1)
        pos = (zf - self.z_min) / self.dt
        idx = pos.floor().long().clamp(0, self.n_points - 2)
        frac = (pos - idx.float()).clamp(0, 1)
    
        F_lo = F_grid[idx]
        w_lo = w_grid[idx]
        F_mid = F_lo + frac * w_lo * self.dt
    
        # linear extrapolation at the ends
        F_low   = F_grid[0]  + (zf - self.z_min) * w_grid[0]
        F_high  = F_grid[-1] + (zf - self.z_max) * w_grid[-1]
    
        Fz = torch.where(zf < self.z_min, F_low,
              torch.where(zf > self.z_max, F_high, F_mid)).view_as(z)
    
        # 4) ── one-time capture of mean / std  (freeze after first large batch)
        if (not self._warp_stats_frozen) and self.training and Fz.numel() >= 1024:
            with torch.no_grad():
                self.warp_mu_fixed.copy_(Fz.mean())
                self.warp_sigma_fixed.copy_(Fz.std().clamp_min(1e-3))
                self._warp_stats_frozen = True
    
        # use frozen statistics once available
        if self._warp_stats_frozen:
            Fz_norm = (Fz - self.warp_mu_fixed) / (self.warp_sigma_fixed + 1e-6)
            logdet_norm = -torch.log(self.warp_sigma_fixed + 1e-6)
        else:                                   # warm-up fall-back
            mu_b  = Fz.mean()
            sig_b = Fz.std().clamp_min(1e-6)
            Fz_norm = (Fz - mu_b) / sig_b
            logdet_norm = -torch.log(sig_b)
    
        # 5) global gain / bias
        a = torch.nn.functional.softplus(self.a_raw) + 1e-3
        Fz_out = a * Fz_norm + self.b
        logdet_gain = torch.log(a)
    
        if return_logdet:
            # total log-det:  norm part + gain part  (same shape as z)
            logdet_total = (logdet_norm + logdet_gain).expand_as(z)
            return Fz_out, logdet_total

        return Fz_out


# -------------------------------
# Mixture of Gaussians Base Distribution
# -------------------------------
class MixtureOfGaussiansBase(nn.Module):
    def __init__(self, dim, num_components=10):
        super().__init__()
        self.num_components = num_components
        self.dim = dim

        self.means = nn.Parameter(torch.linspace(-2, 2, num_components).view(-1, 1))
        self.stds = nn.Parameter(torch.full((num_components, 1), 1.0))
        self.weights = nn.Parameter(torch.ones(num_components) / num_components)

        # Optional latent init param (can be removed if unused)
        self.latent_space = nn.Parameter(torch.randn(dim))

    def log_prob(self, x):
        x = x.unsqueeze(1)  # (B, 1, D)
        log_probs = -0.5 * (((x - self.means) / self.stds) ** 2) \
                    - torch.log(self.stds) - 0.5 * np.log(2 * np.pi)
        log_probs = log_probs.sum(dim=-1)
        log_probs += torch.log(self.weights)
        return torch.logsumexp(log_probs, dim=1)

    def sample(self, num_samples):
        temp = 0.85
        scaled_weights = F.softmax(self.weights / temp, dim=0)
        indices = torch.multinomial(scaled_weights, num_samples, replacement=True)

        means = self.means[indices].squeeze(-1)
        stds = torch.clamp(self.stds[indices].squeeze(-1) + 1.5, min=1.5, max=5.0)

        samples = means + torch.randn(num_samples, device=means.device) * stds
        samples = torch.clamp(samples, min=-12.0, max=12.0)

        return samples.view(-1, self.dim)

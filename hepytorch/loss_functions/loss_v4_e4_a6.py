from .abs_lossfn import AbsLossFn
from collections import OrderedDict
from scipy.stats import gaussian_kde
from geomloss import SamplesLoss  # Wasserstein Loss for mass matching
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch


class AdaptiveHybridLoss(AbsLossFn):
    def get_loss_fn(self):
        return adaptive_hybrid_loss 

def adaptive_hybrid_loss(
    model,
    pred_mass,          # [B,1]   – network output in *normalised* units
    true_mass,          # [B,1]   – truth in *normalised* units
    obs_mass,           # [B,1]   – (kept for API; not used here)
    z_raw,              # [B,1]
    z_bent,             # [B,1]
    epoch: int,
    max_epochs: int,
    running_mag: dict,
    ema_beta: float = 0.95,
):
    device = pred_mass.device
    # ─────────────────────────────── phase schedule
    if   epoch <=  5: phase = 0
    elif epoch <= 10: phase = 1
    elif epoch <= 15: phase = 2
    else:             phase = 3

    PHASES = [
        dict(wasserstein=0.30, tail_floor=0.02),   # free S-line + light tail guard
        dict(wasserstein=0.30, curvature=0.03, fold_clamp=0.02),
        dict(wasserstein=0.25, curvature=0.05, fold_clamp=0.03,
             slope_survival=0.03, z_std_survival=0.03),
        dict(wasserstein=0.15, curvature=0.04, slope_survival=0.05,
             fold_clamp=0.02, z_std_survival=0.04,
             peak_alignment=0.08, mean_alignment=0.06,
             shift_alignment=0.06, width_alignment=0.10),
    ]
    target_rel = PHASES[phase]

    # ─────────────────────────────── always-tracked keys
    ALWAYS = {
        "warp_span", "density_shape_kde",
        "tail_floor", "positive_slope",
        "latent_std_reg", "latent_range_reg", "latent_mean_reg",
        "curvature_mag", "early_span_protect",
    }
    ALL_KEYS = sorted({k for p in PHASES for k in p} | ALWAYS)


    # ─────────────────────────────── raw (no-grad) measurement
    raw = OrderedDict((k, torch.tensor(0., device=device)) for k in ALL_KEYS)
    with torch.no_grad():
        raw["wasserstein"]     = wasserstein_loss2(pred_mass, true_mass)
        raw["curvature"]       = safe_curvature_penalty(z_raw, pred_mass)
        raw["warp_span"]       = warp_span_loss(z_bent)
        raw["fold_clamp"]      = fold_prevention_penalty(z_raw, pred_mass)
        raw["slope_survival"]  = slope_survival_penalty(z_raw, pred_mass)
        raw["z_std_survival"]  = z_std_survival_penalty(z_bent)
        raw["mean_alignment"]  = mean_alignment_loss(pred_mass, true_mass)
        raw["width_alignment"] = width_alignment_loss(pred_mass, true_mass)
        raw["shift_alignment"] = shift_alignment_loss(z_raw, model.z_shifted)
        raw["peak_alignment"]  = peak_alignment_loss(pred_mass, true_mass)
        raw["density_shape_kde"] = density_shape_penalty_kde(pred_mass, true_mass)

        # tail-floor diagnostic
        zz, idx = torch.sort(z_raw.view(-1))
        pp      = pred_mass.view(-1)[idx]
        slope   = (pp[1:] - pp[:-1]) / (zz[1:] - zz[:-1] + 1e-6)
        mask    = (zz[:-1].abs() > 1.0).float()
        raw["tail_floor"] = (mask * F.relu(0.02 - slope.abs())).mean()

    # ─────────────────────────────── update running magnitudes
    with torch.no_grad():
        for k in ALL_KEYS:
            if k not in running_mag:
                running_mag[k] = raw[k].detach().clone().clamp(min=1e-6)
            if k in target_rel:
                running_mag[k] = ema_beta * running_mag[k] + (1 - ema_beta) * raw[k]

    # weights = running_mag / target_rel  (clamped 0.1 … 10)
    weights = {k: ((running_mag[k] / (target_rel[k] + 1e-6)).clamp(0.1, 10.).item()
                   if k in target_rel else 0.0)
               for k in ALL_KEYS}

    # ─────────────────────────────── differentiable pass
    total = torch.tensor(0., device=device)
    loss_components = {}

    def _add(tag, term):
        loss_components[tag] = term * weights.get(tag, 1.0)
        return loss_components[tag]

    # main geometry & alignment terms
    total += _add("wasserstein",     wasserstein_loss2(pred_mass, true_mass))
    total += _add("curvature",       safe_curvature_penalty(z_raw, pred_mass))
    total += _add("warp_span",       warp_span_loss(z_bent))
    total += _add("fold_clamp",      fold_prevention_penalty(z_raw, pred_mass))
    total += _add("slope_survival",  slope_survival_penalty(z_raw, pred_mass))
    total += _add("z_std_survival",  z_std_survival_penalty(z_bent))

    total += _add("mean_alignment",  mean_alignment_loss(pred_mass, true_mass))
    total += _add("width_alignment", width_alignment_loss(pred_mass, true_mass))
    total += _add("shift_alignment", shift_alignment_loss(z_raw, model.z_shifted))
    total += _add("density_shape_kde", density_shape_penalty_kde(pred_mass, true_mass))

    # tail-floor term (use stored raw val for grad safety)
    if weights["tail_floor"] > 0:
        total += _add("tail_floor", raw["tail_floor"].detach().clone().requires_grad_(True))

    # ── self-tuning peak sharpen / tail suppress block ────────────────
    with torch.no_grad():
        tail_frac   = (pred_mass > 2.5).float().mean()
        target_tail = (true_mass  > 2.5).float().mean()
        excess_tail = F.relu(tail_frac - target_tail)
        alpha_peak  = (1.0 + 9.0 * excess_tail).clamp(1.0, 10.0)

    peak_loss  = peak_alignment_loss(pred_mass, true_mass)
    sharp_loss = positive_slope_penalty(z_raw, pred_mass)

    peak_term  = alpha_peak * weights.get("peak_alignment", 1.0) * peak_loss
    sharp_term = alpha_peak * 0.05 * sharp_loss               # fixed 0.05 weight

    loss_components["peak_alignment"] = peak_term
    loss_components["sharp_slope"]    = sharp_term
    total += peak_term + sharp_term

    # extras
    if epoch <= 3:
        prot = F.relu(2.0 - (z_raw.max() - z_raw.min())) * 0.5
        total += prot
        loss_components["early_span_protect"] = prot

    if epoch >= 3:
        std_reg   = (z_bent.std() - 1).pow(2)
        range_reg = (z_raw.max() - z_raw.min()).pow(2) * 1e-3
        mean_reg  = 1e-3 * z_bent.mean().pow(2)
        mag_reg   = curvature_magnitude(z_raw, pred_mass)

        total += std_reg + range_reg + mean_reg + mag_reg
        loss_components.update({
            "latent_std_reg":   std_reg,
            "latent_range_reg": range_reg,
            "latent_mean_reg":  mean_reg,
            "curvature_mag":    mag_reg,
        })

    if epoch < 10:
        pos_slope  = 0.05 * positive_slope_penalty(z_raw, pred_mass)
        surv_rate  = 0.3 if epoch >= 3 else 0.1
        slope_surv = 0.05 * slope_survival_penalty(z_raw, pred_mass, surv_rate)
        total += pos_slope + slope_surv
        loss_components["positive_slope"] = pos_slope
        loss_components["slope_survival"] = \
            loss_components.get("slope_survival", torch.tensor(0., device=device)) + slope_surv

    # --- if tails look good, crank up sharpening / alignment ---
    if epoch >= 3 and raw["tail_floor"] < 3e-3:          # same relaxed cut
        weights["peak_alignment"]    *=  4.0
        weights["density_shape_kde"] *=  4.0
        weights["width_alignment"]   *=  2.0

    if epoch >= 18 and raw["tail_floor"] < 3e-3:      # keep the same cut
        # make sure we do it only once
        if not hasattr(adaptive_hybrid_loss, "_late_boost_done"):
            weights["peak_alignment"]    *= 2.0       # on top of the ×4 above
            weights["density_shape_kde"] *= 2.0
            weights["width_alignment"]   *= 1.5
            adaptive_hybrid_loss._late_boost_done = True
            if epoch % 5 == 0:                        # small debug print
                print(f"[loss] late peak-sharpen boost applied at epoch {epoch}")
    
    raw["excess_tail"] = excess_tail        # <-- add this

    return total, loss_components, raw


def wasserstein_loss2(pred_mass, true_mass, chunk_size=1000):
    loss = 0.0
    num_chunks = (pred_mass.size(0) + chunk_size - 1) // chunk_size

    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, pred_mass.size(0))
        pred_chunk = pred_mass[start:end]
        true_chunk = true_mass[start:end]
        
        # Re-instantiate a smaller SamplesLoss
        loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.01)
        loss = loss + loss_fn(pred_chunk.view(-1, 1), true_chunk.view(-1, 1))

    return loss / num_chunks

def safe_curvature_penalty(z, pred_mass, min_slope=0.3):
    z = z.view(-1)
    pred_mass = pred_mass.view(-1)
    z_sorted, indices = torch.sort(z)
    mass_sorted = pred_mass[indices]

    dz = z_sorted[1:] - z_sorted[:-1]
    dm = mass_sorted[1:] - mass_sorted[:-1]
    slope = dm / (dz + 1e-6)

    penalty = F.relu(min_slope - slope.abs())
    return penalty.mean()

# -----------------------------------------------------------------
#  Warp-span loss  (keeps overall bend from collapsing or exploding)
# -----------------------------------------------------------------
def warp_span_loss(z_bent: torch.Tensor,
                   min_span: float = 1.5,
                   max_span: float = 12.0) -> torch.Tensor:
    """
    Penalise if span = (max − min) of z_bent falls outside [min_span, max_span].

    • Below `min_span`  → pushes gain up (avoids flat centre collapse)  
    • Above `max_span`  → pushes gain down (avoids run-away stretching)
    """
    span = z_bent.max() - z_bent.min()
    low  = F.relu(min_span - span)          # positive if too small
    high = F.relu(span - max_span)          # positive if too large
    return low + high                       # (scalar tensor)


def fold_prevention_penalty(z, mass, focus=0.0, radius=0.3, min_slope=0.2, weight=20.0):
    """
    Penalize regions in z→mass mapping that become too flat or folded
    near the specified focus region (typically around z ≈ 0).
    """
    z = z.view(-1)
    mass = mass.view(-1)

    # Sort by z
    idx = torch.argsort(z)
    z_sorted = z[idx]
    mass_sorted = mass[idx]

    # Compute finite differences
    dz = z_sorted[1:] - z_sorted[:-1]
    dm = mass_sorted[1:] - mass_sorted[:-1]
    slope = dm / (dz + torch.finfo(z.dtype).eps)

    # Focus on critical region
    z_mid = 0.5 * (z_sorted[1:] + z_sorted[:-1])
    mask = (z_mid > (focus - radius)) & (z_mid < (focus + radius))

    if mask.sum() == 0:
        return torch.tensor(0.0, device=z.device)

    bad_slope = F.relu(min_slope - slope[mask])
    return weight * bad_slope.mean()

def slope_survival_penalty(z, mass, min_slope=0.1, z0=5.0):
    """
    Penalize slopes below min_slope, weighted by a Gaussian kernel in z.
    Near z=0 the weight→1; in the tails |z|>>z0 the weight→0.
    """
    # sort and compute slope as before
    z_flat = z.view(-1)
    mass_flat = mass.view(-1)
    z_sorted, idx = torch.sort(z_flat)
    m_sorted = mass_flat[idx]
    dz = z_sorted[1:] - z_sorted[:-1]
    dm = m_sorted[1:]   - m_sorted[:-1]
    slope = dm / dz.clamp(min=1e-4)

    # hinge loss
    hinge = F.relu(min_slope - slope.abs())  # [N-1]

    # Gaussian weight centered at zero
    z_mid = 0.5*(z_sorted[1:] + z_sorted[:-1])  # midpoints
    weight = torch.exp(-(z_mid**2) / (2*z0*z0))  # [N-1]

    # weighted average
    return (weight * hinge).sum() / (weight.sum() + 1e-6)

def z_std_survival_penalty(z, min_std=1.0, weight=1.0):
    """
    Penalize z if its std is too small, encouraging a healthy latent spread.
    """
    std = torch.std(z)
    penalty = torch.relu(min_std - std)
    return weight * penalty

def mean_alignment_loss(pred_mass, true_mass):
    """
    Penalizes the mean difference between predicted and true mass.
    This helps center the predicted distribution.
    """
    pred_mean = pred_mass.mean()
    true_mean = true_mass.mean()
    return torch.abs(pred_mean - true_mean)

def width_alignment_loss(pred_mass, true_mass, mode="relative", epsilon=1e-6):
    """
    Penalizes mismatch in distribution width (std).
    Mode:
      - "relative": penalize fractional deviation
      - "symmetric": penalize both over/under equally
    """
    pred_std = pred_mass.std()
    true_std = true_mass.std()

    if mode == "relative":
        return torch.abs(pred_std - true_std) / (true_std + epsilon)
    elif mode == "symmetric":
        return (pred_std - true_std).pow(2)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def shift_alignment_loss(z_raw, z_shifted):
    """
    Penalizes large deviation between z_raw and z_shifted.
    Encourages z_shift to not be overly dominant or erratic.
    """
    return torch.mean((z_raw - z_shifted)**2)


def density_shape_penalty_kde(pred, true, bandwidth=0.5):
    import numpy as np
    from scipy.stats import gaussian_kde

    pred_np = pred.detach().cpu().numpy().reshape(1, -1)
    true_np = true.detach().cpu().numpy().reshape(1, -1)

    if pred.numel() < 5 or true.numel() < 5:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    try:
        p = gaussian_kde(pred_np, bw_method=bandwidth)
        q = gaussian_kde(true_np, bw_method=bandwidth)
        xs = np.linspace(min(true.min(), pred.min()).item(),
                         max(true.max(), pred.max()).item(), 200)
        p_vals = p(xs)
        q_vals = q(xs)
        return torch.tensor(np.mean((p_vals - q_vals) ** 2), device=pred.device, dtype=pred.dtype)
    except Exception as e:
        print("[KDE ERROR]", e)
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

def peak_alignment_loss(pred_mass, true_mass):
    pred_mass_np = pred_mass.detach().cpu().numpy().flatten()
    true_mass_np = true_mass.detach().cpu().numpy().flatten()

    if np.any(np.isnan(pred_mass_np)) or np.any(np.isinf(pred_mass_np)) or pred_mass_np.std() < 1e-5:
        return torch.tensor(0.0, device=pred_mass.device)

    x = np.linspace(min(pred_mass_np.min(), true_mass_np.min()),
                    max(pred_mass_np.max(), true_mass_np.max()), 100)

    bandwidth = 0.4  # or whatever value you use
    kde_p = gaussian_kde(pred_mass_np, bw_method=bandwidth / (pred_mass_np.std() + 1e-6))
    kde_t = gaussian_kde(true_mass_np, bw_method=bandwidth / (true_mass_np.std() + 1e-6))

    diff = kde_p(x) - kde_t(x)
    loss = np.sum(diff ** 2)

    return torch.tensor(loss, dtype=torch.float32, device=pred_mass.device)

def positive_slope_penalty(z, pred_mass, min_slope=0.01, reduction="mean"):
    """
    Penalizes negative or near-zero slope between latent z and predicted mass.
    Encourages globally positive monotonic mapping.

    Args:
        z (Tensor): latent variable, shape (B, 1)
        pred_mass (Tensor): predicted mass, shape (B, 1)
        min_slope (float): soft lower bound for slope
        reduction (str): 'mean' or 'sum'
    Returns:
        penalty (Tensor)
    """
    z = z.view(-1)
    pred_mass = pred_mass.view(-1)

    # Sort by z for monotonicity check
    z_sorted, indices = torch.sort(z)
    mass_sorted = pred_mass[indices]

    # Finite difference slope
    dz = z_sorted[1:] - z_sorted[:-1] + 1e-6  # avoid divide-by-zero
    dm = mass_sorted[1:] - mass_sorted[:-1]
    slope = dm / dz

    # Penalty: how far below min_slope we are (ReLU)
    penalty = F.relu(min_slope - slope)

    if reduction == "mean":
        return penalty.mean()
    elif reduction == "sum":
        return penalty.sum()
    else:
        return penalty  # (N-1,)

def curvature_magnitude(z, mass, min_curv=0.01):
    z_sorted, _ = torch.sort(z.view(-1))
    m_sorted = mass.view(-1)[_]

    dz = z_sorted[1:] - z_sorted[:-1]
    dm = m_sorted[1:] - m_sorted[:-1]
    slope = dm / dz.clamp(min=1e-4)

    dslope = slope[1:] - slope[:-1]
    d2 = dslope / dz[1:].clamp(min=1e-4)

    penalty = F.relu(min_curv - d2.abs())
    return penalty.mean()
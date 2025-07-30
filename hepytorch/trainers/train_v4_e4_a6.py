from .abs_trainer import AbsTrainer
from collections import defaultdict
from nflows.transforms import ActNorm 
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
import copy
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import torch


class TrainerV4E4A6(AbsTrainer):
    def __init__(self, **kwargs):
        self.bins = kwargs.pop("bins", 200)
        self.train_true_mean = kwargs.pop("train_true_mean", 0.0)
        self.train_true_std = kwargs.pop("train_true_std", 1.0)
        self.train_size = kwargs.pop("train_size", 0.8)
        self.test_size = kwargs.pop("test_size", 0.2)
        self.batch_size = kwargs.pop("batch_size", 500)
        self.epochs = kwargs.pop("epochs", 1024)
        

    def train(self, device, data, target, model, loss_fn, optimizer):
        num_bins = 200
        mass_range = [120, 220]
        bins = torch.linspace(mass_range[0], mass_range[1], num_bins + 1, device=device)
        train_indices = (0, int(len(data) * self.train_size))
        test_indices = (int(len(data) * self.train_size), len(data))

        # split data into train and test sets
        train_data = data[train_indices[0]:train_indices[1]]
        train_target = target[train_indices[0]:train_indices[1]]
        test_data = data[test_indices[0]:test_indices[1]]
        test_target = target[test_indices[0]:test_indices[1]]
        train_dataset = TensorDataset(train_data, train_target)
        test_dataset = TensorDataset(test_data, test_target)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True) 
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        #train_loader = DataLoader(Subset(dataset, train_indices), batch_size=self.batch_size, shuffle=True) 
        #test_loader = DataLoader(Subset(dataset, test_indices), batch_size=self.batch_size, shuffle=False)
        train_inn(device, model, optimizer, loss_fn, train_loader, test_loader, bins, self.train_true_mean, self.train_true_std, self.epochs)
        return {"model": model, "losses": []}  # Placeholder for losses 

# -------------------------------
# Training Function
# -------------------------------
    #def train(self, device, data, target, model, loss_fn, optimizer):
def train_inn(device, model, optimizer, loss_fn, train_loader, test_loader, bins, train_mean, train_std, epochs=1024):
    running_mag = {k: torch.tensor(1.0, device=device) for k in loss_dict}
    running_mag["local_slope"] = torch.tensor(1.0, device=device)

    model_ema = copy.deepcopy(model).eval()
    for p in model_ema.parameters():
        p.requires_grad_(False)            # ensures optimiser ignores it
    ema_beta = 0.995                       # decay factor you’ll use below


    loss_raw_dict = defaultdict(float)

    with torch.no_grad():
        model.z_scale.data.fill_(1.0)
        model.z_scale.requires_grad_(False)

    span_ema = None

    total_loss = 0
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for key in loss_dict:
            loss_dict[key] = 0
    
        # ========== Compressor warm-up ==========
        COMPRESSOR_START_EPOCH = UNFREEZE_WARP_BODY_EPOCH + 4
        if epoch >= COMPRESSOR_START_EPOCH:
            compressor_k = 0.15 * smooth_schedule(
                epoch,
                mid_epoch=COMPRESSOR_START_EPOCH + 5,
                max_epochs=COMPRESSOR_START_EPOCH + 15
            )
        else:
            compressor_k = 0.0
        
        desired_span       = (MIN_SPAN + MAX_SPAN) * 0.5

        # === 5. z_scale logic ===
        schedule_z_scale(model, epoch, enable_grad_at=20, clamp_min=2.0, clamp_max=6.0, decay_epochs=(30, 60))
    
        # === 6. z_shift logic ===
        if epoch < 5:
            model.z_shift.requires_grad_(False)
            model.z_shift.zero_()
        elif 5 <= epoch <= 10:
            model.z_shift.requires_grad_(True)
            with torch.no_grad():
                model.z_shift.data *= (epoch - 4) / 6.0

        if epoch == 5:
            # drop inner-warp and main LR by 10×
            optimizer.param_groups[0]["lr"] *= 0.1   # main_params
            optimizer.param_groups[1]["lr"] *= 0.1   # warp_body_params
            print(f"[epoch {epoch}] LR cut to {[g['lr'] for g in optimizer.param_groups]}")

        # Clamp z_scale in early phase
        if epoch <= 10:
            model.z_scale.data.clamp_(min=0.5, max=2.0)
        # in train loop ─ before backward()
#       if epoch < 20:           # let it float after that
#           clamp_warp_gain(model)

        for batch_idx, (observed_batch, true_batch) in enumerate(train_loader):
            # 🧠 Move data to device
            observed_batch = observed_batch.to(device)
            true_batch     = true_batch.to(device)

            # 🧭 Step 1: Encode observed → latent z_raw (with autograd)
            # Step 1: Get z_raw and ensure it's a leaf with grad tracking
            z_raw = model.flow._transform.forward(observed_batch)[0].detach().requires_grad_() 

            if epoch == 0 and batch_idx == 0:
                for name, p in model.named_parameters():
                    if "output_gain" in name:
                        print(f"[debug] {name}  -> {p.item():.3f}")
             
            #  EARLY HARD CLAMP ON  z_raw
            if epoch < HARD_CLAMP_STOP_EPOCH:
                clamp_limit = CLAMP_BASE + 2.0 * epoch / HARD_CLAMP_STOP_EPOCH
            else:
                clamp_limit = CLAMP_BASE + 2.0
            z_raw = clamp_limit * torch.tanh((z_raw - z_raw.mean()) / clamp_limit / (z_raw.std()+1e-6))
            # --- soft compressor (only after COMPRESSOR_START_EPOCH) ---

            if compressor_k > 0:
                z_raw = z_raw / (1 + compressor_k * z_raw.abs())

            # early phase guard: mild clamp only in the *very* first epochs
            if epoch < 3:
                z_raw = 6.0 * torch.tanh(z_raw / 6.0)      # soft ±6 envelope
                # do NOT call .detach(); we want gradients from here on

            # --- soft compression after peak is formed ---
            if epoch >= 3:         # start clipping after a few warm‑up epochs
                z_raw = safe_soft_clip(z_raw, thresh=4.0, sharpness=0.5)

            if epoch <20:
                decay = 0.5 * (1 + math.cos(math.pi * epoch / 20)) 
                noise_strength = 0.01 * decay
                z_raw += noise_strength * torch.randn_like(z_raw)
                z_raw.retain_grad()

            z_raw.retain_grad()  # Must be called on the actual tensor used for loss

            if epoch >= 3:
                dynamic_freeze_base(model, z_raw, epoch)
            # Step 2: Use z_raw to compute z_center for diagnostics (only once)
            if epoch == 0 and batch_idx == 0:
                model.z_center.data = 0.5 * model.z_center.data + 0.5 * z_raw.mean().detach()

            # Step 3: Save clean z_raw for diagnostics
            z_raw_unaltered = z_raw.clone().detach()
            
            # Step 4: Optionally perturb z_raw, but do it **in-place**
            p_weight = max(0, 0.01*(1-epoch/20))
            z_raw = z_raw + p_weight * torch.randn_like(z_raw)
            z_raw.retain_grad()  # ?~F~P RETAIN AGAIN if you're modifying the tensor

            # Step 5: Proceed to inverse
            pred_mass, _, z_bent = model.inverse(
                observed_batch=observed_batch, z_raw=z_raw, alpha=1.0, epoch=epoch
            )
            assert pred_mass.requires_grad, "Inverse output is not differentiable!"



            with torch.no_grad():
                if epoch < 50:                     # free to adapt in the first 50 epochs
                    model.monotonic_warp.output_gain.clamp_(1.0, 4.0)
                else:                              # lock it down afterwards
                    model.monotonic_warp.output_gain.clamp_(min=1.0, max=4.0)

            # === z_bent implosion safeguard ======================================
            with torch.no_grad():
                std_bent = z_bent.std().item()
                
                if span_ema is None:
                    span_ema = std_bent
                else:
                    span_ema = 0.99 * span_ema + 0.01 * std_bent if 'span_ema' in locals() else std_bent

            if batch_idx % 1 == 0:
                plt.hist(z_raw.detach().cpu().numpy(), bins=100, range=(z_raw.min().item(), z_raw.max().item()))
                plt.title(f"z_raw  Epoch {epoch}, Batch {batch_idx}")
                plt.savefig("INN_unbinned_v4_e4_a6/z_raw.png")
                plt.clf()
                plt.close()

            # 🎯 Final weighted loss computation
            loss, loss_comp, loss_comp_raw = loss_fn(
                model=model,
                pred_mass=pred_mass,
                true_mass=true_batch,
                obs_mass=observed_batch,
                z_raw=z_raw,
                z_bent=z_bent,
                epoch=epoch,
                max_epochs=epochs,
                running_mag=running_mag,
                ema_beta=ema_beta,
            )
            # early latent regularisers (z_range / z_std_survival) 
            if epoch >= 3:            # previously 8
                extra_loss  = L2_LATENT_REG_WEIGHT * (z_bent.std() - 1).pow(2)
                extra_loss += L2_LATENT_REG_WEIGHT * (z_raw.max() - z_raw.min()).pow(2) * 1e-3
                loss = loss + extra_loss

            # --------------- in train_inn() ---------------
            #  after you have `raw` from adaptive_hybrid_loss
            tail_ok = loss_comp_raw["tail_floor"] < 3e-3          # <- more relaxed cut
            if epoch >= 3 and tail_ok and not hasattr(model, "_freeze_done"):
        
                # a) freeze the last affine / ActNorm so bias & scale stop drifting
                last_tf = model.flow._transform._transforms[-1]
                for p in last_tf.parameters():
                    p.requires_grad_(False)
        
                # b) also freeze z_shift (optional – stops small global drifts)
                model.z_shift.requires_grad_(False)
        
                # c) cut the LR of *all* tail-related terms to half
                for g in optimizer.param_groups:
                    if "tail" in g.get("name",""):
                        g["lr"] *= 0.5
        
                model._freeze_done = True
                print(f"[e{epoch}] last affine frozen, tail OK – peak locked")

            #if epoch < 10:
            #   loss = loss + 0.05 * LF.positive_slope_penalty(z_raw, pred_mass)
            
            total_loss += loss.item()

            # 🔧 Step 5: Backprop
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            # ── place this right after loss.backward() and before optim.step() ──
            with torch.no_grad():
                last_tf = model.flow._transform._transforms[-1]
                # nflows’ autoregressive layer has ._shift and ._log_scale
                if hasattr(last_tf, "_shift") and hasattr(last_tf, "_log_scale"):
                    # log-scale in [−1, 0.7]  ⇒  scale in [0.37, 2.01]
                    last_tf._log_scale.clamp_(min=-1.0, max=0.7)
                    # shift in ±5 keeps the mass window reasonable
                    last_tf._shift.clamp_(min=-5.0, max=5.0)

            # Inside training loop
            for k in loss_comp:
                raw_val = loss_comp[k].item()
                norm    = running_mag.get(k, torch.tensor(1., device=device))
                loss_raw_dict[k] += raw_val
                loss_dict[k]     += raw_val / (norm.item() + 1e-8)
            
            # 🔁 ActNorm safety clamp (keeps scale in [-5,5] every step)
            for m in model.modules():
                if isinstance(m, ActNorm):
                    m.scale.data.clamp_(-5.0, 5.0)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            excess_tail = loss_comp_raw["excess_tail"].item()

            # ------------------------------------------------------------
            # place directly after optimizer.step()
            # ------------------------------------------------------------
            with torch.no_grad():
            
                # 0) pick the very last transform only once
                if not hasattr(model, "_last_tf"):
                    model._last_tf = model.flow._transform._transforms[-1]
            
                last_tf = model._last_tf
            
                # 1) hard-clamp its scale / bias every step
                if hasattr(last_tf, "_log_scale"):          # MAF / IAF
                    last_tf._log_scale.clamp_(-1.0, 0.7)
                    last_tf._shift    .clamp_(-5.0, 5.0)
                elif hasattr(last_tf, "log_scale"):         # ActNorm / AffineConst
                    last_tf.log_scale.clamp_(-1.0, 0.7)
                    last_tf.bias     .clamp_(-5.0, 5.0)
            
                # 2) ONE-TIME freeze when tail is clean
                if (epoch >= 3 and excess_tail < 0.01
                    and not getattr(model, "_affine_frozen", False)):
                    for p in last_tf.parameters():
                        p.requires_grad_(False)
                    model._affine_frozen = True
                    print(f"[e{epoch:03d} b{batch_idx:04d}] last affine frozen ✅")
            # ------------------------------------------------------------
            # at the start of some later epoch, e.g. epoch == 30
            if epoch == 30 and getattr(model, "_affine_frozen", False):
                last_tf = model._last_tf                        # cached earlier
            
                # 1) un-freeze its parameters
                for p in last_tf.parameters():
                    p.requires_grad_(True)
            
                # 2) drop LR **only** for those very parameters
                last_param_ids = {id(p) for p in last_tf.parameters()}   # identity set
            
                for g in optimizer.param_groups:
                    # does this group contain (any of) the last-tf params?
                    if any(id(p) in last_param_ids for p in g["params"]):
                        g["lr"] = 1e-4                                # new LR
                print("[info] last affine unfrozen — learning-rate pinned to 1e-4")

            optimizer.zero_grad()     # usual hygiene

            if epoch % 1 == 0 and batch_idx % 1== 0:
                plt.scatter(z_raw.detach().cpu().detach().numpy(), pred_mass.cpu().detach().numpy(), s=5, alpha=0.3)
                plt.xlabel("Latent z")
                plt.ylabel("Predicted Mass")
                plt.title(f"Inverse Mapping: z -> Pred Mass epoch {epoch} batch {batch_idx}")
                plt.grid(True)
                plt.savefig(f"INN_unbinned_v4_e4_a6/z_to_pred_mass_epoch_{epoch}.png")
                plt.savefig("INN_unbinned_v4_e4_a6/z_to_pred_mass_current.png")
                plt.close()
 
            print(f"[Epoch {epoch}, Batch {batch_idx}] pred_mass: mean={pred_mass.mean():.3f}, std={pred_mass.std():.3f}, "
                  f"min={pred_mass.min():.3f}, max={pred_mass.max():.3f}" ,
                  "z_raw mean/std:", z_raw.mean().item(), z_raw.std().item())
#           print(f"[E{epoch}] gain={model.monotonic_warp.final_gain.item():.3f} | span_ema={span_ema:.3f} | final_gain grad={model.monotonic_warp.final_gain.grad}")

        # At the end of epoch:
        # ✅ Track actual weighted (backprop-influencing) losses
        
        ema_beta = 0.995
        with torch.no_grad():
            for p, p_ema in zip(model.parameters(), model_ema.parameters()):
                p_ema.data.mul_(ema_beta).add_(p.data, alpha=1-ema_beta)

        update_loss_tracking(epoch, total_loss / len(train_loader), loss_comp)
        plot_loss_evolution()
        test_inn_conditionally(model, train_loader, test_loader, train_mean, train_std, bins, epoch)

        torch.save(model.state_dict(), "INN_unbinned_v4_e4_a6/INN_unbinned_v4_e4_a6.pt")
        with open("INN_unbinned_v4_e4_a6/INN_unbinned_v4_e4_a6.json", "w") as f:
            json.dump({
                "epoch": epoch,
                "z_shift": model.z_shift.item(),
            }, f, indent=2)
        torch.cuda.empty_cache()


# -- Schedule configs
MIN_SPAN, MAX_SPAN       = 2.0, 6.0
HARD_CLAMP_STOP_EPOCH    = 6          # was 4
UNFREEZE_GAIN_EPOCH      = 3      #  ?~F~R two epochs *before* body
UNFREEZE_WARP_BODY_EPOCH = UNFREEZE_GAIN_EPOCH
CLAMP_BASE              = 4.0        # ±4 ?~F~R soft-schedule up to ±6
L2_LATENT_REG_WEIGHT    = 0.02       # very small, activated early

loss_tracking = {
    "epoch"                : [],
    "total_loss"           : [],
    "wasserstein"          : [],
    "curvature"            : [],
    "z_mass_shift"         : [],
    "fold_clamp"           : [],
    "force_s_shape"           : [],
    "shift_alignment"      : [],
    "width_alignment"      : [],
    "peak_alignment"      : [],
    "warp_span"            : [],
    "mean_alignment"       : [],
    "z_bent_survival"       : [],
    "z_std_survival"       : [],
    "early_span_protect"   : [],
    "density_shape_kde"    : [],
    "positive_slope"       : [],
    "latent_std_reg"       : [],
    "latent_range_reg"     : [],
    "latent_mean_reg"     : [],
    "curvature_mag"        : [],
    "slope_survival"        : [],
    "center_flatness"        : [],
    "target_slope"         : [],
    "tail_floor"      : [],
    "hi_tail"      : [],
    "sharp_slope"      : [],
}

loss_dict = {
    "total_loss": 0,
    "wasserstein"          : 0,
    "curvature"            : 0,
    "z_mass_shift"         : 0,
    "fold_clamp"           : 0,
    "force_s_shape"           : 0,
    "shift_alignment"      : 0,
    "width_alignment"      : 0,
    "peak_alignment"      : 0,
    "warp_span"            : 0,
    "mean_alignment"       : 0,
    "z_bent_survival"       : 0,
    "z_std_survival"       : 0,
    "early_span_protect"   : 0,
    "density_shape_kde"    : 0,
    "positive_slope"       : 0,
    "latent_std_reg"       : 0,
    "latent_range_reg"     : 0,
    "latent_mean_reg"     : 0,
    "curvature_mag"        : 0,
    "slope_survival"        : 0,
    "center_flatness"        : 0,
    "target_slope"         : 0,
    "tail_floor"      : 0,
    "hi_tail"      : 0,
    "sharp_slope"      : 0,
}


@torch.no_grad()
def test_inn_conditionally(model,
                           train_loader, test_loader,
                           train_mean, train_std, bins,
                           epoch,
                           chunk_size: int = 512):
    """
    Evaluate the INN on test data *in the same way it was trained* and
    draw a quick comparison histogram.  Structure identical to the
    original function – only the strictly-necessary fixes are applied.
    """
    device = next(model.parameters()).device

    # ------------------------------------------------------------------
    # 0. switch to eval -- we'll restore the flag at the end
    # ------------------------------------------------------------------
    was_training = model.training
    model.eval()

    # ------------------------------------------------------------------
    # 1. collect test (obs, truth) and train truth                    ##
    #    NOTE:  we assume the dataloader returns (obs , true) ##
    # ------------------------------------------------------------------
    test_obs_raw  = []
    test_true_raw = []
    for obs, truth in test_loader:          # <-- keep loader order
        test_obs_raw.append(obs)
        test_true_raw.append(truth)

    test_obs_raw  = torch.cat(test_obs_raw , 0).to(device)
    test_true_raw = torch.cat(test_true_raw, 0).to(device)

    train_true_raw = torch.cat([t for t, _ in train_loader], 0).to(device)

    # ------------------------------------------------------------------
    # 2. invert in chunks                                              ##
    #    *pass epoch so compression / alpha logic is identical to train*
    # ------------------------------------------------------------------
    preds_normed = []
    zbent_normed = []
    for i in range(0, len(test_obs_raw), chunk_size):
        chunk = test_obs_raw[i:i + chunk_size]
        pred_c, _, zb_c = model.inverse(chunk, epoch=epoch, alpha=1.0)
        preds_normed.append(pred_c.cpu())
        zbent_normed.append(zb_c.cpu())

    pred_mass_normed = torch.cat(preds_normed, 0)

    print(f"[Epoch {epoch}] pred_mass range (normed): "
          f"{pred_mass_normed.min():.3f} … {pred_mass_normed.max():.3f}")

    # --- 3. undo train-time normalisation
    def _unnorm(arr, mu, sigma):
        """
        arr : np.ndarray
        mu, sigma : Python scalars or 0-D tensors (CPU)
        """
        return arr * float(sigma) + float(mu)
    
    pred_mass  = _unnorm(pred_mass_normed.numpy(),  train_mean,  train_std)
    test_true  = _unnorm(test_true_raw.cpu().numpy(),  train_mean,  train_std)
    test_obs   = _unnorm(test_obs_raw.cpu().numpy(),   train_mean,  train_std)
    train_true = _unnorm(train_true_raw.cpu().numpy(), train_mean,  train_std)

    # ------------------------------------------------------------------
    # 4. quick histogram                                               ##
    # ------------------------------------------------------------------
    plot_histogram(
        train_mass=train_true,
        true_mass=test_true,
        pred_mass=pred_mass,
        obs_mass=test_obs,
        epoch=epoch,
        bins=200,
        range=(120, 220)
    )

    # ------------------------------------------------------------------
    # 5. save prediction snapshot (optional)                           ##
    # ------------------------------------------------------------------
    np.save(f"INN_unbinned_v4_e4_a6/pred_mass_epoch_{epoch}.npy", pred_mass)

    # ------------------------------------------------------------------
    # 6. restore original training mode                                ##
    # ------------------------------------------------------------------
    if was_training:
        model.train()


# -------------------------------
# ✅ Fine-Tuned Translational Perturbation
# -------------------------------
def update_loss_tracking(epoch, total_loss, loss_dict):
    """
    Update loss tracking dictionary with new values.
    Fills in missing keys with 0 to ensure consistent lengths.
    """
    loss_tracking["epoch"].append(epoch)
    loss_tracking["total_loss"].append(total_loss)

    for key in loss_tracking:
        if key in ["epoch", "total_loss"]:
            continue
        val = loss_dict.get(key, 0.0)
        if torch.is_tensor(val):
            val = val.detach().cpu().item()
        loss_tracking[key].append(val)

def plot_loss_evolution():
    """
    Plot the evolution of each loss term over epochs.
    """
    epochs = loss_tracking["epoch"]
    plt.figure(figsize=(12, 6))

    # ?~\~E Plot each loss component
    index = 0
    for key in loss_tracking:
        if key == "epoch":
            continue
        loss_arr = [v.detach().cpu().item() if torch.is_tensor(v) else v for v in loss_tracking[key]]
        if len(loss_arr) != len(epochs):
            print(f"⚠️ Skipping {key}: mismatched length ({len(loss_arr)} vs {len(epochs)})")
            continue
        linestyle = '--' if index >= 10 else '-'
        plt.plot(epochs, np.array(loss_arr), label=key, linestyle=linestyle)
        index += 1

    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.title("INN_unbinned_v4_e4_a6/Loss Component Evolution Over Time")
    plt.legend()
    plt.yscale("log")  # Use log scale to handle different magnitudes
    plt.grid(True)
    plt.savefig("INN_unbinned_v4_e4_a6/loss_evolution.png")
    plt.clf()
    plt.close()

# ------------------------------------------------------------
# plot_histogram  –  last-resort guard against tensor inputs
# ------------------------------------------------------------
def plot_histogram(train_mass, true_mass, pred_mass, obs_mass,
                   epoch, bins, range):

    # ------------------------------------------------------------------
    # force `bins` into a Matplotlib-friendly type
    # ------------------------------------------------------------------
    if torch.is_tensor(bins):
        if bins.numel() == 1:                          # scalar tensor
            bins = int(bins.item())
        else:                                          # array of edges
            bins = bins.detach().cpu().numpy()

    # ── be equally safe with the four arrays (GPU tensor → NumPy)
    def _to_np(x):
        return x.detach().cpu().numpy() if torch.is_tensor(x) else x

    train_mass = _to_np(train_mass)
    true_mass  = _to_np(true_mass)
    pred_mass  = _to_np(pred_mass)
    obs_mass   = _to_np(obs_mass)

    plt.figure(figsize=(12, 6))
    plt.hist(pred_mass.flatten(), bins=bins, range=range, alpha=0.3, label='Predicted', density=True)
    if train_mass is not None:
        plt.hist(train_mass.flatten(), bins=bins, range=range, alpha=1.0, histtype="step",label='Train true', density=True)
    plt.hist(true_mass.flatten(), bins=bins, range=range, alpha=1.0, histtype="step",label='Test true', density=True)
#   plt.hist(train_mass.flatten(), bins=bins, range=range, alpha=1.0, histtype="step",label='Test true', density=True)
    plt.hist(obs_mass.flatten(), bins=bins, range=range, alpha=1.0, histtype="step",label='Test observed', density=True)
    plt.xlabel('Mass')
    plt.ylabel('Density')
    plt.legend()
    plt.title(f'Epoch {epoch} - Conditional Test Prediction')
    plt.savefig(f'INN_unbinned_v4_e4_a6/INN_unbinned_v4_e4_a6_{epoch}.png')
    plt.savefig('INN_unbinned_v4_e4_a6/INN_unbinned_v4_e4_a6_current.png')
    plt.clf()
    plt.close()


def dynamic_freeze_base(model, z_raw, epoch):
    std = z_raw.std().item()
    skew = ((z_raw - z_raw.mean())**3).mean().sign().item()

    freeze = std > 4.0 or abs(skew) > 1.5

    for param in [model.flow._distribution.means,
                  model.flow._distribution.stds,
                  model.flow._distribution.weights]:
        param.requires_grad_(not freeze)




def schedule_z_scale(model, epoch,
                     enable_grad_at=10,          # ← push back
                     clamp_min=2.0, clamp_max=6.0,   # ← keep it large
                     decay_epochs=(30, 50),     # ← decay much later
                     decay_factor=0.97):
    """
    Controls z_scale behavior across epochs:
    - Freezes early on
    - Decays gently after unfreezing
    - Clamps aggressively to avoid explosion

    Args:
        model: TranslationalINN instance.
        epoch: Current epoch.
        enable_grad_at: Epoch to enable z_scale training.
        clamp_min/clamp_max: Bounds on z_scale values.
        decay_epochs: Tuple (start, end) for applying multiplicative decay.
        decay_factor: Decay multiplier during ramp-up period.
    """
    if epoch < enable_grad_at:
        model.z_scale.requires_grad = False
        model.z_scale.data.clamp_(min=clamp_min, max=clamp_max)
    else:
        model.z_scale.requires_grad = True

        # Optional: decay over ramp-up window
        start, end = decay_epochs
        if start <= epoch <= end:
            with torch.no_grad():
                model.z_scale.data *= decay_factor

        # Always clamp
        with torch.no_grad():
            model.z_scale.data.clamp_(min=clamp_min, max=clamp_max)

def smooth_schedule(epoch, mid_epoch, max_epochs, start=0.0, end=1.0, shape="sigmoid"):
    """
    Returns the kicker strength based on the current epoch using various ramp-up strategies.

    Args:
        epoch (int): Current epoch number.
        max_epochs (int): Total number of epochs.
        start (float): Starting kicker strength (typically 0.0).
        end (float): Final kicker strength (typically 1.0).
        shape (str): "sigmoid", "linear", or "none".
    
    Returns:
        float: Scaled kicker strength at this epoch.
    """
    if shape == "none":
        return end
    elif shape == "linear":
        alpha = min(epoch / max_epochs, 1.0)
        return start + (end - start) * alpha
    elif shape == "sigmoid":
        # Smooth sigmoid ramp: start near `start`, end near `end`, steep around halfway
        alpha = epoch / max_epochs
        sigmoid = 1 / (1 + math.exp(-10 * (alpha - mid_epoch/max_epochs)))  # 12 is the steepness
        return start + (end - start) * sigmoid
    else:
        raise ValueError(f"Unknown kicker schedule shape: {shape}")


def safe_soft_clip(x, thresh=4.0, sharpness=1.0, eps=1e-6):
    """
    Smoothly compresses |x| > thresh while keeping the function differentiable.
    Numerically safe version to avoid NaNs from log1p.
    
    Args:
        x (Tensor): Input tensor.
        thresh (float): Threshold beyond which soft clipping is applied.
        sharpness (float): Controls how rapidly the curve bends beyond thresh.
        eps (float): Small constant to prevent log1p(-1) when delta < -1.
    
    Returns:
        Tensor: Clipped output with smooth compression in the tails.
    """
    s = torch.sign(x)
    a = torch.abs(x)
    delta = sharpness * (a - thresh)

    # Clamp to avoid NaN in torch.log1p
    delta_clamped = torch.clamp(delta, min=-1.0 + eps)

    compressed = thresh + (1.0 / sharpness) * torch.log1p(delta_clamped)

    return s * torch.where(a < thresh, a, compressed)


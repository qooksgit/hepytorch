from .abs_optimizer import AbsOptimizer
import torch.optim as optim


class OptV4E4A6(AbsOptimizer):
    def get_optimizer(self, model):
        main_params = []
        warp_body_params = []
        for name, p in model.named_parameters():
            if   "monotonic_warp.final_gain" in name: continue  # ← explicitly skip here
            elif "monotonic_warp" in name:       warp_body_params.append(p)
            elif name == "z_shift":              pass  # handled separately
            else:                                main_params.append(p)

        return optim.Adam([
            {"params": main_params,      "lr": 1e-4},
            {"params": warp_body_params, "lr": 3e-4},
            {"params": [model.z_shift],  "lr": 1e-3},
        ])

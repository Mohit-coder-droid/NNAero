# This code has been copied and modified from the repository https://github.com/peterdsharpe/NeuralFoil/

import torch
import torch.nn as nn
import numpy as np
from typing import Union
from huggingface_hub import hf_hub_download, utils

HF_REPO_ID = "MohitAndSahu/NNAero"

# Cache for distribution tensors to avoid moving to GPU every inference call
_DIST_TENSORS_CACHE = {} 
_MODEL_CACHE = {}

class AeroNet(nn.Module):
    def __init__(self, state_dict: dict):
        """
        Initializes the network structure dynamically based on the loaded weights.
        """
        super().__init__()
        
        # 1. Parse architecture from state_dict keys
        # keys are like "net.0.weight", "net.2.weight"
        layer_indices = sorted(list(set(
            int(k.split('.')[1]) for k in state_dict.keys() if "weight" in k
        )))
        
        layers = []
        for i, idx in enumerate(layer_indices):
            weight_key = f"net.{idx}.weight"
            bias_key = f"net.{idx}.bias"
            
            w = state_dict[weight_key]
            b = state_dict[bias_key]
            
            in_dim = w.shape[1]
            out_dim = w.shape[0]
            
            lin = nn.Linear(in_dim, out_dim)
            lin.weight = nn.Parameter(w)
            lin.bias = nn.Parameter(b)
            
            layers.append(lin)
            
            # Add Activation SiLU for all but the last layer
            if i < len(layer_indices) - 1:
                layers.append(nn.SiLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)    

_eps: float = 10 / np.finfo(np.array(1.0).dtype).max
_ln_eps: float = np.log(_eps)

### For speed, pre-loads parameters with statistics about the training distribution
# Includes the mean, covariance, and inverse covariance of training data in the input latent space (25-dim)
try:
    if "scaled_input_dist" not in _DIST_TENSORS_CACHE:
        dist_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="neuralfoil/scaled_input_distribution.pth"
        )
        _scaled_input_distribution = dict(torch.load(dist_path, map_location="cpu"))
        
        # Cache it in memory so we don't reload dictionary every time
        _DIST_TENSORS_CACHE["scaled_input_dist"] = _scaled_input_distribution

    # Access from memory cache
    d = _DIST_TENSORS_CACHE["scaled_input_dist"]
    N_INPUTS = len(d["mean_inputs_scaled"])

except (utils.EntryNotFoundError, FileNotFoundError, Exception) as e:
    print(f"Warning: 'scaled_input_distribution.pth' failed to load from {HF_REPO_ID}. \nError: {e}")
    print("Mahalanobis distance correction will be disabled.")
    _scaled_input_distribution = {}
    N_INPUTS = 25

# --- PyTorch Helpers ---
def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    # Clip to suppress overflow 
    limit = abs(_ln_eps)
    x = torch.clamp(x, -limit, limit)
    return 1 / (1 + torch.exp(-x))

def _squared_mahalanobis_distance(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Computes Mahalanobis distance using PyTorch tensors.
    """
    # 1. Check if stats are already loaded on this device
    if device not in _DIST_TENSORS_CACHE:
        if not _scaled_input_distribution:
            raise RuntimeError("Input distribution stats not loaded.")
            
        mean_cpu = _scaled_input_distribution["mean_inputs_scaled"]
        inv_cov_cpu = _scaled_input_distribution["inv_cov_inputs_scaled"]
        
        _DIST_TENSORS_CACHE[device] = {
            "mean": mean_cpu.to(torch.float32).to(device),
            "inv_cov": inv_cov_cpu.to(torch.float32).to(device),
        }
    
    stats = _DIST_TENSORS_CACHE[device]
    mean = stats["mean"]      # Shape: (N_inputs,)
    inv_cov = stats["inv_cov"] # Shape: (N_inputs, N_inputs)
    
    # 2. Compute Distance
    # x shape: (Batch, N_inputs)
    # Mean broadcasting: (Batch, N_inputs) - (1, N_inputs)
    x_minus_mean = x - mean.unsqueeze(0)
    
    # (Batch, N_inputs) @ (N_inputs, N_inputs) -> (Batch, N_inputs)
    left_term = x_minus_mean @ inv_cov
    
    # Dot product: sum( (Batch, N_inputs) * (Batch, N_inputs), dim=1 )
    dist_sq = (left_term * x_minus_mean).sum(dim=1)
    
    return dist_sq

def get_aero_from_neuralfoil(
    kulfan_parameters: dict[str, Union[float, np.ndarray, list]],
    alpha: Union[float, np.ndarray],
    Re: Union[float, np.ndarray],
    n_crit: Union[float, np.ndarray] = 9.0,
    xtr_upper: Union[float, np.ndarray] = 1.0,
    xtr_lower: Union[float, np.ndarray] = 1.0,
    model_size: str = "medium",
    device: str = "cpu",
    model_path: str = None 
) -> dict[str, np.ndarray]:
    """
    PyTorch implementation of NeuralFoil inference
    """
    device = torch.device(device)
    
    # --- 1. Load Model (Lazy Loading) ---
    if model_size not in _MODEL_CACHE:
        if model_path is None:
            try:
                cached_path = hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=f"neuralfoil/nn-{model_size}.pth"
                )
                state_dict = torch.load(cached_path, map_location=device)
            except Exception as e:
                raise FileNotFoundError(f"Could not download model. Error: {e}")
        else:
            state_dict = torch.load(model_path, map_location=device)

        model = AeroNet(state_dict).to(device)
        model.eval()
        _MODEL_CACHE[model_size] = model
    
    model = _MODEL_CACHE[model_size].to(device)

    # --- 2. Vectorize & Tensorize Inputs ---
    def as_t(val):
        if isinstance(val, list):
            val = np.array(val)
        arr = np.atleast_1d(val)
        return torch.tensor(arr, dtype=torch.float32, device=device)

    alpha_t = as_t(alpha)
    re_t    = as_t(Re)
    ncrit_t = as_t(n_crit)
    xtr_u_t = as_t(xtr_upper)
    xtr_l_t = as_t(xtr_lower)

    # Handle Weights: (8,) -> (1, 8) or (N, 8)
    w_u = as_t(kulfan_parameters["upper_weights"])
    w_l = as_t(kulfan_parameters["lower_weights"])
    
    if w_u.ndim == 1: w_u = w_u.unsqueeze(0)
    if w_l.ndim == 1: w_l = w_l.unsqueeze(0)

    # Handle Scalars: LE and TE
    le_w = as_t(kulfan_parameters["leading_edge_weight"])
    
    te_val = kulfan_parameters["TE_thickness"]
    if isinstance(te_val, list): te_val = np.array(te_val)
    te_t = as_t(te_val * 50.0) 

    batch_candidates = [w_u.shape[0], w_l.shape[0], alpha_t.shape[0]]
    if le_w.ndim > 0: batch_candidates.append(le_w.shape[0])
    
    N_cases = max(batch_candidates)

    def expand(t, target_n):
        # t is at least 1D: (1, ...) or (N, ...)
        if t.shape[0] == 1 and target_n > 1:
            # Repeat along 0-th dim
            repeat_dims = [target_n] + [1] * (t.ndim - 1)
            return t.repeat(*repeat_dims)
        return t

    w_u = expand(w_u, N_cases)
    w_l = expand(w_l, N_cases)
    
    def make_col(t):
        t = expand(t, N_cases)
        return t.unsqueeze(1) if t.ndim == 1 else t

    le_w_col = make_col(le_w)
    te_t_col = make_col(te_t)
    
    alpha_rad = torch.deg2rad(expand(alpha_t, N_cases))
    re_expanded = expand(re_t, N_cases)
    ncrit_expanded = expand(ncrit_t, N_cases)
    xtr_u_expanded = expand(xtr_u_t, N_cases)
    xtr_l_expanded = expand(xtr_l_t, N_cases)
    re_log = (torch.log(re_expanded) - 12.5) / 3.5
    ncrit_sc = (ncrit_expanded - 9) / 4.5

    # Construct Input Matrix X: [N_cases, 25]
    x = torch.cat([
        w_u,                                        # 0-7
        w_l,                                        # 8-15
        le_w_col,                                   # 16
        te_t_col,                                   # 17
        torch.sin(2 * alpha_rad).unsqueeze(1),      # 18
        torch.cos(alpha_rad).unsqueeze(1),          # 19
        (1 - torch.cos(alpha_rad)**2).unsqueeze(1), # 20
        re_log.unsqueeze(1),                        # 21
        ncrit_sc.unsqueeze(1),                      # 22
        xtr_u_expanded.unsqueeze(1),                # 23
        xtr_l_expanded.unsqueeze(1)                 # 24
    ], dim=1)
    
    with torch.no_grad():
        y = model(x)
        
        # Apply Mahalanobis Correction
        dist = _squared_mahalanobis_distance(x, device)
        y[:, 0] -= dist / (2 * N_INPUTS)
        
        # --- 4. Forward Pass 2 (Flipped/Symmetric) ---
        x_flipped = x.clone()
        
        # Swap Upper/Lower weights (0-7 <-> 8-15) and negate
        x_flipped[:, 0:8] = x[:, 8:16] * -1
        x_flipped[:, 8:16] = x[:, 0:8] * -1
        
        # Flip LE weight (16) and sin(2a) (18)
        x_flipped[:, 16] *= -1
        x_flipped[:, 18] *= -1
        
        # Swap Xtr (23 <-> 24)
        x_flipped[:, 23] = x[:, 24]
        x_flipped[:, 24] = x[:, 23]

        y_flipped = model(x_flipped)
        
        # Apply Mahalanobis Correction to Flipped
        dist_flipped = _squared_mahalanobis_distance(x_flipped, device)
        y_flipped[:, 0] -= dist_flipped / (2 * N_INPUTS)
            
        y_unflipped = y_flipped.clone()
        
        y_unflipped[:, 1] *= -1 
        y_unflipped[:, 3] *= -1 
        
        # Swap Xtr predictions (4 <-> 5)
        y_unflipped[:, 4] = y_flipped[:, 5]
        y_unflipped[:, 5] = y_flipped[:, 4]

        # Swap BL Parameter Blocks
        N_bl = 32
        base = 6
        u_slice = slice(base, base + 3*N_bl)          # Upper Block
        l_slice = slice(base + 3*N_bl, base + 6*N_bl) # Lower Block
        y_unflipped[:, u_slice] = y_flipped[:, l_slice]
        y_unflipped[:, l_slice] = y_flipped[:, u_slice]
        
        u_ue_idx = base + 2*N_bl
        l_ue_idx = base + 5*N_bl
        
        y_unflipped[:, u_ue_idx : u_ue_idx + N_bl] *= -1
        y_unflipped[:, l_ue_idx : l_ue_idx + N_bl] *= -1

        y_fused = (y + y_unflipped) / 2

        def to_np(t): return t.cpu().numpy()

        results = {
            "analysis_confidence": to_np(_sigmoid(y_fused[:, 0])),
            "CL": to_np(y_fused[:, 1] / 2.0), # Maintained / 2.0 scaling
            "CD": to_np(torch.exp((y_fused[:, 2] - 2) * 2)),
            "CM": to_np(y_fused[:, 3] / 20.0),
            "Top_Xtr": to_np(torch.clamp(y_fused[:, 4], 0, 1)),
            "Bot_Xtr": to_np(torch.clamp(y_fused[:, 5], 0, 1)),
        }

        # Extract Raw BL outputs
        u_theta_raw = y_fused[:, 6 : 6 + N_bl]
        u_H_raw     = y_fused[:, 6 + N_bl : 6 + 2*N_bl]
        u_ue_raw    = y_fused[:, 6 + 2*N_bl : 6 + 3*N_bl]
        
        l_theta_raw = y_fused[:, 6 + 3*N_bl : 6 + 4*N_bl]
        l_H_raw     = y_fused[:, 6 + 4*N_bl : 6 + 5*N_bl]
        l_ue_raw    = y_fused[:, 6 + 5*N_bl : 6 + 6*N_bl]

        # Physical Conversions for BL
        re_col = re_expanded.unsqueeze(1)
        
        u_theta = ((10 ** u_theta_raw) - 0.1) / (torch.abs(u_ue_raw) * re_col)
        u_H = 2.6 * torch.exp(u_H_raw)
        
        l_theta = ((10 ** l_theta_raw) - 0.1) / (torch.abs(l_ue_raw) * re_col)
        l_H = 2.6 * torch.exp(l_H_raw)

        for i in range(N_bl):
            results[f"upper_bl_theta_{i}"] = to_np(u_theta[:, i])
            results[f"upper_bl_H_{i}"] = to_np(u_H[:, i])
            results[f"upper_bl_ue/vinf_{i}"] = to_np(u_ue_raw[:, i])
            
            results[f"lower_bl_theta_{i}"] = to_np(l_theta[:, i])
            results[f"lower_bl_H_{i}"] = to_np(l_H[:, i])
            results[f"lower_bl_ue/vinf_{i}"] = to_np(l_ue_raw[:, i])

        if N_cases == 1:
            for k, v in results.items():
                results[k] = v.flatten()

        return results
    

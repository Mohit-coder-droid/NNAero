import torch
import torch.nn as nn
import numpy as np
from typing import Union, Dict
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
    """PyTorch equivalent of your clipped sigmoid."""
    # Clip to suppress overflow (using the calculated _ln_eps)
    # Note: _ln_eps is negative (~-36), so we clamp between -abs and +abs
    limit = abs(_ln_eps)
    x = torch.clamp(x, -limit, limit)
    return 1 / (1 + torch.exp(-x))

def _squared_mahalanobis_distance(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Computes Mahalanobis distance using PyTorch tensors.
    Automatically handles device placement and caching of the statistics.
    """
    # 1. Check if stats are already loaded on this device
    if device not in _DIST_TENSORS_CACHE:
        if not _scaled_input_distribution:
            raise RuntimeError("Input distribution stats not loaded.")
            
        mean_np = _scaled_input_distribution["mean_inputs_scaled"]
        inv_cov_np = _scaled_input_distribution["inv_cov_inputs_scaled"]
        
        _DIST_TENSORS_CACHE[device] = {
            "mean": torch.tensor(mean_np, dtype=torch.float32, device=device),
            "inv_cov": torch.tensor(inv_cov_np, dtype=torch.float32, device=device)
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

def get_aero_from_kulfan_parameters_torch(
    kulfan_parameters: dict[str, Union[float, np.ndarray]],
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
    PyTorch implementation of NeuralFoil inference.
    """
    device = torch.device(device)
    
    # --- 1. Load Model (Lazy Loading) ---
    if model_size not in _MODEL_CACHE:
        # Determine path
        if model_path is None:
            try:
                # This downloads the file if missing, or uses cached version
                # It returns the absolute path to the local cached file
                cached_path = hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=f"neuralfoil/nn-{model_size}.pth"
                )
                
                state_dict = torch.load(cached_path, map_location=device)
                
            except (utils.EntryNotFoundError, utils.RepositoryNotFoundError, Exception) as e:
                raise FileNotFoundError(
                    f"Could not download model '{model_size}' from Hugging Face ({HF_REPO_ID}). "
                    f"Error: {e}"
                )
        else:
            # User provided a custom local path override
            state_dict = torch.load(model_path, map_location=device)

        model = AeroNet(state_dict).to(device)
        model.eval()
        _MODEL_CACHE[model_size] = model
    
    model = _MODEL_CACHE[model_size]

    # --- 2. Vectorize & Tensorize Inputs ---
    # Helper to convert inputs to tensors
    def as_t(val):
        arr = np.atleast_1d(val)
        return torch.tensor(arr, dtype=torch.float32, device=device)

    # Convert scalar/array inputs
    alpha_t = as_t(alpha)
    re_t = as_t(Re)
    ncrit_t = as_t(n_crit)
    xtr_u_t = as_t(xtr_upper)
    xtr_l_t = as_t(xtr_lower)
    
    # Handle Kulfan arrays
    w_u = torch.tensor(kulfan_parameters["upper_weights"], dtype=torch.float32, device=device)
    w_l = torch.tensor(kulfan_parameters["lower_weights"], dtype=torch.float32, device=device)
    
    # Ensure dimensions [Batch, 8]
    if w_u.ndim == 1: w_u = w_u.unsqueeze(0)
    if w_l.ndim == 1: w_l = w_l.unsqueeze(0)

    # Determine Batch Size
    N_cases = max(alpha_t.shape[0], w_u.shape[0])

    # Broadcast helper
    def expand(t, dim=0):
        if t.shape[0] == 1 and N_cases > 1:
            return t.repeat(N_cases, *([1]*(t.ndim-1)))
        return t

    # Expand all inputs to [N_cases, ...]
    w_u = expand(w_u)
    w_l = expand(w_l)
    le_w = expand(as_t(kulfan_parameters["leading_edge_weight"])).unsqueeze(1)
    te_t = expand(as_t(kulfan_parameters["TE_thickness"] * 50)).unsqueeze(1) # Note the *50 scaling
    
    # Pre-calculations
    alpha_rad = torch.deg2rad(expand(alpha_t))
    re_log = (torch.log(expand(re_t)) - 12.5) / 3.5
    ncrit_sc = (expand(ncrit_t) - 9) / 4.5
    
    # Construct Input Matrix X: [N_cases, 25]
    # Order: [Upper(8), Lower(8), LE(1), TE(1), sin2a, cosa, 1-cos2a, Re, ncrit, xtr_u, xtr_l]
    x = torch.cat([
        w_u, 
        w_l, 
        le_w, 
        te_t,
        torch.sin(2 * alpha_rad).unsqueeze(1),
        torch.cos(alpha_rad).unsqueeze(1),
        (1 - torch.cos(alpha_rad)**2).unsqueeze(1),
        re_log.unsqueeze(1),
        ncrit_sc.unsqueeze(1),
        expand(xtr_u_t).unsqueeze(1),
        expand(xtr_l_t).unsqueeze(1)
    ], dim=1)

    # --- 3. Forward Pass 1 (Standard) ---
    with torch.no_grad():
        y = model(x)
        
        # Apply Mahalanobis Correction
        # Formula: y[0] = y[0] - dist / (2 * N_inputs)
        dist = _squared_mahalanobis_distance(x, device)
        y[:, 0] -= dist / (2 * N_INPUTS)
        
        # --- 4. Forward Pass 2 (Flipped/Symmetric) ---
        x_flipped = x.clone()
        
        # Symmetrize Inputs:
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
            
        # --- 5. Fuse Results (Un-flip) ---
        y_unflipped = y_flipped.clone()
        
        # Negate anti-symmetric outputs: CL (1) and CM (3)
        y_unflipped[:, 1] *= -1 
        y_unflipped[:, 3] *= -1 
        
        # Swap Xtr predictions (4 <-> 5)
        y_unflipped[:, 4] = y_flipped[:, 5]
        y_unflipped[:, 5] = y_flipped[:, 4]

        # Swap BL Parameter Blocks
        # Structure based on Data.N = 32
        # [0:6] Scalars
        # [6:38] U_Theta | [38:70] U_H | [70:102] U_Ue
        # [102:134] L_Theta | [134:166] L_H | [166:198] L_Ue
        
        N_bl = 32
        base = 6
        
        # Define Slices
        u_slice = slice(base, base + 3*N_bl)          # Upper Block (Theta, H, Ue)
        l_slice = slice(base + 3*N_bl, base + 6*N_bl) # Lower Block (Theta, H, Ue)
        
        # Swap the entire upper and lower chunks
        y_unflipped[:, u_slice] = y_flipped[:, l_slice]
        y_unflipped[:, l_slice] = y_flipped[:, u_slice]
        
        # Fix Ue/Vinf: These need to be negated after swapping
        # Indices relative to the full array:
        u_ue_idx = base + 2*N_bl # Start of Upper Ue
        l_ue_idx = base + 5*N_bl # Start of Lower Ue
        
        # Negate velocity ratios
        y_unflipped[:, u_ue_idx : u_ue_idx + N_bl] *= -1
        y_unflipped[:, l_ue_idx : l_ue_idx + N_bl] *= -1

        # Average the two passes
        y_fused = (y + y_unflipped) / 2

        # --- 6. Post-Processing & Output ---
        
        # Confidence (Sigmoid)
        analysis_conf = _sigmoid(y_fused[:, 0])
        
        # Aero Coefficients
        cl = y_fused[:, 1] / 2.0  
        cd = torch.exp((y_fused[:, 2] - 2) * 2)
        cm = y_fused[:, 3] / 20.0
        top_xtr = torch.clamp(y_fused[:, 4], 0, 1)
        bot_xtr = torch.clamp(y_fused[:, 5], 0, 1)

        # Extract Raw BL outputs
        u_theta_raw = y_fused[:, 6 : 6 + N_bl]
        u_H_raw = y_fused[:, 6 + N_bl : 6 + 2*N_bl]
        u_ue_raw = y_fused[:, 6 + 2*N_bl : 6 + 3*N_bl]
        
        l_theta_raw = y_fused[:, 6 + 3*N_bl : 6 + 4*N_bl]
        l_H_raw = y_fused[:, 6 + 4*N_bl : 6 + 5*N_bl]
        l_ue_raw = y_fused[:, 6 + 5*N_bl : 6 + 6*N_bl]

        # Physical Conversions for BL
        # Re must be broadcast to (N, 1) for division
        re_col = expand(re_t).unsqueeze(1)
        
        u_theta = ((10 ** u_theta_raw) - 0.1) / (torch.abs(u_ue_raw) * re_col)
        u_H = 2.6 * torch.exp(u_H_raw)
        
        l_theta = ((10 ** l_theta_raw) - 0.1) / (torch.abs(l_ue_raw) * re_col)
        l_H = 2.6 * torch.exp(l_H_raw)

        def to_np(t): return t.cpu().numpy()
        
        results = {
            "analysis_confidence": to_np(analysis_conf),
            "CL": to_np(cl),
            "CD": to_np(cd),
            "CM": to_np(cm),
            "Top_Xtr": to_np(top_xtr),
            "Bot_Xtr": to_np(bot_xtr),
        }

        for i in range(N_bl):
            results[f"upper_bl_theta_{i}"] = to_np(u_theta[:, i])
            results[f"upper_bl_H_{i}"] = to_np(u_H[:, i])
            results[f"upper_bl_ue/vinf_{i}"] = to_np(u_ue_raw[:, i])
            
            results[f"lower_bl_theta_{i}"] = to_np(l_theta[:, i])
            results[f"lower_bl_H_{i}"] = to_np(l_H[:, i])
            results[f"lower_bl_ue/vinf_{i}"] = to_np(l_ue_raw[:, i])

        for k, v in results.items():
            results[k] = v.flatten()

        return results
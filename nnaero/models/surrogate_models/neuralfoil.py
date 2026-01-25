# This code has been copied and modified from the repository https://github.com/peterdsharpe/NeuralFoil/

# TODO: Handle multi-point optimization as for each airfoil we will be calculating aero at different angles
# Making a separate class for each different angle is not a good idea as it's going to consume too much storage, while we only require those storage at the time of calculating aero. 

# TODO: Modify the get_aero function to handle compressible flow (see Aerosandbox)

import torch
import torch.nn as nn
import numpy as np
from typing import Union
from huggingface_hub import hf_hub_download, utils

from nnaero.utils import *
from nnaero.modelling.splines import cosine_hermite_patch

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

def _mesh_kulfan_and_conditions(
    kulfan_parameters: dict[str, Union[float, np.ndarray, list]],
    alpha: Union[float, np.ndarray],
    Re: Union[float, np.ndarray]
):
    """
    Creates a cross-product (meshgrid) of Airfoils and Operating Conditions.
    
    Args:
        kulfan_parameters: Dict with weights of shape (N_airfoils, 8)
        alpha: Array of shape (N_alpha,)
        Re: Array of shape (N_alpha,) - Must be paired with alpha
        
    Returns:
        tuple: (meshed_kulfan, meshed_alpha, meshed_Re, (N_airfoils, N_alpha))
        
    Resulting shapes will be (N_airfoils * N_alpha, ...)
    """
    # 1. Determine Sizes
    w_u = np.atleast_2d(kulfan_parameters["upper_weights"]) 
    N_airfoils = w_u.shape[0]
    
    alpha = np.atleast_1d(alpha)
    Re = np.atleast_1d(Re)
    N_alpha = alpha.shape[0]

    # Validate Re/Alpha pairing
    if Re.shape[0] != N_alpha:
        raise ValueError(f"Alpha and Re must have same length. Got {N_alpha} and {Re.shape[0]}")

    # (N, 8) -> (N * M, 8)
    meshed_kulfan = {}
    for key, val in kulfan_parameters.items():
        val_arr = np.array(val)
        
        if val_arr.ndim == 1 and val_arr.shape[0] == N_airfoils:
            meshed_kulfan[key] = np.repeat(val_arr, N_alpha, axis=0)
        elif val_arr.ndim > 1 and val_arr.shape[0] == N_airfoils:
            meshed_kulfan[key] = np.repeat(val_arr, N_alpha, axis=0)
        elif val_arr.ndim == 0 or val_arr.shape[0] == 1:
             meshed_kulfan[key] = np.repeat(val_arr, N_airfoils * N_alpha, axis=0)
        else:
             meshed_kulfan[key] = val_arr

    # (M,) -> (N * M,)
    meshed_alpha = np.tile(alpha, N_airfoils)
    meshed_Re = np.tile(Re, N_airfoils)
    
    return meshed_kulfan, meshed_alpha, meshed_Re, (N_airfoils, N_alpha)

def get_aero_from_kulfan_single(
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
    
def get_aero_from_kulfan(
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
    n_airfoils = np.atleast_2d(kulfan_parameters["upper_weights"]).shape[0]
    n_conds = np.atleast_1d(alpha).shape[0]
    
    needs_meshing = (n_airfoils > 1) and (n_conds > 1) and (n_airfoils != n_conds)

    if needs_meshing:
        # 1. Expand inputs to (N*M)
        m_kulfan, m_alpha, m_Re, original_shape = _mesh_kulfan_and_conditions(
            kulfan_parameters, alpha, Re
        )
        
        # 2. Run Inference on flattened arrays
        # The model sees one massive batch of size N*M
        flat_results = get_aero_from_kulfan_single(m_kulfan, m_alpha, m_Re,
                                                n_crit=n_crit,
                                                xtr_upper=xtr_upper,
                                                xtr_lower=xtr_lower,
                                                model_size=model_size,
                                                device=device,
                                                model_path=model_path
                                                   )
        
        # 3. Reshape Outputs back to (N_airfoils, N_alpha)
        shaped_results = {}
        N, M = original_shape
        
        for key, val in flat_results.items():
            if val.size == N * M:
                shaped_results[key] = val.reshape(N, M)
            
            else:
                 shaped_results[key] = val
                 
        return shaped_results

    else:
        # (1-to-1 or 1-to-N broadcasting)
        return get_aero_from_kulfan_single(kulfan_parameters, alpha, Re,
                                                n_crit=n_crit,
                                                xtr_upper=xtr_upper,
                                                xtr_lower=xtr_lower,
                                                model_size=model_size,
                                                device=device,
                                                model_path=model_path)

# TODO: Include 360 degree effects
# TODO: Implement control surfaces
def get_aero(
    kulfan_parameters: dict[str, Union[float, np.ndarray, list]],
    alpha: Union[float, np.ndarray],
    Re: Union[float, np.ndarray],
    mach: Union[float, np.ndarray] = 0.0,
    n_crit: Union[float, np.ndarray] = 9.0,
    xtr_upper: Union[float, np.ndarray] = 1.0,
    xtr_lower: Union[float, np.ndarray] = 1.0,
    max_thickness: Union[float, np.ndarray] = None,
    model_size: str = "medium",
    device: str = "cpu",
    model_path: str = None ,
    ):
    alpha = np.atleast_1d(alpha)
    Re = np.atleast_1d(Re)
    mach = np.atleast_1d(mach)
    
    w_u = np.atleast_2d(kulfan_parameters["upper_weights"])
    N_airfoils = w_u.shape[0]
    N_conds = alpha.shape[0]
    
    # --- 3. NeuralFoil Inference (The "Raw" Pass) ---
    alpha_input = np.mod(alpha  + 180, 360) - 180

    raw_results = get_aero_from_kulfan(
        kulfan_parameters=kulfan_parameters,
        alpha=alpha_input,
        Re=Re,
        n_crit=n_crit,
        xtr_upper=xtr_upper,
        xtr_lower=xtr_lower,
        model_size=model_size,
        device=device,
        model_path=model_path
    )
    
    CL = raw_results["CL"]
    CD = raw_results["CD"]
    CM = raw_results["CM"]
    
    N_bl = 32 
    
    if N_conds!=1:
        u_ue = np.stack([raw_results[f"upper_bl_ue/vinf_{i}"] for i in range(N_bl)],axis=2)
        l_ue = np.stack([raw_results[f"lower_bl_ue/vinf_{i}"] for i in range(N_bl)],axis=2)
    else:
        u_ue = np.stack([raw_results[f"upper_bl_ue/vinf_{i}"] for i in range(N_bl)],axis=1)
        l_ue = np.stack([raw_results[f"lower_bl_ue/vinf_{i}"] for i in range(N_bl)],axis=1)
    
    if u_ue.ndim==3:
        u_ue = u_ue.reshape(-1, N_bl)
        l_ue = l_ue.reshape(-1, N_bl)
    
    Cpmin_0 = [softmin( 
        *np.concatenate([1 - u_ue**2, 1 - l_ue**2], axis=1)[i], softness=0.01
    ) for i in range(u_ue.shape[0])]
    
    Top_Xtr = raw_results["Top_Xtr"]
    Bot_Xtr = raw_results["Bot_Xtr"]

    # --- 4B. Compressibility Effects ---
    Cpmin_0 = np.array([softmin(Cpmin_0[i], 0, softness=0.001) for i in range(len(Cpmin_0))])
    
    if N_conds!=1:
        Cpmin_0 = Cpmin_0.reshape(N_airfoils, N_conds)
    
    mach_crit = (
        1.011571026701678
        - Cpmin_0
        + 0.6582431351007195 * (-Cpmin_0) ** 0.6724789439840343
    ) ** -0.5504677038358711
    
    mach_dd = mach_crit + (0.1 / 320) ** (1 / 3)

    # Beta / Prandtl-Glauert
    gamma = 1.4
    beta_sq = 1 - mach**2
    beta = (
            softmax(
                beta_sq,
                -beta_sq,
                softness=0.5,  
            )
            ** 0.5
        )

    CL = CL / beta
    CM = CM / beta
    Cpmin = Cpmin_0 / beta
        
    ### Step 3: modify CL based on buffet and supersonic considerations
    # Accounts approximately for the lift drop due to buffet.
    mach = np.atleast_1d(mach) # Shape: (N_cond,) or (1,)
    mach_dd = np.atleast_2d(mach_dd) # Shape: (N_airfoils, N_cond)
    max_thickness = np.atleast_1d(max_thickness) # Shape: (N_airfoils,) or (N_airfoils, 1) 

    if mach.ndim == 1 and mach.shape[0] == N_conds:
        # (N_conds,) -> (1, N_conds) -> (N_airfoils, N_conds)
        mach_grid = np.tile(mach, (N_airfoils, 1))
    elif mach.ndim == 1 and mach.shape[0] == 1:
        # Scalar case: fill grid
        mach_grid = np.full((N_airfoils, N_conds), mach[0])
    else:
        mach_grid = mach

    if max_thickness.ndim == 1 and max_thickness.shape[0] == N_airfoils:
        # (N_airfoils,) -> (N_airfoils, 1) -> (N_airfoils, N_conds)
        max_thickness_grid = np.tile(max_thickness[:, None], (1, N_conds))
    elif max_thickness.ndim == 0:
        max_thickness_grid = np.full((N_airfoils, N_conds), max_thickness)
    else:
        max_thickness_grid = max_thickness


    buffet_factor = blend(
        50 * (mach_grid - (mach_dd + 0.04)),  
        blend((mach_grid - 1) / 0.1, 1, 0.5),
        1,
    )

    cla_supersonic_ratio_factor = blend(
        (mach_grid - 1) / 0.1,
        4 / (2 * np.pi),
        1,
    )

    CL = CL * buffet_factor * cla_supersonic_ratio_factor

    if max_thickness is not None:
        term_quartic = 80 * (mach_grid - mach_crit) ** 4
        
        term_hermite = cosine_hermite_patch(
            mach_grid,
            x_a=mach_dd,
            x_b=1.1,
            f_a=80 * (0.1 / 320) ** (4 / 3),
            f_b=0.8 * max_thickness_grid, 
            dfdx_a=0.1,
            dfdx_b=-0.8 * max_thickness_grid * 8, 
        )
        
        term_supersonic = blend(
            8 * 2 * (mach_grid - 1.1) / (1.2 - 0.8),
            0.8 * 0.8 * max_thickness_grid, 
            1.2 * 0.8 * max_thickness_grid, 
        )

        mask_subcrit = mach_grid < mach_crit
        mask_drag_rise = (mach_grid >= mach_crit) & (mach_grid < mach_dd)
        mask_hermite = (mach_grid >= mach_dd) & (mach_grid < 1.1)
        
        CD_wave = np.select(
            condlist=[mask_subcrit, mask_drag_rise, mask_hermite],
            choicelist=[0.0, term_quartic, term_hermite],
            default=term_supersonic
        )
        
        CD = CD + CD_wave


    has_ac_shift = np.clip((mach_grid - (mach_dd + 0.06)) / 0.06, 0, 1)

    if np.ndim(alpha) == 1 and len(alpha) == N_conds:
        alpha_grid = np.tile(alpha, (N_airfoils, 1))
    else:
        alpha_grid = alpha

    CM_shift = -0.25 * cosd(alpha_grid) * CL - 0.25 * sind(alpha_grid) * CD

    CM = CM + blend(
        has_ac_shift,
        CM_shift,
        0,
    )

    results = {
        "analysis_confidence": raw_results["analysis_confidence"],
        "CL": CL,
        "CD": CD,
        "CM": CM,
        "Cpmin": Cpmin,
        "Top_Xtr": Top_Xtr,
        "Bot_Xtr": Bot_Xtr,
        "mach_crit": mach_crit,
        "mach_dd": mach_dd,
        "Cpmin_0": Cpmin_0
    }
    
    # Add BL arrays (pass through)
    for k, v in raw_results.items():
        if "bl_" in k:
            results[k] = v

    return results

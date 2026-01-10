import numpy as np
import torch
from collections import OrderedDict

class DesignVariables:
    def __init__(self):
        self._vars_meta = OrderedDict()
        self._total_dof = 0
        self._dtype = np.float64
        self._reserved_names = set(dir(self))

    def add(self, name_or_list, initial_value=None, lower_bound=None, upper_bound=None, requires_grad=True):
        """
        Add variables with optional bounds.
        
        Args:
            name_or_list: Variable name (str) or list of tuples.
            initial_value: Value (if adding single variable).
            lower_bound: Lower bound (scalar or array). Default: -inf
            upper_bound: Upper_bound bound (scalar or array). Default: +inf
            requires_grad: Whether to track gradients.
            
        Tuple formats for list input:
            (name, val)                          -> Uses defaults/kwargs for bounds
            (name, val, lb, ub)                  -> Specific bounds
            (name, val, lb, ub, requires_grad)   -> Specific bounds + grad
        """
        # Case 1: List of variables
        if isinstance(name_or_list, (list, tuple)) and initial_value is None:
            for item in name_or_list:
                # Defaults come from the function kwargs (e.g., global bounds for this list)
                n, v = item[0], item[1]
                lb, ub, rg = lower_bound, upper_bound, requires_grad

                if len(item) == 4:
                    lb, ub = item[2], item[3]
                elif len(item) == 5:
                    lb, ub, rg = item[2], item[3], item[4]
                elif len(item) != 2:
                    raise ValueError(f"Tuple must be (name, val), (name, val, lb, ub) or (name, val, lb, ub, grad). Got: {len(item)} items.")

                self._add_single(n, v, lb, ub, rg)
        
        # Case 2: Single variable
        elif isinstance(name_or_list, str) and initial_value is not None:
             self._add_single(name_or_list, initial_value, lower_bound, upper_bound, requires_grad)
        else:
             raise ValueError("Invalid usage. Use add('name', val) or add([('name', val), ...])")

    def _add_single(self, name, value, lower_bound, upper_bound, requires_grad):
        """Internal helper to register a single variable and process bounds."""
        if name in self._reserved_names:
            raise ValueError(f"'{name}' is a reserved method name.")
        if hasattr(self, name):
            raise ValueError(f"Variable '{name}' already exists.")

        # 1. Standardize Data
        data = np.array(value, dtype=self._dtype)
        size = data.size
        shape = data.shape
        
        # 2. Process Bounds (Broadcasting Logic)
        lb_vec = self._expand_bound(lower_bound, size, default=-np.inf)
        ub_vec = self._expand_bound(upper_bound, size, default=np.inf)

        # 3. Store Metadata
        self._vars_meta[name] = {
            'shape': shape,
            'size': size,
            'start_idx': self._total_dof,
            'end_idx': self._total_dof + size,
            'lower_bound': lb_vec,
            'upper_bound': ub_vec,
            'requires_grad': requires_grad
        }
        self._total_dof += size
        
        # 4. Set Attribute
        setattr(self, name, data)

    def _expand_bound(self, bound, size, default):
        """Helper to broadcast scalar bounds to array size."""
        if bound is None:
            return np.full(size, default, dtype=self._dtype)
        
        b = np.array(bound, dtype=self._dtype)
        
        if b.size == 1:
            # Scalar -> Expand to full size
            return np.full(size, b.item(), dtype=self._dtype)
        elif b.size == size:
            # Array -> Flatten to match vector
            return b.flatten()
        else:
            raise ValueError(f"Bound size mismatch. Var size: {size}, Bound size: {b.size}")

    def to_vector(self):
        """Convert attributes to flat vector."""
        vector = np.zeros(self._total_dof, dtype=self._dtype)
        for name, meta in self._vars_meta.items():
            val = getattr(self, name)
            if isinstance(val, torch.Tensor):
                val = val.detach().cpu().numpy()
            vector[meta['start_idx']:meta['end_idx']] = val.flatten()
        return vector

    def update_from_vector(self, vector, to_torch=False, device='cpu'):
        """Update attributes from flat vector."""
        if len(vector) != self._total_dof:
            raise ValueError(f"Size mismatch: Expected {self._total_dof}, got {len(vector)}")

        for name, meta in self._vars_meta.items():
            flat_data = vector[meta['start_idx']:meta['end_idx']]
            reshaped = flat_data.reshape(meta['shape'])
            
            if to_torch:
                t = torch.tensor(reshaped, dtype=torch.float32, device=device)
                if meta['requires_grad']:
                    t.requires_grad = True
                setattr(self, name, t)
            else:
                setattr(self, name, reshaped)

    def get_bounds_vectors(self):
        """
        Returns (xl, xu) - the Full Lower and Upper_bound bound vectors.
        Perfect for passing directly to pymoo or scipy.optimize.
        """
        xl = np.zeros(self._total_dof, dtype=self._dtype)
        xu = np.zeros(self._total_dof, dtype=self._dtype)
        
        for meta in self._vars_meta.values():
            xl[meta['start_idx']:meta['end_idx']] = meta['lower_bound']
            xu[meta['start_idx']:meta['end_idx']] = meta['upper_bound']
            
        return xl, xu

    def get_gradient_vector(self):
        """Collect gradients from torch tensors."""
        grad_vec = np.zeros(self._total_dof, dtype=self._dtype)
        for name, meta in self._vars_meta.items():
            val = getattr(self, name)
            if isinstance(val, torch.Tensor) and val.grad is not None:
                grad_vec[meta['start_idx']:meta['end_idx']] = val.grad.detach().cpu().numpy().flatten()
        return grad_vec
    
    def __repr__(self):
        return f"<DesignVariables: {list(self._vars_meta.keys())} (Total DoF: {self._total_dof})>"
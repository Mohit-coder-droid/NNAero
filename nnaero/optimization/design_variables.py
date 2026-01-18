import numpy as np
import torch
from collections import OrderedDict

class DesignVariables:
    def __init__(self):
        self._vars_meta = OrderedDict()
        self._total_feature_dof = 0
        self._dtype = np.float64
        self._batch_size = None
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
                n, v = item[0], item[1]
                lb, ub, rg = lower_bound, upper_bound, requires_grad
                
                if len(item) == 4:
                    lb, ub = item[2], item[3]
                elif len(item) == 5:
                    lb, ub, rg = item[2], item[3], item[4]
                elif len(item) != 2:
                    raise ValueError(f"Tuple must be (name, val) or (name, val, lb, ub, [grad]).")

                self._add_single(n, v, lb, ub, rg)
        
        # Case 2: Single variable
        elif isinstance(name_or_list, str) and initial_value is not None:
             self._add_single(name_or_list, initial_value, lower_bound, upper_bound, requires_grad)
        else:
             raise ValueError("Invalid usage. Use add('name', val) or add([('name', val), ...])")

    def _add_single(self, name, value, lower_bound, upper_bound, requires_grad):
        if name in self._reserved_names:
            raise ValueError(f"'{name}' is a reserved method name.")
        if hasattr(self, name):
            raise ValueError(f"Variable '{name}' already exists.")

        # 1. Standardize Data
        data = np.array(value, dtype=self._dtype)
        shape = data.shape
        
        if len(shape) < 1:
            raise ValueError(f"Variable '{name}' must have at least one dimension (the batch dimension).")

        # 2. Batch Consistency Check
        current_batch = shape[0]
        if self._batch_size is None:
            self._batch_size = current_batch
        elif current_batch != self._batch_size:
            raise ValueError(f"Batch size mismatch for '{name}'. Expected {self._batch_size}, got {current_batch}.")

        # 3. Feature Dimensions (excluding batch dim)
        # If input is (Batch, 10), feature is (10,). If input is (Batch,), feature is scalar (1,)
        feature_shape = shape[1:]
        feature_size = int(np.prod(feature_shape)) if feature_shape else 1

        # 4. Process Bounds (Broadcast to FEATURE size, not total size)
        # Bounds are usually constraints on the geometry, independent of the batch index.
        lb_vec = self._expand_bound(lower_bound, feature_size, default=-np.inf)
        ub_vec = self._expand_bound(upper_bound, feature_size, default=np.inf)

        # 5. Store Metadata
        self._vars_meta[name] = {
            'shape': shape,                 # Full shape: (B, D1, D2...)
            'feature_shape': feature_shape, # Feature shape: (D1, D2...)
            'feature_size': feature_size,   # Flattened feature length
            'start_idx': self._total_feature_dof,
            'end_idx': self._total_feature_dof + feature_size,
            'lower_bound': lb_vec,
            'upper_bound': ub_vec,
            'requires_grad': requires_grad
        }
        self._total_feature_dof += feature_size
        
        # 6. Set Attribute
        setattr(self, name, data)

    def _expand_bound(self, bound, size, default):
        """Helper to broadcast scalar bounds to FEATURE size."""
        if bound is None:
            return np.full(size, default, dtype=self._dtype)
        
        b = np.array(bound, dtype=self._dtype)
        
        if b.size == 1:
            return np.full(size, b.item(), dtype=self._dtype)
        elif b.size == size:
            return b.flatten()
        else:
            raise ValueError(f"Bound size mismatch. Feature size: {size}, Bound size: {b.size}")

    def to_vector(self):
        """
        Convert attributes to a Batch x Features matrix.
        Returns: numpy array of shape (n_batch, n_total_features)
        """
        if self._batch_size is None:
            return np.empty((0, 0))

        cols = []
        for name, meta in self._vars_meta.items():
            val = getattr(self, name)
            if isinstance(val, torch.Tensor):
                val = val.detach().cpu().numpy()
            
            # Reshape (Batch, D1, D2...) -> (Batch, Feature_Size)
            # If val is (Batch,), reshape -> (Batch, 1)
            flat_features = val.reshape(self._batch_size, -1)
            cols.append(flat_features)
            
        return np.hstack(cols)

    def update_from_vector(self, vector, to_torch=False, device='cpu'):
        """
        Update attributes from a matrix.
        Args:
            vector: Array of shape (n_batch, n_total_features)
        """
        # Validation
        if vector.shape[0] != self._batch_size:
            raise ValueError(f"Batch dimension mismatch: Expected {self._batch_size}, got {vector.shape[0]}")
        if vector.shape[1] != self._total_feature_dof:
            raise ValueError(f"Feature dimension mismatch: Expected {self._total_feature_dof}, got {vector.shape[1]}")

        for name, meta in self._vars_meta.items():
            # Slice the columns corresponding to this variable
            # Shape: (Batch, Feature_Size)
            flat_data = vector[:, meta['start_idx']:meta['end_idx']]
            
            # Reshape back to original dimensions: (Batch, D1, D2...)
            original_shape = (self._batch_size, *meta['feature_shape'])
            reshaped = flat_data.reshape(original_shape)
            
            if to_torch:
                t = torch.tensor(reshaped, dtype=torch.float32, device=device)
                if meta['requires_grad']:
                    t.requires_grad = True
                setattr(self, name, t)
            else:
                setattr(self, name, reshaped)

    def get_bounds_vectors(self):
        """
        Returns (xl, xu) - the Feature Lower and Upper bound vectors.
        Returns shapes: (n_total_features,), (n_total_features,)
        
        Note: These are 1D vectors representing the bounds for a single sample.
        Most optimizers expect bounds for the design variables of one individual.
        """
        xl = np.zeros(self._total_feature_dof, dtype=self._dtype)
        xu = np.zeros(self._total_feature_dof, dtype=self._dtype)
        
        for meta in self._vars_meta.values():
            xl[meta['start_idx']:meta['end_idx']] = meta['lower_bound']
            xu[meta['start_idx']:meta['end_idx']] = meta['upper_bound']
            
        return xl, xu
    
    @property
    def batch_size(self):
        return self._batch_size if self._batch_size is not None else 0
        
    @property
    def num_features(self):
        return self._total_feature_dof

    # TODO: See whether this function is going to be helpful anywhere or not 
    def get_gradient_vector(self):
        """Collect gradients from torch tensors."""
        grad_vec = np.zeros(self._total_feature_dof, dtype=self._dtype)
        for name, meta in self._vars_meta.items():
            val = getattr(self, name)
            if isinstance(val, torch.Tensor) and val.grad is not None:
                grad_vec[meta['start_idx']:meta['end_idx']] = val.grad.detach().cpu().numpy().flatten()
        return grad_vec
    
    def __repr__(self):
        return f"DesignVariables: (Total DoF: {self._total_feature_dof};)"
# TODO: I can add some statistical analysis for this airfoil batch, which can be done in case of some airfoil datasets


from nnaero.geometry.airfoil import Airfoil
from typing import Union, List, Dict
import numpy as np

class AirfoilBatch:
    """
    A batch of airfoils stored as a (B, N, 2) array.
    """
    def __init__(
        self,
        names: List[str] = None,
        coordinates: Union[List, np.ndarray] = None
    ):
        self.names = names if names else ["Untitled"]*len(coordinates)
        self.coordinates = None

        # Case A: Input is already a 3D Batch Array (B, N, 2)
        if isinstance(coordinates, np.ndarray) and coordinates.ndim == 3:
            # Ensure (Batch, N, 2)
            if coordinates.shape[2] != 2 and coordinates.shape[1] == 2:
                self.coordinates = np.transpose(coordinates, (0, 2, 1))
            else:
                self.coordinates = coordinates
        
        # Case B: Input is a list (Files, Names, or Arrays)
        elif isinstance(coordinates, list) or isinstance(names, list):
            n_items = len(self.names)
            coords_list = coordinates if coordinates else [None] * n_items
            
            if len(coords_list) != n_items:
                 raise ValueError("Batch mismatch: len(names) != len(coordinates)")

            resolved_coords = []
            for n, c in zip(self.names, coords_list):
                af = Airfoil(name=n, coordinates=c)
                if af.coordinates is None:
                    raise ValueError(f"Could not resolve airfoil: {n}")
                resolved_coords.append(af.coordinates)

            self.coordinates = np.array(resolved_coords)
            
        self.count = len(self.coordinates)
            
    def LE_index(self) -> Union[int, np.ndarray]:
        """
        Returns the index(es) of the leading edge point in the airfoil coordinates.
        """
        # Handle Batch (B, N, 2) -> Return shape (B,)
        return np.argmin(self.coordinates[..., 0], axis=1)
        
    def lower_coordinates(self) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Returns [x, y] coordinates of the lower surface (LE to TE).
        For Batch: Returns a List of arrays.
        """
        le_idx = self.LE_index()
        
        # Must return a list because length of surface may vary per airfoil
        return [c[idx:, :] for c, idx in zip(self.coordinates, le_idx)]

    def upper_coordinates(self) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Returns [x, y] coordinates of the upper surface (TE to LE).
        For Batch: Returns a List of arrays.
        """
        le_idx = self.LE_index()

        # Must return a list because length of surface may vary per airfoil
        return [c[: idx + 1, :] for c, idx in zip(self.coordinates, le_idx)]

    def local_camber(
        self, x_over_c: Union[float, np.ndarray] = np.linspace(0, 1, 101)
    ) -> Union[float, np.ndarray]:
        """
        Returns the local camber of the airfoil(s).
        Returns shape (Batch_Size, N_points) for batches, or (N_points,) for single.
        """
        # Ensure x_over_c is an array
        x_vals = np.atleast_1d(x_over_c)

        batch_size = self.coordinates.shape[0]
        camber = np.zeros((batch_size, len(x_vals)))

        uppers = self.upper_coordinates()
        lowers = self.lower_coordinates()

        for i in range(batch_size):
            # Flip upper to ensure increasing x for np.interp (LE -> TE)
            u_pts = uppers[i][::-1]
            l_pts = lowers[i]

            u_interp = np.interp(x_vals, u_pts[:, 0], u_pts[:, 1])
            l_interp = np.interp(x_vals, l_pts[:, 0], l_pts[:, 1])
            
            camber[i, :] = (u_interp + l_interp) / 2
        
        return camber
    
    def local_thickness(
        self, x_over_c: Union[float, np.ndarray] = np.linspace(0, 1, 101)
    ) -> Union[float, np.ndarray]:
        """
        Returns the local thickness of the airfoil(s).
        Returns shape (Batch_Size, N_points) for batches, or (N_points,) for single.
        """
        # Ensure x_over_c is an array
        x_vals = np.atleast_1d(x_over_c)

        batch_size = self.coordinates.shape[0]
        thickness = np.zeros((batch_size, len(x_vals)))

        uppers = self.upper_coordinates()
        lowers = self.lower_coordinates()

        for i in range(batch_size):
            # Flip upper to ensure increasing x for np.interp (LE -> TE)
            u_pts = uppers[i][::-1]
            l_pts = lowers[i]

            u_interp = np.interp(x_vals, u_pts[:, 0], u_pts[:, 1])
            l_interp = np.interp(x_vals, l_pts[:, 0], l_pts[:, 1])
            
            thickness[i, :] = (u_interp - l_interp) 
        
        return thickness
    
    def max_camber(
        self, x_over_c_sample: np.ndarray = np.linspace(0, 1, 101)
    ) -> Union[float, np.ndarray]:
        """
        Returns the maximum camber.
        Returns array of shape (Batch_Size,) for batches, or float for single.
        """
        camber_dist = self.local_camber(x_over_c=x_over_c_sample)
        
        return np.max(camber_dist, axis=1)
        
    def max_thickness(
        self, x_over_c_sample: np.ndarray = np.linspace(0, 1, 101)
    ) -> Union[float, np.ndarray]:
        """
        Returns the maximum thickness.
        Returns array of shape (Batch_Size,) for batches, or float for single.
        """
        thickness_dist = self.local_thickness(x_over_c=x_over_c_sample)
        
        return np.max(thickness_dist, axis=1)
    
    def TE_thickness(self) -> np.ndarray:
        """
        Returns the thickness of the trailing edge (Euclidean distance).
        """
        # Vectorized for (B, N, 2)
        p_first = self.coordinates[:, 0, :]
        p_last = self.coordinates[:, -1, :]
        
        return np.linalg.norm(p_first - p_last, axis=1)

    def TE_angle(self) -> np.ndarray:
        """
        Returns the trailing edge angle in degrees.
        Calculates angle between the last panel on upper and lower surfaces.
        """
        # Vector 1: Upper surface TE segment (pointing aft)
        # Point 0 - Point 1
        upper_vec = self.coordinates[:, 0, :] - self.coordinates[:, 1, :]
        
        # Vector 2: Lower surface TE segment (pointing aft)
        # Point -1 - Point -2
        lower_vec = self.coordinates[:, -1, :] - self.coordinates[:, -2, :]

        # 2D Cross Product (Determinant)
        cross = upper_vec[:, 0] * lower_vec[:, 1] - upper_vec[:, 1] * lower_vec[:, 0]
        
        dot = upper_vec[:, 0] * lower_vec[:, 0] + upper_vec[:, 1] * lower_vec[:, 1]

        return np.degrees(np.arctan2(cross, dot))
    
    def LE_radius(self, npts: int = 7) -> np.ndarray:
        """
        Returns the leading edge radius of curvature.
        Uses a parabolic fit x = f(y) to handle vertical tangents at the nose.
        """
        # Get LE indices for the batch
        indices = self.LE_index() # Shape (B,)
        
        radii = np.empty(len(indices))
        
        # Loop required because the slicing window [i-n : i+n] moves per airfoil
        for k, center_idx in enumerate(indices):
            # Extract points around LE
            start = center_idx - npts // 2
            end = center_idx + npts // 2 + 1
            
            # Handle edge cases where indices might go out of bounds (unlikely for LE, but safe)
            pts = self.coordinates[k, max(0, start):end, :]

            # Fit x = ay^2 + by + c (Fitting x as function of y)
            x = pts[:, 0]
            y = pts[:, 1]
            
            try:
                coeffs = np.polyfit(y, x, 2)
                a = coeffs[0]
                
                # R = 1 / (2*a) at the vertex
                val = 1.0 / (2.0 * abs(a)) if abs(a) > 1e-12 else np.inf
                radii[k] = val
            except np.linalg.LinAlgError:
                radii[k] = np.nan

        return radii

    # TODO: Implement rest of the functions; IG I had implemented all the functions that requires for constraints calculations

    def __len__(self):
        return len(self.names)
    
    def __repr__(self)->str:
        return f"Airfoil Batch Size ({self.count}) ({self.coordinates.shape[1]}) points"
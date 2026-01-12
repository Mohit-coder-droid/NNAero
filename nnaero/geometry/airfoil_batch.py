# TODO: I can add some statistical analysis for this airfoil batch, which can be done in case of some airfoil datasets
from nnaero.geometry.airfoil import Airfoil
from nnaero.utils import rotation_matrix_2D, cosspace

from typing import Union, List, Dict
from pathlib import Path
import numpy as np
from scipy import interpolate

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

    def translate(
        self,
        translate_x: float = 0.0,
        translate_y: float = 0.0,
    ) -> "AirfoilBatch":
        """
        Translates batch of Airfoils by a given amount.
        Args:
            translate_x: Amount to translate in the x-direction
            translate_y: Amount to translate in the y-direction

        Returns: The translated batch of Airfoils.
        """
        # Broadcasting adds (2,) to (Batch, N, 2)
        new_coords = self.coordinates + np.array([translate_x, translate_y])

        return AirfoilBatch(names=self.names, coordinates=new_coords)

    def rotate(
        self, angle: float, x_center: float = 0.0, y_center: float = 0.0
    ) -> "AirfoilBatch":
        """
        Rotates batch of Airfoils clockwise by the specified amount, in radians.

        Rotates about the point (x_center, y_center), which is (0, 0) by default.

        Args:
            angle: Angle to rotate, counterclockwise, in radians.

            x_center: The x-coordinate of the center of rotation.

            y_center: The y-coordinate of the center of rotation.

        Returns: The rotated batch of Airfoils.
        """
        rotation_matrix = rotation_matrix_2D(angle)
        
        center_point = np.array([x_center, y_center])

        # 1. Translate to origin
        centered_coords = self.coordinates - center_point

        # 2. Rotate
        # (B, N, 2) @ (2, 2).T -> (B, N, 2)
        rotated_coords = centered_coords @ rotation_matrix.T

        # 3. Translate back
        final_coords = rotated_coords + center_point

        return AirfoilBatch(names=self.names, coordinates=final_coords)

    def scale(
        self,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
    ) -> "AirfoilBatch":
        """
        Scales batch of Airfoils about the origin.

        Args:

            scale_x: Amount to scale in the x-direction.

            scale_y: Amount to scale in the y-direction. Scaling by a negative y-value will result in coordinates
                being re-ordered such that the order of the coordinates is still correct (i.e., starts from the
                upper-surface trailing edge, continues along the upper surface to the nose, then continues along the
                lower surface to the trailing edge).

        Returns: A copy of batch of Airfoils with appropriate scaling applied.
        """
        # Extract X and Y (Shape: Batch, N)
        x = self.coordinates[:, :, 0] * scale_x
        y = self.coordinates[:, :, 1] * scale_y

        if scale_x < 0:
            # Must loop because argmax (split point) varies per airfoil
            new_x_list = []
            new_y_list = []
            
            for i in range(x.shape[0]):
                xi = x[i]
                yi = y[i]
                
                TE_index = np.argmax(xi)
                
                xi = np.concatenate([xi[TE_index::-1], xi[-2 : TE_index - 1 : -1]])
                yi = np.concatenate([yi[TE_index::-1], yi[-2 : TE_index - 1 : -1]])
                
                new_x_list.append(xi)
                new_y_list.append(yi)
            
            x = np.array(new_x_list)
            y = np.array(new_y_list)

        if scale_y < 0:
            x = x[:, ::-1]
            y = y[:, ::-1]

        new_coords = np.stack((x, y), axis=2)

        return AirfoilBatch(names=self.names, coordinates=new_coords)

    def repanel(
        self,
        n_points_per_side: int = 100,
        spacing_function_per_side=cosspace,
    ) -> "AirfoilBatch":
        """
        Returns a repaneled copy of the airfoil batch with cosine-spaced coordinates on the upper and lower surfaces.

        Args:

            n_points_per_side: Number of points per side (upper and lower) of the airfoil [int]

                Notes: The number of points defining the final airfoil will be `n_points_per_side * 2 - 1`,
                since one point (the leading edge point) is shared by both the upper and lower surfaces.

            spacing_function_per_side: Determines how to space the points on each side of the airfoil. Can be
                `np.linspace` or `cosspace`, or any other function of the call signature `f(a, b, n)` that returns
                a spaced array of `n` points between `a` and `b`. [function]

        Returns: A copy of the airfoil batch with the new coordinates.
        """

        batch_size = len(self.names)
        new_coords_list = []
        
        uppers = self.upper_coordinates()
        lowers = self.lower_coordinates()

        for i in range(batch_size):
            old_upper = uppers[i]
            old_lower = lowers[i]

            # --- Arc Length Calculation ---
            upper_dist = np.linalg.norm(np.diff(old_upper, axis=0), axis=1)
            lower_dist = np.linalg.norm(np.diff(old_lower, axis=0), axis=1)
            
            # Distance from TE/LE
            u_s = np.concatenate(([0], np.cumsum(upper_dist)))
            l_s = np.concatenate(([0], np.cumsum(lower_dist)))

            # --- Spline Interpolation ---
            try:
                # Upper Surface
                new_u = interpolate.CubicSpline(
                    x=u_s, y=old_upper, axis=0,
                    bc_type=((2, (0, 0)), (1, (0, -1)))
                )(spacing_function_per_side(0, u_s[-1], n_points_per_side))

                # Lower Surface
                new_l = interpolate.CubicSpline(
                    x=l_s, y=old_lower, axis=0,
                    bc_type=((1, (0, -1)), (2, (0, 0)))
                )(spacing_function_per_side(0, l_s[-1], n_points_per_side))
            
            except ValueError as e:
                # Check for duplicates (zero distance)
                if np.any(upper_dist == 0) or np.any(lower_dist == 0):
                    raise ValueError(f"Airfoil {self.names[i]} has duplicate points.")
                raise e

            # Concatenate (remove duplicate LE at start of lower)
            # Standard order: Upper TE -> LE -> Lower TE
            joined = np.concatenate((new_u, new_l[1:]), axis=0)
            new_coords_list.append(joined)

        return AirfoilBatch(names=self.names, coordinates=np.array(new_coords_list))
    
    def normalize(
        self,
        return_dict: bool = False,
    ) -> Union["AirfoilBatch", Dict[str, Union["AirfoilBatch", np.ndarray]]]:
        """
        Returns a copy of the Airfoil batch with a new set of `coordinates`, such that:
            - The leading edge (LE) is at (0, 0)
            - The trailing edge (TE) is at (1, 0)
            - The chord length is equal to 1

        The trailing-edge (TE) point is defined as the midpoint of the line segment connecting the first and last coordinate points (upper and lower surface TE points, respectively). The TE point is not necessarily one of the original points in the airfoil coordinates (`Airfoil.coordinates`); in general, it will not be one of the points if the TE thickness is nonzero.

        The leading-edge (LE) point is defined as the coordinate point with the largest Euclidian distance from the trailing edge. (In other words, if you were to center a circle on the trailing edge and progressively grow it, what's the last coordinate point that it would intersect?) The LE point is always one of the original points in the airfoil coordinates.

        The chord is defined as the Euclidian distance between the LE and TE points.

        Coordinate modifications to achieve the constraints described above (LE @ origin, TE at (1, 0), and chord of 1) are done by means of a translation and rotation.

        Args:

            return_dict: Determines the output type of the function.
                - If `False` (default), returns a copy of the Airfoil with the new coordinates.
                - If `True`, returns a dictionary with keys:

                        - "airfoil": a copy of the Airfoil with the new coordinates

                        - "x_translation": the amount by which the airfoil's LE was translated in the x-direction

                        - "y_translation": the amount by which the airfoil's LE was translated in the y-direction

                        - "scale_factor": the amount by which the airfoil was scaled (if >1, the airfoil had to get
                            bigger)

                        - "rotation_angle": the angle (in degrees) by which the airfoil was rotated about the LE.
                            Sign convention is that positive angles rotate the airfoil counter-clockwise.

                    All of thes values represent the "required change", e.g.:

                        - "x_translation" is the amount by which the airfoil's LE had to be translated in the
                            x-direction to get it to the origin.

                        - "rotation_angle" is the angle (in degrees) by which the airfoil had to be rotated (CCW).

        Returns: Depending on the value of `return_dict`, either:

            - A copy of the airfoil batch with the new coordinates (default), or

            - A dictionary with keys "airfoil", "x_translation", "y_translation", "scale_factor", and "rotation_angle".
                documentation for `return_tuple` for more information.
        """

        coords = self.coordinates.copy()
        
        ### Step 1: Translate LE to (0, 0)
        # Find geometric TE midpoint (Batch, 2)
        p_first = coords[:, 0, :]
        p_last = coords[:, -1, :]
        te_mid = (p_first + p_last) / 2
        
        # Calculate distance of all points to TE (Batch, N)
        # (B, N, 2) - (B, 1, 2) -> Norm
        dist_to_te = np.linalg.norm(coords - te_mid[:, None, :], axis=2)
        
        # Find LE index (point furthest from TE)
        le_indices = np.argmax(dist_to_te, axis=1)
        
        # Extract LE coordinates (Batch, 2)
        batch_indices = np.arange(len(le_indices))
        le_coords = coords[batch_indices, le_indices]
        
        coords -= le_coords[:, None, :]
        
        x_trans = -le_coords[:, 0]
        y_trans = -le_coords[:, 1]

        ### Step 2: Scale Chord to 1.0
        # The chord length is the max distance we found earlier
        chord_lengths = dist_to_te[batch_indices, le_indices]
        scale_factors = 1.0 / chord_lengths
        
        # Apply scale (Batch, N, 2) * (Batch, 1, 1)
        coords *= scale_factors[:, None, None]

        ### Step 3: Rotate TE to (1, 0)
        # Recalculate TE position after translation/scaling
        # Note: LE is at (0,0), so TE vector is just the TE point coordinate
        new_te_mid = (coords[:, 0, :] + coords[:, -1, :]) / 2
        
        # Angle to rotate (negative of current angle)
        angles = -np.arctan2(new_te_mid[:, 1], new_te_mid[:, 0])
        
        c, s = np.cos(angles), np.sin(angles)
        
        # Construct rotation matrices: Shape (Batch, 2, 2)
        # [[c, -s], [s, c]]
        rot_matrices = np.empty((len(angles), 2, 2))
        rot_matrices[:, 0, 0] = c
        rot_matrices[:, 0, 1] = -s
        rot_matrices[:, 1, 0] = s
        rot_matrices[:, 1, 1] = c
        
        # Matrix Multiply: (Batch, N, 2) @ (Batch, 2, 2).T
        # Einstein summation for batch matrix mult: bni, bij -> bnj
        coords = np.einsum('bni,bij->bnj', coords, rot_matrices)

        new_batch = AirfoilBatch(names=self.names, coordinates=coords)

        if not return_dict:
            return new_batch
        else:
            return {
                "airfoil": new_batch,
                "x_translation": x_trans,
                "y_translation": y_trans,
                "scale_factor": scale_factors,
                "rotation_angle": np.degrees(angles),
            }
            
    def set_TE_thickness(
        self,
        thickness: float = 0.0,
    ) -> "AirfoilBatch":
        """
        Creates a new modified batch that has a specified trailing-edge thickness.

        Note that the trailing-edge thickness is given nondimensionally (e.g., as a fraction of chord).

        Args:
            thickness: The target trailing-edge thickness, given nondimensionally (e.g., as a fraction of chord).

        Returns: The modified airfoil batch.
        """
        # Get surfaces (List of arrays)
        uppers = self.upper_coordinates()
        lowers = self.lower_coordinates()
        
        new_coords_list = []
        
        # We must iterate because geometries differ per airfoil
        for i in range(len(self.names)):
            # --- 1. Compute existing TE properties ---
            # Using raw coordinates for gap calculation
            # Point 0 is Upper TE, Point -1 is Lower TE
            p_upper_te = self.coordinates[i, 0, :]
            p_lower_te = self.coordinates[i, -1, :]
            
            x_gap = p_upper_te[0] - p_lower_te[0]
            y_gap = p_upper_te[1] - p_lower_te[1]

            s_gap = (x_gap**2 + y_gap**2) ** 0.5
            
            # Current TE thickness needed for adjustment calculation
            current_te_thick = np.linalg.norm(p_upper_te - p_lower_te)
            s_adjustment = (thickness - current_te_thick) / 2

            # --- 2. Determine shift vector ---
            if s_gap != 0:
                x_adjustment = s_adjustment * x_gap / s_gap
                y_adjustment = s_adjustment * y_gap / s_gap
            else:
                x_adjustment = 0.0
                y_adjustment = s_adjustment

            # --- 3. Decompose & Adjust Surfaces ---
            u = uppers[i]
            # Standard: Upper is TE -> LE
            ux, uy = u[:, 0], u[:, 1]
            le_x = ux[-1] # LE is last point of upper

            l = lowers[i][1:] # Skip duplicate LE
            # Standard: Lower is LE -> TE
            lx, ly = l[:, 0], l[:, 1]

            # Average TE X-location
            te_x = (ux[0] + lx[-1]) / 2
            
            # Avoid division by zero if airfoil is a vertical line (unlikely)
            denom = te_x - le_x if (te_x - le_x) != 0 else 1.0

            # Linearly scale adjustment from LE (0) to TE (1)
            # Upper adjustment
            new_u = np.stack([
                ux + x_adjustment * (ux - le_x) / denom,
                uy + y_adjustment * (ux - le_x) / denom,
            ], axis=1)

            # Lower adjustment (subtracted to widen gap)
            new_l = np.stack([
                lx - x_adjustment * (lx - le_x) / denom,
                ly - y_adjustment * (lx - le_x) / denom,
            ], axis=1)

            # --- 4. Force exact closure if thickness is 0 ---
            if thickness == 0:
                new_l[-1] = new_u[0]

            # --- 5. Recombine ---
            new_coords_list.append(np.concatenate([new_u, new_l], axis=0))

        return AirfoilBatch(names=self.names, coordinates=np.array(new_coords_list))

    def write_dat(
        self,
        folder_path: Union[str, Path],
        include_name: bool = True,
    ) -> None:
        """
        Writes .dat files for all airfoils in the batch to the specified folder.
        Filenames are derived from the airfoil names.
        """
        folder = Path(folder_path)
        folder.mkdir(parents=True, exist_ok=True)

        for name, coords in zip(self.names, self.coordinates):
            # Sanitize name for filename
            safe_name = "".join([c for c in name if c.isalnum() or c in (' ', '.', '_')]).strip()
            filepath = folder / f"{safe_name}.dat"

            contents = []
            if include_name:
                contents.append(name)
            
            contents.extend([f"{c[0]:.6f} {c[1]:.6f}" for c in coords])
            
            with open(filepath, "w") as f:
                f.write("\n".join(contents))

    # TODO: Implement rest of the functions; IG I had implemented all the functions except add_control_surface and blend_with_another_airfoil

    def __len__(self):
        return len(self.names)
    
    def __repr__(self)->str:
        return f"Airfoil Batch Size ({self.count}) ({self.coordinates.shape[1]})    points"
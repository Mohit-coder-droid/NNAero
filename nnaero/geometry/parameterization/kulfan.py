# this file has been copied and modified from the repository https://github.com/peterdsharpe/AeroSandbox/blob/master/aerosandbox/geometry/airfoil/airfoil_families.py

from nnaero.geometry import Airfoil
import numpy as np
from scipy.special import comb
from typing import Dict, Union, List

def _get_single_kulfan_parameters(
    coordinates: np.ndarray,
    n_weights_per_side: int = 8,
    N1: float = 0.5,
    N2: float = 1.0,
) -> Dict[str, Union[np.ndarray, float]]:
    
    n_coordinates = len(coordinates)

    x = coordinates[:, 0]
    y = coordinates[:, 1]

    LE_index = np.argmin(x)
    is_upper = np.arange(len(x)) <= LE_index

    # Class function
    C = (x) ** N1 * (1 - x) ** N2

    # Shape function (Bernstein polynomials)
    N = n_weights_per_side - 1  # Order of Bernstein polynomials

    K = comb(N, np.arange(N + 1))  # Bernstein polynomial coefficients

    dims = (n_weights_per_side, n_coordinates)

    def wide(vector):
        return np.tile(np.reshape(vector, (1, dims[1])), (dims[0], 1))

    def tall(vector):
        return np.tile(np.reshape(vector, (dims[0], 1)), (1, dims[1]))

    S_matrix = (
        tall(K)
        * wide(x) ** tall(np.arange(N + 1))
        * wide(1 - x) ** tall(N - np.arange(N + 1))
    )  # Bernstein polynomial coefficients * weight matrix

    leading_edge_weight_row = x * np.maximum(1 - x, 0) ** (n_weights_per_side + 0.5)

    trailing_edge_thickness_row = np.where(is_upper, x / 2, -x / 2)
    
    A = np.concatenate(
        [
            np.where(wide(is_upper), 0, wide(C) * S_matrix).T,
            np.where(wide(is_upper), wide(C) * S_matrix, 0).T,
            np.reshape(leading_edge_weight_row, (n_coordinates, 1)),
            np.reshape(trailing_edge_thickness_row, (n_coordinates, 1)),
        ],
        axis=1,
    )

    b = y

    # Solve least-squares problem
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    lower_weights = x[:n_weights_per_side]
    upper_weights = x[n_weights_per_side : 2 * n_weights_per_side]
    leading_edge_weight = x[-2]
    trailing_edge_thickness = x[-1]

    # If you got a negative trailing-edge thickness, then resolve the problem with a TE_thickness = 0 constraint.
    if trailing_edge_thickness < 0:
        x, _, _, _ = np.linalg.lstsq(A[:, :-1], b, rcond=None)

        lower_weights = x[:n_weights_per_side]
        upper_weights = x[n_weights_per_side : 2 * n_weights_per_side]
        leading_edge_weight = x[-1]
        trailing_edge_thickness = 0

    return {
        "lower_weights": lower_weights,
        "upper_weights": upper_weights,
        "TE_thickness": trailing_edge_thickness,
        "leading_edge_weight": leading_edge_weight,
    }

def get_kulfan_parameters(
    airfoils:"Airfoil",
    n_weights_per_side: int = 8,
    N1: float = 0.5,
    N2: float = 1.0,
    normalize_coordinates: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Given a batch of airfoil, constructs the Kulfan parameters that would recreate that airfoil. Uses a curve fitting (optimization) process.

    Kulfan parameters are a highly-efficient and flexible way to parameterize the shape of an airfoil. The particular
    flavor of Kulfan parameterization used in AeroSandbox is the "CST with LEM" method, which is described in various
    papers linked below. In total, the Kulfan parameterization consists of:

    * A vector of weights corresponding to the lower surface of the airfoil
    * A vector of weights corresponding to the upper surface of the airfoil
    * A scalar weight corresponding to the strength of a leading-edge camber mode shape of the airfoil (optional)
    * The trailing-edge (TE) thickness of the airfoil (optional)

    These Kulfan parameters are also referred to as CST (Class/Shape Transformation) parameters.

    References on Kulfan (CST) airfoils:

    * Kulfan, Brenda "Universal Parametric Geometry Representation Method" (2008). AIAA Journal of Aircraft.
        Describes the basic Kulfan (CST) airfoil parameterization.
        Mirrors:
            * https://arc.aiaa.org/doi/10.2514/1.29958
            * https://www.brendakulfan.com/_files/ugd/169bff_6738e0f8d9074610942c53dfaea8e30c.pdf
            * https://www.researchgate.net/publication/245430684_Universal_Parametric_Geometry_Representation_Method

    * Kulfan, Brenda "Modification of CST Airfoil Representation Methodology" (2020). Unpublished note:
        Describes the optional "Leading-Edge Modification" (LEM) addition to the Kulfan (CST) airfoil parameterization.
        Mirrors:
            * https://www.brendakulfan.com/_files/ugd/169bff_16a868ad06af4fea946d299c6028fb13.pdf
            * https://www.researchgate.net/publication/343615711_Modification_of_CST_Airfoil_Representation_Methodology

    * Masters, D.A. "Geometric Comparison of Aerofoil Shape Parameterization Methods" (2017). AIAA Journal.
        Compares the Kulfan (CST) airfoil parameterization to other airfoil parameterizations. Also has further notes
        on the LEM addition.
        Mirrors:
            * https://arc.aiaa.org/doi/10.2514/1.J054943
            * https://research-information.bris.ac.uk/ws/portalfiles/portal/91793513/SP_Journal_RED.pdf

    Notes on N1, N2 (shape factor) combinations:
        * 0.5, 1: Conventional airfoil
        * 0.5, 0.5: Elliptic airfoil
        * 1, 1: Biconvex airfoil
        * 0.75, 0.75: Sears-Haack body (radius distribution)
        * 0.75, 0.25: Low-drag projectile
        * 1, 0.001: Cone or wedge airfoil
        * 0.001, 0.001: Rectangle, circular duct, or circular rod.

    The following demonstrates the reversibility of this function:

    >>> import aerosandbox as asb
    >>> from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_parameters
    >>>
    >>> af = asb.Airfoil("dae11")  # A conventional airfoil
    >>> params = get_kulfan_parameters(
    >>>     coordinates=af.coordinates,
    >>> )
    >>> af_reconstructed = asb.Airfoil(
    >>>     name="Reconstructed Airfoil",
    >>>     coordinates=get_kulfan_coordinates(
    >>>         **params
    >>>     )

    Args:

        airfoils (Airfoil): Batch of Airfoil

        n_weights_per_side (int): The number of Kulfan weights to use per side of the airfoil.

        N1 (float): The shape factor corresponding to the leading edge of the airfoil. See above for examples.

        N2 (float): The shape factor corresponding to the trailing edge of the airfoil. See above for examples.

    Returns:
        A dictionary containing the Kulfan parameters. The keys are:
            * "lower_weights" (np.ndarray): The weights corresponding to the lower surface of the airfoil.
            * "upper_weights" (np.ndarray): The weights corresponding to the upper surface of the airfoil.
            * "TE_thickness" (float): The trailing-edge thickness of the airfoil.
            * "leading_edge_weight" (float): The strength of the leading-edge camber mode shape of the airfoil.

        These can be passed directly into `get_kulfan_coordinates()` to reconstruct the airfoil.
    """
    
    if normalize_coordinates:
        airfoils = airfoils.normalize()
        
    coordinates = airfoils.coordinates
    
    n_batch = coordinates.shape[0]
    results_list = []

    # Loop over batch (Clean parallelization via simple iteration)
    # Note: We loop because each airfoil might trigger the 'TE < 0' condition 
    # independently, requiring a different matrix size solve.
    for i in range(n_batch):
        res = _get_single_kulfan_parameters(
            coordinates[i],
            n_weights_per_side,
            N1,
            N2,
        )
        results_list.append(res)

    # Aggregate results into a dictionary of arrays
    batch_results = {
        "lower_weights": np.array([r["lower_weights"] for r in results_list]),
        "upper_weights": np.array([r["upper_weights"] for r in results_list]),
        "TE_thickness": np.array([r["TE_thickness"] for r in results_list]),
        "leading_edge_weight": np.array([r["leading_edge_weight"] for r in results_list]),
    }

    return batch_results


# this file has been copied and modified from the repository https://github.com/peterdsharpe/AeroSandbox/blob/790ac4f2ed6754e61046fe8cfa5ebb2004ab0c61/aerosandbox/geometry/airfoil/airfoil.py

import numpy as np
from nnaero.geometry.polygon import Polygon
from nnaero.geometry.airfoil_families import (
    get_NACA_coordinates,
    get_UIUC_coordinates,
    get_file_coordinates,
)
from scipy import interpolate
from typing import Union, Dict
from pathlib import Path


class Airfoil(Polygon):
    """
    An airfoil. See constructor docstring for usage details.
    """

    def __init__(
        self,
        name: str = "Untitled",
        coordinates: Union[None, str, Path, np.ndarray] = None,
        **deprecated_keyword_arguments,
    ):
        """
        Creates an Airfoil object.

        Args:

            name: Name of the airfoil [string]. Can also be used to auto-generate coordinates; see docstring for
            `coordinates` below.

            coordinates: A representation of the coordinates that define the airfoil. Can be one of several types of
            input; the following sequence of operations is used to interpret the meaning of the parameter:

                If `coordinates` is an Nx2 array of the [x, y] coordinates that define the airfoil, these are used
                as-is. Points are expected to be provided in standard airfoil order:

                    * Points should start on the upper surface at the trailing edge, continue forward over the upper
                    surface, wrap around the nose, continue aft over the lower surface, and then end at the trailing
                    edge on the lower surface.

                    * The trailing edge need not be closed, but many analyses implicitly assume that this gap is small.

                    * Take care to ensure that the point at the leading edge of the airfoil, usually (0, 0),
                    is not duplicated.

                If `coordinates` is provided as a string, it assumed to be the filepath to a *.dat file containing
                the coordinates; we attempt to load coordinates from this.

                If the coordinates are not specified and instead left as None, the constructor will attempt to
                auto-populate the coordinates based on the `name` parameter provided, in the following order of
                priority:

                    * If `name` is a 4-digit NACA airfoil (e.g. "naca2412"), coordinates will be created based on the
                    analytical equation.

                    * If `name` is the name of an airfoil in the UIUC airfoil database (e.g. "s1223", "e216",
                    "dae11"), coordinates will be loaded from that. Note that the string you provide must be exactly
                    the name of the associated *.dat file in the UIUC database.

        """
        ### Handle the airfoil name
        self.name = name

        ### Handle the coordinates
        self.coordinates = None
        if coordinates is None:  # If no coordinates are given
            try:  # See if it's a NACA airfoil
                self.coordinates = get_NACA_coordinates(name=self.name)
            except (ValueError, NotImplementedError):
                try:  # See if it's in the UIUC airfoil database
                    self.coordinates = get_UIUC_coordinates(name=self.name)
                except FileNotFoundError:
                    pass
                except UnicodeDecodeError:
                    import warnings

                    warnings.warn(
                        f"Airfoil {self.name} was found in the UIUC airfoil database, but could not be parsed.\n"
                        f"Check for any non-Unicode-compatible characters in the file, or specify the airfoil "
                        f"coordinates yourself.",
                    )
        else:
            try:  # If coordinates is a string, assume it's a filepath to a .dat file
                self.coordinates = get_file_coordinates(filepath=coordinates)
            except (OSError, FileNotFoundError, TypeError, UnicodeDecodeError):
                try:
                    shape = coordinates.shape
                    assert len(shape) == 2
                    assert shape[0] == 2 or shape[1] == 2
                    if not shape[1] == 2:
                        coordinates = np.transpose(shape)

                    self.coordinates = coordinates
                except AttributeError:
                    pass

        if self.coordinates is None:
            import warnings

            warnings.warn(
                f"Airfoil {self.name} had no coordinates assigned, and could not parse the `coordinates` input!",
                UserWarning,
                stacklevel=2,
            )

        ### Handle deprecated keyword arguments
        if len(deprecated_keyword_arguments) > 0:
            import warnings

            warnings.warn(
                "The `generate_polars`, `CL_function`, `CD_function`, and `CM_function` keyword arguments to the "
                "Airfoil constructor will be deprecated in an upcoming release. Their functionality is replaced"
                "by `Airfoil.get_aero_from_neuralfoil()`, which is faster and has better properties for optimization.",
                DeprecationWarning,
            )

            generate_polars = deprecated_keyword_arguments.get("generate_polars", False)
            CL_function = deprecated_keyword_arguments.get("CL_function", None)
            CD_function = deprecated_keyword_arguments.get("CD_function", None)
            CM_function = deprecated_keyword_arguments.get("CM_function", None)

            ### Handle getting default polars
            if generate_polars:
                self.generate_polars()
            else:
                from aerosandbox.library.aerodynamics.viscous import Cf_flat_plate

                def print_default_warning():
                    warnings.warn(
                        "\n".join(
                            [
                                "Warning: Using a placeholder aerodynamics model for this Airfoil!",
                                "It's highly recommended that you either:",
                                "\ta) Specify polar functions in the Airfoil constructor, or",
                                "\tb) Call Airfoil.generate_polars() to auto-generate these polar functions with XFoil.",
                            ]
                        ),
                        stacklevel=3,
                    )

                def default_CL_function(alpha, Re, mach=0, deflection=0):
                    """
                    Lift coefficient.
                    """
                    print_default_warning()
                    Cl_inc = np.pi * np.sind(2 * alpha)
                    beta = (1 - mach) ** 2

                    Cl = Cl_inc * beta
                    return Cl

                def default_CD_function(alpha, Re, mach=0, deflection=0):
                    """
                    Drag coefficient.
                    """
                    print_default_warning()
                    Cf = Cf_flat_plate(Re_L=Re, method="hybrid-sharpe-convex")

                    ### Form factor model from Raymer, "Aircraft Design". Section 12.5, Eq. 12.30
                    t_over_c = 0.12
                    FF = 1 + 2 * t_over_c * 100 * t_over_c**4

                    Cd_inc = 2 * Cf * FF * (1 + (np.sind(alpha) * 180 / np.pi / 5) ** 2)
                    beta = (1 - mach) ** 2

                    Cd = Cd_inc * beta
                    return Cd

                def default_CM_function(alpha, Re, mach=0, deflection=0):
                    """
                    Pitching moment coefficient, as measured about quarter-chord.
                    """
                    print_default_warning()
                    return np.zeros_like(alpha)

                self.CL_function = default_CL_function
                self.CD_function = default_CD_function
                self.CM_function = default_CM_function

            ### Overwrite any default polars with those provided
            if CL_function is not None:
                self.CL_function = CL_function

            if CD_function is not None:
                self.CD_function = CD_function

            if CM_function is not None:
                self.CM_function = CM_function

    def __repr__(self) -> str:
        return f"Airfoil {self.name} ({self.n_points()} points)"

    # modify this function to return just the kulfan coordinates
    def to_kulfan_airfoil(
        self,
        n_weights_per_side: int = 8,
        N1: float = 0.5,
        N2: float = 1.0,
        normalize_coordinates: bool = True,
        use_leading_edge_modification: bool = True,
    ):
        from aerosandbox.geometry.airfoil.kulfan_airfoil import KulfanAirfoil
        from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_parameters

        parameters = get_kulfan_parameters(
            coordinates=self.coordinates,
            n_weights_per_side=n_weights_per_side,
            N1=N1,
            N2=N2,
            normalize_coordinates=normalize_coordinates,
            use_leading_edge_modification=use_leading_edge_modification,
        )

        return KulfanAirfoil(
            name=self.name,
            lower_weights=parameters["lower_weights"],
            upper_weights=parameters["upper_weights"],
            leading_edge_weight=parameters["leading_edge_weight"],
            TE_thickness=parameters["TE_thickness"],
            N1=N1,
            N2=N2,
        )

    def local_camber(
        self, x_over_c: Union[float, np.ndarray] = np.linspace(0, 1, 101)
    ) -> Union[float, np.ndarray]:
        """
        Returns the local camber of the airfoil at a given point or points.

        Args:
            x_over_c: The x/c locations to calculate the camber at [1D array, more generally, an iterable of floats]

        Returns:
            Local camber of the airfoil (y/c) [1D array].
        """
        upper = self.upper_coordinates()[::-1]
        lower = self.lower_coordinates()

        upper_interpolated = np.interp(
            x_over_c,
            upper[:, 0],
            upper[:, 1],
        )
        lower_interpolated = np.interp(
            x_over_c,
            lower[:, 0],
            lower[:, 1],
        )

        return (upper_interpolated + lower_interpolated) / 2

    def local_thickness(
        self, x_over_c: Union[float, np.ndarray] = np.linspace(0, 1, 101)
    ) -> Union[float, np.ndarray]:
        """
        Returns the local thickness of the airfoil at a given point or points.

        Args:
            x_over_c: The x/c locations to calculate the thickness at [1D array, more generally, an iterable of floats]

        Returns:
            Local thickness of the airfoil (y/c) [1D array].
        """
        upper = self.upper_coordinates()[::-1]
        lower = self.lower_coordinates()

        upper_interpolated = np.interp(
            x_over_c,
            upper[:, 0],
            upper[:, 1],
        )
        lower_interpolated = np.interp(
            x_over_c,
            lower[:, 0],
            lower[:, 1],
        )

        return upper_interpolated - lower_interpolated

    def max_camber(self, x_over_c_sample: np.ndarray = np.linspace(0, 1, 101)) -> float:
        """
        Returns the maximum camber of the airfoil.

        Args:
            x_over_c_sample: Where should the airfoil be sampled to determine the max camber?

        Returns: The maximum thickness, as a fraction of chord.

        """
        return np.max(self.local_camber(x_over_c=x_over_c_sample))

    def max_thickness(
        self, x_over_c_sample: np.ndarray = np.linspace(0, 1, 101)
    ) -> float:
        """
        Returns the maximum thickness of the airfoil.

        Args:
            x_over_c_sample: Where should the airfoil be sampled to determine the max thickness?

        Returns: The maximum thickness, as a fraction of chord.

        """
        return np.max(self.local_thickness(x_over_c=x_over_c_sample))

    def draw(
        self, draw_mcl=False, draw_markers=True, backend="matplotlib", show=True
    ) -> None:
        """
        Draw the airfoil object.

        Args:
            draw_mcl: Should we draw the mean camber line (MCL)? [boolean]

            backend: Which backend should we use? "plotly" or "matplotlib"

            show: Should we show the plot? [boolean]

        Returns: None
        """
        x = np.reshape(np.array(self.x()), -1)
        y = np.reshape(np.array(self.y()), -1)
        if draw_mcl:
            x_mcl = np.linspace(np.min(x), np.max(x), len(x))
            y_mcl = self.local_camber(x_mcl)

        if backend == "matplotlib":
            import matplotlib.pyplot as plt
            import aerosandbox.tools.pretty_plots as p

            color = "#280887"
            plt.plot(x, y, ".-" if draw_markers else "-", zorder=11, color=color)
            plt.fill(x, y, zorder=10, color=color, alpha=0.2)
            if draw_mcl:
                plt.plot(x_mcl, y_mcl, "-", zorder=4, color=color, alpha=0.4)
            plt.axis("equal")
            if show:
                p.show_plot(
                    title=f"{self.name} Airfoil",
                    xlabel=r"$x/c$",
                    ylabel=r"$y/c$",
                )

        elif backend == "plotly":
            from aerosandbox.visualization.plotly import go

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines+markers" if draw_markers else "lines",
                    name="Airfoil",
                    fill="toself",
                    line=dict(color="blue"),
                ),
            )
            if draw_mcl:
                fig.add_trace(
                    go.Scatter(
                        x=x_mcl,
                        y=y_mcl,
                        mode="lines",
                        name="Mean Camber Line (MCL)",
                        line=dict(color="navy"),
                    )
                )
            fig.update_layout(
                xaxis_title="x/c",
                yaxis_title="y/c",
                yaxis=dict(scaleanchor="x", scaleratio=1),
                title=f"{self.name} Airfoil",
            )
            if show:
                fig.show()
            else:
                return fig

    def LE_index(self) -> int:
        """
        Returns the index of the leading edge point in the airfoil coordinates.
        """
        return int(np.argmin(self.x()))

    def lower_coordinates(self) -> np.ndarray:
        """
        Returns an Nx2 ndarray of [x, y] coordinates that describe the lower surface of the airfoil.

        Order is from the leading edge to the trailing edge.

        Includes the leading edge point; be careful about duplicates if using this method in conjunction with
        Airfoil.upper_coordinates().
        """
        return self.coordinates[self.LE_index() :, :]

    def upper_coordinates(self) -> np.ndarray:
        """
        Returns an Nx2 ndarray of [x, y] coordinates that describe the upper surface of the airfoil.

        Order is from the trailing edge to the leading edge.

        Includes the leading edge point; be careful about duplicates if using this method in conjunction with
        Airfoil.lower_coordinates().
        """
        return self.coordinates[: self.LE_index() + 1, :]

    # modify this function later, to include a more better version of this, as sometimes this gives -ve values for radius
    def LE_radius(self, softness: float = 1e-6):
        LE_index = self.LE_index()

        # The three points closest to the leading edge
        LE_points = self.coordinates[LE_index - 1 : LE_index + 2, :]

        # Make these 3 points into a triangle; these are the vectors representing edges
        edge_vectors = LE_points - np.roll(LE_points, 1, axis=0)
        edge_lengths = np.linalg.norm(edge_vectors, axis=1)

        # Now use a variant of Heron's formula for the circumcircle diameter
        a = edge_lengths[0]
        b = edge_lengths[1]
        c = edge_lengths[2]

        s = (a + b + c) / 2

        diameter = (a * b * c) / (2 * np.sqrt(s * (s - a) * (s - b) * (s - c)))

        return diameter / 2

    def TE_thickness(self) -> float:
        """
        Returns the thickness of the trailing edge of the airfoil.
        """
        x_gap = self.coordinates[0, 0] - self.coordinates[-1, 0]
        y_gap = self.coordinates[0, 1] - self.coordinates[-1, 1]

        return (x_gap**2 + y_gap**2) ** 0.5

    def TE_angle(self) -> float:
        """
        Returns the trailing edge angle of the airfoil, in degrees.
        """
        upper_TE_vec = self.coordinates[0, :] - self.coordinates[1, :]
        lower_TE_vec = self.coordinates[-1, :] - self.coordinates[-2, :]

        return np.arctan2d(
            upper_TE_vec[0] * lower_TE_vec[1] - upper_TE_vec[1] * lower_TE_vec[0],
            upper_TE_vec[0] * lower_TE_vec[0] + upper_TE_vec[1] * lower_TE_vec[1],
        )

    def repanel(
        self,
        n_points_per_side: int = 100,
        # spacing_function_per_side=np.cosspace,
        spacing_function_per_side=np.linspace,
    ) -> "Airfoil":
        """
        Returns a repaneled copy of the airfoil with cosine-spaced coordinates on the upper and lower surfaces.

        Args:

            n_points_per_side: Number of points per side (upper and lower) of the airfoil [int]

                Notes: The number of points defining the final airfoil will be `n_points_per_side * 2 - 1`,
                since one point (the leading edge point) is shared by both the upper and lower surfaces.

            spacing_function_per_side: Determines how to space the points on each side of the airfoil. Can be
                `np.linspace` or `np.cosspace`, or any other function of the call signature `f(a, b, n)` that returns
                a spaced array of `n` points between `a` and `b`. [function]

        Returns: A copy of the airfoil with the new coordinates.
        """

        old_upper_coordinates = (
            self.upper_coordinates()
        )  # Note: includes leading edge point, be careful about duplicates
        old_lower_coordinates = (
            self.lower_coordinates()
        )  # Note: includes leading edge point, be careful about duplicates

        # Find the streamwise distances between coordinates, assuming linear interpolation
        upper_distances_between_points = np.linalg.norm(
            np.diff(old_upper_coordinates, axis=0), axis=1
        )
        lower_distances_between_points = np.linalg.norm(
            np.diff(old_lower_coordinates, axis=0), axis=1
        )
        upper_distances_from_TE = np.concatenate(
            ([0], np.cumsum(upper_distances_between_points))
        )
        lower_distances_from_LE = np.concatenate(
            ([0], np.cumsum(lower_distances_between_points))
        )

        try:
            new_upper_coordinates = interpolate.CubicSpline(
                x=upper_distances_from_TE,
                y=old_upper_coordinates,
                axis=0,
                bc_type=(
                    (2, (0, 0)),
                    (1, (0, -1)),
                ),
            )(
                spacing_function_per_side(
                    0, upper_distances_from_TE[-1], n_points_per_side
                )
            )

            new_lower_coordinates = interpolate.CubicSpline(
                x=lower_distances_from_LE,
                y=old_lower_coordinates,
                axis=0,
                bc_type=(
                    (1, (0, -1)),
                    (2, (0, 0)),
                ),
            )(
                spacing_function_per_side(
                    0, lower_distances_from_LE[-1], n_points_per_side
                )
            )

        except ValueError as e:
            if not (
                (np.all(np.diff(upper_distances_from_TE)) > 0)
                and (np.all(np.diff(lower_distances_from_LE)) > 0)
            ):
                raise ValueError(
                    "It looks like your Airfoil has a duplicate point. Try removing the duplicate point and "
                    "re-running Airfoil.repanel()."
                )
            else:
                raise e

        return Airfoil(
            name=self.name,
            coordinates=np.concatenate(
                (new_upper_coordinates, new_lower_coordinates[1:, :]), axis=0
            ),
        )

    def normalize(
        self,
        return_dict: bool = False,
    ) -> Union["Airfoil", Dict[str, Union["Airfoil", float]]]:
        """
        Returns a copy of the Airfoil with a new set of `coordinates`, such that:
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

            - A copy of the airfoil with the new coordinates (default), or

            - A dictionary with keys "airfoil", "x_translation", "y_translation", "scale_factor", and "rotation_angle".
                documentation for `return_tuple` for more information.
        """

        ### Step 1: Translate so that the LE point is at (0, 0).
        x_te = (self.x()[0] + self.x()[-1]) / 2
        y_te = (self.y()[0] + self.y()[-1]) / 2

        distance_to_te = ((self.x() - x_te) ** 2 + (self.y() - y_te) ** 2) ** 0.5

        le_index = np.argmax(distance_to_te)

        x_translation = -self.x()[le_index]
        y_translation = -self.y()[le_index]

        newfoil = self.translate(
            translate_x=x_translation,
            translate_y=y_translation,
        )

        ### Step 2: Scale so that the chord length is 1.
        scale_factor = 1 / distance_to_te[le_index]

        newfoil = newfoil.scale(
            scale_x=scale_factor,
            scale_y=scale_factor,
        )

        ### Step 3: Rotate so that the trailing edge is at (1, 0).

        x_te = (newfoil.x()[0] + newfoil.x()[-1]) / 2
        y_te = (newfoil.y()[0] + newfoil.y()[-1]) / 2

        rotation_angle = -np.arctan2(y_te, x_te)

        newfoil = newfoil.rotate(
            angle=rotation_angle,
        )

        if not return_dict:
            return newfoil
        else:
            return {
                "airfoil": newfoil,
                "x_translation": x_translation,
                "y_translation": y_translation,
                "scale_factor": scale_factor,
                "rotation_angle": np.degrees(rotation_angle),
            }

    def add_control_surface(
        self,
        deflection: float = 0.0,
        hinge_point_x: float = 0.75,
        modify_coordinates: bool = True,
        modify_polars: bool = True,
    ) -> "Airfoil":
        """
        Returns a version of the airfoil with a trailing-edge control surface added at a given point. Implicitly
        repanels the airfoil as part of this operation.

        Args:
            deflection: Deflection angle [degrees]. Downwards-positive.
            hinge_point_x: Chordwise location of the hinge, as a fraction of chord (x/c) [float]

        Returns: an Airfoil object with the new control deflection.

        """
        if modify_coordinates:
            # Find the hinge point
            hinge_point_y = np.where(
                deflection > 0,
                self.local_camber(hinge_point_x)
                - self.local_thickness(hinge_point_x) / 2,
                self.local_camber(hinge_point_x)
                + self.local_thickness(hinge_point_x) / 2,
            )

            # hinge_point_y = self.local_camber(hinge_point_x)
            hinge_point = np.reshape(np.array([hinge_point_x, hinge_point_y]), (1, 2))

            def is_behind_hinge(xy: np.ndarray) -> np.ndarray:
                return (xy[:, 0] - hinge_point_x) * np.cosd(deflection / 2) - (
                    xy[:, 1] - hinge_point_y
                ) * np.sind(deflection / 2) > 0

            orig_u = self.upper_coordinates()
            orig_l = self.lower_coordinates()[1:, :]

            rotation_matrix = np.rotation_matrix_2D(
                angle=-np.radians(deflection),
            )

            def T(xy):
                return np.transpose(xy)

            hinge_point_u = np.tile(hinge_point, (np.length(orig_u), 1))
            hinge_point_l = np.tile(hinge_point, (np.length(orig_l), 1))

            rot_u = T(rotation_matrix @ T(orig_u - hinge_point_u)) + hinge_point_u
            rot_l = T(rotation_matrix @ T(orig_l - hinge_point_l)) + hinge_point_l

            coordinates_x = np.concatenate(
                [
                    np.where(is_behind_hinge(rot_u), rot_u[:, 0], orig_u[:, 0]),
                    np.where(is_behind_hinge(rot_l), rot_l[:, 0], orig_l[:, 0]),
                ]
            )
            coordinates_y = np.concatenate(
                [
                    np.where(is_behind_hinge(rot_u), rot_u[:, 1], orig_u[:, 1]),
                    np.where(is_behind_hinge(rot_l), rot_l[:, 1], orig_l[:, 1]),
                ]
            )

            coordinates = np.stack([coordinates_x, coordinates_y], axis=1)
        else:
            coordinates = self.coordinates

        if modify_polars:
            effectiveness = (
                1 - np.maximum(0, hinge_point_x + 1e-16) ** 2.751428551177291
            )
            dalpha = deflection * effectiveness

            def CL_function(alpha: float, Re: float, mach: float) -> float:
                return self.CL_function(
                    alpha=alpha + dalpha,
                    Re=Re,
                    mach=mach,
                )

            def CD_function(alpha: float, Re: float, mach: float) -> float:
                return self.CD_function(
                    alpha=alpha + dalpha,
                    Re=Re,
                    mach=mach,
                )

            def CM_function(alpha: float, Re: float, mach: float) -> float:
                return self.CM_function(
                    alpha=alpha + dalpha,
                    Re=Re,
                    mach=mach,
                )

        else:
            CL_function = self.CL_function
            CD_function = self.CD_function
            CM_function = self.CM_function

        return Airfoil(
            name=self.name,
            coordinates=coordinates,
            CL_function=CL_function,
            CD_function=CD_function,
            CM_function=CM_function,
        )

    def set_TE_thickness(
        self,
        thickness: float = 0.0,
    ) -> "Airfoil":
        """
        Creates a modified copy of the Airfoil that has a specified trailing-edge thickness.

        Note that the trailing-edge thickness is given nondimensionally (e.g., as a fraction of chord).

        Args:
            thickness: The target trailing-edge thickness, given nondimensionally (e.g., as a fraction of chord).

        Returns: The modified airfoil.

        """
        ### Compute existing trailing-edge properties
        x_gap = self.coordinates[0, 0] - self.coordinates[-1, 0]
        y_gap = self.coordinates[0, 1] - self.coordinates[-1, 1]

        s_gap = (x_gap**2 + y_gap**2) ** 0.5

        s_adjustment = (thickness - self.TE_thickness()) / 2

        ### Determine how much the trailing edge should move by in X and Y.
        if s_gap != 0:
            x_adjustment = s_adjustment * x_gap / s_gap
            y_adjustment = s_adjustment * y_gap / s_gap
        else:
            x_adjustment = 0
            y_adjustment = s_adjustment

        ### Decompose the existing airfoil coordinates to upper and lower sides, and x and y.
        u = self.upper_coordinates()
        ux = u[:, 0]
        uy = u[:, 1]

        le_x = ux[-1]

        l = self.lower_coordinates()[1:]
        lx = l[:, 0]
        ly = l[:, 1]

        te_x = (ux[0] + lx[-1]) / 2

        ### Create modified versions of the upper and lower coordinates
        new_u = np.stack(
            arrays=[
                ux + x_adjustment * (ux - le_x) / (te_x - le_x),
                uy + y_adjustment * (ux - le_x) / (te_x - le_x),
            ],
            axis=1,
        )
        new_l = np.stack(
            arrays=[
                lx - x_adjustment * (lx - le_x) / (te_x - le_x),
                ly - y_adjustment * (lx - le_x) / (te_x - le_x),
            ],
            axis=1,
        )

        ### If the desired thickness is zero, ensure that is precisely reached.
        if thickness == 0:
            new_l[-1] = new_u[0]

        ### Combine the upper and lower surface coordinates into a single array.
        new_coordinates = np.concatenate([new_u, new_l], axis=0)

        ### Return a new Airfoil with the desired coordinates.
        return Airfoil(name=self.name, coordinates=new_coordinates)

    def scale(
        self,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
    ) -> "Airfoil":
        """
        Scales an Airfoil about the origin.

        Args:

            scale_x: Amount to scale in the x-direction.

            scale_y: Amount to scale in the y-direction. Scaling by a negative y-value will result in coordinates
                being re-ordered such that the order of the coordinates is still correct (i.e., starts from the
                upper-surface trailing edge, continues along the upper surface to the nose, then continues along the
                lower surface to the trailing edge).

        Returns: A copy of the Airfoil with appropriate scaling applied.
        """
        x = self.x() * scale_x
        y = self.y() * scale_y

        if scale_x < 0:
            TE_index = np.argmax(x)
            x = np.concatenate([x[TE_index::-1], x[-2 : TE_index - 1 : -1]])
            y = np.concatenate([y[TE_index::-1], y[-2 : TE_index - 1 : -1]])

        if scale_y < 0:
            x = x[::-1]
            y = y[::-1]

        coordinates = np.stack((x, y), axis=1)

        return Airfoil(
            name=self.name,
            coordinates=coordinates,
        )

    def translate(
        self,
        translate_x: float = 0.0,
        translate_y: float = 0.0,
    ) -> "Airfoil":
        """
        Translates an Airfoil by a given amount.
        Args:
            translate_x: Amount to translate in the x-direction
            translate_y: Amount to translate in the y-direction

        Returns: The translated Airfoil.

        """
        x = self.x() + translate_x
        y = self.y() + translate_y

        return Airfoil(name=self.name, coordinates=np.stack((x, y), axis=1))

    def rotate(
        self, angle: float, x_center: float = 0.0, y_center: float = 0.0
    ) -> "Airfoil":
        """
        Rotates the airfoil clockwise by the specified amount, in radians.

        Rotates about the point (x_center, y_center), which is (0, 0) by default.

        Args:
            angle: Angle to rotate, counterclockwise, in radians.

            x_center: The x-coordinate of the center of rotation.

            y_center: The y-coordinate of the center of rotation.

        Returns: The rotated Airfoil.

        """

        coordinates = np.copy(self.coordinates)

        ### Translate
        translation = np.array([x_center, y_center])
        coordinates -= translation

        ### Rotate
        rotation_matrix = np.rotation_matrix_2D(
            angle=angle,
        )
        coordinates = (rotation_matrix @ coordinates.T).T

        ### Translate
        coordinates += translation

        return Airfoil(name=self.name, coordinates=coordinates)

    def blend_with_another_airfoil(
        self,
        airfoil: "Airfoil",
        blend_fraction: float = 0.5,
        n_points_per_side: int = 100,
    ) -> "Airfoil":
        """
        Blends this airfoil with another airfoil. Merges both the coordinates and the aerodynamic functions.

        Args:

            airfoil: The other airfoil to blend with.

            blend_fraction: The fraction of the other airfoil to use when blending. Defaults to 0.5 (50%).

                * A blend fraction of 0 will return an identical airfoil to this one (self).

                * A blend fraction of 1 will return an identical airfoil to the other one (`airfoil` parameter).

            n_points_per_side: The number of points per side to use when blending the coordinates of the two airfoils.

        Returns: A new airfoil that is a blend of this airfoil and another one.

        """
        foil_a = self.repanel(n_points_per_side=n_points_per_side)
        foil_b = airfoil.repanel(n_points_per_side=n_points_per_side)
        a_fraction = 1 - blend_fraction
        b_fraction = blend_fraction

        name = f"{a_fraction * 100:.0f}% {self.name}, {b_fraction * 100:.0f}% {airfoil.name}"

        coordinates = a_fraction * foil_a.coordinates + b_fraction * foil_b.coordinates

        return Airfoil(
            name=name,
            coordinates=coordinates,
        )

    def write_dat(
        self,
        filepath: Union[str, Path] = None,
        include_name: bool = True,
    ) -> str:
        """
        Writes a .dat file corresponding to this airfoil to a filepath.

        Args:
            filepath: filepath (including the filename and .dat extension) [string]
                If None, this function returns the .dat file as a string.

            include_name: Should the name be included in the .dat file? (In a standard *.dat file, it usually is.)

        Returns: None

        """
        contents = []

        if include_name:
            contents += [self.name]

        contents += ["%f %f" % tuple(coordinate) for coordinate in self.coordinates]

        string = "\n".join(contents)

        if filepath is not None:
            with open(filepath, "w+") as f:
                f.write(string)

        return string

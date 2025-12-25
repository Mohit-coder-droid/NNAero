import numpy as np
from NNAero.optimization.opti import Opti
from abc import abstractmethod, ABC
import copy
from typing import Any
import dill
from pathlib import Path
import sys
import warnings


class NNAeroObject(ABC):
    _nnaero_metadata: dict[str, str] = None

    @abstractmethod
    def __init__(self):
        """
        Denotes NNAeroObject as an abstract class, meaning you can't instantiate it directly - you must subclass
        (extend) it instead.
        """
        pass

    def __eq__(self, other):
        """
        Checks if two NNAero objects are value-equivalent. A more sensible default for classes that represent
        physical objects than checking for memory equivalence.

        This is done by checking if the two objects are of the same type and have the same __dict__.

        Args:
            other: Another object.

        Returns:
            bool: True if the objects are equal, False otherwise.
        """
        if self is other:  # If they point to the same object in memory, they're equal
            return True

        if type(self) != type(
            other
        ):  # If they are of different types, they cannot be equal
            return False

        if set(self.__dict__.keys()) != set(
            other.__dict__.keys()
        ):  # If they have differing dict keys, don't bother checking values
            return False

        for key in self.__dict__.keys():  # Check equality of all values
            if np.all(self.__dict__[key] == other.__dict__[key]):
                continue
            else:
                return False

        return True

    def save(
        self,
        filename: str | Path | None = None,
        verbose: bool = True,
        automatically_add_extension: bool = True,
    ) -> None:
        """
        Saves the object to a binary file, using the `dill` library.

        Creates a .nnaero file, which is a binary file that can be loaded with `NNAero.load()`. This can be loaded
            into memory in a different Python session or a different computer, and it will be exactly the same as when it
            was saved.

        Args:

            filename: The filename to save this object to. Should be a .nnaero file.

            verbose: If True, prints messages to console on successful save.

            automatically_add_extension: If True, automatically adds the .nnaero extension to the filename if it doesn't
                already have it. If False, does not add the extension.

        Returns: None (writes to file)

        """

        if filename is None:
            try:
                filename = self.name
            except AttributeError:
                filename = "untitled"

        filename = Path(filename)

        if filename.suffix == "" and automatically_add_extension:
            filename = filename.with_suffix(".nnaero")

        if verbose:
            print(f"Saving {str(self)} to:\n\t{filename}...")

        import nnaero as nnaero

        self._nnaero_metadata = {
            "python_version": ".".join(
                [
                    str(sys.version_info.major),
                    str(sys.version_info.minor),
                    str(sys.version_info.micro),
                ]
            ),
            "nnaero_version": nnaero.__version__,
        }
        with open(filename, "wb") as f:
            dill.dump(
                obj=self,
                file=f,
            )

    def copy(self):
        """
        Returns a shallow copy of the object.
        """
        return copy.copy(self)

    def deepcopy(self):
        """
        Returns a deep copy of the object.
        """
        return copy.deepcopy(self)


def load(
    filename: str | Path,
    verbose: bool = True,
) -> NNAeroObject:
    """
    Loads an NNAeroObject from a file.

    Upon load, will compare metadata from the file to the current Python version and NNAero version. If there are
    any discrepancies, will raise a warning.

    Args:

        filename: The filename to load from. Should be a .nnaero file.

        verbose: If True, prints messages to console on successful load.

    Returns: An NNAeroObject.

    """
    filename = Path(filename)

    # Load the object from file
    with open(filename, "rb") as f:
        obj = dill.load(f)

    # At this point, the object is loaded
    try:
        metadata = obj._nnaero_metadata
    except AttributeError:
        warnings.warn(
            "This object was saved without metadata. This may cause compatibility issues.",
            stacklevel=2,
        )
        return obj

    # Check if the Python version is different
    try:
        saved_python_version = metadata["python_version"]
        current_python_version = ".".join(
            [
                str(sys.version_info.major),
                str(sys.version_info.minor),
                str(sys.version_info.micro),
            ]
        )

        saved_python_version_split = saved_python_version.split(".")
        current_python_version_split = current_python_version.split(".")

        if any(
            [
                saved_python_version_split[0] != current_python_version_split[0],
                saved_python_version_split[1] != current_python_version_split[1],
            ]
        ):
            warnings.warn(
                f"This object was saved with Python {saved_python_version}, but you are currently using Python {current_python_version}.\n"
                f"This may cause compatibility issues.",
                stacklevel=2,
            )

    except KeyError:
        warnings.warn(
            "This object was saved without Python version info metadata. This may cause compatibility issues.",
            stacklevel=2,
        )

    # Check if the NNAero version is different
    import NNAero as nnaero

    try:
        saved_nnaero_version = metadata["nnaero_version"]

        if saved_nnaero_version != nnaero.__version__:
            warnings.warn(
                f"This object was saved with NNAero {saved_nnaero_version}, but you are currently using NNAero {nnaero.__version__}.\n"
                f"This may cause compatibility issues.",
                stacklevel=2,
            )

    except KeyError:
        warnings.warn(
            "This object was saved without NNAero version info metadata. This may cause compatibility issues.",
            stacklevel=2,
        )

    if verbose:
        print(f"Loaded {str(obj)} from:\n\t{filename}")

    return obj


class ExplicitAnalysis(NNAeroObject):
    default_analysis_specific_options: dict[type, dict[str, Any]] = {}
    """This is part of NNAero's "analysis-specific options" feature, which lets you "tag" geometry objects with 
    flags that change how different analyses act on them. 
    
    This variable, `default_analysis_specific_options`, allows you to specify default values for options that can be used for 
    specific problems. 
    
    This should be a dictionary, where: * keys are the geometry-like types that you might be interested in defining 
    parameters for. * values are dictionaries, where: * keys are strings that label a given option * values are 
    anything. These are used as the default values, in the event that the associated geometry doesn't override those. 
    
    An example of what this variable might look like, for a vortex-lattice method aerodynamic analysis:
    
    >>> default_analysis_specific_options = {
    >>>     Airplane: dict(
    >>>         profile_drag_coefficient=0
    >>>     ),
    >>>     Wing    : dict(
    >>>         wing_level_spanwise_spacing=True,
    >>>         spanwise_resolution=12,
    >>>         spanwise_spacing="cosine",
    >>>         chordwise_resolution=12,
    >>>         chordwise_spacing="cosine",
    >>>         component=None,  # type: int
    >>>         no_wake=False,
    >>>         no_alpha_beta=False,
    >>>         no_load=False,
    >>>         drag_polar=dict(
    >>>             CL1=0,
    >>>             CD1=0,
    >>>             CL2=0,
    >>>             CD2=0,
    >>>             CL3=0,
    >>>             CD3=0,
    >>>         ),
    >>>     )
    >>> }
    
    """

    def get_options(
        self,
        geometry_object: NNAeroObject,
    ) -> dict[str, Any]:
        """
        Retrieves the analysis-specific options that correspond to both:

            * An analysis type (which is this object, "self"), and

            * A specific geometry object, such as an Airplane or Wing.

        Args:
            geometry_object: An instance of an NNAero geometry object, such as an Airplane, Wing, etc.

                * In order for this function to do something useful, you probably want this option to have
                `analysis_specific_options` defined. See the nnaero.Airplane constructor for an example of this.

        Returns: A dictionary that combines:

            * This analysis's default options for this geometry, if any exist.

            * The geometry's declared analysis-specific-options for this analysis, if it exists. These geometry
            options will override the defaults from the analysis.

            This dictionary has the format:

                * keys are strings, listing specific options

                * values can be any type, and simply state the value of the analysis-specific option following the
                logic above.

        Note: if this analysis defines a set of default options for the geometry type in question (by using
        `self.default_analysis_specific_options`), all keys from the geometry object's `analysis_specific_options`
        will be validated against those in the default options set. A warning will be raised if keys do not
        correspond to those in the defaults, as this (potentially) indicates a typo, which would otherwise be
        difficult to debug.

        """

        ### Determine the types of both this analysis and the geometry object.
        analysis_type: type = self.__class__
        geometry_type: type = geometry_object.__class__

        ### Determine whether this analysis and the geometry object have options that specifically reference each other or not.
        try:
            analysis_options_for_this_geometry = self.default_analysis_specific_options[
                geometry_type
            ]
            assert hasattr(analysis_options_for_this_geometry, "items")
        except (AttributeError, KeyError, AssertionError):
            analysis_options_for_this_geometry = None

        try:
            geometry_options_for_this_analysis = (
                geometry_object.analysis_specific_options[analysis_type]
            )
            assert hasattr(geometry_options_for_this_analysis, "items")
        except (AttributeError, KeyError, AssertionError):
            geometry_options_for_this_analysis = None

        ### Now, merge those options (with logic depending on whether they exist or not)
        if analysis_options_for_this_geometry is not None:
            options = copy.deepcopy(analysis_options_for_this_geometry)
            if geometry_options_for_this_analysis is not None:
                for k, v in geometry_options_for_this_analysis.items():
                    if k in analysis_options_for_this_geometry.keys():
                        options[k] = v
                    else:
                        import warnings

                        allowable_keys = [
                            f'"{k}"' for k in analysis_options_for_this_geometry.keys()
                        ]
                        warnings.warn(
                            f"\nAn object of type '{geometry_type.__name__}' declared the analysis-specific option '{k}' for use with analysis '{analysis_type.__name__}'.\n"
                            f"This was unexpected! Allowable analysis-specific options for '{geometry_type.__name__}' with '{analysis_type.__name__}' are:\n"
                            "\t" + "\n\t".join(allowable_keys) + "\n"
                            "Did you make a typo?",
                            stacklevel=2,
                        )

        else:
            if geometry_options_for_this_analysis is not None:
                options = geometry_options_for_this_analysis
            else:
                options = {}

        return options


class ImplicitAnalysis(NNAeroObject):
    @staticmethod
    def initialize(init_method):
        """
        A decorator that should be applied to the __init__ method of ImplicitAnalysis or any subclass of it.

        Usage example:

        >>> class MyAnalysis(ImplicitAnalysis):
        >>>
        >>>     @ImplicitAnalysis.initialize
        >>>     def __init__(self):
        >>>         self.a = self.opti.variable(init_guess = 1)
        >>>         self.b = self.opti.variable(init_guess = 2)
        >>>
        >>>         self.opti.subject_to(
        >>>             self.a == self.b ** 2
        >>>         ) # Add a nonlinear governing equation

        Functionality:

        The basic purpose of this wrapper is to ensure that every ImplicitAnalysis has an `opti` property that points to
        an optimization environment (nnaero.Opti type) that it can work in.

        How do we obtain an nnaero.Opti environment to work in? Well, this decorator adds an optional `opti` parameter to
        the __init__ method that it is applied to.

            1. If this `opti` parameter is not provided, then a new empty `nnaero.Opti` environment is created and stored as
            `ImplicitAnalysis.opti`.

            2. If the `opti` parameter is provided, then we simply assign the given `nnaero.Opti` environment (which may
            already contain other variables/constraints/objective) to `ImplicitAnalysis.opti`.

        In addition, a property called `ImplicitAnalysis.opti_provided` is stored, which records whether the user
        provided an Opti environment or if one was instead created for them.

        If the user did not provide an Opti environment (Option 1 from our list above), we assume that the user basically
        just wants to perform a normal, single-disciplinary analysis. So, in this case, we proceed to solve the analysis as-is
        and do an in-place substitution of the solution.

        If the user did provide an Opti environment (Option 2 from our list above), we assume that the user might potentially want
        to add other implicit analyses to the problem. So, in this case, we don't solve the analysis, and the user must later
        solve the analysis by calling `sol = opti.solve()` or similar.

        """

        def init_wrapped(self, *args, opti=None, **kwargs):
            if opti is None:
                self.opti = Opti()
                self.opti_provided = False
            else:
                self.opti = opti
                self.opti_provided = True

            init_method(self, *args, **kwargs)

            if not self.opti_provided and not self.opti.x.shape == (0, 1):
                sol = self.opti.solve()
                self.__dict__ = sol(self.__dict__)

        return init_wrapped

    class ImplicitAnalysisInitError(Exception):
        def __init__(
            self,
            message="""
    Your ImplicitAnalysis object doesn't have an `opti` property!
    This is almost certainly because you didn't decorate your object's __init__ method with 
    `@ImplicitAnalysis.initialize`, which you should go do.
                     """,
        ):
            self.message = message
            super().__init__(self.message)

    @property
    def opti(self):
        try:
            return self._opti
        except AttributeError:
            raise self.ImplicitAnalysisInitError()

    @opti.setter
    def opti(self, value: Opti):
        self._opti = value

    @property
    def opti_provided(self):
        try:
            return self._opti_provided
        except AttributeError:
            raise self.ImplicitAnalysisInitError()

    @opti_provided.setter
    def opti_provided(self, value: bool):
        self._opti_provided = value
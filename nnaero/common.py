import numpy as np
from abc import abstractmethod, ABC
import copy
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
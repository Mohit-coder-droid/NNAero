def length(array) -> int:
    """
    Returns the length of an 1D-array-like object. An extension of len() with slightly different functionality.
    Args:
        array:

    Returns:

    """
    try:
        return len(array)
    except TypeError:
        return 1
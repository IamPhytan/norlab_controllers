import numpy as np


def wrap_to_pi(angle: float):
    """Wrap angle to PI, return a angle value between `-PI` and `PI`

    Args:
        angle (float): Angle in radians

    Returns:
        float: Wrapped angle, in radians
    """
    return np.arctan2(np.sin(angle), np.cos(angle))

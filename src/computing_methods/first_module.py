import numpy as np
import math as math

def rot_axis(vector: np.ndarray, angle: float, axis: np.ndarray) -> np.ndarray:
    """Rotate a vector [x, y, z] around a given axis by a certain angle

    Parameters
    ----------
    vector (np.ndarray): 
        Vector to rotate
    angle (float): 
        Angle in radians of the rotation
    axis (np.ndarray): 
        Axis around which the rotation is made

    Returns
    -------
    np.ndarray
        Rotated vector
    """
    axis = axis / np.linalg.norm(axis)
    cos_theta = math.cos(angle)
    sin_theta = math.sin(angle)
    return cos_theta * vector + sin_theta * np.cross(axis, vector) + (1 - cos_theta) * np.dot(axis, vector) * axis

def rot_angle(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Computes the angle between two vectors [x, y, z]

    Parameters
    ----------
    vec1 : np.ndarray
        First vector
    vec2 : np.ndarray
        Second vector

    Returns
    -------
    float
        Angle between the vectors in radians
    """
    return np.arccos(np.clip(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)), -1.0, 1.0))

def find_perp(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """Finds a vector perpendicular to two given vectors [x, y, z] (right hand rule)

    Parameters
    ----------
    vec1 : np.ndarray
        First vector
        
    vec2 : np.ndarray
        Second vector

    Returns
    -------
    np.ndarray
        Perpendicular vector
    """
    return np.cross(vec1, vec2)


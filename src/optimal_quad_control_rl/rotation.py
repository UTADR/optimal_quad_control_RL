import numpy as np
from numpy.typing import NDArray


def angle_rotate_point2d(angle: float, point: NDArray) -> NDArray:
    c, s = np.cos(angle), np.sin(angle)
    res = np.empty_like(point)
    res[..., 0] = c * point[..., 0] - s * point[..., 1]
    res[..., 1] = s * point[..., 0] + c * point[..., 1]
    return res


def angle_rotate_point2d_inv(angle: float, point: NDArray) -> NDArray:
    c, s = np.cos(angle), np.sin(-angle)
    res = np.empty_like(point)
    res[..., 0] = c * point[..., 0] - s * point[..., 1]
    res[..., 1] = s * point[..., 0] + c * point[..., 1]
    return res


def euler_angles_rotate_point_inv(angles: NDArray, point: NDArray) -> NDArray:
    c_phi, s_phi = np.cos(angles[..., 0]), np.sin(angles[..., 0])
    c_theta, s_theta = np.cos(angles[..., 1]), np.sin(angles[..., 1])
    c_psi, s_psi = np.cos(angles[..., 2]), np.sin(angles[..., 2])

    spst = s_psi * s_theta
    cpst = c_psi * s_theta

    r10 = cpst * s_phi - s_psi * c_phi
    r11 = spst * s_phi + c_psi * c_phi
    r12 = c_theta * s_phi
    r20 = cpst * c_phi + s_psi * s_phi
    r21 = spst * c_phi - c_psi * s_phi
    r22 = c_theta * c_phi
    r00 = c_psi * c_theta
    r01 = s_psi * c_theta
    r02 = -s_theta

    res = np.empty_like(point)
    res[..., 0] = r00 * point[..., 0] + r10 * point[..., 1] + r20 * point[..., 2]
    res[..., 1] = r01 * point[..., 0] + r11 * point[..., 1] + r21 * point[..., 2]
    res[..., 2] = r02 * point[..., 0] + r12 * point[..., 1] + r22 * point[..., 2]
    return res


def euler_angles_rotate_point(angles: NDArray, point: NDArray) -> NDArray:
    c_phi, s_phi = np.cos(angles[..., 0]), np.sin(angles[..., 0])
    c_theta, s_theta = np.cos(angles[..., 1]), np.sin(angles[..., 1])
    c_psi, s_psi = np.cos(angles[..., 2]), np.sin(angles[..., 2])

    spst = s_psi * s_theta
    cpst = c_psi * s_theta

    r10 = cpst * s_phi - s_psi * c_phi
    r11 = spst * s_phi + c_psi * c_phi
    r12 = c_theta * s_phi
    r20 = cpst * c_phi + s_psi * s_phi
    r21 = spst * c_phi - c_psi * s_phi
    r22 = c_theta * c_phi
    r00 = c_psi * c_theta
    r01 = s_psi * c_theta
    r02 = -s_theta

    res = np.empty_like(point)

    res[..., 0] = r00 * point[..., 0] + r01 * point[..., 1] + r02 * point[..., 2]
    res[..., 1] = r10 * point[..., 0] + r11 * point[..., 1] + r12 * point[..., 2]
    res[..., 2] = r20 * point[..., 0] + r21 * point[..., 1] + r22 * point[..., 2]
    return res


def euler_angles_kinematic_rates(angles: NDArray, rates: NDArray) -> NDArray:
    c_phi, s_phi = np.cos(angles[..., 0]), np.sin(angles[..., 0])
    c_theta = np.cos(angles[..., 1])
    tan_theta = np.tan(angles[..., 1])

    res = np.empty_like(rates)

    res[..., 0] = (
        rates[..., 0]
        + rates[..., 1] * s_phi * tan_theta
        + rates[..., 2] * c_phi * tan_theta
    )
    res[..., 1] = rates[..., 1] * c_phi - rates[..., 2] * s_phi
    res[..., 2] = rates[..., 1] * s_phi / c_theta + rates[..., 2] * c_phi / c_theta
    return res

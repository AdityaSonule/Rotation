import numpy as np

# Use a single complex dtype for numpy everywhere.
DTYPE = np.complex128

INV_SQRT2 = 1.0 / np.sqrt(2.0)
H = INV_SQRT2 * np.array([[1, 1], [1, -1]], dtype=DTYPE)

# LAMBDA_PI is the base rotation angle realized by the H/T building blocks:
# cos(LAMBDA_PI) = cos^2(pi/8) = (1 + 1/sqrt2)/2. Because LAMBDA_PI / (2 pi) is
# irrational, the multiples {k * LAMBDA_PI mod 2 pi} densely fill [0, 2 pi).
LAMBDA_PI = np.arccos((1.0 + INV_SQRT2) / 2.0)
TWO_PI = 2.0 * np.pi


class Bloch:
    """Axis-angle (Bloch) form of a 2x2 unitary G:

        G = e^{i alpha} (cos(theta/2) I - i sin(theta/2) (n . sigma))

    i.e. a global phase e^{i alpha} times a rotation by angle `theta` about the
    Bloch-sphere axis `n`. Here (n . sigma) = n_x X + n_y Y + n_z Z.
    """

    alpha: float  # global phase
    n: np.ndarray  # unit rotation axis, shape (3,): [n_x, n_y, n_z]
    theta: float  # rotation angle

    def __init__(self, alpha: float, n: np.ndarray, theta: float):
        self.alpha = alpha
        self.n = np.array(n, dtype=float)
        self.theta = theta


def to_bloch(g: np.ndarray) -> Bloch:
    """Recover the Bloch form (alpha, n, theta) of a 2x2 unitary `g`."""

    g = np.array(g, dtype=DTYPE)

    X = np.array([[0, 1], [1, 0]], dtype=DTYPE)
    Y = np.array([[0, -1j], [1j, 0]], dtype=DTYPE)
    Z = np.array([[1, 0], [0, -1]], dtype=DTYPE)

    det = np.linalg.det(g)
    arg = np.angle(det)
    alpha = arg / 2.0

    phase = np.exp(-1j * alpha)
    g = phase * g

    tr = np.trace(g)
    cos_half_theta = np.real(tr) / 2.0
    cos_half_theta = np.clip(cos_half_theta, -1.0, 1.0)

    theta = 2.0 * np.arccos(cos_half_theta)
    sin_half_theta = np.sin(theta / 2.0)

    if np.isclose(sin_half_theta, 0.0):
        n = np.array([1.0, 0.0, 0.0])
        return Bloch(alpha, n, theta)

    n_x = -np.imag(np.trace(X @ g)) / (2.0 * sin_half_theta)
    n_y = -np.imag(np.trace(Y @ g)) / (2.0 * sin_half_theta)
    n_z = -np.imag(np.trace(Z @ g)) / (2.0 * sin_half_theta)

    n = np.array([n_x, n_y, n_z], dtype=float)

    norm = np.linalg.norm(n)
    if not np.isclose(norm, 0.0):
        n = n / norm

    return Bloch(alpha, n, theta)


# n1, n2 are two orthogonal Bloch-sphere axes (n1 . n2 == 0)
# TODO: fill in the two orthogonal rotation axes (each a length-3
# unit vector [x, y, z])
n1 = np.array([1.0, 0.0, 0.0])
n2 = np.array([0.0, 1.0, 0.0])

# frame derived from the axes (given)
# take the dot product of the Bloch axis with these
# the minus sign arises from the double cover issue
a1 = -n1
a2 = -n2
a3 = np.cross(a1, a2)


def n1n2n1_angles(b: Bloch) -> tuple[float, float, float, float]:
    """Factor the rotation part of a unitary (given as its Bloch form `b`) as
        u = e^{i global_phase} * Rn1(alpha) * Rn2(beta) * Rn1(gamma)

    where Ra(angle) is a rotation by `angle` about axis a, and {a1, a2, a3} is
    the orthonormal frame defined above. Returns (alpha, beta, gamma, global_phase).
    """

    def wrap_angle(x: float) -> float:
        return x % TWO_PI

    def skew(v: np.ndarray) -> np.ndarray:
        x, y, z = v
        return np.array([
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0]
        ], dtype=float)

    def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
        axis = np.array(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)

        K = skew(axis)
        I3 = np.eye(3)

        return (
            np.cos(angle) * I3
            + (1.0 - np.cos(angle)) * np.outer(axis, axis)
            + np.sin(angle) * K
        )

    n = np.array(b.n, dtype=float)
    n = n / np.linalg.norm(n)

    R_global = rotation_matrix(n, b.theta)

    e1 = a1 / np.linalg.norm(a1)
    e2 = a2 / np.linalg.norm(a2)
    e3 = a3 / np.linalg.norm(a3)

    B = np.column_stack([e1, e2, e3])

    R = B.T @ R_global @ B

    cos_beta = np.clip(R[0, 0], -1.0, 1.0)
    beta = np.arccos(cos_beta)
    sin_beta = np.sin(beta)

    if not np.isclose(sin_beta, 0.0):
        gamma = np.arctan2(R[0, 1], R[0, 2])
        alpha = np.arctan2(R[1, 0], -R[2, 0])
    else:
        gamma = 0.0

        if cos_beta > 0:
            beta = 0.0
            alpha = np.arctan2(R[2, 1], R[1, 1])
        else:
            beta = np.pi
            alpha = np.arctan2(R[1, 2], R[1, 1])

    alpha = wrap_angle(alpha)
    beta = wrap_angle(beta)
    gamma = wrap_angle(gamma)

    return alpha, beta, gamma, b.alpha


def approx_angle_with_tolerance(angle: float, tolerance: float) -> int:
    """Find an integer multiple k such that
        (k * LAMBDA_PI) mod 2*pi  ~=  angle   (within `tolerance`)
    Since LAMBDA_PI / (2 pi) is irrational, such a k always exists; search
    k = 1, 2, 3, ... and return the first one whose wrapped multiple lands within
    `tolerance` of `angle` (compare both as angles in [0, 2 pi)).

    Hint:
      * wrap an angle into [0, 2 pi)
      * the angular distance between two wrapped angles a, b is
        min(|a - b|, TWO_PI - |a - b|) (so 0.01 and 2*pi - 0.01 count as close).
    """

    def wrap_angle(x: float) -> float:
        return x % TWO_PI

    def angular_distance(a: float, b: float) -> float:
        diff = abs(a - b)
        return min(diff, TWO_PI - diff)

    target = wrap_angle(angle)

    if angular_distance(0.0, target) <= tolerance:
        return 0

    k = 1
    while True:
        candidate = wrap_angle(k * LAMBDA_PI)

        if angular_distance(candidate, target) <= tolerance:
            return k

        k += 1


def decompose_2x2(u: np.ndarray, tolerance: float) -> tuple[int, int, int]:
    """Approximate a 2x2 unitary `u` as a product of powers of M1 and M2:

        u  ~=  M1^k * M2^l * M1^m     (up to a global phase)

    where M1 is a rotation about axis a1 and M2 a rotation about axis a2, each by
    the base angle realized by the H/T building blocks. Returns the powers
    (k, l, m).

    Steps (combine the two functions above):

      1. Get the Bloch form of u (to_bloch), then factor its rotation into the
         three frame angles with n1n2n1_angles:
             alpha, beta, gamma, _global_phase = n1n2n1_angles(to_bloch(u))
         alpha and gamma are rotations about a1 (realized by powers of M1);
         beta is a rotation about a2 (realized by powers of M2).

      2. Convert each angle to an integer power with approx_angle_with_tolerance:
             k = approx_angle_with_tolerance(alpha, tolerance)   # power of M1
             l = approx_angle_with_tolerance(beta,  tolerance)   # power of M2
             m = approx_angle_with_tolerance(gamma, tolerance)   # power of M1
         (Mind the relationship between a target rotation angle and the base
         angle each application of M1/M2 adds.)

      3. Return (k, l, m).
    """

    b = to_bloch(u)

    alpha, beta, gamma, _global_phase = n1n2n1_angles(b)

    k = approx_angle_with_tolerance(alpha, tolerance)
    l = approx_angle_with_tolerance(beta, tolerance)
    m = approx_angle_with_tolerance(gamma, tolerance)

    return k, l, m
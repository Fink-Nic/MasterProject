# type: ignore
import torch
from madnis.integrator import Integrator, Integrand
from vectors import LorentzVector, Vector
import math

m_s = 0.0
m_psi = 0.0
q = 0.
p = 0.


def cartesian_parameterize(xs: list[float], origin: Vector | None = None) -> tuple[Vector, float]:
    x, y, z = xs
    scale = 10*m_s

    # All coordinates use the same conformal mapping log(c/(1-c))
    def conf(c: float):
        return math.log(c/(1-c))

    # Derivate of the conformal mapping
    def d_conf(c: float):
        return 1/(c-c**2)

    cartesian = scale*Vector(conf(x), conf(y), conf(z))
    # Scale factor enters determinant as a third power
    jacobian = scale**3*d_conf(x)*d_conf(y)*d_conf(z)

    if origin is not None:
        return cartesian-origin, jacobian

    return cartesian, jacobian


def python_integrand(loop_momentum: Vector) -> float:

    ose = [
        math.sqrt(loop_momentum.squared() + m_psi**2),
        math.sqrt((loop_momentum-q.spatial()
                   ).squared() + m_psi**2),
        math.sqrt((loop_momentum+p.spatial()
                   ).squared() + m_psi**2),
    ]

    q_i = [0, -q.t, p.t]

    def eta(i, j, sig_i=1, sig_j=1):
        return sig_i*(ose[i] + q_i[j] - q_i[i]) + sig_j*ose[j]

    return 1/(8*math.pi**3)/(8*ose[0]*ose[1]*ose[2])*(
        1/eta(2, 0)/eta(2, 1) + 1/eta(0, 1) /
        eta(2, 1) + 1/eta(0, 1)/eta(0, 2)
        + 1/eta(0, 2)/eta(1, 2) + 1/eta(1, 0) /
        eta(1, 2) + 1/eta(1, 0)/eta(2, 0)
    )

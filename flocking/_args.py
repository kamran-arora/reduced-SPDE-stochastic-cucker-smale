import equinox as eqx
from jaxtyping import ArrayLike


class CuckerSmaleArgs(eqx.Module):
    L: ArrayLike
    lambda_: float
    r: float
    sigma: float
    C_a: float
    vbar0: ArrayLike
    """
    Equinox module to store CuckerSmale arguments

    L: domain length such that we work on [-L/2, L/2]
    lambda_ : CS kernel strength parameter
    r: CS kernel localisation parameter
    sigma: noise strength
    C_a: constant kernel strength
    vbar0: initial alignment velocity
    """

    def __init__(
        self,
        L: ArrayLike,
        lambda_: float,
        r: any,
        sigma: float,
        C_a: float = 10.0,
        vbar0: ArrayLike = None,
    ):
        self.L = L
        self.lambda_ = lambda_
        self.r = r
        self.sigma = sigma
        self.C_a = C_a
        self.vbar0 = vbar0

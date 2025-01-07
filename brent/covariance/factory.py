# Internal imports
from .covariance import Covariance
from brent.util import Factory


class CovarianceFactory(Factory):

    @classmethod
    def create(cls, name, *args, **kwargs) -> Covariance:
        return super().create(name, *args, **kwargs)

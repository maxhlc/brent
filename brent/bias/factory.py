# Internal imports
from .bias import Bias
from brent.util import Factory


class BiasFactory(Factory):
   
    @classmethod
    def create(cls, name, *args, **kwargs) -> Bias:
        return super().create(name, *args, **kwargs)


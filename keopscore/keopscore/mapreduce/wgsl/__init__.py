from .WgslAssignZero import WgslAssignZero
from .WgslReduc1D import WgslReduc1D

_exports = [
    WgslAssignZero,
    WgslReduc1D,
]

__all__ = [cls.__name__ for cls in _exports]

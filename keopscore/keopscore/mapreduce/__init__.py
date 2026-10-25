from keopscore.config import cuda

from .cpu import *
from .wgsl import *

if cuda.get_use_cuda():
    from .gpu import *

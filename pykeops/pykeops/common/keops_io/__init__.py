import pykeops.config as pykeopsconfig
from .nvrtc import LoadKeOps_nvrtc

if pykeopsconfig.cuda.get_use_cuda():
    from .cpp import LoadKeOps_cpp

    keops_binder = {
        "nvrtc": LoadKeOps_nvrtc.LoadKeOps_nvrtc,
        "cpp": LoadKeOps_cpp.LoadKeOps_cpp,
    }
else:
    from .cpp import LoadKeOps_cpp

    keops_binder = {"cpp": LoadKeOps_cpp.LoadKeOps_cpp}

try:
    from .webgpu import LoadKeOps_webgpu

    keops_binder["wgpu"] = LoadKeOps_webgpu.LoadKeOps_webgpu
except ImportError:
    pass


def get_keops_binder(tagCPUGPU):
    # tagCPUGPU is 0 for Cpu, 1 for Gpu (Cuda), 2 for WebGPU
    binder_name = ("cpp", "nvrtc", "wgpu")[tagCPUGPU]
    if binder_name not in keops_binder:
        raise ValueError(f"[pyKeOps] The {binder_name} binder is not available.")
    return keops_binder[binder_name]

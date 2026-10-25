import numpy as np
import wgpu

import pykeops.config as pykeopsconfig

from keopscore.mapreduce.wgsl.WgslReduc1D import wgsl_block_size
from keopscore.utils.Cache import Cache_partial

from pykeops.common.keops_io.LoadKeOps import LoadKeOps

# max number of workgroups per dimension guaranteed by WebGPU
max_workgroups = 65535

# max number of (i,j) interactions computed in a single submission to the GPU
max_interactions = 2**28

_device = None


def get_device():
    # The WebGPU device is created once and shared by all formulas.
    # We request the adapter limits, to allow more than 8 input tensors.
    global _device
    if _device is None:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        limits = adapter.limits
        _device = adapter.request_device_sync(
            required_limits={
                key: limits[key]
                for key in (
                    "max-storage-buffers-per-shader-stage",
                    "max-storage-buffer-binding-size",
                    "max-buffer-size",
                )
            }
        )
    return _device


class LoadKeOps_webgpu_class(LoadKeOps):
    def __init__(self, *args, fast_init=False):
        super().__init__(*args, fast_init=fast_init)

    def init(self, *args):
        tagCPUGPU, tag1D2D, tagHostDevice, use_ranges, device_id_request = args[:5]
        dtype = args[8]
        if use_ranges:
            raise ValueError(
                "[pyKeOps] The WebGPU backend does not support ranges or batch dimensions."
            )
        if dtype != "float32":
            raise ValueError(
                f"[pyKeOps] The WebGPU backend only supports float32 inputs (got {dtype})."
            )
        super().init(*args)

    def init_phase2(self):
        device = get_device()
        with open(self.params.low_level_code_file, "r") as f:
            code = f.read()
        shader_module = device.create_shader_module(code=code)
        self.pipeline = device.create_compute_pipeline(
            layout="auto", compute={"module": shader_module, "entry_point": "main"}
        )
        self.block_size = wgsl_block_size(self.params.dimy)
        # indices of the input tensors used by the kernel (see wgsl_bindings)
        self.inds = sorted(
            set(self.params.indsi + self.params.indsj + self.params.indsp)
        )

    def genred(self, device_args, ranges, nx, ny, nbatchdims, out, *args):
        # the base class only keeps pointers to the data, so we keep the arrays here
        self.args = args
        if out is None:
            M = nx if self.params.tagI == 0 else ny
            out = self.tools.empty(
                (M, self.params.dim), dtype=args[0].dtype, device=device_args
            )
        self.out = out
        return super().genred(device_args, ranges, nx, ny, nbatchdims, out, *args)

    genred_pytorch = genred
    genred_numpy = genred

    def call_keops(self, nx, ny):
        if self.params.tagI == 1:
            nx, ny = ny, nx
        if nx == 0:
            return
        device = get_device()

        def storage_buffer(data, usage=wgpu.BufferUsage.STORAGE):
            # N.B. WebGPU does not allow empty bindings
            data = np.ascontiguousarray(data, dtype=np.float32).reshape(-1)
            if data.size == 0:
                data = np.zeros(1, dtype=np.float32)
            return device.create_buffer_with_data(data=data, usage=usage)

        sizes = device.create_buffer(
            size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        out_size = nx * self.params.dim * 4
        out_buffer = device.create_buffer(
            size=max(out_size, 4),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        entries = [
            {"binding": 0, "resource": {"buffer": sizes}},
            {"binding": 1, "resource": {"buffer": out_buffer}},
        ]
        for k in self.inds:
            buffer = storage_buffer(self.tools.numpy(self.args[k]))
            entries.append({"binding": k + 2, "resource": {"buffer": buffer}})
        bind_group = device.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(0), entries=entries
        )

        # The workgroups are dispatched by chunks of at most max_interactions
        # (i,j) pairs, each in a separate submission : the OS may silently kill
        # long running GPU commands (e.g. the GPU watchdog on macOS).
        nblocks = (nx + self.block_size - 1) // self.block_size
        chunk = max(1, max_interactions // (self.block_size * max(ny, 1)))
        for block_offset in range(0, nblocks, chunk):
            nblocks_chunk = min(chunk, nblocks - block_offset)
            # 1D grid of workgroups, folded in 2D if needed
            grid_x = min(nblocks_chunk, max_workgroups)
            grid_y = (nblocks_chunk + grid_x - 1) // grid_x
            device.queue.write_buffer(
                sizes, 0, np.array([nx, ny, block_offset, 0], dtype=np.int32)
            )
            command_encoder = device.create_command_encoder()
            compute_pass = command_encoder.begin_compute_pass()
            compute_pass.set_pipeline(self.pipeline)
            compute_pass.set_bind_group(0, bind_group)
            compute_pass.dispatch_workgroups(grid_x, grid_y, 1)
            compute_pass.end()
            device.queue.submit([command_encoder.finish()])

        res = np.frombuffer(device.queue.read_buffer(out_buffer), dtype=np.float32)
        res = res[: nx * self.params.dim].reshape(self.out.shape)
        if self.params.lang == "torch":
            import torch

            self.out.copy_(torch.from_numpy(res))
        else:
            self.out[...] = res


LoadKeOps_webgpu = Cache_partial(
    LoadKeOps_webgpu_class,
    use_cache_file=True,
    save_folder=pykeopsconfig.get_build_folder(),
)

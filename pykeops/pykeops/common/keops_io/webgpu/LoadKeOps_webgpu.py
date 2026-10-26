import numpy as np
import wgpu

import pykeops.config as pykeopsconfig

from keopscore.mapreduce.wgsl.WgslReduc1D import wgsl_block_size
from keopscore.utils.Cache import Cache_partial

from pykeops.common.keops_io.LoadKeOps import LoadKeOps

# max number of workgroups per dimension guaranteed by WebGPU
max_workgroups = 65535

# max number of (i,j) interactions computed in a single submission to the GPU,
# for variables of total dimension <= 8 (it is reduced for higher dimensions)
max_interactions = 2**30

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
            size=32, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        out_size = nx * self.params.dim * 4
        out_buffer = device.create_buffer(
            size=max(out_size, 4),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        acc_buffer = device.create_buffer(
            size=max(nx * self.params.dimred * 4, 4), usage=wgpu.BufferUsage.STORAGE
        )
        entries = [
            {"binding": 0, "resource": {"buffer": sizes}},
            {"binding": 1, "resource": {"buffer": out_buffer}},
            {"binding": 2, "resource": {"buffer": acc_buffer}},
        ]
        for k in self.inds:
            buffer = storage_buffer(self.tools.numpy(self.args[k]))
            entries.append({"binding": k + 3, "resource": {"buffer": buffer}})
        bind_group = device.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(0), entries=entries
        )

        # The computation is split into dispatches of a bounded number of (i,j)
        # pairs, each in a separate submission : the OS may silently kill long
        # running GPU commands (e.g. the GPU watchdog on macOS). Each dispatch
        # handles all the rows (up to rows_chunk) for a range of j indices,
        # to keep enough workgroups busy.
        bs = self.block_size
        dimvars = sum(self.params.dimsx) + sum(self.params.dimsy)
        budget = max_interactions * 8 // max(8, dimvars)
        rows_chunk = max(1, budget // bs // bs) * bs
        for i0 in range(0, nx, rows_chunk):
            nrows = min(rows_chunk, nx - i0)
            cols_chunk = max(1, budget // nrows // bs) * bs
            # 1D grid of workgroups, folded in 2D if needed
            nblocks = (nrows + bs - 1) // bs
            grid_x = min(nblocks, max_workgroups)
            grid_y = (nblocks + grid_x - 1) // grid_x
            # N.B. if ny=0, we still need one dispatch to initialize the output
            for j0 in range(0, max(ny, 1), cols_chunk):
                j1 = min(ny, j0 + cols_chunk)
                device.queue.write_buffer(
                    sizes,
                    0,
                    np.array([nx, ny, i0 // bs, j0, j1, 0, 0, 0], dtype=np.int32),
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

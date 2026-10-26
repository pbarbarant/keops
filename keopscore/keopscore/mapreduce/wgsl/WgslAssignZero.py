from keopscore.binders.wgsl.Wgsl_link_compile import Wgsl_link_compile
from keopscore.mapreduce.MapReduce import MapReduce
from keopscore.utils.code_gen_utils import (
    c_array,
    c_zero_float,
    wgsl_codegen,
)


def wgsl_bindings(inds=(), acc=False):
    # WGSL declarations of the kernel inputs : sizes at binding 0, output at
    # binding 1, accumulators at binding 2 and input tensor k at binding k+3.
    # N.B. only the inputs used by the kernel can be declared, since unused
    # bindings are removed by the WGSL compiler.
    # The computation may be split into several dispatches, each one for
    # the workgroups starting at block_offset and the j indices in [jbegin, jend).
    # Between two dispatches the accumulators are saved in keops_acc.
    string = """
                        struct Sizes {
                          nx: i32,
                          ny: i32,
                          block_offset: i32,
                          jbegin: i32,
                          jend: i32,
                        }
                        @group(0) @binding(0) var<uniform> sizes: Sizes;
                        @group(0) @binding(1) var<storage, read_write> keops_out: array<f32>;
            """
    if acc:
        string += """
                        @group(0) @binding(2) var<storage, read_write> keops_acc: array<f32>;"""
    for k in sorted(set(inds)):
        string += f"""
                        @group(0) @binding({k + 3}) var<storage, read> arg_{k}: array<f32>;"""
    return string


class WgslAssignZero(MapReduce, Wgsl_link_compile):
    # class for generating the final WGSL code, WebGPU version

    def __init__(self, *args):
        MapReduce.__init__(self, *args)
        Wgsl_link_compile.__init__(self)
        self.dimy = self.varloader.dimy

    def get_code(self):
        with wgsl_codegen():
            super().get_code()

            dim = self.red_formula.dim
            outi = c_array(self.dtype, dim, f"(keops_out + i * {dim})")

            self.code = f"""
                        {wgsl_bindings()}

                        @compute @workgroup_size(64)
                        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{

                          // get the index of the current thread
                          let i = i32(gid.x);

                          if (i < sizes.nx) {{
                            {outi.assign(c_zero_float)}
                          }}

                        }}
                    """

from keopscore.binders.wgsl.Wgsl_link_compile import Wgsl_link_compile
from keopscore.mapreduce.wgsl.WgslAssignZero import WgslAssignZero, wgsl_bindings
from keopscore.mapreduce.MapReduce import MapReduce
from keopscore.utils.code_gen_utils import (
    c_variable,
    c_array,
    pointer,
    wgsl_codegen,
)
from keopscore.utils.messages import KeOps_Error


def wgsl_block_size(dimy):
    # Number of threads per workgroup. The workgroup (shared) memory must hold
    # block_size*dimy floats, within the 16kB guaranteed by WebGPU
    # (maxComputeWorkgroupStorageSize). N.B. this function is also used by the
    # pykeops launcher to dispatch the kernel.
    block_size = min(256, 16384 // (4 * max(dimy, 1)))
    if block_size < 1:
        KeOps_Error(
            f"Total dimension of j variables ({dimy}) is too large for the WebGPU backend."
        )
    return block_size


class WgslReduc1D(MapReduce, Wgsl_link_compile):
    # class for generating the final WGSL code, WebGPU version.
    # This is a transcription of GpuReduc1D.

    AssignZero = WgslAssignZero

    def __init__(self, *args):
        MapReduce.__init__(self, *args)
        Wgsl_link_compile.__init__(self)
        self.dimy = self.varloader.dimy

    def get_code(self):
        with wgsl_codegen():
            super().get_code()

            red_formula = self.red_formula
            dtype = self.dtype
            varloader = self.varloader

            i = self.i
            j = self.j
            fout = self.fout
            acc = self.acc
            sum_scheme = self.sum_scheme

            dimy = varloader.dimy
            block_size = wgsl_block_size(dimy)

            # each input tensor is bound to a separate storage buffer arg_k
            args = [c_variable(pointer(dtype), f"arg_{k}") for k in range(self.nargs)]
            outi = c_array(
                dtype, red_formula.dim, f"(keops_out + i * {red_formula.dim})"
            )

            param_loc = self.param_loc
            xi = self.xi
            yjloc = c_array(dtype, dimy, f"(yj + tid * {dimy})")
            yjrel = c_array(dtype, dimy, f"(yj + jrel * {dimy})")
            table = varloader.table(self.xi, yjrel, self.param_loc)
            jreltile = c_variable("signed long int", "(jrel + jstart)")

            self.code = f"""
                        {wgsl_bindings(varloader.indsi + varloader.indsj + varloader.indsp)}

                        var<workgroup> yj: array<f32, {block_size * max(dimy, 1)}>;

                        @compute @workgroup_size({block_size})
                        fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{

                          let nx = sizes.nx;
                          let ny = sizes.ny;

                          // get the index of the current thread
                          let tid = i32(lid.x);
                          let {i.id} = (sizes.block_offset + i32(wid.x + wid.y * nwg.x)) * {block_size} + tid;

                          // load parameters variables from global memory to local thread memory
                          {param_loc.declare()}
                          {varloader.load_vars("p", param_loc, args)}

                          {fout.declare()}
                          {xi.declare()}
                          {acc.declare()}
                          {sum_scheme.declare_temporary_accumulator()}

                          if ({i.id} < nx) {{
                            {red_formula.InitializeReduction(acc)} // acc = 0
                            {sum_scheme.initialize_temporary_accumulator_first_init()}
                            {varloader.load_vars('i', xi, args, row_index=i)} // load xi variables from global memory to local thread memory
                          }}

                          for (var jstart: i32 = 0; jstart < ny; jstart += {block_size}) {{

                            // get the current column
                            let {j.id} = jstart + tid;

                            if ({j.id} < ny) {{ // we load yj from device global memory only if j<ny
                              {varloader.load_vars("j", yjloc, args, row_index=j)}
                            }}
                            workgroupBarrier();

                            if ({i.id} < nx) {{ // we compute x1i only if needed
                              {sum_scheme.initialize_temporary_accumulator_block_init()}
                              for (var jrel: i32 = 0; (jrel < {block_size}) && (jrel < ny - jstart); jrel++) {{
                                {red_formula.formula(fout, table)} // Call the function, which outputs results in fout
                                {sum_scheme.accumulate_result(acc, fout, jreltile)}
                              }}
                              {sum_scheme.final_operation(acc)}
                            }}
                            workgroupBarrier();
                          }}
                          if ({i.id} < nx) {{
                            {red_formula.FinalizeOutput(acc, outi, i)}
                          }}

                        }}
                    """

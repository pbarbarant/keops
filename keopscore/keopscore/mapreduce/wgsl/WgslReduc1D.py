from keopscore.binders.wgsl.Wgsl_link_compile import Wgsl_link_compile
from keopscore.mapreduce.wgsl.WgslAssignZero import WgslAssignZero, wgsl_bindings
from keopscore.mapreduce.MapReduce import MapReduce
from keopscore.utils.code_gen_utils import (
    c_variable,
    c_array,
    pointer,
    VectCopy,
    wgsl_codegen,
)
from keopscore.utils.messages import KeOps_Error


def wgsl_yj_stride(dimy):
    # number of floats used to store the j variables of one thread in workgroup
    # memory, padded to a multiple of 4 : the rows are then 16 bytes aligned and
    # the GPU compiler can load them with vec4 (128 bits) instructions.
    return 4 * max(1, (dimy + 3) // 4)


def wgsl_block_size(dimy):
    # Number of threads per workgroup. The workgroup (shared) memory must hold
    # block_size*wgsl_yj_stride(dimy) floats, within the 16kB guaranteed by WebGPU
    # (maxComputeWorkgroupStorageSize). N.B. this function is also used by the
    # pykeops launcher to dispatch the kernel.
    block_size = min(256, 16384 // (4 * wgsl_yj_stride(dimy)))
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
            stride = wgsl_yj_stride(dimy)
            yjloc = c_array(dtype, dimy, f"(yj + tid * {stride})")
            yjrel = c_array(dtype, dimy, f"(yj + jrel * {stride})")
            table = varloader.table(self.xi, yjrel, self.param_loc)
            jreltile = c_variable("signed long int", "(jrel + jstart)")
            # accumulators saved between two dispatches
            acci = c_array(
                acc.dtype, red_formula.dimred, f"(keops_acc + i * {red_formula.dimred})"
            )

            self.code = f"""
                        {wgsl_bindings(varloader.indsi + varloader.indsj + varloader.indsp, acc=True)}

                        var<workgroup> yj: array<f32, {block_size * stride}>;

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
                            if (sizes.jbegin == 0) {{
                              {red_formula.InitializeReduction(acc)} // acc = 0
                            }} else {{
                              {VectCopy(acc, acci)} // resume the reduction of the previous dispatch
                            }}
                            {sum_scheme.initialize_temporary_accumulator_first_init()}
                            {varloader.load_vars('i', xi, args, row_index=i)} // load xi variables from global memory to local thread memory
                          }}

                          for (var jstart: i32 = sizes.jbegin; jstart < sizes.jend; jstart += {block_size}) {{

                            // get the current column
                            let {j.id} = jstart + tid;

                            if ({j.id} < sizes.jend) {{ // we load yj from device global memory only if j<jend
                              {varloader.load_vars("j", yjloc, args, row_index=j)}
                            }}
                            workgroupBarrier();

                            if ({i.id} < nx) {{ // we compute x1i only if needed
                              {sum_scheme.initialize_temporary_accumulator_block_init()}
                              // a single loop bound, which compiles to a much faster loop
                              let jlim = min({block_size}, sizes.jend - jstart);
                              for (var jrel: i32 = 0; jrel < jlim; jrel++) {{
                                {red_formula.formula(fout, table)} // Call the function, which outputs results in fout
                                {sum_scheme.accumulate_result(acc, fout, jreltile)}
                              }}
                              {sum_scheme.final_operation(acc)}
                            }}
                            workgroupBarrier();
                          }}
                          if ({i.id} < nx) {{
                            if (sizes.jend == ny) {{
                              {red_formula.FinalizeOutput(acc, outi, i)}
                            }} else {{
                              {VectCopy(acci, acc)} // save the reduction for the next dispatch
                            }}
                          }}

                        }}
                    """

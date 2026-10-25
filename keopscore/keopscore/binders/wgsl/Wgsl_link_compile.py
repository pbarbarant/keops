from keopscore.binders.LinkCompile import LinkCompile


class Wgsl_link_compile(LinkCompile):
    """
    Binder for the WebGPU backend. There is no compilation step here : the WGSL code
    is written in a .wgsl file, which is compiled at run time by the wgpu library
    (see pykeops/common/keops_io/wgpu/LoadKeOps_wgpu.py).
    """

    source_code_extension = "wgsl"

    def __init__(self):
        LinkCompile.__init__(self)
        # low_level_code_file is the WGSL source file itself, e.g. 7b9a611f7e.wgsl
        self.low_level_code_file = self.gencode_file

        # file to check for existence to detect code generation is needed
        self.file_to_check = self.gencode_file

    def generate_code(self):
        # generate the code and save it in self.code, by calling get_code method from WgslReduc class :
        self.get_code()
        # write the code in the source .wgsl file
        self.write_code()
        # retreive some parameters that will be saved into info_file.
        self.tagI = self.red_formula.tagI
        self.dim = self.red_formula.dim

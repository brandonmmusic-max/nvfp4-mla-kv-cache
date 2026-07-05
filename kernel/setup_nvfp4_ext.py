from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="klc_nvfp4_mla_ext",
    ext_modules=[
        CUDAExtension(
            name="_C_nvfp4",
            sources=["nvfp4_mla_ext.cu"],
            include_dirs=["/opt/vllm/csrc"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-gencode=arch=compute_120a,code=sm_120a",
                    "--expt-relaxed-constexpr",
                    "-DENABLE_NVFP4_SM120",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='farthest_point_sample',
    ext_modules=[
        CUDAExtension(
            name='farthest_point_sample_cuda',
            sources=[
                'src/farthest_point_sample.cpp',
                'src/farthest_point_sample_kernel.cu',
            ],
            extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

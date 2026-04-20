from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, include_paths, library_paths
import os.path as osp
import torch
import os

import os.path as osp
ROOT = osp.dirname(osp.abspath(__file__))
torch_include_dirs = include_paths()
torch_library_dirs = library_paths()
conda_prefix = os.environ.get("PREFIX", os.environ.get("CONDA_PREFIX", ""))
eigen_path = osp.join(conda_prefix, 'include', 'eigen3')

setup(
    name='droid_backends',
    ext_modules=[
        CUDAExtension('droid_backends',
            include_dirs=torch_include_dirs + [
                eigen_path,
                osp.join(ROOT, 'src')
            ],
            sources=[
                'src/lib/droid.cpp',
                'src/lib/droid_kernels.cu',
                'src/lib/correlation_kernels.cu',
                'src/lib/altcorr_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3', '-D_GLIBCXX_USE_CXX11_ABI=1'],
                'nvcc': ['-O3',
                    '-D_GLIBCXX_USE_CXX11_ABI=1',
                    '-gencode=arch=compute_60,code=sm_60',
                    '-gencode=arch=compute_61,code=sm_61',
                    '-gencode=arch=compute_70,code=sm_70',
                    '-gencode=arch=compute_75,code=sm_75',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_86,code=sm_86',
                ]
            }),
    ],
    cmdclass={'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)},
)

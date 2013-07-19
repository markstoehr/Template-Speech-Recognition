#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
import os.path

cython_req = (0, 16)
try:
    import Cython
    from Cython.Distutils import build_ext
    import re
    cython_ok = tuple(map(int, re.sub(r"[^\d.]*", "", Cython.__version__).split('.')[:2])) >= cython_req 
except ImportError:
    cython_ok = False 

if not cython_ok:
    raise ImportError("At least Cython {0} is required".format(".".join(map(str, cython_req))))


def cython_extension(modpath, mp=False):
    extra_compile_args = []
    extra_link_args = []
    if mp:
        extra_compile_args.append('-fopenmp')
        extra_link_args.append('-fopenmp') 
    filepath = os.path.join(*modpath.split('.')) + ".pyx"
    return Extension(modpath, [filepath], extra_compile_args=extra_compile_args, extra_link_args=extra_link_args)

setup(name='amitgroup',
    cmdclass = {'build_ext': build_ext},
    version='0',
    url="https://github.com/Template-Speech-Rec/Template-Speech_rec",
    description="Code for Mark Stoehr's research",
    packages=[    ],
    ext_modules = [
        cython_extension("compute_likelihood_linear_filter"),
        cython_extension("code_parts"),
        cython_extension("spread_waliji_patches"),        
        cython_extension("get_mistakes"),
        cython_extension("cluster_times"),
        
    ]
)

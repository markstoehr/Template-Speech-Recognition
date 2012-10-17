
def configuration(parent_package='', top_path=None):
    import numpy
    from numpy.distutils.misc_util import Configuration

    config = Configuration('.',
                           parent_package,
                           top_path)
    config.add_extension('npufunc_log_quantizer', ['single_type_log_quantizer.c'])

    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(configuration=configuration)

    from distutils.core import setup
    from Cython.Build import cythonize

    setup(
        name = "code_parts",
        ext_modules = cythonize('code_parts.pyx'), # accepts a glob pattern
        )

    setup(
        name = "spread_waliji_patches",
        ext_modules = cythonize('spread_waliji_patches.pyx'), # accepts a glob pattern
        )

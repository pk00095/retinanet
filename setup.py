from importlib.machinery import SourceFileLoader
import fnmatch, setuptools,os,sys
from distutils.command.build_ext import build_ext as DistUtilsBuildExt

from setuptools import setup, find_packages,Extension
from setuptools.command.build_py import build_py as build_py_orig

from Cython.Build import cythonize
import numpy

version = SourceFileLoader('retinanet.version', os.path.join('retinanet', 'version.py')).load_module().VERSION


class BuildExtension(setuptools.Command):
    description     = DistUtilsBuildExt.description
    user_options    = DistUtilsBuildExt.user_options
    boolean_options = DistUtilsBuildExt.boolean_options
    help_options    = DistUtilsBuildExt.help_options

    def __init__(self, *args, **kwargs):
        from setuptools.command.build_ext import build_ext as SetupToolsBuildExt

        # Bypass __setatrr__ to avoid infinite recursion.
        self.__dict__['_command'] = SetupToolsBuildExt(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._command, name)

    def __setattr__(self, name, value):
        setattr(self._command, name, value)

    def initialize_options(self, *args, **kwargs):
        return self._command.initialize_options(*args, **kwargs)

    def finalize_options(self, *args, **kwargs):
        ret = self._command.finalize_options(*args, **kwargs)
        import numpy
        self.include_dirs.append(numpy.get_include())
        return ret

    def run(self, *args, **kwargs):
        return self._command.run(*args, **kwargs)


extensions = [Extension('retinanet.compute_overlap', ['retinanet/compute_overlap.pyx'])]

cython_excludes = []

def not_cythonized(tup):
    (package, module, filepath) = tup
    return any(
        fnmatch.fnmatchcase(filepath, pat=pattern) for pattern in cython_excludes
    ) or not any(
        fnmatch.fnmatchcase(filepath, pat=pattern)
        for ext in extensions
        for pattern in ext.sources
    )
def not_cythonized(tup):
    (package, module, filepath) = tup
    return any(
        fnmatch.fnmatchcase(filepath, pat=pattern) for pattern in cython_excludes
    ) or not any(
        fnmatch.fnmatchcase(filepath, pat=pattern)
        for ext in extensions
        for pattern in ext.sources
    )
class build_py(build_py_orig):
    def find_modules(self):
        modules = super().find_modules()
        return list(filter(not_cythonized, modules))
    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return list(filter(not_cythonized, modules))


setuptools.setup(
    name             = 'retinanet',
    version          = version,
    description      = 'Tensorflow implementation of RetinaNet object detection using keras api.',
    url              = 'https://github.com/pk00095/retinanet',
    author           = 'T Pratik',
    author_email     = 'pk00095@gmail.com',
    maintainer       = 'T Pratik',
    maintainer_email = 'pk00095@gmail.com',
    cmdclass         = {'build_ext': BuildExtension},
    packages         = setuptools.find_packages(),
    install_requires = ['cython', 'Pillow', 'opencv-python', 'tqdm', 'albumentations==0.4.5'],
    ext_modules    = cythonize(extensions, exclude=cython_excludes, compiler_directives={'language_level' : "3"}),
    include_dirs=[numpy.get_include()],
    setup_requires = ["cython>=0.28", "numpy>=1.14.0"]
)
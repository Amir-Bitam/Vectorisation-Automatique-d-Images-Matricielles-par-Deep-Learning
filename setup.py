# Adapted from https://github.com/pybind/cmake_example/blob/master/setup.py
import os
import re
import sys
import platform
import subprocess
import importlib
import tempfile
import hashlib
from sysconfig import get_paths

import importlib
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
from distutils.sysconfig import get_config_var
from distutils.version import LooseVersion

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir, build_with_cuda):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        self.build_with_cuda = build_with_cuda

class Build(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        super().run()

    def build_extension(self, ext):
        if isinstance(ext, CMakeExtension):
            extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
            info = get_paths()
            include_path = info['include']
            python_library = get_config_var('LIBDIR')
            library_name = get_config_var('LIBRARY')
            if python_library and library_name and os.path.isdir(python_library):
                python_library = os.path.join(python_library, library_name)
            if platform.system() == "Windows" and (not python_library or not os.path.exists(python_library)):
                python_library = os.path.join(
                    sys.prefix,
                    'libs',
                    'python{}{}.lib'.format(sys.version_info.major, sys.version_info.minor))
            cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                          '-DCMAKE_POLICY_VERSION_MINIMUM=3.5',
                          '-DPYTHON_EXECUTABLE=' + sys.executable,
                          '-DPython_EXECUTABLE=' + sys.executable,
                          '-DPYTHON_LIBRARY=' + python_library,
                          '-DPYTHON_INCLUDE_DIR=' + include_path,
                          '-DPYTHON_INCLUDE_PATH=' + include_path]

            cfg = 'Debug' if self.debug else 'Release'
            build_args = ['--config', cfg]

            if platform.system() == "Windows":
                cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir),
                               '-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
                generator = os.environ.get('CMAKE_GENERATOR', '')
                single_config_generators = ('Ninja', 'NMake', 'MinGW')
                if sys.maxsize > 2**32 and not any(g in generator for g in single_config_generators):
                    cmake_args += ['-A', 'x64']
                build_args += ['--', '/m']
            else:
                cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
                build_args += ['--', '-j8']

            if ext.build_with_cuda:
                cmake_args += ['-DDIFFVG_CUDA=1']
            else:
                cmake_args += ['-DDIFFVG_CUDA=0']

            env = os.environ.copy()
            env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                                  self.distribution.get_version())
            build_temp = self.build_temp
            if platform.system() == "Windows":
                source_hash = hashlib.sha1(ext.sourcedir.encode('utf-8')).hexdigest()[:8]
                build_temp = os.path.join(
                    tempfile.gettempdir(),
                    'diffvg-cmake',
                    'py{}{}-{}'.format(sys.version_info.major, sys.version_info.minor, source_hash))
                print('Using short CMake build directory: {}'.format(build_temp))
                print('Python executable: {}'.format(sys.executable))
                print('Python include: {}'.format(include_path))
                print('Python library: {}'.format(python_library))
            if not os.path.exists(build_temp):
                os.makedirs(build_temp)
            subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=build_temp, env=env)
            subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=build_temp)
            expected_ext_path = os.path.abspath(self.get_ext_fullpath(ext.name))
            extensionless_path = os.path.join(extdir, ext.name)
            if platform.system() == "Windows" and not os.path.exists(expected_ext_path) and os.path.exists(extensionless_path):
                os.replace(extensionless_path, expected_ext_path)
                print('Renamed {} to {}'.format(extensionless_path, expected_ext_path))
        else:
            super().build_extension(ext)

torch_spec = importlib.util.find_spec("torch")
tf_spec = importlib.util.find_spec("tensorflow")
packages = []
build_with_cuda = False
if torch_spec is not None:
    packages.append('pydiffvg')
    import torch
    if torch.cuda.is_available():
        build_with_cuda = True
if tf_spec is not None and sys.platform != 'win32':
    packages.append('pydiffvg_tensorflow')
    if not build_with_cuda:
        import tensorflow as tf
        if tf.test.is_gpu_available(cuda_only=True, min_cuda_compute_capability=None):
            build_with_cuda = True
if len(packages) == 0:
    print('Error: PyTorch or Tensorflow must be installed. For Windows platform only PyTorch is supported.')
    exit()
# Override build_with_cuda with environment variable
if 'DIFFVG_CUDA' in os.environ:
    build_with_cuda = os.environ['DIFFVG_CUDA'] == '1'

setup(name = 'diffvg',
      version = '0.0.1',
      install_requires = ["svgpathtools"],
      description = 'Differentiable Vector Graphics',
      ext_modules = [CMakeExtension('diffvg', '', build_with_cuda)],
      cmdclass = dict(build_ext=Build, install=install),
      packages = packages,
      zip_safe = False)

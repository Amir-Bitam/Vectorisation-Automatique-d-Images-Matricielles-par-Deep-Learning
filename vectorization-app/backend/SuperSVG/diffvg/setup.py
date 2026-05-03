# Adapted from https://github.com/pybind/cmake_example/blob/master/setup.py
import os
import re
import sys
import platform
import subprocess
import importlib
import shutil
import sysconfig
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
    def _cmake_command(self):
        candidates = []
        if platform.system() == "Windows":
            candidates.append(os.path.join(sys.prefix, 'Scripts', 'cmake.exe'))
        else:
            candidates.append(os.path.join(sys.prefix, 'bin', 'cmake'))

        cmake_on_path = shutil.which('cmake')
        if cmake_on_path is not None:
            candidates.append(cmake_on_path)

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

        return 'cmake'

    def _msbuild_safe_library_path(self, path):
        if platform.system() != "Windows" or "'" not in path:
            return path

        safe_root = os.environ.get('PUBLIC')
        if safe_root is None or "'" in safe_root:
            system_drive = os.environ.get('SystemDrive') or os.path.splitdrive(path)[0] or 'C:'
            safe_root = os.path.join(system_drive + os.sep, 'Users', 'Public')

        safe_dir = os.path.join(safe_root, 'diffvg_build_libs',
                                'python{}{}'.format(sys.version_info.major, sys.version_info.minor))
        os.makedirs(safe_dir, exist_ok=True)
        safe_path = os.path.join(safe_dir, os.path.basename(path))
        if (not os.path.exists(safe_path) or
                os.path.getmtime(safe_path) < os.path.getmtime(path) or
                os.path.getsize(safe_path) != os.path.getsize(path)):
            shutil.copy2(path, safe_path)
        print("Copied Python import library for MSBuild-safe path: {}".format(safe_path))
        return safe_path

    def run(self):
        cmake_cmd = self._cmake_command()
        try:
            out = subprocess.check_output([cmake_cmd, '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        self.cmake_cmd = cmake_cmd
        print("Using CMake executable: {}".format(cmake_cmd))
        super().run()

    def _python_library(self):
        if platform.system() == "Windows":
            lib_name = 'python{}{}.lib'.format(sys.version_info.major, sys.version_info.minor)
            for prefix in (getattr(sys, 'base_prefix', sys.prefix), sys.prefix):
                candidate = os.path.join(prefix, 'libs', lib_name)
                if os.path.exists(candidate):
                    return self._msbuild_safe_library_path(candidate)

        libdir = get_config_var('LIBDIR')
        libname = get_config_var('LDLIBRARY') or get_config_var('LIBRARY')
        if libdir is not None and libname is not None:
            candidate = os.path.join(libdir, libname)
            if os.path.exists(candidate):
                return self._msbuild_safe_library_path(candidate)

        if libdir is not None:
            return libdir

        return ''

    def build_extension(self, ext):
        if isinstance(ext, CMakeExtension):
            extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
            info = get_paths()
            include_path = info['include']
            python_library = self._python_library()
            python_root = getattr(sys, 'base_prefix', sys.prefix)
            python_module_extension = sysconfig.get_config_var('EXT_SUFFIX') or '.pyd'
            cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                          '-DPYTHON_LIBRARY=' + python_library,
                          '-DPYTHON_INCLUDE_PATH=' + include_path,
                          '-DPYTHON_INCLUDE_DIR=' + include_path,
                          '-DPYTHON_INCLUDE_DIRS=' + include_path,
                          '-DPYTHON_EXECUTABLE=' + sys.executable,
                          '-DPYTHONLIBS_VERSION_STRING={}.{}.{}'.format(
                              sys.version_info.major, sys.version_info.minor, sys.version_info.micro),
                          '-DPYTHON_MODULE_PREFIX=',
                          '-DPYTHON_MODULE_EXTENSION=' + python_module_extension,
                          '-DPython_EXECUTABLE=' + sys.executable,
                          '-DPython_ROOT_DIR=' + python_root,
                          '-DPython_INCLUDE_DIR=' + include_path,
                          '-DPython_LIBRARY=' + python_library,
                          '-DPython_FIND_STRATEGY=LOCATION',
                          '-DPython_FIND_REGISTRY=NEVER',
                          '-DPython_FIND_FRAMEWORK=NEVER',
                          '-DCMAKE_POLICY_VERSION_MINIMUM=3.5']

            print("Using Python executable: {}".format(sys.executable))
            print("Using Python include: {}".format(include_path))
            print("Using Python library: {}".format(python_library))
            print("Using Python extension suffix: {}".format(python_module_extension))
            print("CMake extension output directory: {}".format(extdir))

            cfg = 'Debug' if self.debug else 'Release'
            build_args = ['--config', cfg]

            if platform.system() == "Windows":
                cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir),
                               '-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
                generator = os.environ.get('CMAKE_GENERATOR', '')
                if sys.maxsize > 2**32 and 'Visual Studio' in generator:
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
            if not os.path.exists(self.build_temp):
                os.makedirs(self.build_temp)
            cmake_cmd = getattr(self, 'cmake_cmd', self._cmake_command())
            subprocess.check_call([cmake_cmd, ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
            subprocess.check_call([cmake_cmd, '--build', '.'] + build_args, cwd=self.build_temp)
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

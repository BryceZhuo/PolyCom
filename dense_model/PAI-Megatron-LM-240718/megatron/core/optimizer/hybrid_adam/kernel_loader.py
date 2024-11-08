# Copyright (c) 2024 Alibaba PAI, ColossalAI and Nvidia Megatron-LM Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import warnings
from typing import List
import hashlib
import os
from abc import ABC, abstractmethod
from typing import Callable, Union
from pathlib import Path
import importlib
import time
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDA_HOME, CUDAExtension
import platform

from ._utils import *

class _Extension(ABC):
    def __init__(self, name: str, support_aot: bool, support_jit: bool, priority: int = 1):
        self._name = name
        self._support_aot = support_aot
        self._support_jit = support_jit
        self.priority = priority

    @property
    def name(self):
        return self._name

    @property
    def support_aot(self):
        return self._support_aot

    @property
    def support_jit(self):
        return self._support_jit

    @staticmethod
    def get_jit_extension_folder_path():
        """
        Kernels which are compiled during runtime will be stored in the same cache folder for reuse.
        The folder is in the path ~/.cache/colossalai/torch_extensions/<cache-folder>.
        The name of the <cache-folder> follows a common format:
            torch<torch_version_major>.<torch_version_minor>_<device_name><device_version>-<hash>

        The <hash> suffix is the hash value of the path of the `colossalai` file.
        """
        import torch

        import colossalai
        from colossalai.accelerator import get_accelerator

        # get torch version
        torch_version_major = torch.__version__.split(".")[0]
        torch_version_minor = torch.__version__.split(".")[1]

        # get device version
        device_name = get_accelerator().name
        device_version = get_accelerator().get_version()

        # use colossalai's file path as hash
        hash_suffix = hashlib.sha256(colossalai.__file__.encode()).hexdigest()

        # concat
        home_directory = os.path.expanduser("~")
        extension_directory = f".cache/colossalai/torch_extensions/torch{torch_version_major}.{torch_version_minor}_{device_name}-{device_version}-{hash_suffix}"
        cache_directory = os.path.join(home_directory, extension_directory)
        return cache_directory

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the hardware required by the kernel is available.
        """

    @abstractmethod
    def assert_compatible(self) -> None:
        """
        Check if the hardware required by the kernel is compatible.
        """

    @abstractmethod
    def build_aot(self) -> Union["CppExtension", "CUDAExtension"]:
        pass

    @abstractmethod
    def build_jit(self) -> Callable:
        pass

    @abstractmethod
    def load(self) -> Callable:
        pass

__all__ = [
    "CPUAdamLoader",
]
# Some constants for installation checks
MIN_PYTORCH_VERSION_MAJOR = 1
MIN_PYTORCH_VERSION_MINOR = 10


class _CppExtension(_Extension):
    def __init__(self, name: str, priority: int = 1):
        super().__init__(name, support_aot=True, support_jit=True, priority=priority)

        # we store the op as an attribute to avoid repeated building and loading
        self.cached_op = None

        # build-related variables
        self.prebuilt_module_path = "megatron_patch._C"
        self.prebuilt_import_path = f"{self.prebuilt_module_path}.{self.name}"
        self.version_dependent_macros = ["-DVERSION_GE_1_1", "-DVERSION_GE_1_3", "-DVERSION_GE_1_5"]

    def csrc_abs_path(self, path):
        return os.path.join(self.relative_to_abs_path("csrc"), path)

    def relative_to_abs_path(self, code_path: str) -> str:
        """
        This function takes in a path relative to the root directory and return the absolute path.
        """

        # get the current file path
        # iteratively check the parent directory
        # if the parent directory is "hybrid_adam", then the current file path is the root directory
        # otherwise, the current file path is inside the root directory
        current_file_path = Path(__file__)
        while True:
            if current_file_path.name == "hybrid_adam":
                break
            else:
                current_file_path = current_file_path.parent
        extension_module_path = current_file_path
        code_abs_path = extension_module_path.joinpath(code_path)
        return str(code_abs_path)

    # functions must be overrided over
    def strip_empty_entries(self, args):
        """
        Drop any empty strings from the list of compile and link flags
        """
        return [x for x in args if len(x) > 0]

    def import_op(self):
        """
        This function will import the op module by its string name.
        """
        return importlib.import_module(self.prebuilt_import_path)

    def build_aot(self) -> "CppExtension":
        
        return CppExtension(
            name=self.prebuilt_import_path,
            sources=self.strip_empty_entries(self.sources_files()),
            include_dirs=self.strip_empty_entries(self.include_dirs()),
            extra_compile_args=self.strip_empty_entries(self.cxx_flags()),
        )

    def build_jit(self) -> None:
        from torch.utils.cpp_extension import load

        build_directory = _Extension.get_jit_extension_folder_path()
        build_directory = Path(build_directory)
        build_directory.mkdir(parents=True, exist_ok=True)

        # check if the kernel has been built
        compiled_before = False
        kernel_file_path = build_directory.joinpath(f"{self.name}.o")
        if kernel_file_path.exists():
            compiled_before = True

        # load the kernel
        if compiled_before:
            print(f"[extension] Loading the JIT-built {self.name} kernel during runtime now")
        else:
            print(f"[extension] Compiling the JIT {self.name} kernel during runtime now")

        build_start = time.time()
        op_kernel = load(
            name=self.name,
            sources=self.strip_empty_entries(self.sources_files()),
            extra_include_paths=self.strip_empty_entries(self.include_dirs()),
            extra_cflags=self.cxx_flags(),
            extra_ldflags=[],
            build_directory=str(build_directory),
        )
        build_duration = time.time() - build_start

        if compiled_before:
            print(f"[extension] Time taken to load {self.name} op: {build_duration} seconds")
        else:
            print(f"[extension] Time taken to compile {self.name} op: {build_duration} seconds")

        return op_kernel

    # functions must be overrided begin
    @abstractmethod
    def sources_files(self) -> List[str]:
        """
        This function should return a list of source files for extensions.
        """

    @abstractmethod
    def include_dirs(self) -> List[str]:
        """
        This function should return a list of include files for extensions.
        """
        return [self.csrc_abs_path("")]

    @abstractmethod
    def cxx_flags(self) -> List[str]:
        """
        This function should return a list of cxx compilation flags for extensions.
        """

    def load(self):
        try:
            op_kernel = self.import_op()
        except (ImportError, ModuleNotFoundError):
            # if import error occurs, it means that the kernel is not pre-built
            # so we build it jit
            op_kernel = self.build_jit()

        return op_kernel

class _CudaExtension(_CppExtension):
    @abstractmethod
    def nvcc_flags(self) -> List[str]:
        """
        This function should return a list of nvcc compilation flags for extensions.
        """
        return ["-DCOLOSSAL_WITH_CUDA"]

    def is_available(self) -> bool:
        # cuda extension can only be built if cuda is available
        try:
            import torch

            # torch.cuda.is_available requires a device to exist, allow building with cuda extension on build nodes without a device
            # but where cuda is actually available.
            cuda_available = torch.cuda.is_available() or bool(os.environ.get("FORCE_CUDA", 0))
        except:
            cuda_available = False
        return cuda_available

    def assert_compatible(self) -> None:
        from torch.utils.cpp_extension import CUDA_HOME

        if not CUDA_HOME:
            raise AssertionError(
                "[extension] CUDA_HOME is not found. You need to export CUDA_HOME environment variable or install CUDA Toolkit first in order to build/load CUDA extensions"
            )
        check_system_pytorch_cuda_match(CUDA_HOME)
        check_pytorch_version(MIN_PYTORCH_VERSION_MAJOR, MIN_PYTORCH_VERSION_MINOR)

    def get_cuda_home_include(self):
        """
        return include path inside the cuda home.
        """
        from torch.utils.cpp_extension import CUDA_HOME

        if CUDA_HOME is None:
            raise RuntimeError("CUDA_HOME is None, please set CUDA_HOME to compile C++/CUDA kernels in ColossalAI.")
        cuda_include = os.path.join(CUDA_HOME, "include")
        return cuda_include

    def include_dirs(self) -> List[str]:
        """
        This function should return a list of include files for extensions.
        """
        return super().include_dirs() + [self.get_cuda_home_include()]

    def build_jit(self) -> None:
        from torch.utils.cpp_extension import CUDA_HOME, load

        set_cuda_arch_list(CUDA_HOME)

        # get build dir
        build_directory = _Extension.get_jit_extension_folder_path()
        build_directory = Path(build_directory)
        build_directory.mkdir(parents=True, exist_ok=True)

        # check if the kernel has been built
        compiled_before = False
        kernel_file_path = build_directory.joinpath(f"{self.name}.o")
        if kernel_file_path.exists():
            compiled_before = True

        # load the kernel
        if compiled_before:
            print(f"[extension] Loading the JIT-built {self.name} kernel during runtime now")
        else:
            print(f"[extension] Compiling the JIT {self.name} kernel during runtime now")

        build_start = time.time()
        op_kernel = load(
            name=self.name,
            sources=self.strip_empty_entries(self.sources_files()),
            extra_include_paths=self.strip_empty_entries(self.include_dirs()),
            extra_cflags=self.cxx_flags(),
            extra_cuda_cflags=self.nvcc_flags(),
            extra_ldflags=[],
            build_directory=str(build_directory),
        )
        build_duration = time.time() - build_start

        if compiled_before:
            print(f"[extension] Time taken to load {self.name} op: {build_duration} seconds")
        else:
            print(f"[extension] Time taken to compile {self.name} op: {build_duration} seconds")

        return op_kernel

    def build_aot(self) -> "CUDAExtension":
        

        set_cuda_arch_list(CUDA_HOME)
        return CUDAExtension(
            name=self.prebuilt_import_path,
            sources=self.strip_empty_entries(self.sources_files()),
            include_dirs=self.strip_empty_entries(self.include_dirs()),
            extra_compile_args={
                "cxx": self.strip_empty_entries(self.cxx_flags()),
                "nvcc": self.strip_empty_entries(self.nvcc_flags()),
            },
        )

class CpuAdamX86Extension(_CudaExtension):
    def __init__(self):
        super().__init__(name="cpu_adam_x86")

    def is_available(self) -> bool:
        return platform.machine() == "x86_64" and super().is_available()

    def assert_compatible(self) -> None:
        arch = platform.machine()
        assert (
            arch == "x86_64"
        ), f"[extension] The {self.name} kernel requires the CPU architecture to be x86_64 but got {arch}"
        super().assert_compatible()

    # necessary 4 functions
    def sources_files(self):
        ret = [
            self.csrc_abs_path("cpu_adam.cpp"),
        ]
        return ret

    def cxx_flags(self):
        extra_cxx_flags = [
            "-std=c++14",
            "-std=c++17",
            "-lcudart",
            "-lcublas",
            "-g",
            "-Wno-reorder",
            "-fopenmp",
            "-march=native",
        ]
        return ["-O3"] + self.version_dependent_macros + extra_cxx_flags

    def nvcc_flags(self):
        extra_cuda_flags = [
            "-std=c++14",
            "-std=c++17",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-DTHRUST_IGNORE_CUB_VERSION_CHECK",
        ]
        ret = ["-O3", "--use_fast_math"] + self.version_dependent_macros + extra_cuda_flags + super().nvcc_flags()
        return append_nvcc_threads(ret)

class CpuAdamArmExtension(_CppExtension):
    def __init__(self):
        super().__init__(name="cpu_adam_arm")

    def is_available(self) -> bool:
        # only arm allowed
        return platform.machine() == "aarch64"

    def assert_compatible(self) -> None:
        arch = platform.machine()
        assert (
            arch == "aarch64"
        ), f"[extension] The {self.name} kernel requires the CPU architecture to be aarch64 but got {arch}"

    # necessary 4 functions
    def sources_files(self):
        ret = [
            self.csrc_abs_path("cpu_adam_arm.cpp"),
        ]
        return ret

    def include_dirs(self) -> List[str]:
        return super().include_dirs()

    def cxx_flags(self):
        extra_cxx_flags = [
            "-std=c++14",
            "-std=c++17",
            "-g",
            "-Wno-reorder",
            "-fopenmp",
        ]
        return ["-O3"] + self.version_dependent_macros + extra_cxx_flags

    def nvcc_flags(self):
        return []

class KernelLoader:
    """
    An abstract class which offers encapsulation to the kernel loading process.

    Usage:
        kernel_loader = KernelLoader()
        kernel = kernel_loader.load()
    """

    REGISTRY: List[_Extension] = []

    @classmethod
    def register_extension(cls, extension: _Extension):
        """
        This classmethod is an extension point which allows users to register their customized
        kernel implementations to the loader.

        Args:
            extension (_Extension): the extension to be registered.
        """
        cls.REGISTRY.append(extension)

    def load(self, ext_name: str = None):
        """
        Load the kernel according to the current machine.

        Args:
            ext_name (str): the name of the extension to be loaded. If not specified, the loader
                will try to look for an kernel available on the current machine.
        """
        exts = [ext_cls() for ext_cls in self.__class__.REGISTRY]

        # look for exts which can be built/loaded on the current machine

        if ext_name:
            usable_exts = list(filter(lambda ext: ext.name == ext_name, exts))
        else:
            usable_exts = []
            for ext in exts:
                if ext.is_available():
                    # make sure the machine is compatible during kernel loading
                    ext.assert_compatible()
                    usable_exts.append(ext)

        assert len(usable_exts) != 0, f"No usable kernel found for {self.__class__.__name__} on the current machine."

        if len(usable_exts) > 1:
            # if more than one usable kernel is found, we will try to load the kernel with the highest priority
            usable_exts = sorted(usable_exts, key=lambda ext: ext.priority, reverse=True)
            warnings.warn(
                f"More than one kernel is available, loading the kernel with the highest priority - {usable_exts[0].__class__.__name__}"
            )
        return usable_exts[0].load()

class CPUAdamLoader(KernelLoader):
    REGISTRY = [CpuAdamX86Extension, CpuAdamArmExtension]


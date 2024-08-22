import os
import platform
from ctypes import *

lib = None

if platform.system().lower() == 'windows':
    lib = CDLL(os.path.join(os.path.dirname(__file__), "../cmake-build-debuglocal", "libtinynpu_support.dll"), RTLD_GLOBAL)
elif platform.system().lower() == 'linux':
    lib = CDLL(os.path.join(os.path.dirname(__file__),"../build", "libtinynpu_support.so"), RTLD_GLOBAL)
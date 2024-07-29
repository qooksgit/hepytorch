"""
Awesome hep pytorch package
~~~~~~~~~~~~~~~~~~~

A nice pytorch package for high energy physics.

:copyright: (c) 2024-present H.S. Kim and W.Q. Choi
:license: MIT, see LICENSE for more details.

"""
__title__ = 'hepytorch'
__author__ = 'H.S. Kim and W.Q. Choi'
__license__ = 'MIT'
__copyright__ = 'Copyright 2024-present H.S. Kim and W.Q. Choi'
__version__ = '0.0.1a'

__path__ = __import__('pkgutil').extend_path(__path__, __name__)

import logging
from typing import NamedTuple, Literal

#from .dir import *

from . import (
    models as models,
)

from .hepytorch import *


class VersionInfo(NamedTuple):
    major: int
    minor: int
    micro: int
    releaselevel: Literal["alpha", "beta", "candidate", "final"]
    serial: int


version_info: VersionInfo = VersionInfo(major=2, minor=5, micro=0, releaselevel='alpha', serial=0)

logging.getLogger(__name__).addHandler(logging.NullHandler())

del logging, NamedTuple, Literal, VersionInfo

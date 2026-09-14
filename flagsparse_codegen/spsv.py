# Copyright 2026 FlagOS Contributors
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

"""Triangular-solve kernels -- re-exported, not copied. No wrappers needed.

The chain-wave route: workers take rows from an atomic counter and spin on a
ready flag until every dependency of their row has been published. That gives
one kernel for the whole solve instead of a level-by-level launch, and the
accumulator type is already a bool constexpr (USE_FP64_ACC), so unlike
spmm_csr.py this module needs no jit wrappers.

Two things the dispatch layer has to guarantee, because the kernel assumes them:

* ``ready`` (n_rows int32) and ``row_counter`` (1 int32) must be ZERO at launch.
  They are the caller's scratch, sized by flagsparseSpSV_bufferSize and cleared
  by the solve.
* Column indices must be sorted ascending within a row. The scan walks a row and
  stops at the diagonal, so an unsorted row stops early and silently drops
  terms. flagsparseSpSV_analysis checks it.

Over-subscribing workers is safe: a program only waits on rows with a lower
logical index, which can only have been claimed by a resident program. See the
kernel docstring.
"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

from flagsparse.sparse_operations.spsv import (  # noqa: E402,F401
    _spsv_csr_cw_kernel,
    _spsv_csr_cw_kernel_complex,
)

__all__ = ["_spsv_csr_cw_kernel", "_spsv_csr_cw_kernel_complex"]

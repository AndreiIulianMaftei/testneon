# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""
NeoN - A framework for CFD software

Python bindings for the NeoN CFD framework.

Note: Type stubs (_neon.pyi) are automatically generated during the build process
using nanobind-stubgen. This enables IDE autocomplete and type checking.
"""

__version__ = "0.1.0"

# Import the C++ extension module
try:
    from ._neon import *  # noqa: F401, F403
    # Explicitly import module attributes that aren't exported by *
    from ._neon import __has_serial__, __has_cpu__, __has_gpu__
except ImportError as e:
    raise ImportError(
        "Failed to import NeoN C++ extension module. "
        "Make sure the package is properly installed. "
        f"Error: {e}"
    ) from e

__all__ = ["__version__", "__has_serial__", "__has_cpu__", "__has_gpu__"]

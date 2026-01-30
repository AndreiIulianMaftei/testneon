# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""
Test configuration for NeoN Python bindings.

This module automatically locates the neon package from the build directory
and configures pytest fixtures for executor parameterization.
"""

import sys
from pathlib import Path

import pytest


def _find_and_setup_neon():
    """
    Find the neon package in the build directory and add it to sys.path.

    Searches common build locations for the compiled neon package.
    """
    test_dir = Path(__file__).resolve().parent
    project_root = test_dir.parent.parent

    build_patterns = [
        "build/develop/bindings",
        "build/release/bindings",
        "build/bindings",
    ]

    build_base = project_root / "build"
    if build_base.exists():
        for subdir in build_base.iterdir():
            if subdir.is_dir() and (subdir / "bindings").exists():
                build_patterns.append(f"build/{subdir.name}/bindings")

    for pattern in build_patterns:
        bindings_dir = project_root / pattern
        neon_dir = bindings_dir / "neon"
        if neon_dir.exists():
            # Check if _neon module exists
            neon_files = list(neon_dir.glob("_neon*.so")) + list(neon_dir.glob("_neon*.pyd"))
            init_file = neon_dir / "__init__.py"
            if neon_files and init_file.exists():
                bindings_path = str(bindings_dir)
                if bindings_path not in sys.path:
                    sys.path.insert(0, bindings_path)
                return True

    raise ImportError(
        "Could not find neon package. Please build the project first.\n"
        "Expected location: build/develop/bindings/neon/\n"
        "Build with: cmake --build --preset develop"
    )


# Set up path before importing neon
_find_and_setup_neon()
import neon


@pytest.fixture(scope="session", autouse=True)
def neon_global_session():
    """Initialize NeoN once for all test files in this session."""
    neon.initialize()
    yield  # This is where all tests run
    neon.finalize()


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU support")


def get_available_executors():
    """Get list of executors available at build time."""
    executors = []
    if neon.__has_serial__:
        executors.append(("serial", neon.SerialExecutor))
    if neon.__has_cpu__:
        executors.append(("cpu", neon.CPUExecutor))
    if neon.__has_gpu__:
        executors.append(("gpu", neon.GPUExecutor))
    return executors


def pytest_generate_tests(metafunc):
    """Auto-parameterize tests using 'executor' fixture."""
    if "executor" in metafunc.fixturenames:
        available = get_available_executors()
        if not available:
            pytest.skip("No executors available")
        # Create instances directly in parametrization
        executor_instances = [(name, exec_class()) for name, exec_class in available]
        metafunc.parametrize(
            "executor",
            executor_instances,
            ids=[name for name, _ in available]
        )


def pytest_collection_modifyitems(config, items):
    """Automatically skip GPU tests if GPU is not available."""
    if hasattr(neon, '__has_gpu__') and not neon.__has_gpu__:
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)

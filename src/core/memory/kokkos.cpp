// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <Kokkos_Core.hpp>

#include "NeoN/core/memory/kokkos.hpp"

namespace NeoN
{

void* KokkosAllocator::alloc(size_t size)
{
    if (memSpace_ == MemorySpace::GPU)
    {
        return Kokkos::kokkos_malloc<GPUMemSpace>("Vector", size);
    }
    else if (memSpace_ == MemorySpace::CPU)
    {
        return Kokkos::kokkos_malloc<CPUMemSpace>("Vector", size);
    }
    // this shouldnt be reached
    return Kokkos::kokkos_malloc<GPUMemSpace>("Vector", size);
}

void* KokkosAllocator::realloc(void* ptr, size_t size)
{
    switch (memSpace_)
    {
    case MemorySpace::GPU:
    {
        return Kokkos::kokkos_realloc<GPUMemSpace>(ptr, size);
    }
    case MemorySpace::CPU:
    {
        return Kokkos::kokkos_realloc<CPUMemSpace>(ptr, size);
    }
    }
}

void KokkosAllocator::free(void* ptr)
{
    switch (memSpace_)
    {
    case MemorySpace::GPU:
    {
        Kokkos::kokkos_free<GPUMemSpace>(ptr);
        break;
    }
    case MemorySpace::CPU:
    {
        Kokkos::kokkos_free<CPUMemSpace>(ptr);
        break;
    }
    }
}

}

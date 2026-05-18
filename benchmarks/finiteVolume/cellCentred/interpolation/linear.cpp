// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "NeoN/NeoN.hpp"
#include "benchmarks/catch_main.hpp"
#include "test/catch2/executorGenerator.hpp"

#include <catch2/catch_template_test_macros.hpp>

using NeoN::finiteVolume::cellCentred::SurfaceInterpolation;
using NeoN::finiteVolume::cellCentred::VolumeField;
using NeoN::finiteVolume::cellCentred::SurfaceField;

TEMPLATE_TEST_CASE("linear", "[bench]", NeoN::scalar, NeoN::Vec3)
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    NeoN::Input input = NeoN::TokenList({std::string("linear")});

    SECTION("2D")
    {
        auto nCellsPerDim = GENERATE(256, 512, 1024);
        NeoN::UnstructuredMesh mesh = NeoN::create2DUniformMesh(exec, nCellsPerDim, nCellsPerDim);
        auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<TestType>>(mesh);
        auto linear = SurfaceInterpolation<TestType>(exec, mesh, input);
        auto in = VolumeField<TestType>(exec, "in", mesh, {});
        auto out = SurfaceField<TestType>(exec, "out", mesh, surfaceBCs);
        NeoN::fill(in.internalVector(), NeoN::one<TestType>());

        DYNAMIC_SECTION(nCellsPerDim << "x" << nCellsPerDim)
        {
            BENCHMARK(std::string(execName)) { linear.interpolate(in, out); };
        }
    }

    SECTION("3D")
    {
        auto nCellsPerDim = GENERATE(32, 64, 128);
        NeoN::UnstructuredMesh mesh =
            NeoN::create3DUniformMesh(exec, nCellsPerDim, nCellsPerDim, nCellsPerDim);
        auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<TestType>>(mesh);
        auto linear = SurfaceInterpolation<TestType>(exec, mesh, input);
        auto in = VolumeField<TestType>(exec, "in", mesh, {});
        auto out = SurfaceField<TestType>(exec, "out", mesh, surfaceBCs);
        NeoN::fill(in.internalVector(), NeoN::one<TestType>());

        DYNAMIC_SECTION(nCellsPerDim << "x" << nCellsPerDim << "x" << nCellsPerDim)
        {
            BENCHMARK(std::string(execName)) { linear.interpolate(in, out); };
        }
    }
}

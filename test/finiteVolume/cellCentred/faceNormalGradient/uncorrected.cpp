// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

using NeoN::finiteVolume::cellCentred::VolumeField;
using NeoN::finiteVolume::cellCentred::SurfaceField;
using NeoN::finiteVolume::cellCentred::FaceNormalGradient;

namespace NeoN
{

template<typename T>
using I = std::initializer_list<T>;

TEMPLATE_TEST_CASE("uncorrected", "[template]", NeoN::scalar, NeoN::Vec3)
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    const NeoN::localIdx nCells = 10;
    auto mesh = create1DUniformMesh(exec, nCells);

    // --- create BCs (Volume only!) ---
    std::vector<fvcc::VolumeBoundary<TestType>> vbcs;

    auto nPatches = mesh.boundaryMesh().offset().size() - 1;

    for (NeoN::localIdx patchi = 0; patchi < nPatches; ++patchi)
    {
        Dictionary dict;
        dict.insert("type", std::string("fixedValue"));

        if (patchi == 0) // left boundary
        {
            dict.insert("fixedValue", 0.5 * one<TestType>());
        }
        else if (patchi == 1) // right boundary
        {
            dict.insert("fixedValue", 10.5 * one<TestType>());
        }
        else // unused in 1D, but keep robust
        {
            dict.insert("fixedValue", 0.0 * one<TestType>());
        }

        vbcs.emplace_back(mesh, dict, patchi);
    }

    auto phi = VolumeField<TestType>(exec, "phi", mesh, vbcs);
    NeoN::parallelFor(
        phi.internalVector(),
        NEON_LAMBDA(const NeoN::localIdx i) { return scalar(i + 1) * one<TestType>(); }
    );
    phi.correctBoundaryConditions();

    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<TestType>>(mesh);

    fvcc::SurfaceField<TestType> phif(exec, "phif", mesh, surfaceBCs);
    fill(phif.internalVector(), zero<TestType>());

    SECTION("Construct from Token" + execName)
    {
        NeoN::Input input = NeoN::TokenList({std::string("uncorrected")});
        fvcc::FaceNormalGradient<TestType> uncorrected(exec, mesh, input);
    }

    SECTION("faceNormalGrad" + execName)
    {
        NeoN::Input input = NeoN::TokenList({std::string("uncorrected")});
        fvcc::FaceNormalGradient<TestType> uncorrected(exec, mesh, input);
        uncorrected.faceNormalGrad(phi, phif);

        auto phifHost = phif.internalVector().copyToHost();
        auto sPhif = phifHost.view();
        for (NeoN::localIdx i = 0; i < nCells - 1; i++)
        {
            // correct value is 10.0
            REQUIRE(
                NeoN::mag(sPhif[i] - 10.0 * one<TestType>()) == Catch::Approx(0.0).margin(1e-8)
            );
        }
        // left boundary is  -10.0
        REQUIRE(
            NeoN::mag(sPhif[nCells - 1] + 10.0 * one<TestType>()) == Catch::Approx(0.0).margin(1e-8)
        );
        // right boundary is 10.0
        REQUIRE(
            NeoN::mag(sPhif[nCells] - 10.0 * one<TestType>()) == Catch::Approx(0.0).margin(1e-8)
        );
    }
}
}

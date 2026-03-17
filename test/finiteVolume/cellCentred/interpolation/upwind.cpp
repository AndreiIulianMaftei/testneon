// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"


using NeoN::finiteVolume::cellCentred::SurfaceInterpolation;
using NeoN::finiteVolume::cellCentred::VolumeField;
using NeoN::finiteVolume::cellCentred::SurfaceField;

namespace NeoN
{

template<typename T>
using I = std::initializer_list<T>;

TEMPLATE_TEST_CASE("upwind", "", NeoN::scalar, NeoN::Vec3)
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto mesh = create1DUniformMesh(exec, 10);
    Input input = TokenList({std::string("upwind")});
    auto upwind = SurfaceInterpolation<TestType>(exec, mesh, input);
    std::vector<fvcc::SurfaceBoundary<TestType>> bcs {};

    auto nPatches = mesh.boundaryMesh().offset().size() - 1;
    for (NeoN::localIdx patchi = 0; patchi < nPatches; ++patchi)
    {
        Dictionary dict;
        dict.insert("type", std::string("fixedValue"));
        dict.insert("fixedValue", one<TestType>());
        bcs.emplace_back(mesh, dict, patchi);
    }

    auto in = VolumeField<TestType>(exec, "in", mesh, {});
    auto flux = SurfaceField<scalar>(exec, "flux", mesh, {});
    auto out = SurfaceField<TestType>(exec, "out", mesh, bcs);

    fill(flux.internalVector(), one<scalar>());
    fill(in.internalVector(), one<TestType>());

    upwind.interpolate(flux, in, out);
    out.correctBoundaryConditions();

    auto outHost = out.internalVector().copyToHost();
    auto nInternal = mesh.nInternalFaces();
    auto nBoundary = mesh.nBoundaryFaces();
    for (NeoN::localIdx i = 0; i < nInternal; i++)
    {
        REQUIRE(outHost.view()[i] == one<TestType>());
    }

    for (NeoN::localIdx i = nInternal; i < nInternal + nBoundary; i++)
    {
        REQUIRE(outHost.view()[i] == one<TestType>());
    }
}

}

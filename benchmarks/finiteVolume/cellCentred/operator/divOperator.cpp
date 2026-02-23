// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "NeoN/NeoN.hpp"
#include "benchmarks/catch_main.hpp"
#include "test/catch2/executorGenerator.hpp"

using Operator = NeoN::dsl::Operator;

TEMPLATE_TEST_CASE("DivOperator::div", "[bench]", NeoN::scalar, NeoN::Vec3)
{
    auto size = GENERATE(1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20);
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    NeoN::UnstructuredMesh mesh = NeoN::create1DUniformMesh(exec, size);
    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(mesh);
    fvcc::SurfaceField<NeoN::scalar> faceFlux(exec, "sf", mesh, surfaceBCs);
    NeoN::fill(faceFlux.internalVector(), 1.0);

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<TestType>>(mesh);
    fvcc::VolumeField<TestType> phi(exec, "vf", mesh, volumeBCs);
    NeoN::fill(phi.internalVector(), NeoN::one<TestType>());

    // capture the value of size as section name
    DYNAMIC_SECTION("" << size)
    {
        NeoN::Input input = NeoN::TokenList({std::string("Gauss"), std::string("linear")});

        SECTION("Explicit")
        {
            // Create a scalar field to hold the div value - output field
            fvcc::VolumeField<TestType> divPhi(exec, "vf", mesh, volumeBCs);
            NeoN::fill(divPhi.internalVector(), NeoN::zero<TestType>());

            auto op = fvcc::DivOperator(Operator::Type::Explicit, faceFlux, phi, input);

            BENCHMARK(std::string(execName) + "_explicit")
            {
                op.explicitOperation(divPhi.internalVector());
            };
        }

        SECTION("Implicit")
        {
            // Build sparsity pattern and allocate linear system once - output goes to ls
            auto ls = la::createEmptyLinearSystem<TestType>(mesh);

            auto op = fvcc::DivOperator(Operator::Type::Implicit, faceFlux, phi, input);

            BENCHMARK(std::string(execName) + "_implicit") { op.implicitOperation(ls); };
        }
    }
}

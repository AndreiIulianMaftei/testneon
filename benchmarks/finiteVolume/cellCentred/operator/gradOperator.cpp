// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "NeoN/NeoN.hpp"
#include "benchmarks/catch_main.hpp"
#include "test/catch2/executorGenerator.hpp"

using Operator = NeoN::dsl::Operator;

TEST_CASE("GradOperator::grad", "[bench]")
{
    auto size = GENERATE(1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20);
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    NeoN::UnstructuredMesh mesh = NeoN::create1DUniformMesh(exec, size);

    // Create a scalar field phi and initialise with 1.0
    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::scalar>>(mesh);
    fvcc::VolumeField<NeoN::scalar> phi(exec, "vf", mesh, volumeBCs);
    NeoN::fill(phi.internalVector(), 1.0);

    // Create a vector field to hold the gradient value
    auto gradVecBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::Vec3>>(mesh);
    fvcc::VolumeField<NeoN::Vec3> gradPhi(exec, "gradPhi", mesh, gradVecBCs);
    NeoN::fill(gradPhi.internalVector(), NeoN::zero<NeoN::Vec3>());

    // capture the value of size as section name
    DYNAMIC_SECTION("" << size)
    {
        NeoN::Input input = NeoN::TokenList({std::string("Gauss"), std::string("linear")});
        auto op = fvcc::GradOperator<NeoN::Vec3>(Operator::Type::Explicit, phi, input);

        BENCHMARK(std::string(execName))
        {
            return (op.explicitOperation(gradPhi.internalVector()));
        };
    }
}

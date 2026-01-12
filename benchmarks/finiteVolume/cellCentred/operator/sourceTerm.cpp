// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "NeoN/NeoN.hpp"
#include "benchmarks/catch_main.hpp"
#include "test/catch2/executorGenerator.hpp"

using Operator = NeoN::dsl::Operator;

TEST_CASE("SourceTerm::sourceTerm", "[bench]")
{
    auto size = GENERATE(1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20);
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    NeoN::UnstructuredMesh mesh = NeoN::create1DUniformMesh(exec, size);

    // Create a scalar field coeff and initialise with 1.0
    auto coeffBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::scalar>>(mesh);
    fvcc::VolumeField<NeoN::scalar> coeff(exec, "coeff", mesh, coeffBCs);
    fill(coeff.internalVector(), 2.0);
    fill(coeff.boundaryData().value(), 0.0);
    coeff.correctBoundaryConditions();

    // Create a scalar field to hold the output field
    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::scalar>>(mesh);
    fvcc::VolumeField<NeoN::scalar> phi(exec, "sf", mesh, volumeBCs);
    fill(phi.internalVector(), 1.0);
    fill(phi.boundaryData().value(), 0.0);
    phi.correctBoundaryConditions();

    // capture the value of size as section name
    DYNAMIC_SECTION("" << size)
    {
        SECTION("Explicit")
        {
            auto op = fvcc::SourceTerm<NeoN::scalar>(Operator::Type::Explicit, coeff, phi);

            BENCHMARK(std::string(execName) + "_explicit")
            {
                op.explicitOperation(phi.internalVector());
            };
        }

        SECTION("Implicit")
        {
            // Build sparsity pattern and allocate linear system once - output goes to ls
            const auto& sp = la::SparsityPattern::readOrCreate(mesh);
            auto ls = la::createEmptyLinearSystem<NeoN::scalar, NeoN::localIdx>(mesh, sp);

            auto op = fvcc::SourceTerm<NeoN::scalar>(Operator::Type::Implicit, coeff, phi);

            BENCHMARK(std::string(execName) + "_implicit") { op.implicitOperation(ls); };
        }
    }
}

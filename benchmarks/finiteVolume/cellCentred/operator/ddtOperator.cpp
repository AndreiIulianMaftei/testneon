// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "NeoN/NeoN.hpp"
#include "benchmarks/catch_main.hpp"
#include "test/catch2/executorGenerator.hpp"
#include "benchmarks/finiteVolume/cellCentred/common.hpp"

using Operator = NeoN::dsl::Operator;

TEMPLATE_TEST_CASE("DdtOperator::ddt", "[bench]", NeoN::scalar, NeoN::Vec3)
{
    auto size = GENERATE(1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20);
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    NeoN::UnstructuredMesh mesh = NeoN::create1DUniformMesh(exec, size);

    NeoN::Database db;
    fvcc::VectorCollection& fieldCollection =
        fvcc::VectorCollection::instance(db, "benchVectorCollection");
    fvcc::VolumeField<TestType>& phi = fieldCollection.registerVector<fvcc::VolumeField<TestType>>(
        CreateVector<TestType> {.name = "phi", .mesh = mesh, .timeIndex = 1}
    );

    NeoN::fill(phi.internalVector(), NeoN::one<TestType>());
    NeoN::fill(phi.boundaryData().value(), NeoN::zero<TestType>());
    phi.correctBoundaryConditions();

    // Ensure old-time storage exists and is initialized
    NeoN::fill(oldTime(phi).internalVector(), 0.5 * NeoN::one<TestType>());           // phi^n
    NeoN::fill(oldTime(oldTime(phi)).internalVector(), 0.25 * NeoN::one<TestType>()); // phi^{n-1}

    const NeoN::scalar t = 1.0;
    const NeoN::scalar dt = 0.5;

    // capture the value of size as section name
    DYNAMIC_SECTION("" << size)
    {
        SECTION("Explicit")
        {
            // Create a scalar field to hold the div value - output field
            NeoN::Vector<TestType> source(exec, phi.size(), NeoN::zero<TestType>());

            auto op = fvcc::DdtOperator(Operator::Type::Explicit, phi);

            BENCHMARK(std::string(execName) + "_explicit") { op.explicitOperation(source, t, dt); };
        }

        SECTION("Implicit_Euler")
        {
            NeoN::Dictionary fvSchemes;
            NeoN::Dictionary ddtSchemes;
            ddtSchemes.insert("ddt(phi)", std::string("BDF1"));
            fvSchemes.insert("ddtSchemes", ddtSchemes);

            // Build sparsity pattern and allocate linear system once - output goes to ls
            auto ls = la::createEmptyLinearSystem<TestType>(mesh);

            auto op = fvcc::DdtOperator(Operator::Type::Implicit, phi);
            op.read(fvSchemes);

            BENCHMARK(std::string(execName) + "_implicit_Euler")
            {
                op.implicitOperation(ls, t, dt);
            };
        }

        SECTION("Implicit_BDF2")
        {
            NeoN::Dictionary fvSchemes;
            NeoN::Dictionary ddtSchemes;
            ddtSchemes.insert("ddt(phi)", std::string("BDF2"));
            fvSchemes.insert("ddtSchemes", ddtSchemes);

            // Build sparsity pattern and allocate linear system once - output goes to ls
            auto ls = la::createEmptyLinearSystem<TestType>(mesh);

            auto op = fvcc::DdtOperator(Operator::Type::Implicit, phi);
            op.read(fvSchemes);

            // Ensure oldTimeLevel >=2 to confirm true BDF2
            NF_ASSERT(
                oldTimeLevel(phi) >= 2,
                std::format("Expected oldTimeLevel(phi) >= 2 but got {}", oldTimeLevel(phi))
            );

            BENCHMARK(std::string(execName) + "_implicit_BDF2")
            {
                op.implicitOperation(ls, t, dt);
            };
        }
    }
}

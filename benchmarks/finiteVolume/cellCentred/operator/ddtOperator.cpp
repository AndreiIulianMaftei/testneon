// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "NeoN/NeoN.hpp"
#include "benchmarks/catch_main.hpp"
#include "test/catch2/executorGenerator.hpp"

using Operator = NeoN::dsl::Operator;

template<typename ValueType>
struct CreateVector
{
    std::string name;
    const NeoN::UnstructuredMesh& mesh;
    std::int64_t timeIndex = 0;
    std::int64_t iterationIndex = 0;
    std::int64_t subCycleIndex = 0;

    NeoN::Document operator()(NeoN::Database& db)
    {
        auto bcs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<ValueType>>(mesh);
        NeoN::Field<ValueType> domainVector(
            mesh.exec(),
            NeoN::Vector<ValueType>(mesh.exec(), mesh.nCells(), NeoN::one<ValueType>()),
            mesh.boundaryMesh().offset()
        );
        fvcc::VolumeField<ValueType> vf(mesh.exec(), name, mesh, domainVector, bcs, db, "", "");

        return NeoN::Document(
            {{"name", vf.name},
             {"timeIndex", timeIndex},
             {"iterationIndex", iterationIndex},
             {"subCycleIndex", subCycleIndex},
             {"field", vf}},
            fvcc::validateVectorDoc
        );
    }
};

TEST_CASE("DdtOperator::ddt", "[bench]")
{
    auto size = GENERATE(1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20);
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    NeoN::UnstructuredMesh mesh = NeoN::create1DUniformMesh(exec, size);

    NeoN::Database db;
    fvcc::VectorCollection& fieldCollection =
        fvcc::VectorCollection::instance(db, "benchVectorCollection");
    fvcc::VolumeField<NeoN::scalar>& phi =
        fieldCollection.registerVector<fvcc::VolumeField<NeoN::scalar>>(
            CreateVector<NeoN::scalar> {.name = "phi", .mesh = mesh, .timeIndex = 1}
        );

    fill(phi.internalVector(), 1.0);
    fill(phi.boundaryData().value(), 0.0);
    phi.correctBoundaryConditions();

    // Ensure old-time storage exists and is initialized
    NeoN::fill(oldTime(phi).internalVector(), 0.5);           // phi^n
    NeoN::fill(oldTime(oldTime(phi)).internalVector(), 0.25); // phi^{n-1}

    const NeoN::scalar t = 1.0;
    const NeoN::scalar dt = 0.5;

    // fvSchemes dictionaries for BDF1/BDF2


    // capture the value of size as section name
    DYNAMIC_SECTION("" << size)
    {
        SECTION("Explicit")
        {
            // Create a scalar field to hold the div value - output field
            NeoN::Vector<NeoN::scalar> source(exec, phi.size(), NeoN::zero<NeoN::scalar>());

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
            const auto& sp = la::SparsityPattern::readOrCreate(mesh);
            auto ls = la::createEmptyLinearSystem<NeoN::scalar, NeoN::localIdx>(mesh, sp);

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
            const auto& sp = la::SparsityPattern::readOrCreate(mesh);
            auto ls = la::createEmptyLinearSystem<NeoN::scalar, NeoN::localIdx>(mesh, sp);

            auto op = fvcc::DdtOperator(Operator::Type::Implicit, phi);
            op.read(fvSchemes);

            BENCHMARK(std::string(execName) + "_implicit_BDF2")
            {
                op.implicitOperation(ls, t, dt);
            };
        }
    }
}

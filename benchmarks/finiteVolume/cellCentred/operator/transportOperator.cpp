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

TEMPLATE_TEST_CASE("TransportOperator::transport", "[bench]", NeoN::scalar, NeoN::Vec3)
{
    auto size = GENERATE(1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20);
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    NeoN::UnstructuredMesh mesh = NeoN::create1DUniformMesh(exec, size);

    // -------------------------
    // Time setup (ddt)
    // -------------------------
    NeoN::Database db;
    fvcc::VectorCollection& fieldCollection =
        fvcc::VectorCollection::instance(db, "benchVectorCollection");
    fvcc::VolumeField<TestType>& phi = fieldCollection.registerVector<fvcc::VolumeField<TestType>>(
        CreateVector<TestType> {.name = "phi", .mesh = mesh, .timeIndex = 1}
    );

    NeoN::fill(phi.internalVector(), NeoN::one<TestType>());
    NeoN::fill(phi.boundaryData().value(), NeoN::zero<TestType>());
    phi.correctBoundaryConditions();

    NeoN::fill(oldTime(phi).internalVector(), NeoN::one<TestType>());          // phi^n
    NeoN::fill(oldTime(oldTime(phi)).internalVector(), NeoN::one<TestType>()); // phi^{n-1}

    const NeoN::scalar t = 1.0;
    const NeoN::scalar dt = 0.5;

    // Boundary fields
    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<TestType>>(mesh);
    auto coeffBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::scalar>>(mesh);
    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(mesh);

    // FaceFlux
    fvcc::SurfaceField<NeoN::scalar> faceFlux(exec, "faceFlux", mesh, surfaceBCs);
    NeoN::fill(faceFlux.internalVector(), 1.0);

    // Diffusion coefficient
    fvcc::SurfaceField<NeoN::scalar> gamma(exec, "gamma", mesh, surfaceBCs);
    NeoN::fill(gamma.internalVector(), 1.0);

    // Source coeff
    fvcc::VolumeField<NeoN::scalar> coeff(exec, "coeff", mesh, coeffBCs);
    NeoN::fill(coeff.internalVector(), 2.0);
    NeoN::fill(coeff.boundaryData().value(), 0.0);
    coeff.correctBoundaryConditions();

    DYNAMIC_SECTION("" << size)
    {
        NeoN::Input divInput = NeoN::TokenList({std::string("Gauss"), std::string("linear")});
        NeoN::Input lapInput = NeoN::TokenList(
            {std::string("Gauss"), std::string("linear"), std::string("uncorrected")}
        );

        SECTION("Explicit")
        {
            NeoN::Vector<TestType> rhs(exec, phi.size(), NeoN::zero<TestType>());

            // Explicit operators
            auto ddtOpExp = fvcc::DdtOperator(Operator::Type::Explicit, phi);
            auto divOpExp = fvcc::DivOperator(Operator::Type::Explicit, faceFlux, phi, divInput);
            auto lapOpExp =
                fvcc::LaplacianOperator<TestType>(Operator::Type::Explicit, gamma, phi, lapInput);
            auto srcOpExp = fvcc::SourceTerm<TestType>(Operator::Type::Explicit, coeff, phi);

            BENCHMARK(std::string(execName) + "_explicit_combined")
            {
                // rhs += ddt(phi) + div(faceFlux, phi) + laplacian(gamma, phi) + source(coeff, phi)
                ddtOpExp.explicitOperation(rhs, t, dt);
                divOpExp.explicitOperation(rhs);
                lapOpExp.explicitOperation(rhs);
                srcOpExp.explicitOperation(rhs);
            };
        }

        SECTION("Implicit_Euler")
        {
            NeoN::Dictionary fvSchemes;
            NeoN::Dictionary ddtSchemes;
            ddtSchemes.insert("ddt(phi)", std::string("BDF1"));
            fvSchemes.insert("ddtSchemes", ddtSchemes);

            // Build sparsity pattern and allocate linear system once - output goes to ls
            const auto& sp1 = la::SparsityPattern::readOrCreate(mesh);
            auto ls1 = la::createEmptyLinearSystem<TestType, NeoN::localIdx>(mesh, sp1);

            // Implicit operators - First order
            auto ddtOpImp1 = fvcc::DdtOperator(Operator::Type::Implicit, phi);
            ddtOpImp1.read(fvSchemes);

            auto divOpImp1 = fvcc::DivOperator(Operator::Type::Implicit, faceFlux, phi, divInput);
            auto lapOpImp1 =
                fvcc::LaplacianOperator<TestType>(Operator::Type::Implicit, gamma, phi, lapInput);
            auto srcOpImp1 = fvcc::SourceTerm<TestType>(Operator::Type::Implicit, coeff, phi);

            BENCHMARK(std::string(execName) + "_implicit_Euler")
            {
                ddtOpImp1.implicitOperation(ls1, t, dt);
                divOpImp1.implicitOperation(ls1);
                lapOpImp1.implicitOperation(ls1);
                srcOpImp1.implicitOperation(ls1);
            };
        }

        SECTION("Implicit_BDF2")
        {
            // Select implicit ddt scheme (example: BDF2).
            NeoN::Dictionary fvSchemes;
            NeoN::Dictionary ddtSchemes;
            ddtSchemes.insert("ddt(phi)", std::string("BDF2"));
            fvSchemes.insert("ddtSchemes", ddtSchemes);

            // Build sparsity pattern and allocate linear system once - output goes to ls
            const auto& sp2 = la::SparsityPattern::readOrCreate(mesh);
            auto ls2 = la::createEmptyLinearSystem<TestType, NeoN::localIdx>(mesh, sp2);

            // Implicit operators - Second order
            auto ddtOpImp2 = fvcc::DdtOperator(Operator::Type::Implicit, phi);
            ddtOpImp2.read(fvSchemes);

            auto divOpImp2 = fvcc::DivOperator(Operator::Type::Implicit, faceFlux, phi, divInput);
            auto lapOpImp2 =
                fvcc::LaplacianOperator<TestType>(Operator::Type::Implicit, gamma, phi, lapInput);
            auto srcOpImp2 = fvcc::SourceTerm<TestType>(Operator::Type::Implicit, coeff, phi);

            BENCHMARK(std::string(execName) + "_implicit_BDF2")
            {
                ddtOpImp2.implicitOperation(ls2, t, dt);
                divOpImp2.implicitOperation(ls2);
                lapOpImp2.implicitOperation(ls2);
                srcOpImp2.implicitOperation(ls2);
            };
        }
    }
}

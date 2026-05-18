// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "NeoN/NeoN.hpp"
#include "benchmarks/catch_main.hpp"
#include "test/catch2/executorGenerator.hpp"
#include "benchmarks/finiteVolume/cellCentred/common.hpp"

#include "NeoN/dsl/explicit.hpp"
#include "NeoN/dsl/implicit.hpp"
#include "NeoN/dsl/expression.hpp"

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
        NeoN::Dictionary fvSchemes;
        NeoN::Dictionary divSchemes;
        divSchemes.insert(
            "div(faceFlux,phi)", NeoN::TokenList({std::string("Gauss"), std::string("linear")})
        );
        fvSchemes.insert("divSchemes", divSchemes);

        NeoN::Dictionary lapSchemes;
        lapSchemes.insert(
            "laplacian(gamma,phi)",
            NeoN::TokenList(
                {std::string("Gauss"), std::string("linear"), std::string("uncorrected")}
            )
        );
        fvSchemes.insert("laplacianSchemes", lapSchemes);

        SECTION("Explicit")
        {
            NeoN::Vector<TestType> rhs(exec, phi.size(), NeoN::zero<TestType>());

            // Build Explicit Expression
            auto expr = NeoN::dsl::exp::ddt(phi) + NeoN::dsl::exp::div(faceFlux, phi)
                      + NeoN::dsl::exp::laplacian(gamma, phi) + NeoN::dsl::exp::source(coeff, phi);

            expr.read(fvSchemes);

            BENCHMARK(std::string(execName) + "_explicit_combined")
            {
                // rhs += ddt(phi)
                expr.explicitOperation(rhs, t, dt);

                // rhs += div(faceFlux, phi) + laplacian(gamma, phi) + source(coeff, phi)
                expr.explicitOperation(rhs);
            };
        }

        SECTION("Implicit_Euler")
        {
            NeoN::Dictionary ddtSchemes;
            ddtSchemes.insert("ddt(phi)", std::string("BDF1"));
            fvSchemes.insert("ddtSchemes", ddtSchemes);

            // Build sparsity pattern and allocate linear system once - output goes to ls
            auto ls1 = la::createEmptyLinearSystem<TestType>(mesh);

            // Build Implicit Expression - First order
            auto expr = NeoN::dsl::imp::ddt(phi) + NeoN::dsl::imp::div(faceFlux, phi)
                      + NeoN::dsl::imp::laplacian(gamma, phi) + NeoN::dsl::imp::source(coeff, phi);

            expr.read(fvSchemes);

            BENCHMARK(std::string(execName) + "_implicit_Euler") { expr.assemble(t, dt, ls1); };
        }

        SECTION("Implicit_BDF2")
        {
            // Select implicit ddt scheme (example: BDF2).
            NeoN::Dictionary ddtSchemes;
            ddtSchemes.insert("ddt(phi)", std::string("BDF2"));
            fvSchemes.insert("ddtSchemes", ddtSchemes);

            // Build sparsity pattern and allocate linear system once - output goes to ls
            auto ls2 = la::createEmptyLinearSystem<TestType>(mesh);

            // Build Implicit Expression - Second order
            auto expr = NeoN::dsl::imp::ddt(phi) + NeoN::dsl::imp::div(faceFlux, phi)
                      + NeoN::dsl::imp::laplacian(gamma, phi) + NeoN::dsl::imp::source(coeff, phi);

            expr.read(fvSchemes);

            BENCHMARK(std::string(execName) + "_implicit_BDF2") { expr.assemble(t, dt, ls2); };
        }
    }
}

// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
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

TEMPLATE_TEST_CASE("laplacianOperator fixedValue", "[template]", scalar, Vec3)
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    const NeoN::localIdx nCells = 10;
    auto mesh = create1DUniformMesh(exec, nCells);

    // Define diffusion coefficient field gamma
    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);
    fvcc::SurfaceField<scalar> gamma(exec, "gamma", mesh, surfaceBCs);
    fill(gamma.internalVector(), 2.0);

    auto [boundaryType, firstValue, lastValue] = GENERATE(
        std::tuple<std::string, scalar, scalar> {"fixedValue", 0.5, 10.5} // Dirichlet BCs
        // std::tuple<std::string, scalar, scalar> {"fixedGradient", -10.0, 10} // Neumann BCs
    );

    SECTION(boundaryType)
    {
        std::vector<fvcc::VolumeBoundary<TestType>> vbcs;
        // vbcs.push_back(fvcc::VolumeBoundary<TestType>(
        //     mesh,
        //     Dictionary(
        //         {{"type", std::string(boundaryType)}, {boundaryType, firstValue *
        //         one<TestType>()}}
        //     ),
        //     0
        // ));
        // vbcs.push_back(fvcc::VolumeBoundary<TestType>(
        //     mesh,
        //     Dictionary(
        //         {{"type", std::string(boundaryType)}, {boundaryType, lastValue *
        //         one<TestType>()}}
        //     ),
        //     1
        // ));
        for (NeoN::localIdx patchi = 0; patchi < mesh.nBoundaries(); ++patchi)
        {
            Dictionary dict;
            if (patchi == 0)
            {
                dict.insert("type", std::string("fixedValue"));
                dict.insert("fixedValue", 0.5 * one<TestType>());
                vbcs.emplace_back(mesh, dict, patchi);
            }
            else if (patchi == 1)
            {
                dict.insert("type", std::string("fixedValue"));
                dict.insert("fixedValue", 10.5 * one<TestType>());
                vbcs.emplace_back(mesh, dict, patchi);
            }
            else
            {
                dict.insert("type", std::string("empty"));
            }
        }

        // Define the field phi: [1, 2, ..., nCells]
        auto phi = fvcc::VolumeField<TestType>(exec, "phi", mesh, vbcs);
        parallelFor(
            phi.internalVector(),
            NEON_LAMBDA(const localIdx i) { return scalar(i + 1) * one<TestType>(); }
        );
        phi.correctBoundaryConditions();

        // Define the Laplacian scheme
        Input input =
            TokenList({std::string("Gauss"), std::string("linear"), std::string("uncorrected")});

        SECTION("explicit laplacian operator for constant field on " + execName)
        {
            dsl::SpatialOperator lapOp = dsl::exp::laplacian(gamma, phi);
            lapOp.read(input);
            Vector<TestType> source(exec, nCells, zero<TestType>());
            lapOp.explicitOperation(source);
            auto sourceHost = source.copyToHost();
            auto sourceV = sourceHost.view();
            std::cout << "Executor: " << execName << std::endl;
            for (NeoN::localIdx i = 0; i < nCells; i++)
            {
                // the laplacian of a linear function is 0
                std::cout << "sourceV[" << i << "] = " << sourceV[i] << std::endl;
                // REQUIRE(mag(sourceV[i]) == Catch::Approx(0.0).margin(1e-8));
            }
        }

        SECTION("Construct from Token" + execName)
        {
            fvcc::LaplacianOperator<TestType> lapOp(
                dsl::Operator::Type::Implicit, gamma, phi, input
            );
        }

        // auto ls = NeoN::la::createEmptyLinearSystem<TestType>(mesh);

        // SECTION("implicit laplacian operator of constant field on " + execName)
        // {
        //     dsl::SpatialOperator lapOp = dsl::imp::laplacian(gamma, phi);
        //     lapOp.read(input);
        //     // currently only defined for scalar types
        //     if constexpr (std::is_same_v<TestType, scalar>)
        //     {
        //         lapOp.implicitOperation(ls);
        //         auto res = Vector<scalar>(phi.internalVector());
        //         fill(res, 1.0);

        //         computeResidual(ls.matrix(), ls.rhs(), phi.internalVector(), res);

        //         auto resHost = res.copyToHost();
        //         auto resV = resHost.view();
        //         for (localIdx celli = 0; celli < resV.size(); celli++)
        //         {
        //             // the laplacian of a linear function is 0
        //             REQUIRE(resV[celli] == Catch::Approx(0.0).margin(1e-8));
        //         }
        //     }
        // }

        // SECTION("implicit laplacian operator scale" + execName)
        // {
        //     if constexpr (std::is_same_v<TestType, scalar>)
        //     {
        //         ls.reset();
        //         dsl::SpatialOperator lapOp = dsl::imp::laplacian(gamma, phi);
        //         lapOp.read(input);
        //         lapOp = dsl::Coeff(-0.5) * lapOp;

        //         lapOp.implicitOperation(ls);

        //         auto res = Vector<scalar>(phi.internalVector());
        //         computeResidual(ls.matrix(), ls.rhs(), phi.internalVector(), res);

        //         auto resHost = res.copyToHost();
        //         auto resV = resHost.view();
        //         for (localIdx celli = 0; celli < resV.size(); celli++)
        //         {
        //             // the laplacian of a linear function is 0
        //             REQUIRE(resV[celli] == Catch::Approx(0.0).margin(1e-8));
        //         }
        //     }
        // }
    }
}

} // namespace NeoN

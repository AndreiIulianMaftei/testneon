// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/fields/field.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/divOperator.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"
#include "NeoN/linearAlgebra/meshIterationStrategies.hpp"

namespace NeoN::finiteVolume::cellCentred
{

/* @brief
 *
 */
template<typename ValueType>
class GaussGreenDiv :
    public DivOperatorFactory<ValueType>::template Register<GaussGreenDiv<ValueType>>
{
    using Base = DivOperatorFactory<ValueType>::template Register<GaussGreenDiv<ValueType>>;

public:

    static std::string name() { return "Gauss"; }

    static std::string doc() { return "Gauss-Green Divergence"; }

    static std::string schema() { return "none"; }

    GaussGreenDiv(const Executor& exec, const UnstructuredMesh& mesh, const Input& inputs)
        : Base(exec, mesh), surfaceInterpolation_(exec, mesh, inputs) {};

    virtual VolumeField<ValueType>
    div(const SurfaceField<scalar>& faceFlux,
        const VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling) const override;

    virtual void
    div(VolumeField<ValueType>& divPhi,
        const SurfaceField<scalar>& faceFlux,
        const VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling) const override;

    virtual void
    div(Vector<ValueType>& divPhi,
        const SurfaceField<scalar>& faceFlux,
        const VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling) const override;

    virtual void
    div(la::LinearSystem<ValueType>& ls,
        const SurfaceField<scalar>& faceFlux,
        const VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling) const override;

    std::unique_ptr<DivOperatorFactory<ValueType>> clone() const override
    {
        return std::make_unique<GaussGreenDiv<ValueType>>(*this);
    }

private:

    SurfaceInterpolation<ValueType> surfaceInterpolation_;
};

// Required on MSVC: without extern template, each TU (DLL and EXE) gets its own
// instantiation of table() static local, so the DLL's addSubType() inserts into
// a different map than the one the test binary queries.
extern template class GaussGreenDiv<scalar>;
extern template class GaussGreenDiv<Vec3>;

} // namespace NeoN

// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/dsl/spatialOperator.hpp"
#include "NeoN/dsl/temporalOperator.hpp"
#include "NeoN/dsl/ddt.hpp"

// TODO: decouple from fvcc
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/ddtOperator.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/divOperator.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/laplacianOperator.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/sourceTerm.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

namespace NeoN::dsl::exp
{

template<typename ValueType>
TemporalOperator<ValueType> ddt(fvcc::VolumeField<ValueType>& phi)
{
    return fvcc::DdtOperator(dsl::Operator::Type::Explicit, phi);
}

template<typename ValueType>
SpatialOperator<ValueType>
source(fvcc::VolumeField<scalar>& coeff, fvcc::VolumeField<ValueType>& phi)
{
    return SpatialOperator<ValueType>(fvcc::SourceTerm(dsl::Operator::Type::Explicit, coeff, phi));
}

template<typename ValueType>
SpatialOperator<ValueType>
div(fvcc::SurfaceField<scalar>& faceFlux, fvcc::VolumeField<ValueType>& phi)
{
    return SpatialOperator<ValueType>(
        fvcc::DivOperator(dsl::Operator::Type::Explicit, faceFlux, phi)
    );
}

template<typename ValueType>
SpatialOperator<ValueType>
laplacian(fvcc::SurfaceField<scalar>& gamma, fvcc::VolumeField<ValueType>& phi)
{
    return SpatialOperator<ValueType>(
        fvcc::LaplacianOperator<ValueType>(dsl::Operator::Type::Explicit, gamma, phi)
    );
}

} // namespace NeoN

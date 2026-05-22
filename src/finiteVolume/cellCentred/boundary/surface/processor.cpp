// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/surface/processor.hpp"

namespace NeoN::finiteVolume::cellCentred::surfaceBoundary::detail
{

template<typename ValueType>
void setProcBoundaryValue(
    Field<ValueType>& domainVector,
    const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range
)
{
    const auto iVector = domainVector.internalVector().view();
    const auto nInternalFaces = static_cast<localIdx>(mesh.nInternalFaces());

    auto [refGradient, value, valueFraction, refValue] = views(
        domainVector.boundaryData().refGrad(),
        domainVector.boundaryData().value(),
        domainVector.boundaryData().valueFraction(),
        domainVector.boundaryData().refValue()
    );

    NeoN::parallelFor(
        domainVector.exec(),
        range,
        NEON_LAMBDA(const localIdx i) {
            refGradient[i] = zero<ValueType>();
            value[i] = iVector[nInternalFaces + i];
            valueFraction[i] = 0.0;
            refValue[i] = zero<ValueType>();
        },
        "setProcBoundaryValue"
    );
}

template void setProcBoundaryValue<
    scalar>(Field<scalar>&, const UnstructuredMesh&, std::pair<localIdx, localIdx>);
template void
setProcBoundaryValue<Vec3>(Field<Vec3>&, const UnstructuredMesh&, std::pair<localIdx, localIdx>);

} // namespace NeoN::finiteVolume::cellCentred::surfaceBoundary::detail

namespace NeoN::finiteVolume::cellCentred::surfaceBoundary
{
template class Processor<scalar>;
template class Processor<Vec3>;
}

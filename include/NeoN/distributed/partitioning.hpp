// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/primitives/label.hpp"

#ifdef NF_WITH_MPI_SUPPORT

namespace NeoN
{

namespace detail
{

/** @brief Set processor boundary types on a copy of the BC vector for a distributed mesh part. */
template<typename BoundaryType>
auto setProcessorBoundaryHelper(
    UnstructuredMesh& mesh, const std::vector<BoundaryType>& bcs, NeoN::mpi::Environment mpiEnviron
)
{
    const auto rank = mpiEnviron.rank();
    const auto ranks = mpiEnviron.sizeRank();
    const bool isMiddle = (rank > 0 && rank < ranks - 1);
    auto ret = std::vector<BoundaryType> {};
    for (int i = 0; i < static_cast<int>(bcs.size()); i++)
    {
        // boundary index 1 is always the proc boundary for first/last ranks;
        // middle ranks have all boundaries as proc.
        const bool isProc = (i == 1) || isMiddle;
        std::string boundaryType = isProc ? "processor" : "calculated";
        auto patchDict = Dictionary({{"type", boundaryType}});
        ret.emplace_back(mesh, patchDict, i);
    }
    return ret;
}

/** @brief Extract the rank-local slice from a global internal vector for a 1D uniform partition.
 *
 *  localSize = globalVector.size() / nRanks  (integer division; works for both cells and faces
 *              because proc-boundary faces between ranks are excluded from the global internal
 * count) firstIdx  = rank * localMesh.nCells()
 */
template<typename ValueType>
Vector<ValueType> partitionInternalVector(
    const Vector<ValueType>& globalVector,
    const UnstructuredMesh& localMesh,
    NeoN::mpi::Environment mpiEnviron
)
{
    const localIdx localSize = static_cast<localIdx>(globalVector.size()) / mpiEnviron.sizeRank();
    const localIdx firstIdx = mpiEnviron.rank() * localMesh.nCells();
    return take(globalVector, {firstIdx, firstIdx + localSize});
}

/** @brief Partition a 1D field into the rank-local slice and replace boundary conditions. */
template<typename FieldType>
FieldType
oneDPartitionField(FieldType field, UnstructuredMesh& mesh, NeoN::mpi::Environment mpiEnviron)
{
    auto internalVector = partitionInternalVector(field.internalVector(), mesh, mpiEnviron);
    auto bcsPart = setProcessorBoundaryHelper(mesh, field.boundaryConditions(), mpiEnviron);
    FieldType ret(field.exec(), field.name + "Part", mesh, bcsPart);
    ret.internalVector() = internalVector;
    return ret;
}

} // namespace detail

} // namespace NeoN

#endif

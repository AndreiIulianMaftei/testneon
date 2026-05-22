// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/primitives/label.hpp"

#ifdef NF_WITH_MPI_SUPPORT

namespace NeoN
{

/** @brief Returns the rank's slice of a VolumeField from a global 1D mesh.
 *
 *  Assumes 3 MPI ranks. Rank r owns cells [r*localCells, (r+1)*localCells).
 */
template<typename FieldType, typename MeshType, typename BcType>
FieldType partitionVolField(
    FieldType field, const MeshType& mesh, BcType bcs, NeoN::mpi::Environment mpiEnviron
)
{
    localIdx localCells = mesh.nCells();
    localIdx firstCell = 0;
    localIdx lastCell = localCells;

    if (mpiEnviron.rank() == 0)
    {
        lastCell = localCells;
    }
    if (mpiEnviron.rank() >= 1 && mpiEnviron.rank() != mpiEnviron.sizeRank() - 1)
    {
        firstCell = localCells;
        lastCell = localCells + localCells;
    }

    if (mpiEnviron.rank() == mpiEnviron.sizeRank() - 1)
    {
        firstCell = localCells + localCells;
        lastCell = localCells + localCells + localCells;
    }

    auto internalVector = take(field.internalVector(), {firstCell, lastCell});

    return {field.exec(), field.name + "Part", mesh, internalVector, bcs};
}

/** @brief Returns the rank's slice of a SurfaceField from a global 1D mesh.
 *
 *  Handles the two extra proc-boundary faces appended to each partition.
 *  Only works for exactly 3 MPI ranks.
 */
template<typename FieldType, typename MeshType, typename BcType>
FieldType partitionSurfaceField(
    FieldType field,
    MeshType& mesh,
    BcType bcs,
    NeoN::mpi::Environment mpiEnviron,
    bool flip = false
)
{
    auto exec = mesh.exec();
    localIdx localCells = mesh.nCells();         // 4
    localIdx localFaces = mesh.nInternalFaces(); // 3
    localIdx firstFace = 0;
    localIdx lastFace = localFaces;

    localIdx firstBoundaryFace = 0;
    localIdx secondBoundaryFace = 0;

    scalar signLeft = 1.0;
    scalar signRight = 1.0;

    // FIXME this only works for 3 ranks
    if (mpiEnviron.rank() == 0)
    {
        firstBoundaryFace = 3 * localFaces + 2;
        secondBoundaryFace = localFaces;

        if (flip)
        {
            signRight = -1.0;
        }
    }
    if (mpiEnviron.rank() == 1)
    {
        firstFace = localFaces + 1;

        firstBoundaryFace = localFaces;
        secondBoundaryFace = firstBoundaryFace + localCells;

        if (flip)
        {
            signLeft = -1.0;
        }
    }
    if (mpiEnviron.rank() == 2)
    {
        firstFace = localCells + localCells;

        firstBoundaryFace = 3 * localFaces + 3;
        secondBoundaryFace = 2 * localFaces + 1;

        if (flip)
        {
            signRight = -1.0;
        }
    }

    lastFace = firstFace + localFaces + 2;

    FieldType ret = {field.exec(), field.name + "Part", mesh, bcs};

    NF_ASSERT(lastFace - firstFace == mesh.nTotalFaces(), "different size");

    auto internalVector = take(field.internalVector(), {firstFace, lastFace});
    auto outV = internalVector.view();
    auto inV = field.internalVector().view();

    NeoN::parallelFor(
        exec,
        {0, 1},
        NEON_LAMBDA(const localIdx i) { outV[localFaces] = signLeft * inV[firstBoundaryFace]; },
        "copyMap"
    );

    NeoN::parallelFor(
        exec,
        {0, 1},
        NEON_LAMBDA(const localIdx i) {
            outV[localFaces + 1] = signRight * inV[secondBoundaryFace];
        },
        "copyMap"
    );

    ret.internalVector() = internalVector;
    return ret;
}

} // namespace NeoN

#endif // NF_WITH_MPI_SUPPORT

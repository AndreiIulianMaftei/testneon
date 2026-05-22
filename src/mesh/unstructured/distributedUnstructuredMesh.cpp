// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <mpi.h>

#include "NeoN/core/mpi/environment.hpp"
#include "NeoN/core/vector/vectorFreeFunctions.hpp"
#include "NeoN/distributed/communicationPattern.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN
{

/** @brief Creates a 1D uniform mesh partition for the current MPI rank.
 *
 *  The global domain spans [0, 1].  Each rank owns a contiguous slab of
 *  nCells cells.  Regular (physical) boundary patches appear on the domain
 *  boundary; processor patches are created where partitions share a face.
 */
UnstructuredMesh create1DUniformMeshPart(const Executor exec, const localIdx nCells)
{
    mpi::Environment mpiEnviron;
    scalar rightBoundary {static_cast<scalar>(1.0) / mpiEnviron.sizeRank()};

    const bool isFirst = mpiEnviron.rank() == 0;
    const bool isLast = mpiEnviron.rank() == mpiEnviron.sizeRank() - 1;
    const localIdx nProcBoundaryFaces = (isFirst || isLast) ? 1 : 2;
    const localIdx nRegularBoundary = static_cast<localIdx>((isFirst || isLast) ? 1 : 0);

    // boundary face owners and weights (regular patches first, proc patches after)
    auto faceCellVec = std::vector<localIdx>();
    auto boundaryWeights = std::vector<scalar>();
    auto neighRanks = std::vector<localIdx>();

    if (!isFirst && !isLast)
    {
        // middle rank: only proc patches (no regular boundary)
        faceCellVec = {0, nCells - 1};
        boundaryWeights = {-1.0, 1.0};
        neighRanks = {
            static_cast<localIdx>(mpiEnviron.rank() - 1),
            static_cast<localIdx>(mpiEnviron.rank() + 1)
        };
    }
    else if (isFirst)
    {
        // rank 0: xmin regular then proc right
        faceCellVec = {0, nCells - 1};
        boundaryWeights = {-1.0, 1.0};
        neighRanks = {static_cast<localIdx>(1)};
    }
    else
    {
        // last rank: xmax regular first then proc left
        faceCellVec = {nCells - 1, 0};
        boundaryWeights = {1.0, -1.0};
        neighRanks = {static_cast<localIdx>(mpiEnviron.rank() - 1)};
    }

    labelVector faceCells(exec, faceCellVec);

    // Face center and normal for each boundary slot (2 slots regardless of rank).
    // For the last rank the boundary face ordering is [xmax, proc-left] — the
    // face data vectors are in that order too (swapped relative to rank-0/mid).
    std::vector<Vec3> bcCfVec {{0.0, 0.0, 0.0}, {rightBoundary, 0.0, 0.0}};
    std::vector<Vec3> bcSfVec {{-1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    std::vector<Vec3> bcNfVec {{-1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    if (isLast)
    {
        std::swap(bcCfVec[0], bcCfVec[1]);
        std::swap(bcSfVec[0], bcSfVec[1]);
        std::swap(bcNfVec[0], bcNfVec[1]);
    }

    auto tmp = create1DUniformMesh(exec, nCells, rightBoundary);

    BoundaryMesh boundaryMesh(
        exec,
        faceCells,
        {exec, bcCfVec},                       // cf
        tmp.boundaryMesh().ownerCellCenters(), // cn
        {exec, bcSfVec},                       // sf
        {exec, {1.0, 1.0}},                    // magSf
        {exec, bcNfVec},                       // nf
        tmp.boundaryMesh().delta(),            // delta
        {exec, boundaryWeights},               // weights
        tmp.boundaryMesh().deltaCoeffs(),      // deltaCoeffs
        {0, 1, 2},                             // offset: 2 patches
        nProcBoundaryFaces,
        neighRanks
    );

    // For the last rank, the tmp boundary faces need to be physically swapped so
    // that regular (xmax) comes before the proc face in the internal face arrays.
    auto faceCentresH = tmp.faceCenters().copyToHost();
    auto faceNormalsH = tmp.faceNormals().copyToHost();
    if (isLast)
    {
        const auto localCells = tmp.nCells();
        std::swap(faceCentresH.view()[localCells - 1], faceCentresH.view()[localCells]);
        faceNormalsH.view()[localCells - 1] = {1.0, 0.0, 0.0}; // xmax
        faceNormalsH.view()[localCells] = {-1.0, 0.0, 0.0};    // proc left
    }

    const localIdx nFacesPart = tmp.nInternalFaces() + nRegularBoundary;

    auto faceCentresPart = take(faceCentresH.copyToExecutor(exec), {0, nFacesPart});
    auto faceNormalsPart = take(faceNormalsH.copyToExecutor(exec), {0, nFacesPart});
    auto faceAreasPart = take(tmp.faceAreas(), {0, nFacesPart});
    auto faceOwnersPart = take(tmp.faceOwners(), {0, nFacesPart});

    UnstructuredMesh mesh(
        exec,
        tmp.points(),
        tmp.cellVolumes(),
        tmp.cellCenters(),
        faceNormalsPart,
        faceCentresPart,
        faceAreasPart,
        faceOwnersPart,
        tmp.faceNeighbors(),
        std::move(boundaryMesh)
    );

    return mesh;
}

CommunicationPattern computeCommunicationPattern(const UnstructuredMesh& mesh)
{
    mpi::Environment mpiEnviron;
    if (!mesh.boundaryMesh().isDistributed())
    {
        return {};
    }

    auto sendCounts = std::vector<int>(static_cast<std::size_t>(mpiEnviron.sizeRank() + 1), 0);

    const auto& neighbourRanks = mesh.boundaryMesh().neighbourRank();
    const auto& offsets = mesh.boundaryMesh().offset();
    const auto nInnerBoundaries =
        mesh.boundaryMesh().nBoundaries() - mesh.boundaryMesh().nProcBoundaryPatches();

    for (int i = 0; i < static_cast<int>(neighbourRanks.size()); i++)
    {
        const auto targetRank = static_cast<int>(neighbourRanks[static_cast<std::size_t>(i)]);
        const auto patchSize = static_cast<int>(
            offsets[static_cast<std::size_t>(nInnerBoundaries + i + 1)]
            - offsets[static_cast<std::size_t>(nInnerBoundaries + i)]
        );
        sendCounts[static_cast<std::size_t>(targetRank)] = patchSize;
        sendCounts[static_cast<std::size_t>(mpiEnviron.sizeRank())] += patchSize;
    }

    const auto globalOffset = mesh.globalOffset();
    const auto faceCellsH = mesh.boundaryMesh().faceOwners().copyToHost();

    auto buffer = std::vector<localIdx>();
    buffer.reserve(static_cast<std::size_t>(mesh.boundaryMesh().nProcBoundaryFaces()));
    const auto procStart = offsets[static_cast<std::size_t>(nInnerBoundaries)];
    for (int i = 0; i < mesh.boundaryMesh().nProcBoundaryFaces(); i++)
    {
        buffer.push_back(faceCellsH.view()[i + procStart] + globalOffset);
    }

    auto sdispl = std::vector<int>(static_cast<std::size_t>(mpiEnviron.sizeRank()), 0);
    for (int i = 0; i < static_cast<int>(neighbourRanks.size()); i++)
    {
        const auto targetRank = static_cast<int>(neighbourRanks[static_cast<std::size_t>(i)]);
        sdispl[static_cast<std::size_t>(targetRank)] =
            static_cast<int>(offsets[static_cast<std::size_t>(nInnerBoundaries + i)] - procStart);
    }

    auto recvCounts = std::vector<int>(static_cast<std::size_t>(mpiEnviron.sizeRank()), 0);
    MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT, mpiEnviron.comm());

    auto rdispl = std::vector<int>(static_cast<std::size_t>(mpiEnviron.sizeRank()), 0);
    for (int r = 1; r < mpiEnviron.sizeRank(); ++r)
    {
        rdispl[static_cast<std::size_t>(r)] =
            rdispl[static_cast<std::size_t>(r - 1)] + recvCounts[static_cast<std::size_t>(r - 1)];
    }
    int totalRecv = rdispl.back() + recvCounts.back();

    auto recvIdx = std::vector<int>(static_cast<std::size_t>(totalRecv));

    std::vector<int> sendBuffer(buffer.size());
    for (std::size_t i = 0; i < buffer.size(); ++i)
        sendBuffer[i] = static_cast<int>(buffer[i]);

    MPI_Alltoallv(
        sendBuffer.data(),
        sendCounts.data(),
        sdispl.data(),
        MPI_INT,
        recvIdx.data(),
        recvCounts.data(),
        rdispl.data(),
        MPI_INT,
        mpiEnviron.comm()
    );

    std::vector<localIdx> boundaryMapVector;
    return CommunicationPattern {sendCounts, recvIdx, boundaryMapVector, mpiEnviron};
}

} // namespace NeoN

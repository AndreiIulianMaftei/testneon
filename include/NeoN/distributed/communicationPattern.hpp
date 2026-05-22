// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>

#include "NeoN/core/primitives/label.hpp"

#ifdef NF_WITH_MPI_SUPPORT
#include "NeoN/core/mpi/environment.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN
{

/** @struct CommunicationPattern
 *  @brief Collects all data required for distributed halo exchange.
 */
struct CommunicationPattern
{
    std::vector<int> sendCounts;

    std::vector<int> recvIdx;

    std::vector<localIdx> boundaryMapVector;

    mpi::Environment env;
};

/** @brief Computes the CommunicationPattern for a distributed mesh partition.
 *
 *  Performs an MPI_Alltoallv exchange so each rank knows the global cell
 *  indices it will receive from its neighbours.
 *
 *  Returns an empty pattern for non-distributed meshes.
 */
CommunicationPattern computeCommunicationPattern(const UnstructuredMesh& mesh);

} // namespace NeoN

#endif // NF_WITH_MPI_SUPPORT

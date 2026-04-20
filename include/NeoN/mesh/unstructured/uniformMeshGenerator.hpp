// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/segmentedVector.hpp"

#include <vector>

namespace NeoN::detail
{

struct MeshParams
{
    localIdx nx, ny, nz;
    scalar Lx, Ly, Lz;

    // Derived spacings
    scalar dx() const { return Lx / static_cast<scalar>(nx); }
    scalar dy() const { return Ly / static_cast<scalar>(ny); }
    scalar dz() const { return Lz / static_cast<scalar>(nz); }

    // Layout of cells in memory: go through x first, then y, then z.
    // So cell indices in the first layer are ordered as:
    // (0,0,0), (1,0,0), ..., (nx-1,0,0),
    // (0,1,0), (1,1,0), ..., (nx-1,1,0),
    // ...,
    // (0,ny-1,0), (1,ny-1,0), ..., (nx-1,ny-1,0)
    localIdx cellIdx(localIdx i, localIdx j, localIdx k) const { return k * nx * ny + j * nx + i; }

    localIdx ptIdx(localIdx i, localIdx j, localIdx k) const
    {
        return k * (nx + 1) * (ny + 1) + j * (nx + 1) + i;
    }
};

struct CellData
{
    std::vector<scalar> volumes;
    std::vector<Vec3> centres;
};

struct FaceData
{
    std::vector<Vec3> areas;
    std::vector<Vec3> centres;
    std::vector<scalar> magnitudes;
    std::vector<label> owner;
    std::vector<label> neighbour;
    // localIdx nInternal;
};

struct BoundaryData
{
    BoundaryMesh mesh;
    // localIdx nBoundary;
};

std::vector<Vec3> generatePoints(const MeshParams& p);

CellData generateCellData(const MeshParams& p);

FaceData
generateInternalFaces(const MeshParams& p, const localIdx nInternal, const localIdx nFaces);

BoundaryData generateBoundaryData(
    const Executor exec,
    const int dim,
    const MeshParams& p,
    const std::vector<Vec3>& centres,
    FaceData& faces,
    const localIdx nInternal,
    const localIdx nBoundary
);

std::vector<std::vector<localIdx>> buildFaceNodes(const MeshParams& p, localIdx nFaces);

void storeFaceNodesInStencilDB(
    UnstructuredMesh& mesh, const std::vector<std::vector<localIdx>>& faceNodesVec
);

} // namespace NeoN::detail

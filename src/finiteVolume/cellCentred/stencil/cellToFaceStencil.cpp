// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/finiteVolume/cellCentred/stencil/cellToFaceStencil.hpp"

namespace NeoN::finiteVolume::cellCentred
{

CellToFaceStencil::CellToFaceStencil(const UnstructuredMesh& mesh) : mesh_(mesh) {}

SegmentedVector<localIdx, localIdx> CellToFaceStencil::computeInternalStencil() const
{
    const auto exec = mesh_.exec();
    const auto nCells = mesh_.nCells();
    const auto [faceOwners, faceNeighbors] =
        views(mesh_.faceOwners(), mesh_.faceNeighbors());

    const auto nInternalFaces = mesh_.nInternalFaces();

    Vector<localIdx> nFacesPerCell(exec, nCells, 0);
    View<localIdx> nFacesPerCellView = nFacesPerCell.view();

    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx i) {
            Kokkos::atomic_inc(&nFacesPerCellView[faceOwners[i]]);
            Kokkos::atomic_inc(&nFacesPerCellView[faceNeighbors[i]]);
        },
        "countFacesPerCellInternal"
    );

    SegmentedVector<localIdx, localIdx> stencil(nFacesPerCell); // guessed
    auto [stencilValues, segment] = stencil.views();

    fill(nFacesPerCell, 0); // reset nFacesPerCell

    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            localIdx owner = faceOwners[facei];
            localIdx neighbour = faceNeighbors[facei];

            const auto segIdxOwn = Kokkos::atomic_fetch_inc(&nFacesPerCellView[owner]);
            const auto segIdxNei = Kokkos::atomic_fetch_inc(&nFacesPerCellView[neighbour]);

            auto segOwn = segment[owner] + segIdxOwn;
            auto segNei = segment[neighbour] + segIdxNei;

            stencilValues[segOwn] = facei;
            stencilValues[segNei] = facei;
        },
        "computeStencilInternal"
    );

    return stencil;
}

} // namespace NeoN::finiteVolume::cellCentred

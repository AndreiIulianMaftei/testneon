// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/finiteVolume/cellCentred/stencil/cellToFaceStencil.hpp"

namespace NeoN::finiteVolume::cellCentred
{

CellToFaceStencil::CellToFaceStencil(const UnstructuredMesh& mesh) : mesh_(mesh) {}

SegmentedVector<localIdx, localIdx> CellToFaceStencil::computeStencil() const
{
    const auto exec = mesh_.exec();
    const auto nCells = mesh_.nCells();
    const auto [faceOwners, faceNeighbors, boundaryFaceOwners] =
        views(mesh_.faceOwners(), mesh_.faceNeighbors(), mesh_.boundaryMesh().faceOwners());

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

    parallelFor(
        exec,
        {0, boundaryFaceOwners.size()},
        NEON_LAMBDA(const localIdx i) {
            Kokkos::atomic_inc(&nFacesPerCellView[boundaryFaceOwners[i]]);
        },
        "countFacesPerCellBoundary"
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

            localIdx segIdxOwn = Kokkos::atomic_fetch_add(&nFacesPerCellView[owner], 1);
            localIdx segIdxNei = Kokkos::atomic_fetch_add(&nFacesPerCellView[neighbour], 1);

            auto startSegOwn = segment[owner];
            auto startSegNei = segment[neighbour];
            Kokkos::atomic_store(&stencilValues[startSegOwn + segIdxOwn], facei);
            Kokkos::atomic_store(&stencilValues[startSegNei + segIdxNei], facei);
        },
        "computeStencilInternal"
    );

    parallelFor(
        exec,
        {nInternalFaces, nInternalFaces + boundaryFaceOwners.size()},
        NEON_LAMBDA(const localIdx facei) {
            localIdx owner = boundaryFaceOwners[facei - nInternalFaces];
            localIdx segIdxOwn = Kokkos::atomic_fetch_add(&nFacesPerCellView[owner], 1);
            localIdx startSegOwn = segment[owner];
            Kokkos::atomic_store(&stencilValues[startSegOwn + segIdxOwn], facei);
        },
        "computeStencilBound"
    );

    return stencil;
}

} // namespace NeoN::finiteVolume::cellCentred

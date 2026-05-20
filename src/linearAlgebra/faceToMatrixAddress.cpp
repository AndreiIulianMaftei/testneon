// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/macros.hpp"
#include "NeoN/core/segmentedVector.hpp"
#include "NeoN/linearAlgebra/utilities.hpp"
#include "NeoN/linearAlgebra/cooSparsityPattern.hpp"
#include "NeoN/linearAlgebra/csrSparsityPattern.hpp"
#include "NeoN/linearAlgebra/faceToMatrixAddress.hpp"

namespace NeoN::la
{

// ---------------------------------------------------------------------------
// FaceToMatrixAddress
// ---------------------------------------------------------------------------

void FaceToMatrixAddress::validate() const
{
    NF_ASSERT(ownerOffset_.exec() == neighbourOffset_.exec(), "Executors are not the same");
    NF_ASSERT(ownerOffset_.exec() == diagOffset_.exec(), "Executors are not the same");
    NF_ASSERT(
        ownerOffset_.size() == neighbourOffset_.size(),
        "ownerOffset and neighbourOffset must have the same size"
    );
}

FaceToMatrixAddress::FaceToMatrixAddress(
    Array<uint8_t> ownerOffset, Array<uint8_t> neighbourOffset, Array<uint8_t> diagOffset
)
    : ownerOffset_(std::move(ownerOffset)), neighbourOffset_(std::move(neighbourOffset)),
      diagOffset_(std::move(diagOffset))
{
    validate();
}

FaceToMatrixAddress::FaceToMatrixAddress(const FaceToMatrixAddress& mi)
    : ownerOffset_(mi.ownerOffset_), neighbourOffset_(mi.neighbourOffset_),
      diagOffset_(mi.diagOffset_)
{
    validate();
}

FaceToMatrixAddress FaceToMatrixAddress::copyToExecutor(Executor dstExec) const
{
    return {
        ownerOffset_.copyToExecutor(dstExec),
        neighbourOffset_.copyToExecutor(dstExec),
        diagOffset_.copyToExecutor(dstExec)
    };
}

FaceToMatrixView FaceToMatrixAddress::view(View<const localIdx> rowOffsView) const
{
    return FaceToMatrixView(
        ownerOffset_.view(), neighbourOffset_.view(), diagOffset_.view(), rowOffsView
    );
}

const NeoN::Array<uint8_t>& FaceToMatrixAddress::ownerOffset() const { return ownerOffset_; }

const NeoN::Array<uint8_t>& FaceToMatrixAddress::neighbourOffset() const
{
    return neighbourOffset_;
}

const NeoN::Array<uint8_t>& FaceToMatrixAddress::diagOffset() const { return diagOffset_; }

NeoN::Array<uint8_t>& FaceToMatrixAddress::ownerOffset() { return ownerOffset_; }

NeoN::Array<uint8_t>& FaceToMatrixAddress::neighbourOffset() { return neighbourOffset_; }

NeoN::Array<uint8_t>& FaceToMatrixAddress::diagOffset() { return diagOffset_; }

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

template<typename IndexType>
void setBoundarySparsityPatternImpl(
    const UnstructuredMesh& mesh,
    const Array<uint8_t>& diagOffs,
    Vector<IndexType>& rowIdx,
    Vector<IndexType>& colIdx
)
{
    const auto exec = mesh.exec();
    const auto nBoundaryFaces = static_cast<std::size_t>(mesh.nBoundaryFaces());
    const auto diagOffsV = diagOffs.view();
    const auto faceCellsV = mesh.boundaryMesh().faceOwners().view();
    const auto nBoundaryFaces = mesh.nBoundaryFaces();
    NF_ASSERT(nBoundaryFaces == rowIdx.size(), "Inconsistent size");
    NF_ASSERT(nBoundaryFaces == colIdx.size(), "Inconsistent size");

    auto rowIdxV = rowIdx.view();
    auto colIdxV = colIdx.view();
    parallelFor(
        exec,
        {0, nBoundaryFaces},
        KOKKOS_LAMBDA(const localIdx bfacei) {
            localIdx celli = faceCellsV[bfacei];
            colIdxV[bfacei] = celli + diagOffsV[celli];
            rowIdxV[bfacei] = celli;
        },
        "setSparsityPatternFaceToMatrixAddress::setBoundarySparsity"
    );
}

template<typename IndexType>
void setProcBoundarySparsityPattern(
    const UnstructuredMesh& mesh,
    const Array<uint8_t>& diagOffs,
    Vector<IndexType>& rowIdx,
    Vector<IndexType>& colIdx
)
{
    const auto exec = mesh.exec();
    const auto faceCellsV = mesh.boundaryMesh().faceOwners().view();
    const auto nBoundaryFaces = mesh.nBoundaryFaces();
    const auto nProcBoundaryFaces = mesh.nProcBoundaryFaces();
    auto rowIdxV = rowIdx.view();

    NF_ASSERT(nBoundaryFaces + nProcBoundaryFaces == faceCellsV.size(), "Inconsistent size");
    NF_ASSERT(nProcBoundaryFaces == rowIdx.size(), "Inconsistent size");
    NF_ASSERT(nProcBoundaryFaces == colIdx.size(), "Inconsistent size");

    parallelFor(
        exec,
        {0, nProcBoundaryFaces},
        KOKKOS_LAMBDA(const localIdx bfacei) {
            rowIdxV[bfacei] = faceCellsV[bfacei + nBoundaryFaces];
        },
        "setSparsityPatternFaceToMatrixAddress::setProcBoundarySparsity"
    );
}

template<typename IndexType>
void setSparsityPatternFaceToMatrixAddressSerial(
    const UnstructuredMesh& mesh,
    Array<uint8_t>& diagOffs,
    Array<uint8_t>& ownOffs,
    Array<uint8_t>& neiOffs,
    Vector<IndexType>& rowOffs,
    Vector<IndexType>& colIdx
)
{
    const auto nInternalFaces = mesh.nInternalFaces();
    auto nCells = mesh.nCells();

    // Start with one to include the diagonal
    auto nFacesPerCellH = Vector<localIdx>(SerialExecutor {}, nCells, 1);
    auto [neiOffsetH, ownOffsetH, diagOffsetH, faceOwnH, faceNeiH] =
        copyToHosts(neiOffs, ownOffs, diagOffs, mesh.faceOwners(), mesh.faceNeighbors());

    auto [nFacesPerCellHV, neiOffsetHV, ownOffsetHV, diagOffsetHV, faceOwnHV, faceNeiHV] =
        views(nFacesPerCellH, neiOffsetH, ownOffsetH, diagOffsetH, faceOwnH, faceNeiH);

    // Accumulate number of non-zeros per row
    parallelFor(
        SerialExecutor {},
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const localIdx facei) {
            auto own = faceOwnHV[facei];
            auto nei = faceNeiHV[facei];
            Kokkos::atomic_inc(&nFacesPerCellHV[own]);
            Kokkos::atomic_inc(&nFacesPerCellHV[nei]);
        },
        "setSparsityPatternFaceToMatrixAddress::accumulateNonZeros"
    );

    auto rowOffsH = rowOffs.copyToHost();
    auto rowOffsHV = rowOffsH.view();
    segmentsFromIntervals(nFacesPerCellH, rowOffsH);
    auto colIdxH = colIdx.copyToHost();
    auto colIdxHV = colIdxH.view();
    fill(nFacesPerCellH, 0); // Reset nFacesPerCell

    // Compute lower triangular part
    parallelFor(
        SerialExecutor {},
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const localIdx facei) {
            auto nei = faceNeiHV[facei];
            auto own = faceOwnHV[facei];
            auto segIdxNei = Kokkos::atomic_fetch_add(&nFacesPerCellHV[nei], 1);
            neiOffsetHV[facei] = static_cast<uint8_t>(segIdxNei);
            auto startSegNei = rowOffsHV[nei];
            Kokkos::atomic_store(&colIdxHV[startSegNei + segIdxNei], own);
        },
        "setSparsityPatternFaceToMatrixAddress::computeLowerTriangular"
    );

    map(
        nFacesPerCellH,
        KOKKOS_LAMBDA(const localIdx celli) {
            auto nFaces = nFacesPerCellHV[celli];
            diagOffsetHV[celli] = static_cast<uint8_t>(nFaces);
            colIdxHV[rowOffsHV[celli] + nFaces] = celli;
            return nFaces + 1;
        }
    );

    // Compute upper triangular part
    parallelFor(
        SerialExecutor {},
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const localIdx facei) {
            auto nei = faceNeiHV[facei];
            auto own = faceOwnHV[facei];
            auto segIdxOwn =
                static_cast<uint8_t>(Kokkos::atomic_fetch_add(&nFacesPerCellHV[own], 1));
            ownOffsetHV[facei] = segIdxOwn;
            auto startSegOwn = rowOffsHV[own];
            Kokkos::atomic_store(&colIdxHV[startSegOwn + segIdxOwn], nei);
        },
        "setSparsityPatternFaceToMatrixAddress::computeUpperTriangular"
    );

    // Copy results back to device
    const auto exec = mesh.exec();
    ownOffs = ownOffsetH.copyToExecutor(exec);
    neiOffs = neiOffsetH.copyToExecutor(exec);
    diagOffs = diagOffsetH.copyToExecutor(exec);
    colIdx = colIdxH.copyToExecutor(exec);
    rowOffs = rowOffsH.copyToExecutor(exec);
}

template<typename SparsityType>
std::pair<std::shared_ptr<const SparsityType>, std::shared_ptr<const FaceToMatrixAddress>>
createSparsityPatternFaceToMatrixAddress(const UnstructuredMesh& mesh)
{
    using IndexType = typename SparsityType::SparsityIndexType;
    const auto exec = mesh.exec();
    const auto nInternalFaces = mesh.nInternalFaces();
    const auto nCells = mesh.nCells();
    Array<uint8_t> diagOffs(exec, nCells, 0);
    Array<uint8_t> ownOffs(exec, nInternalFaces, 0);
    Array<uint8_t> neiOffs(exec, nInternalFaces, 0);
    Vector<IndexType> rowOffs(exec, nCells + 1, 0);
    Vector<IndexType> colIdx(exec, nCells + 2 * nInternalFaces, 0);

    setSparsityPatternFaceToMatrixAddressSerial(mesh, diagOffs, ownOffs, neiOffs, rowOffs, colIdx);
    auto sp = std::make_shared<const SparsityType>(
        std::move(colIdx), std::move(rowOffs), Dimensions {nCells, nCells}
    );
    auto ftma = std::make_shared<const FaceToMatrixAddress>(ownOffs, neiOffs, diagOffs);
    return {sp, ftma};
}

// COO specialization: internal algorithm produces CSR-style rowOffs, so we must expand
// them to flat per-NNZ row indices as required by CooSparsityPattern.
template<>
std::pair<
    std::shared_ptr<const CooSparsityPattern<localIdx>>,
    std::shared_ptr<const FaceToMatrixAddress>>
createSparsityPatternFaceToMatrixAddress<CooSparsityPattern<localIdx>>(const UnstructuredMesh& mesh)
{
    const auto exec = mesh.exec();
    const auto nInternalFaces = mesh.nInternalFaces();
    const auto nCells = mesh.nCells();
    const localIdx nnz = nCells + 2 * nInternalFaces;
    Array<uint8_t> diagOffs(exec, nCells, 0);
    Array<uint8_t> ownOffs(exec, nInternalFaces, 0);
    Array<uint8_t> neiOffs(exec, nInternalFaces, 0);
    Vector<localIdx> rowOffs(exec, nCells + 1, 0);
    Vector<localIdx> colIdx(exec, nnz, 0);

    setSparsityPatternFaceToMatrixAddressSerial(mesh, diagOffs, ownOffs, neiOffs, rowOffs, colIdx);

    // Convert CSR rowOffs → COO flat row indices
    Vector<localIdx> cooRowIdx(exec, nnz, 0);
    {
        auto rowOffsH = rowOffs.copyToHost();
        auto cooRowIdxH = cooRowIdx.copyToHost();
        const auto rowOffsHV = rowOffsH.view();
        auto cooRowIdxHV = cooRowIdxH.view();
        for (localIdx r = 0; r < nCells; r++)
            for (localIdx j = rowOffsHV[r]; j < rowOffsHV[r + 1]; j++)
                cooRowIdxHV[j] = r;
        cooRowIdx = cooRowIdxH.copyToExecutor(exec);
    }

    auto sp = std::make_shared<const CooSparsityPattern<localIdx>>(
        std::move(colIdx), std::move(cooRowIdx), Dimensions {nCells, nCells}
    );
    auto ftma = std::make_shared<const FaceToMatrixAddress>(ownOffs, neiOffs, diagOffs);
    return {sp, ftma};
}

// ---------------------------------------------------------------------------
// createBoundarySparsityPattern specializations
// ---------------------------------------------------------------------------

template<>
std::shared_ptr<const CooSparsityPattern<localIdx>>
createBoundarySparsityPattern<CooSparsityPattern<localIdx>>(
    const UnstructuredMesh& mesh, const FaceToMatrixAddress& ftma
)
{
    const auto exec = mesh.exec();
    const auto nBoundaryFaces = static_cast<localIdx>(mesh.nBoundaryFaces());
    Vector<localIdx> rowIdx(exec, nBoundaryFaces, 0);
    Vector<localIdx> colIdx(exec, nBoundaryFaces, 0);
    setBoundarySparsityPatternImpl(mesh, ftma.diagOffset(), rowIdx, colIdx);
    return std::make_shared<const CooSparsityPattern<localIdx>>(
        std::move(colIdx), std::move(rowIdx), Dimensions {mesh.nCells(), mesh.nCells()}
    );
}

template<>
std::shared_ptr<const CsrSparsityPattern<localIdx>>
createBoundarySparsityPattern<CsrSparsityPattern<localIdx>>(
    const UnstructuredMesh& mesh, const FaceToMatrixAddress& ftma
)
{
    const auto exec = mesh.exec();
    const auto nBoundaryFaces = static_cast<localIdx>(mesh.nBoundaryFaces());
    Vector<localIdx> rowIdx(exec, nBoundaryFaces, 0);
    Vector<localIdx> colIdx(exec, nBoundaryFaces, 0);
    setBoundarySparsityPatternImpl(mesh, ftma.diagOffset(), rowIdx, colIdx);
    auto rowPtrs = rowsToRowOffs(rowIdx);
    return std::make_shared<const CsrSparsityPattern<localIdx>>(
        std::move(colIdx), std::move(rowPtrs), Dimensions {mesh.nCells(), nBoundaryFaces}
    );
}

// TODO currently CSR is hardcoded here
template std::pair<
    std::shared_ptr<const CsrSparsityPattern<localIdx>>,
    std::shared_ptr<const FaceToMatrixAddress>>
createSparsityPatternFaceToMatrixAddress<CsrSparsityPattern<localIdx>>(const UnstructuredMesh&);

} // namespace NeoN::la

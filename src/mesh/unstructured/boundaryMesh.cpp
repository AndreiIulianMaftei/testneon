// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/mesh/unstructured/boundaryMesh.hpp"

namespace NeoN
{

BoundaryMesh::BoundaryMesh(
    const Executor& exec,
    labelVector faceCells,
    vectorVector faceCenters,
    vectorVector cn,
    vectorVector faceNormals,
    scalarVector faceAreas,
    vectorVector nf,
    vectorVector delta,
    scalarVector weights,
    scalarVector deltaCoeffs,
    std::vector<localIdx> offset
)
    : exec_(exec), faceCells_(faceCells), faceCenters_(faceCenters), Cn_(cn),
      faceNormals_(faceNormals), faceAreas_(faceAreas), nf_(nf), delta_(delta), weights_(weights),
      deltaCoeffs_(deltaCoeffs), offset_(offset) {};

// Accessor methods
const labelVector& BoundaryMesh::faceCells() const { return faceCells_; }


template<typename ValueType>
View<const ValueType>
extractSubView(const Vector<ValueType>& vec, const std::vector<localIdx>& offs, localIdx i)
{
    // TODO make offset a Vector<localIdx> instead of std::vector
    auto j = static_cast<std::size_t>(i);
    return vec.view({offs[j], offs[j + 1]});
}


View<const label> BoundaryMesh::faceCells(const localIdx i) const
{
    return extractSubView(faceCells_, offset_, i);
}

const vectorVector& BoundaryMesh::faceCenters() const { return faceCenters_; }

View<const Vec3> BoundaryMesh::faceCenters(const localIdx i) const
{
    return extractSubView(faceCenters_, offset_, i);
}

const vectorVector& BoundaryMesh::cn() const { return Cn_; }

View<const Vec3> BoundaryMesh::cn(const localIdx i) const
{
    return extractSubView(Cn_, offset_, i);
}

const vectorVector& BoundaryMesh::faceNormals() const { return faceNormals_; }

View<const Vec3> BoundaryMesh::faceNormals(const localIdx i) const
{
    return extractSubView(faceNormals_, offset_, i);
}

const scalarVector& BoundaryMesh::faceAreas() const { return faceAreas_; }

View<const scalar> BoundaryMesh::faceAreas(const localIdx i) const
{
    return extractSubView(faceAreas_, offset_, i);
}

const vectorVector& BoundaryMesh::nf() const { return nf_; }

View<const Vec3> BoundaryMesh::nf(const localIdx i) const
{
    return extractSubView(nf_, offset_, i);
}

const vectorVector& BoundaryMesh::delta() const { return delta_; }

View<const Vec3> BoundaryMesh::delta(const localIdx i) const
{
    return extractSubView(delta_, offset_, i);
}

const scalarVector& BoundaryMesh::weights() const { return weights_; }

View<const scalar> BoundaryMesh::weights(const localIdx i) const
{
    return extractSubView(weights_, offset_, i);
}

const scalarVector& BoundaryMesh::deltaCoeffs() const { return deltaCoeffs_; }

View<const scalar> BoundaryMesh::deltaCoeffs(const localIdx i) const
{
    return extractSubView(deltaCoeffs_, offset_, i);
}

const std::vector<localIdx>& BoundaryMesh::offset() const { return offset_; }


} // namespace NeoN

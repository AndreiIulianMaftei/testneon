// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/mesh/unstructured/uniformMeshGenerator.hpp"

#include "NeoN/core/primitives/vec3.hpp" // for Vec3


namespace NeoN
{
UnstructuredMesh::UnstructuredMesh(
    Executor exec,
    vectorVector points,
    scalarVector cellVolumes,
    vectorVector cellCentres,
    vectorVector faceAreas,
    vectorVector faceCentres,
    scalarVector magFaceAreas,
    labelVector faceOwner,
    labelVector faceNeighbour,
    localIdx nCells,
    localIdx nInternalFaces,
    localIdx nBoundaryFaces,
    localIdx nBoundaries,
    localIdx nFaces,
    BoundaryMesh boundaryMesh
)
    : exec_(exec), points_(points), cellVolumes_(cellVolumes), cellCentres_(cellCentres),
      faceAreas_(faceAreas), faceCentres_(faceCentres), magFaceAreas_(magFaceAreas),
      faceOwner_(faceOwner), faceNeighbour_(faceNeighbour), nCells_(nCells),
      nInternalFaces_(nInternalFaces), nBoundaryFaces_(nBoundaryFaces), nBoundaries_(nBoundaries),
      nFaces_(nFaces), boundaryMesh_(boundaryMesh), stencilDataBase_()
{}

UnstructuredMesh::UnstructuredMesh(
    vectorVector points,
    scalarVector cellVolumes,
    vectorVector cellCentres,
    vectorVector faceAreas,
    vectorVector faceCentres,
    scalarVector magFaceAreas,
    labelVector faceOwner,
    labelVector faceNeighbour,
    localIdx nCells,
    localIdx nInternalFaces,
    localIdx nBoundaryFaces,
    localIdx nBoundaries,
    localIdx nFaces,
    BoundaryMesh boundaryMesh
)
    : UnstructuredMesh(
        faceOwner.exec(),
        points,
        cellVolumes,
        cellCentres,
        faceAreas,
        faceCentres,
        magFaceAreas,
        faceOwner,
        faceNeighbour,
        nCells,
        nInternalFaces,
        nBoundaryFaces,
        nBoundaries,
        nFaces,
        boundaryMesh
    )
{}


const vectorVector& UnstructuredMesh::points() const { return points_; }

vectorVector& UnstructuredMesh::points() { return points_; }

const scalarVector& UnstructuredMesh::cellVolumes() const { return cellVolumes_; }

scalarVector& UnstructuredMesh::cellVolumes() { return cellVolumes_; }

const vectorVector& UnstructuredMesh::cellCentres() const { return cellCentres_; }

vectorVector& UnstructuredMesh::cellCentres() { return cellCentres_; }

const vectorVector& UnstructuredMesh::faceCentres() const { return faceCentres_; }

vectorVector& UnstructuredMesh::faceCentres() { return faceCentres_; }

const vectorVector& UnstructuredMesh::faceAreas() const { return faceAreas_; }

vectorVector& UnstructuredMesh::faceAreas() { return faceAreas_; }

const scalarVector& UnstructuredMesh::magFaceAreas() const { return magFaceAreas_; }

scalarVector& UnstructuredMesh::magFaceAreas() { return magFaceAreas_; }

const labelVector& UnstructuredMesh::faceOwner() const { return faceOwner_; }

labelVector& UnstructuredMesh::faceOwner() { return faceOwner_; }

const labelVector& UnstructuredMesh::faceNeighbour() const { return faceNeighbour_; }

labelVector& UnstructuredMesh::faceNeighbour() { return faceNeighbour_; }

localIdx UnstructuredMesh::nCells() const { return nCells_; }

localIdx UnstructuredMesh::nInternalFaces() const { return nInternalFaces_; }

localIdx UnstructuredMesh::nBoundaryFaces() const { return nBoundaryFaces_; }

localIdx UnstructuredMesh::nBoundaries() const { return nBoundaries_; }

localIdx UnstructuredMesh::nFaces() const { return nFaces_; }

const BoundaryMesh& UnstructuredMesh::boundaryMesh() const { return boundaryMesh_; }

Dictionary& UnstructuredMesh::stencilDB() const { return stencilDataBase_; }

const Executor& UnstructuredMesh::exec() const { return exec_; }

UnstructuredMesh createSingleCellMesh(const Executor exec)
{
    // a 2D mesh in 3D space with left, right, top, bottom boundary faces
    // with the centre at (0.5, 0.5, 0.0)
    // left, top, right, bottom faces
    // and four boundaries one left, right, top, bottom

    vectorVector faceAreasVec3s(exec, {{-1, 0, 0}, {0, 1, 0}, {1, 0, 0}, {0, -1, 0}});
    vectorVector faceCentresVec3s(
        exec, {{0.0, 0.5, 0.0}, {0.5, 1.0, 0.0}, {1.0, 0.5, 0.0}, {0.5, 0.0, 0.0}}
    );
    scalarVector magFaceAreas(exec, {1, 1, 1, 1});

    BoundaryMesh boundaryMesh(
        exec,
        {exec, {0, 0, 0, 0}},                                                           // faceCells
        faceCentresVec3s,                                                               // cf
        faceAreasVec3s,                                                                 // cn,
        faceAreasVec3s,                                                                 // sf,
        magFaceAreas,                                                                   // magSf,
        faceAreasVec3s,                                                                 // nf,
        {exec, {{-0.5, 0.0, 0.0}, {0.0, 0.5, 0.0}, {0.5, 0.0, 0.0}, {0.0, -0.5, 0.0}}}, // delta
        {exec, {1, 1, 1, 1}},                                                           // weights
        {exec, {2.0, 2.0, 2.0, 2.0}}, // deltaCoeffs --> mag(1 / delta)
        {0, 1, 2, 3, 4}               // offset
    );
    return UnstructuredMesh(
        {exec, {{0, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}}}, // points,
        {exec, 1, 1.0},                                       // cellVolumes
        {exec, {{0.5, 0.5, 0.0}}},                            // cellCentres
        faceAreasVec3s,
        faceCentresVec3s,
        magFaceAreas,
        {exec, {0, 0, 0, 0}}, // faceOwner
        {exec, {}},           // faceNeighbour,
        1,                    // nCells
        0,                    // nInternalFaces,
        4,                    // nBoundaryFaces,
        4,                    // nBoundaries,
        4,                    // nFaces,
        boundaryMesh
    );
}

// UnstructuredMesh create1DUniformMesh(const Executor exec, const localIdx nCells)
// {
//     // mesh points are stored in the following layout
//     // [ internal points | left boundary point | right boundary point ]

//     // Define the left and right boundaries of the mesh
//     const Vec3 leftBoundary = {0.0, 0.0, 0.0};
//     const Vec3 rightBoundary = {1.0, 0.0, 0.0};

//     // Define the spacing between the mesh points
//     scalar meshSpacing = (rightBoundary[0] - leftBoundary[0]) / static_cast<scalar>(nCells);

//     // Create a host view for the mesh points and initialize the boundary points
//     auto hostExec = SerialExecutor {};
//     vectorVector meshPointsHost(hostExec, nCells + 1, {0.0, 0.0, 0.0});
//     auto meshPointsHostView = meshPointsHost.view();
//     meshPointsHostView[nCells - 1] = leftBoundary;
//     meshPointsHostView[nCells] = rightBoundary;

//     // Copy the mesh points to the executor
//     auto meshPoints = meshPointsHost.copyToExecutor(exec);

//     // Compute internal mesh points
//     auto meshPointsView = meshPoints.view();
//     auto leftBoundaryX = leftBoundary[0];
//     parallelFor(
//         exec,
//         {0, nCells - 1},
//         NEON_LAMBDA(const localIdx i) {
//             meshPointsView[i][0] = leftBoundaryX + static_cast<scalar>(i + 1) * meshSpacing;
//         },
//         "computeMeshPoints"
//     );

//     // Create the cell volumes
//     scalarVector cellVolumes(exec, nCells, meshSpacing);

//     // Create and compute the cell centers
//     vectorVector cellCenters(exec, nCells, {0.0, 0.0, 0.0});
//     auto cellCentersView = cellCenters.view();
//     parallelFor(
//         exec,
//         {0, nCells},
//         NEON_LAMBDA(const localIdx i) {
//             cellCentersView[i][0] = 0.5 * meshSpacing + meshSpacing * static_cast<scalar>(i);
//         },
//         "computeCellCenters"
//     );

//     // Create the face normals
//     vectorVector faceAreasHost(hostExec, nCells + 1, {1.0, 0.0, 0.0});
//     auto faceAreasHostView = faceAreasHost.view();
//     faceAreasHostView[nCells - 1] = {-1.0, 0.0, 0.0}; // left boundary face
//     auto faceAreas = faceAreasHost.copyToExecutor(exec);

//     // Create the face centers
//     vectorVector faceCenters(exec, meshPoints);
//     scalarVector magFaceAreas(exec, nCells + 1, 1.0);

//     // Create the face owner and neighbor lists
//     labelVector faceOwnerHost(hostExec, nCells + 1);
//     labelVector faceNeighbor(exec, nCells - 1);
//     auto faceOwnerHostView = faceOwnerHost.view();
//     faceOwnerHostView[nCells - 1] = 0;                          // left boundary face
//     faceOwnerHostView[nCells] = static_cast<label>(nCells) - 1; // right boundary face
//     auto faceOwner = faceOwnerHost.copyToExecutor(exec);

//     // Compute the face owner and neighbor lists for internal faces
//     auto faceOwnerView = faceOwner.view();
//     auto faceNeighborView = faceNeighbor.view();
//     parallelFor(
//         exec,
//         {0, nCells - 1},
//         NEON_LAMBDA(const localIdx i) {
//             faceOwnerView[i] = i;
//             faceNeighborView[i] = i + 1;
//         },
//         "computeFaceOwnerAndNeighbors"
//     );

//     // Create the delta vectors and delta coefficients for the boundary faces
//     vectorVector deltaHost(hostExec, 2);
//     auto deltaHostView = deltaHost.view();
//     auto cellCentersHost = cellCenters.copyToHost();
//     auto cellCentersHostView = cellCentersHost.view();
//     deltaHostView[0] = {leftBoundary[0] - cellCentersHostView[0][0], 0.0, 0.0};
//     deltaHostView[1] = {rightBoundary[0] - cellCentersHostView[nCells - 1][0], 0.0, 0.0};
//     auto delta = deltaHost.copyToExecutor(exec);

//     scalarVector deltaCoeffsHost(hostExec, 2);
//     auto deltaCoeffsHostView = deltaCoeffsHost.view();
//     deltaCoeffsHostView[0] = 1 / mag(deltaHostView[0]);
//     deltaCoeffsHostView[1] = 1 / mag(deltaHostView[1]);
//     auto deltaCoeffs = deltaCoeffsHost.copyToExecutor(exec);

//     BoundaryMesh boundaryMesh(
//         exec,
//         {exec, {0, nCells - 1}},                                           // faceCells
//         {exec, {leftBoundary, rightBoundary}},                             // face centers
//         {exec, {cellCentersHostView[0], cellCentersHostView[nCells - 1]}}, // neighbor cell
//         centers {exec, {{-1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}}},                       // face
//         normals {exec, {1.0, 1.0}},                                                // face areas
//         {exec, {{-1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}}},                       // face unit normals
//         delta,                                                             // delta vectors
//         {exec, {1.0, 1.0}},                                                // weights
//         deltaCoeffs, // inverse of magnitude of delta vectors
//         {0, 1, 2}    // offset of the faces of each boundary
//     );

//     return UnstructuredMesh(
//         meshPoints,
//         cellVolumes,
//         cellCenters,
//         faceAreas,
//         faceCenters,
//         magFaceAreas,
//         faceOwner,
//         faceNeighbor,
//         nCells,
//         nCells - 1,
//         2,
//         2,
//         nCells + 1,
//         boundaryMesh
//     );
// }

UnstructuredMesh create1DUniformMesh(const Executor exec, const localIdx nCells)
{
    return create3DUniformMesh(exec, nCells, 1, 1);
}

UnstructuredMesh
create2DUniformMesh(const Executor exec, localIdx nx, localIdx ny, scalar Lx, scalar Ly)
{
    return create3DUniformMesh(exec, nx, ny, 1, Lx, Ly, 1.0);
}

UnstructuredMesh create3DUniformMesh(
    const Executor exec, localIdx nx, localIdx ny, localIdx nz, scalar Lx, scalar Ly, scalar Lz
)
{
    // Validate input parameters
    NF_ASSERT(nx > 0 && ny > 0 && nz > 0, "Number of cells in each direction must be positive");
    NF_ASSERT(Lx > 0 && Ly > 0 && Lz > 0, "Domain lengths must be positive");

    // Hold the mesh parameters
    detail::MeshParams p {nx, ny, nz, Lx, Ly, Lz};

    const auto points = detail::generatePoints(p);
    const auto [cellVolumes, cellCentres] = detail::generateCellData(p);

    // Judge the dimension based on the input parameters
    int dim = 0;
    if (ny == 1 && nz == 1)
    {
        dim = 1;
    }
    else if (nz == 1)
    {
        dim = 2;
    }
    else
    {
        dim = 3;
    }

    // Compute the number of internal faces and boundary faces based on the mesh parameters
    const localIdx nXInternalFaces = (p.nx - 1) * p.ny * p.nz;
    const localIdx nYInternalFaces = p.nx * (p.ny - 1) * p.nz;
    const localIdx nZInternalFaces = p.nx * p.ny * (p.nz - 1);
    const localIdx nInternalFaces = nXInternalFaces + nYInternalFaces + nZInternalFaces;

    const localIdx nBndLeft = p.ny * p.nz;
    const localIdx nBndRight = p.ny * p.nz;

    std::vector<localIdx> offset = {0, nBndLeft, nBndLeft + nBndRight};

    // If the mesh is more than 1D, there are bottom and top boundary faces
    if (dim > 1)
    {
        const localIdx nBndBottom = p.nx * p.nz;
        const localIdx nBndTop = p.nx * p.nz;
        offset.push_back(offset.back() + nBndBottom);
        offset.push_back(offset.back() + nBndTop);
    }

    // If the mesh is more than 2D, there are front and back boundary faces
    if (dim > 2)
    {
        const localIdx nBndFront = p.nx * p.ny;
        const localIdx nBndBack = p.nx * p.ny;
        offset.push_back(offset.back() + nBndFront);
        offset.push_back(offset.back() + nBndBack);
    }

    const localIdx nBoundaryFaces = offset.back();
    const localIdx nFaces = nInternalFaces + nBoundaryFaces;

    auto faces = detail::generateInternalFaces(p, nInternalFaces, nFaces);
    const auto boundaryMesh = detail::generateBoundaryData(
        exec, dim, p, cellCentres, faces, nInternalFaces, nBoundaryFaces, offset
    );

    // Note: With localIdx, the safer limit is ~700 million cells
    const localIdx nCells = nx * ny * nz;

    UnstructuredMesh mesh(
        vectorVector(exec, points),
        scalarVector(exec, cellVolumes),
        vectorVector(exec, cellCentres),
        {exec, faces.areas},
        {exec, faces.centres},
        {exec, faces.magnitudes},
        {exec, faces.owner},
        labelVector(exec, faces.neighbour),
        nCells,
        nInternalFaces,
        nBoundaryFaces,
        6,
        nFaces,
        boundaryMesh
    );

    return mesh;
}
} // namespace NeoN

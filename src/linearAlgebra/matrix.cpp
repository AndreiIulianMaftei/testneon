// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/primitives/vec3.hpp" // for Vec3
#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/linearAlgebra/CSRMatrix.hpp"

namespace NeoN::la
{

template<typename ValueType, typename IndexType>
Vector<ValueType> CSRMatrix<ValueType, IndexType>::diag() const
{
    auto diag = Vector<ValueType>(values_.exec(), nRows());
    auto [diagV, rowOffsV, colIdxV, matrixV] = views(diag, rowOffs(), colIdxs(), values_);

    parallelFor(
        values_.exec(),
        {0, nRows()},
        NEON_LAMBDA(const localIdx rowi) {
            for (auto i = rowOffsV[rowi]; i < rowOffsV[rowi + 1]; i++)
            {
                if (rowi == colIdxV[i])
                {
                    diagV[rowi] = matrixV[i];
                    break;
                }
            }
        },
        "copyDiag"
    );
    return diag;
}

template<typename ValueType, typename IndexType>
Vector<ValueType> CSRMatrix<ValueType, IndexType>::upper() const
{
    localIdx nUpper = (nNonZeros() - nRows()) / 2;

    auto upper = Vector<ValueType>(values_.exec(), nUpper);
    auto count = Vector<IndexType>(values_.exec(), nRows(), 0);
    auto offset = Vector<IndexType>(values_.exec(), nRows() + 1, 0);

    auto [upperV, rowOffsV, colIdxV, matrixV, countV, offsetV] =
        views(upper, rowOffs(), colIdxs(), values_, count, offset);

    parallelFor(
        values_.exec(),
        {0, nRows()},
        NEON_LAMBDA(const localIdx rowi) {
            for (auto i = rowOffsV[rowi]; i < rowOffsV[rowi + 1]; i++)
            {
                if (colIdxV[i] > rowi)
                {
                    Kokkos::atomic_inc(&countV[rowi]);
                }
            }
        },
        "computeNumUpperValues"
    );

    parallelScan(
        values_.exec(),
        {1, offsetV.size()},
        NEON_LAMBDA(const NeoN::localIdx i, NeoN::localIdx& update, const bool final) {
            update += countV[i - 1];
            if (final)
            {
                offsetV[i] = update;
            }
        }
    );

    parallelFor(
        values_.exec(),
        {0, nRows()},
        NEON_LAMBDA(const localIdx rowi) {
            label j = 0; // index of nth element found in this row
            for (auto i = rowOffsV[rowi]; i < rowOffsV[rowi + 1]; i++)
            {
                if (colIdxV[i] > rowi)
                {
                    upperV[offsetV[rowi] + j] = matrixV[i];
                    j++;
                }
            }
        }
    );
    return upper;
}

#define NN_DECLARE_CSRMATRIX(VALUETYPE, INTEGERTYPE)                                               \
    template class CSRMatrix<VALUETYPE, INTEGERTYPE>

NN_DECLARE_CSRMATRIX(scalar, localIdx);
NN_DECLARE_CSRMATRIX(Vec3, int);

}

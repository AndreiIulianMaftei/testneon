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
        }
    );
    return diag;
}

#define NN_DECLARE_CSRMATRIX(VALUETYPE, INTEGERTYPE)                                               \
    template class CSRMatrix<VALUETYPE, INTEGERTYPE>

NN_DECLARE_CSRMATRIX(scalar, localIdx);
NN_DECLARE_CSRMATRIX(Vec3, int);

}

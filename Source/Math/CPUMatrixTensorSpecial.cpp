#include "stdafx.h"

#ifdef USE_MKL

#include "CPUMatrixTensorImpl.h"
#include "mkl_cblas.h"
#include "mkl_vml.h"

#pragma warning(disable:4100) // warning C4100: 'var': unreferenced formal parameter

namespace Microsoft { namespace MSR { namespace CNTK {

template<>
bool CPUMatrixSpecialUnaryTensorOpImpl<float>(float beta, const CPUMatrix<float>& a, CPUMatrix<float>& o, float alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
    const array<size_t, 2>& offsets,
    const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 2>& regularStrides,
    const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 2>& reducingStrides)
{
    if (a.GetNumElements() == o.GetNumElements())
    {
        int N = (int)a.GetNumElements();
        switch(op)
        {
        case ElementWiseOperator::opLinearRectifier:
            vsAbs(N, a.Data(), o.Data());
            cblas_saxpby(N, 0.5f, a.Data(), 1, 0.5f, o.Data(), 1); // o = (a + abs(a))/2
            return true;
        }
    }
    return false;
}

template<>
bool CPUMatrixSpecialBinaryTensorOpImpl<float>(float beta, const CPUMatrix<float>& a, const CPUMatrix<float>& b, CPUMatrix<float>& o, float alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
    const array<size_t, 3>& offsets,
    const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 3>& regularStrides,
    const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 3>& reducingStrides)
{
    if (op == ElementWiseOperator::opSum &&
        offsets[0] == 0 && offsets[1] == 0 && offsets[2] == 0 &&
        a.GetNumRows() == b.GetNumRows() &&
        a.GetNumRows() == o.GetNumRows() &&
        ((a.GetNumCols() == 1 && o.GetNumCols() == b.GetNumCols()) ||
         (b.GetNumCols() == 1 && o.GetNumCols() == a.GetNumCols())))
    {
        // plus parameter (no dynamic axes, or GetNumCols() == 1)
        float* dataWithDynamicAxes = (a.GetNumCols() == 1 ? b.Data() : a.Data());
        float* dataParameter = (a.GetNumCols() == 1 ? a.Data() : b.Data());
        int N = (int)a.GetNumRows();
        for (int col = 0; col < o.GetNumCols(); ++col)
        {
            vsAdd(N, dataWithDynamicAxes + col * N, dataParameter, o.Data());
        }
        return true;
    }
    else if (op == ElementWiseOperator::opElementwiseProduct && a.GetNumElements() == 1)
    {
        cblas_saxpby((int)o.GetNumElements(), a.Data()[0], b.Data(), 1, 0.0f, o.Data(), 1);
    }
    else if (op == ElementWiseOperator::opDifference)
    {
        memcpy(o.Data(), b.Data(), o.GetNumElements() * sizeof(float));
        cblas_saxpby((int)o.GetNumElements(), 1.0f, a.Data(), (a.GetNumElements() == 1 ? 0 : 1), -1.0f, o.Data(), 1);
        return true;
    }
    return false;
}

template<>
bool CPUMatrixSpecialTernaryTensorOpImpl<float>(float beta, const CPUMatrix<float>& a, const CPUMatrix<float>& b, const CPUMatrix<float>& c, CPUMatrix<float>& o, float alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
    const array<size_t, 4>& offsets,
    const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 4>& regularStrides,
    const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 4>& reducingStrides)
{
    return false;
}

template<>
bool CPUMatrixSpecialUnaryTensorOpImpl<double>(double beta, const CPUMatrix<double>& a, CPUMatrix<double>& o, double alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
    const array<size_t, 2>& offsets,
    const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 2>& regularStrides,
    const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 2>& reducingStrides)
{
    return false;
}

template<>
bool CPUMatrixSpecialBinaryTensorOpImpl<double>(double beta, const CPUMatrix<double>& a, const CPUMatrix<double>& b, CPUMatrix<double>& o, double alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
    const array<size_t, 3>& offsets,
    const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 3>& regularStrides,
    const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 3>& reducingStrides)
{
    return false;
}

template<>
bool CPUMatrixSpecialTernaryTensorOpImpl<double>(double beta, const CPUMatrix<double>& a, const CPUMatrix<double>& b, const CPUMatrix<double>& c, CPUMatrix<double>& o, double alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
    const array<size_t, 4>& offsets,
    const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 4>& regularStrides,
    const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 4>& reducingStrides)
{
    return false;
}

template<>
bool CPUMatrixSpecialUnaryTensorOpImpl<half>(half beta, const CPUMatrix<half>& a, CPUMatrix<half>& o, half alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
    const array<size_t, 2>& offsets,
    const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 2>& regularStrides,
    const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 2>& reducingStrides)
{
    return false;
}

template<>
bool CPUMatrixSpecialBinaryTensorOpImpl<half>(half beta, const CPUMatrix<half>& a, const CPUMatrix<half>& b, CPUMatrix<half>& o, half alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
    const array<size_t, 3>& offsets,
    const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 3>& regularStrides,
    const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 3>& reducingStrides)
{
    return false;
}

template<>
bool CPUMatrixSpecialTernaryTensorOpImpl<half>(half beta, const CPUMatrix<half>& a, const CPUMatrix<half>& b, const CPUMatrix<half>& c, CPUMatrix<half>& o, half alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
    const array<size_t, 4>& offsets,
    const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 4>& regularStrides,
    const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 4>& reducingStrides)
{
    return false;
}

}}}

#endif
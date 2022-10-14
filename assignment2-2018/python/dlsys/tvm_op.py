from __future__ import absolute_import, print_function

import tvm
import numpy as np
import topi

# Global declarations of environment.

# llvm
tgt_host="llvm"
# llvm, cuda, opencl, metal
# Change it to respective GPU if gpu is enabled Ex: cuda, opencl
tgt="llvm"


def make_elemwise_add(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.placeholder(shape, dtype=dtype, name="B")
    C = tvm.compute(A.shape, lambda *i: A(*i) + B(*i))

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_elemwise_mul(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.placeholder(shape, dtype=dtype, name="B")
    C = tvm.compute(A.shape, lambda *i: A(*i) * B(*i))

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f

def make_elemwise_add_by_const(shape, const_k, tgt, tgt_host, func_name,
                               dtype="float32"):
    """TODO: Your code here"""
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = tvm.compute(A.shape, lambda *i: A(*i) + const_k)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_elemwise_mul_by_const(shape, const_k, tgt, tgt_host, func_name,
                            dtype="float32"):
    """TODO: Your code here"""
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = tvm.compute(A.shape, lambda *i: A(*i) * const_k)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f

def make_relu(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.max, tvm.const(0, A.dtype)"""
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    t = tvm.const(0, A.dtype)
    C = tvm.compute(A.shape, lambda *i: tvm.max(A(*i),t) )

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f



def make_relu_gradient(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.select"""
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.placeholder(shape, dtype=dtype, name="B")
    t = tvm.const(0, A.dtype)
    C = tvm.compute(A.shape, lambda *i: tvm.select((A(*i)>0),B(*i),t) )

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_matrix_mul(shapeA, transposeA, shapeB, transposeB, tgt, tgt_host,
                    func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum"""
    """Hint: treat 4 cases of transposeA, transposeB separately"""
    """Hint: for tvm schedule, use split, reorder, vectorize, parallel"""
    """Hint: debug tvm schedule using tvm.lower"""
    if ((transposeA == False) and (transposeB == False)):
        m = shapeA[0]
        n = shapeA[1]
        k = shapeB[1]
        assert(shapeA[1] == n)

        A = tvm.placeholder((m, n), name="A")
        B = tvm.placeholder((n, k), name="B")
        r = tvm.reduce_axis((0, n), name="k")
        C = tvm.compute((m, k), lambda i,j: tvm.sum(A[i, r] * B[r, j], axis=r), name="C")

        s = tvm.create_schedule(C.op)
        xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], x_factor=32, y_factor=64)
        xk, yk = s[C].split(r, factor=8)
        s[C].reorder(xo, yo, xk, xi, yi, yk)
        s[C].parallel(xo)
        s[C].unroll(yk)
        f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
        return f
    elif ((transposeA == False) and (transposeB == True)):
        m = shapeA[0]
        n = shapeA[1]
        k = shapeB[0]
        assert(shapeB[1] == n)

        A = tvm.placeholder((m, n), name="A")
        B = tvm.placeholder((k, n), name="B")
        r = tvm.reduce_axis((0, n), name="k")
        C = tvm.compute((m, k), lambda i,j: tvm.sum(A[i, r] * B[j, r], axis=r), name="C")

        s = tvm.create_schedule(C.op)
        xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], x_factor=32, y_factor=64)
        xk, yk = s[C].split(r, factor=8)
        s[C].reorder(xo, yo, xk, xi, yi, yk)
        s[C].parallel(xo)
        s[C].unroll(yk)
        f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
        return f
    elif ((transposeA == True) and (transposeB == False)):
        m = shapeA[1]
        n = shapeA[0]
        k = shapeB[1]
        assert(shapeB[0] == n)

        A = tvm.placeholder((n, m), name="A")
        B = tvm.placeholder((n, k), name="B")
        r = tvm.reduce_axis((0, n), name="k")
        C = tvm.compute((m, k), lambda i,j: tvm.sum(A[r, i] * B[r, j], axis=r), name="C")

        s = tvm.create_schedule(C.op)
        xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], x_factor=32, y_factor=64)
        xk, yk = s[C].split(r, factor=8)
        s[C].reorder(xo, yo, xk, xi, yi, yk)
        s[C].parallel(xo)
        s[C].unroll(yk)
        f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
        return f
    elif ((transposeA == True) and (transposeB == True)):
        m = shapeA[1]
        n = shapeA[0]
        k = shapeB[0]
        assert(shapeB[1] == n)

        A = tvm.placeholder((n, m), name="A")
        B = tvm.placeholder((k, n), name="B")
        r = tvm.reduce_axis((0, n), name="k")
        C = tvm.compute((m, k), lambda i,j: tvm.sum(A[r, i] * B[j, r], axis=r), name="C")

        s = tvm.create_schedule(C.op)
        xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], x_factor=32, y_factor=64)
        xk, yk = s[C].split(r, factor=8)
        s[C].reorder(xo, yo, xk, xi, yi, yk)
        s[C].parallel(xo)
        s[C].unroll(yk)
        f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
        return f



def make_conv2d(shapeX, shapeF, tgt, tgt_host, func_name, dtype="float32"):
    assert(shapeX[1] == shapeF[1])
    N, C, H, W = shapeX
    M, C, R, S = shapeF

    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum"""
    """Hint: go by conv2d definition. Treat stride=1, padding=0 case only."""
    """For a challenge, treat the general case for stride and padding."""
    X = tvm.placeholder((N, C, H, W), name="X")
    F = tvm.placeholder((M, C, R, S), name="F")
    r = tvm.reduce_axis((0, R*S), name="r")
    p = tvm.reduce_axis((0, C), name="p")
    Y = tvm.compute((N, M, H-R+1, W-S+1), lambda i,j,h,w: tvm.sum(X[i,p,h+r/S,w+r%S] * F[j,p,r/S,r%S], axis=[r,p]), name="Y")

    s = tvm.create_schedule(Y.op)
    f = tvm.build(s, [X, F, Y], tgt, target_host=tgt_host, name=func_name)
    return f

    # Input = tvm.placeholder(shapeX, dtype=dtype, name="A")
    # Filter = tvm.placeholder(shapeF, dtype=dtype, name="B")

    # di = tvm.reduce_axis((0, R), name='di')
    # dj = tvm.reduce_axis((0, S), name='dj')
    # dc = tvm.reduce_axis((0, C), name='dc')

    # Output = tvm.compute((N, M, H - R + 1, W - S + 1),
    #                     lambda n, m, i, j: tvm.sum(Input[n, dc, i + di, j + dj] * Filter[m, dc, di, dj], axis=[di, dj, dc]),
    #                     name='Output')
    # s = tvm.create_schedule(Output.op)
    # f = tvm.build(s, [Input, Filter, Output], tgt, target_host=tgt_host, name=func_name)
    # return f

def make_matrix_softmax(shape, tgt, tgt_host, func_name, dtype="float32"):

    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum, tvm.max, tvm.exp"""
    """Hint: do not reuse the same reduction axis j."""
    """Hint: implement the following version for better stability
        e_x = np.exp(x - np.max(x))
        softmax(x)= e_x / e_x.sum()
    """
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    r = tvm.reduce_axis((0, shape[1]), name="r")
    mx = tvm.compute((shape[0],), lambda i: tvm.max(A[i,r], axis=r), name="mx")
    ex = tvm.compute(shape, lambda i,j: tvm.exp(A[i, j] - mx[i] ), name="ex" )
    r = tvm.reduce_axis((0, shape[1]), name="r")
    ex_sum = tvm.compute((shape[0],), lambda i: tvm.sum(ex[i,r], axis=r), name="ex_sum")
    B = tvm.compute(shape, lambda i,j: ex[i, j] / ex_sum[i], name="B" )

    s = tvm.create_schedule(B.op)
    f = tvm.build(s, [A, B], tgt, target_host=tgt_host, name=func_name)
    return f


def make_matrix_softmax_cross_entropy(shape, tgt, tgt_host, func_name,
                                      dtype="float32"):
    """TODO: Your code here"""
    """Hint: output shape should be (1,)"""
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    A1 = tvm.placeholder(shape, dtype=dtype, name="A1")
    r = tvm.reduce_axis((0, shape[1]), name="r")
    mx = tvm.compute((shape[0],), lambda i: tvm.max(A[i,r], axis=r), name="mx")
    ex = tvm.compute(shape, lambda i,j: tvm.exp(A[i, j] - mx[i] ), name="ex" )
    r = tvm.reduce_axis((0, shape[1]), name="r")
    ex_sum = tvm.compute((shape[0],), lambda i: tvm.sum(ex[i,r], axis=r), name="ex_sum")
    B = tvm.compute(shape, lambda i,j: ex[i, j] / ex_sum[i], name="B" )
    r = tvm.reduce_axis((0, shape[1]), name="r")
    C = tvm.compute((shape[0],), lambda i: tvm.sum(A1[i, r] * tvm.log(B[i, r]) , axis=r), name="C" )
    r = tvm.reduce_axis((0, shape[0]), name="r")
    d = tvm.compute((1,), lambda i: tvm.sum(-C[r]/shape[0], axis=r), name="d" )

    s = tvm.create_schedule(d.op)
    f = tvm.build(s, [A, A1, d], tgt, target_host=tgt_host, name=func_name)
    return f


def make_reduce_sum_axis_zero(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = topi.sum(A, axis=0, keepdims=False)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_broadcast_to(shape, to_shape, tgt, tgt_host, func_name,
                      dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = topi.broadcast_to(A, to_shape)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_sgd_update(shape, learning_rate, tgt, tgt_host, func_name,
                    dtype="float32"):
    X = tvm.placeholder(shape, dtype=dtype, name="A")
    grad = tvm.placeholder(shape, dtype=dtype, name="grad")
    Y = tvm.compute(shape, lambda *i: X(*i) - learning_rate * grad(*i))

    s = tvm.create_schedule(Y.op)
    f = tvm.build(s, [X, grad, Y], tgt, target_host=tgt_host, name=func_name)
    return f
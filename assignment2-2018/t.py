from __future__ import absolute_import, print_function


import tvm
import numpy as np
# # declare some variables for use later
# n = tvm.var("n")
# m = tvm.var("m")

# # declare a matrix element-wise multiply
# A = tvm.placeholder((m, n), name="A")
# B = tvm.placeholder((m, n), name="B")
# C = tvm.compute((m, n), lambda i, j: A[i, j] * B[i, j], name="C")

# s = tvm.create_schedule([C.op])
# # lower will transform the computation from definition to the real
# # callable function. With argument `simple_mode=True`, it will
# # return you a readable C like statvmment, we use it here to print the
# # schedule result.
# print(tvm.lower(s, [A, B, C], simple_mode=True))

# llvm
tgt_host="llvm"
# llvm, cuda, opencl, metal
# Change it to respective GPU if gpu is enabled Ex: cuda, opencl
tgt="llvm"


shapeA=(200,200)
shapeB=(200,200)
dtype = "float32"

m = shapeA[0]
n = shapeA[1]
k = shapeB[1]
assert(shapeA[1] == n)

A = tvm.placeholder((m, n), name="A")
B = tvm.placeholder((n, k), name="B")
r = tvm.reduce_axis((0, n), name="k")
C = tvm.compute((m, k), lambda i,j: tvm.sum(A[i, r] * B[r, j], axis=r), name="C")

s = tvm.create_schedule(C.op)
# f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
# return f
print(C.op.axis[0])
print(C.op.axis[1])
xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], x_factor=32, y_factor=64)
xk, yk = s[C].split(r, factor=8)
print(xo)#i.outer
print(yo)#j.outer
print(xi)#i.inner
print(yi)#j.inner
print(xk)#k.outer
print(yk)#k.inner
s[C].reorder(xo, yo, xk, xi, yi, yk)
s[C].parallel(xo)
s[C].unroll(yk)
print(tvm.lower(s, [A,B,C], simple_mode=True))
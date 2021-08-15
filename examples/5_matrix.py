import phidnet



a = [[1, 2, 3],
     [9, 4, 3],
     [4, 9, 3],
     [7, 7, 3]]

b = [[2, 3, 4, 5],
     [2, 3, 4, 5],
     [2, 3, 4, 5]]

a = phidnet.array(a)
b = phidnet.array(b)



add = a + a
mul = a * a
dot = phidnet.matrix.dot(a, b)

sliced = phidnet.matrix.slice_full(a, 1, 2, 1, 1)   # 1~2 row, 1~1 column (0 based index)
sliced_2 = a["1:3,1:2"]   # 1~2 row, 1~1 column (0 based index)
sliced_3 = a[",1:2"]   # all row, 1~1 column (0 based index)

trans = phidnet.Matrix.trans(a)
trans_2 = a.trans()

def f(x): return 2*x
mapped = phidnet.Matrix.map(a, f)

print("========================")
print(a)
print(b)
print(add)
print(mul)
print(dot)
print(sliced)
print(sliced_2)
print(sliced_3)
print(trans)
print(trans_2)
print(mapped)
print(a["0,1"])
print("========================")

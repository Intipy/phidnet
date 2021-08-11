import phidnet



a = [[1, 2, 3],
     [9, 4, 3],
     [4, 9, 3],
     [7, 7, 3]]

b = [[2, 3, 4, 5],
     [2, 3, 4, 5],
     [2, 3, 4, 5]]

a = phidnet.matrix.matrix(a)
b = phidnet.matrix.matrix(b)

print(a)
print(b)

add = a + a
mul = a * a
dot = phidnet.matrix.Matrix.dot(a, b)
slice = phidnet.matrix.Matrix.slice(a, 1, 3, 1, 2)   # 1~3 row, 1~2 column (

print("--------")
print(add)
print(mul)
print(dot)
print(slice)
print("--------")
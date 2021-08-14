from phidnet.matrix import matrix_class



def array(*pargs, **kwargs):   # convert to matrix class.
    if isinstance(pargs[0], int):
        return matrix_class.Matrix.identity(pargs[0])
    elif isinstance(pargs[0], str):
        return matrix_class.Matrix.from_string(*pargs, **kwargs)
    elif isinstance(pargs[0], list):
        return matrix_class.Matrix.from_list(*pargs, **kwargs)
    else:
        raise NotImplementedError



def cross(u, v):   # Returns u x v - the vector product of 3D column vectors u and v.
    w = matrix_class.Matrix(3, 1)
    w[0][0] = u[1][0] * v[2][0] - u[2][0] * v[1][0]
    w[1][0] = u[2][0] * v[0][0] - u[0][0] * v[2][0]
    w[2][0] = u[0][0] * v[1][0] - u[1][0] * v[0][0]
    return w



def dot(self, other):   # Multiplication: `dot(self, other)`.
    if isinstance(other, matrix_class.Matrix):
        if self.numcols != other.numrows:
            raise matrix_class.MatrixError('incompatible sizes for multiplication')
        m = matrix_class.Matrix(self.numrows, other.numcols)
        for row, col, element in m:
            for re, ce in zip(self.row(row), other.col(col)):
                m[row][col] += re * ce
        return m
    else:
        return self.map(lambda element: element * other)



def slice_full(self, s1, e1, s2, e2):
    sliced = []
    mat = self.to_list()

    for i in range(s1, e1 + 1):   # s1 ~ e1 slice
        sliced.append(mat[i])

    index = 0
    for i in sliced:
        sliced[index] = sliced[index][s2:e2 + 1]
        index += 1

    return array(sliced)




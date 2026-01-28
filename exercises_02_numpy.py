import numpy as np
from numpy.ma.core import ones_like, identity
from numpy.matlib import zeros
from numpy.matrixlib.defmatrix import matrix

zeros_array = np.array(zeros(10))
print("array of 10 zeros:",zeros_array)

ones_array = np.ones(10)
print("\narray of 10 ones:", ones_array)

fives_array = ones_array*5
print("\narray of 10 fives:", fives_array)

array_2 = np.arange(10, 51)
print("\narray of the integers from 10 to 50", array_2)

array_3 = np.arange(10, 51, 2)
print("\narray of all the even integers from 10 to 50", array_3)

matrix_1 = np.array([np.arange(0, 3), np.arange(3, 6), np.arange(6, 9)])
print("\n3x3 matrix with values ranging from 0 to 8:\n", matrix_1)

m_identity = identity(3)
print("\n3x3 identity matrix:\n", m_identity)

r = np.random.rand(1)
print("\nrandom number between 0 and 1:", r)

normal_array = np.random.normal(size=25)
print("\nan array of 25 random numbers sampled from a standard normal distribution:\n", normal_array)

matrix_2 = (np.arange(1, 101)/100).reshape(10, 10)
print("\nmatrix 0.01 - 1:\n", matrix_2)

matrix_3 = np.arange(1, 26).reshape(5,5)
print("\nmatrix 1-25:\n", matrix_3)

matrix_4 = np.delete(np.arange(12, 27).reshape(3,5), -1, axis=1)
print("\nmatrix 12-15, 17-20, 22-25:\n", matrix_4)
print("\nelement 20 from the matrix: ", matrix_4[1][3])

matrix_5 = np.arange(2, 13, 5).reshape(3,1)
print("\nmatrix 2 7 12:\n", matrix_5)

array_4 = np.arange(21,26)
print("\narray 21-25:\n", array_4)

matrix_6 = np.arange(16, 26).reshape(2, 5)
print("\nmatrix 16-20, 21-25:\n", matrix_6)
print("\nsum of all the values in this matrix:", matrix_6.sum())
print("\nstandard deviation of the values in this matrix:", np.std(matrix_6))
print("\nsum of all the columns:", np.sum(matrix_6, axis=0))
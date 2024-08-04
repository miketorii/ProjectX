import numpy as np

arr = np.array([1,2,3,4,5])
print(arr)

arr = np.array([1,2,3,4], ndmin=2)
print(arr)

arr = np.array([])
print(arr.ndim)

arr = np.array([[1,2,3,4,5], [6,7,8,9,10]])
print('2nd element on 1st row: ', arr[1, 2])




#arr21 = np.arange(4).reshape((2,2))
#print(arr21)

#arr21_inv = np.linalg.inv(arr21)
#print(arr21_inv)

#
#arr31 = np.arange(6).reshape((2,3))
#print(arr31)
#print(arr31.T)
# 0 1 2
# 3 4 5
#mat31 = np.matrix(arr31)
#print(mat31)

#print('----------')
#b = np.random.randint(-10, 10, size=(3,3)) 
#print(b)
#c = np.linalg.inv(b)
#print(c)
#print('----------')

#
#arr32 = np.arange(0,12,2).reshape((2,3))
#print(arr32)
# 0 2 4
# 6 8 10
#mat32 = np.matrix(arr32)
#print(mat32)

#ans = np.dot(arr21, arr31)
#print(ans)

#ans = arr21 @ arr31
#print(ans)

#ans = np.matmul(arr21, arr31)
#print(ans)

#ans = np.matmul(arr31, arr32.T)
#print(ans)




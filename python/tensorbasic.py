import tensorflow as tf
import cProfile

print(10*'-','start',10*'-')

print(tf.__version__)

#tensor
r0 = tf.constant(4)
print(r0)
r1 = tf.constant([2.,3.,4.])
print(r1)
r2 = tf.constant([[1,2],[3,4],[5,6]])
print(r2)
r3 = tf.constant([ [ [0,1,2,3,4],[5,6,7,8,9] ],
                   [ [10,11,12,13,14],[15,16,17,18,19] ],
                   [ [20,21,22,23,24],[25,26,27,28,29] ] ])
print(r3)

#calc
a = tf.constant([[1,2],[3,4]])
b = tf.constant([[1,1],[1,1]])

c = tf.add(a,b)
print(c)
c = tf.multiply(a,b)
print(c)
c = tf.matmul(a,b)
print(c)


print(10*'-','end',10*'-')

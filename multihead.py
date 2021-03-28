import tensorflow as tf

#############################################
#
# Scaled Dot Product Attention
#   Attention(Q, K, V) = softmax( Q*K.T / sqrt(d)) * V
#
def scaled_dot_product_attention(q, k, v):
 	# Q * K.T
	matmul_qk = tf.matmul(q, k, transpose_b=True)

	# Q*K.T / sqrt(d)
	dk = tf.cast(tf.shape(k)[-1], tf.float32)
	scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

	# softmax( Q*K.T / sqrt(d) )
	attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

	# softmax( Q*K.T / sqrt(d) ) * V
	output = tf.matmul(attention_weights, v)

	return output, attention_weights

#############################################
#
#	Multi Head Attention
#
class MultiHeadAttention(tf.keras.layers.Layer):
	def __init__(self, d_model, num_heads):
		#print('__init__')
		super(MultiHeadAttention, self).__init__()
		self.num_heads = num_heads
		self.d_model   = d_model

		self.depth = d_model

		self.wq = tf.keras.layers.Dense(d_model)
		self.wk = tf.keras.layers.Dense(d_model)
		self.wv = tf.keras.layers.Dense(d_model)	

		self.dense = tf.keras.layers.Dense(d_model)

#		print(self.dense)

	def split_heads(self, x, batch_size):
		#print('split_heads')
		x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
		return tf.transpose(x, perm=[0, 2, 1, 3])

	def call(self, v, k, q):
		print('call')
		batch_size = tf.shape(q)[0]

		q = self.wq(q)
		k = self.wk(k)
		v = self.wv(v)
		#print(q)

		q = self.split_heads(q, batch_size)
		k = self.split_heads(k, batch_size)
		v = self.split_heads(v, batch_size)
		#print(k)

		#############################################
		#
		# Scaled-Dot Ptoduct Attention
		#
		scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v)
		#print(scaled_attention)
		#print(attention_weights)

		concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
		#print(concat_attention)

		output = self.dense(concat_attention)
		#print(output)

		return output, attention_weights

#############################################
#
# Test "MultiHeadAttention class" method
#
def main():	
	mha = MultiHeadAttention(d_model=512, num_heads=8)

	y = tf.random.uniform((1,64,512))
	#print(y)

	output, attention_weights = mha(y, k=y, q=y)

	print('------------------')
	print(attention_weights)
	print('------------------')
	print(output)

#############################################
#
if __name__ == '__main__':
	main()


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import scipy

# Función para inicializar pesos con una distribución normal truncada
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

# Función para inicializar sesgos con un valor constante
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# Función para realizar una convolución 2D
def conv2d(x, W, stride):
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

# Placeholders para las entradas, etiquetas y probabilidad de dropout
x = tf.placeholder(tf.float32, shape=[None, 66, 200, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 1])
keep_prob = tf.placeholder(tf.float32)

# Reshape de la entrada a la forma de imagen
x_image = x

# Capa 1: Convolución + ELU + Normalización por lotes
W_conv1 = weight_variable([5, 5, 3, 24])
b_conv1 = bias_variable([24])
h_conv1 = tf.nn.elu(conv2d(x_image, W_conv1, stride=2) + b_conv1)
h_conv1 = tf.keras.layers.BatchNormalization()(h_conv1, training=True)

# Capa 2: Convolución + ELU + Normalización por lotes
W_conv2 = weight_variable([5, 5, 24, 36])
b_conv2 = bias_variable([36])
h_conv2 = tf.nn.elu(conv2d(h_conv1, W_conv2, stride=2) + b_conv2)
h_conv2 = tf.keras.layers.BatchNormalization()(h_conv2, training=True)

# Capa 3: Convolución + ELU + Normalización por lotes
W_conv3 = weight_variable([5, 5, 36, 48])
b_conv3 = bias_variable([48])
h_conv3 = tf.nn.elu(conv2d(h_conv2, W_conv3, stride=2) + b_conv3)
h_conv3 = tf.keras.layers.BatchNormalization()(h_conv3, training=True)

# Capa 4: Convolución + ELU + Normalización por lotes
W_conv4 = weight_variable([3, 3, 48, 64])
b_conv4 = bias_variable([64])
h_conv4 = tf.nn.elu(conv2d(h_conv3, W_conv4, stride=1) + b_conv4)
h_conv4 = tf.keras.layers.BatchNormalization()(h_conv4, training=True)

# Capa 5: Convolución + ELU + Normalización por lotes
W_conv5 = weight_variable([3, 3, 64, 64])
b_conv5 = bias_variable([64])
h_conv5 = tf.nn.elu(conv2d(h_conv4, W_conv5, stride=1) + b_conv5)
h_conv5 = tf.keras.layers.BatchNormalization()(h_conv5, training=True)

# Aplanar la salida de las capas convolucionales
h_conv5_flat = tf.reshape(h_conv5, [-1, 9*25*64])

# Capa completamente conectada 1: Dense + ELU + Dropout
W_fc1 = weight_variable([9*25*64, 1164])
b_fc1 = bias_variable([1164])
h_fc1 = tf.nn.elu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Capa completamente conectada 2: Dense + ELU + Dropout
W_fc2 = weight_variable([1164, 100])
b_fc2 = bias_variable([100])
h_fc2 = tf.nn.elu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

# Capa completamente conectada 3: Dense + ELU + Dropout
W_fc3 = weight_variable([100, 50])
b_fc3 = bias_variable([50])
h_fc3 = tf.nn.elu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

# Capa de salida: Dense sin activación
W_out = weight_variable([50, 1])
b_out = bias_variable([1])
y = tf.identity(tf.matmul(h_fc3_drop, W_out) + b_out)
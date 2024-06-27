import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.core.protobuf import saver_pb2
import driving_data
import model
import numpy as np

# Directorio para guardar el modelo entrenado
LOGDIR = './save'

# Iniciar una sesión interactiva de TensorFlow
sess = tf.InteractiveSession()

# Hiperparámetro que se puede ajustar (Constante L2 para la regularización)
##### SE PUEDE CAMBIAR #####
L2NormConst = 0.0001
############################

# Obtener las variables entrenables del modelo
train_vars = tf.trainable_variables()

# Función de pérdida que combina el error cuadrático medio entre la salida deseada (y_) 
# y la salida del modelo (y) con la regularización L2 sobre los pesos del modelo
loss = tf.reduce_mean(tf.square(tf.subtract(model.y_, model.y))) + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst

# Parámetros para el aprendizaje adaptativo (learning rate decay)
##### SE PUEDE CAMBIAR #####
initial_learning_rate = 1e-3  # Tasa de aprendizaje inicial (1e-3 = 0.001)
lr_decay = 0.95                 # Factor de decrecimiento de la tasa de aprendizaje
global_step = tf.Variable(0, trainable=False)  # Variable para contabilizar los pasos de entrenamiento
learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, 1000, lr_decay, staircase=True)
# learning_rate decrece exponencialmente con el número de pasos de entrenamiento
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
# Optimizador Adam que minimiza la función de pérdida para actualizar los pesos del modelo

# Inicializar las variables globales del grafo de cómputo
sess.run(tf.global_variables_initializer())

# Definir un summary para el calculo de pérdida 
tf.summary.scalar("loss", loss)

# Combinar todos los summaries en una única operación 
merged_summary_op = tf.summary.merge_all()

# Guardador para persistir el modelo entrenado
saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V2)

# Directorio para guardar los summaries de TensorBoard (no usado en este caso)
logs_path = './logs'
summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

# Hiperparámetros que se pueden ajustar (número de épocas y tamaño de lote)
##### SE PUEDE CAMBIAR #####
epochs = 30  # Número de épocas de entrenamiento
batch_size = 64  # Tamaño del lote de datos para entrenamiento

# Implementación básica de early stopping
best_val_loss = float('inf')  # Inicializar mejor pérdida de validación con infinito positivo
patience = 5                  # Número máximo de épocas sin mejora en la validación para detener el entrenamiento
wait = 0                       # Contador de épocas sin mejora en la validación


# Entrenamiento por épocas
for epoch in range(epochs):  # Recorre el número especificado de épocas
  
  # Bucle sobre lotes de datos de entrenamiento
  for i in range(int(driving_data.num_images/batch_size)):  # Recorre todos los lotes de entrenamiento
    # Carga un lote de entrenamiento (imágenes y ángulos de volante deseados)
    xs, ys = driving_data.LoadTrainBatch(batch_size)

    # Realiza un paso de entrenamiento actualizando los pesos del modelo para minimizar la pérdida
    train_step.run(feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 0.8})  # Ejecuta un paso de entrenamiento

    # Imprime información de pérdida cada cierto número de lotes
    if i % 10 == 0:  # Cada 10 lotes
      # Carga un lote de validación (imágenes y ángulos de volante deseados)
      xs, ys = driving_data.LoadValBatch(batch_size)
      # Evalúa la pérdida en el lote de validación
      loss_value = loss.eval(feed_dict={model.x:xs, model.y_: ys, model.keep_prob: 1.0})
      # Imprime información de la época, el paso, y la pérdida de validación
      print("Epoch: %d, Paso: %d, Loss: %g" % (epoch, epoch * batch_size + i, loss_value))

      summary = merged_summary_op.eval(feed_dict={model.x:xs, model.y_: ys, model.keep_prob: 1.0})
      summary_writer.add_summary(summary, epoch * driving_data.num_images/batch_size + i)

    # Guarda el modelo cada cierto número de lotes
    if i % batch_size == 0:  # Cada cierto número de lotes
      # Crea el directorio para guardar el modelo si no existe
      if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
      # Define la ruta completa del archivo del modelo
      checkpoint_path = os.path.join(LOGDIR, "model.ckpt")
      # Guarda el modelo entrenado en el archivo
      filename = saver.save(sess, checkpoint_path)

  # Validación al final de cada época
  val_loss = 0  # Variable para acumular la pérdida de validación
  val_batches = 0  # Contador de lotes de validación procesados
  for j in range(int(driving_data.num_val_images/batch_size)):  # Recorre todos los lotes de validación
    # Carga un lote de validación (imágenes y ángulos de volante deseados)
    xs, ys = driving_data.LoadValBatch(batch_size)
    # Evalúa la pérdida en el lote de validación y la suma al acumulador
    val_loss += loss.eval(feed_dict={model.x:xs, model.y_: ys, model.keep_prob: 1.0})
    val_batches += 1  # Incrementa el contador de lotes procesados
  # Calcula la pérdida de validación promedio
  val_loss /= val_batches

  # Imprime información de la época y la pérdida de validación promedio
  print("Epoch: %d, Validation Loss: %g" % (epoch, val_loss))

  # Early stopping (parada temprana)
  if val_loss < best_val_loss:  # Si la pérdida de validación mejora
    best_val_loss = val_loss  # Actualiza la mejor pérdida de validación
    wait = 0  # Reinicia el contador de espera
  else:
    wait += 1  # Incrementa el contador de espera si no mejora la validación
    if wait >= patience:  # Si se supera el número máximo de épocas sin mejora
      print("Early stopping triggered. Stopping training.")
      break  # Detener el entrenamiento

# Imprime mensaje de confirmación al guardar el modelo final
print("Modelo guardado en archivo: %s" % filename)
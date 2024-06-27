import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import model
import cv2
import numpy as np
from subprocess import call
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Definir los bins de ángulos y etiquetas
angle_bins = np.linspace(-90, 90, num=10)
bin_labels = range(len(angle_bins) - 1)

def assign_bin(angle):
    return np.digitize(angle, angle_bins) - 1

def read_true_angles(file_path):
    true_angles = {}
    with open(file_path, 'r') as f:
        for line in f:
            image_file, angle = line.split()
            true_angles[image_file] = float(angle)
    return true_angles

# Leer los ángulos verdaderos
true_angles = read_true_angles("test_dataset/data.txt")

# Verificar si estamos en Windows
windows = False
if os.name == 'nt':
    windows = True

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "trained_model/model.ckpt")

img = cv2.imread('steering_wheel_image.jpg', 0)
rows, cols = img.shape

smoothed_angle = 0

# Inicializar listas para métricas
true_labels = []
pred_labels = []
true_angles_list = []
pred_angles_list = []

i = 0
while(cv2.waitKey(10) != ord('q')):
    image_file = f"{i}.jpg"
    full_image = cv2.imread(f"test_dataset/data/{image_file}")
    if full_image is None:
        break
    
    image = cv2.resize(full_image[-150:], (200, 66)) / 255.0
    degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180.0 / 3.14159265
    
    true_angle = true_angles[image_file]
    
    true_label = assign_bin(true_angle)
    pred_label = assign_bin(degrees)
    
    true_labels.append(true_label)
    pred_labels.append(pred_label)
    true_angles_list.append(true_angle)
    
    
    if not windows:
        call("clear")
    print(f"Angulo predecido: {degrees} grados, Angulo Real: {true_angle} grados")
    
    cv2.imshow("frame", full_image)
    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    
    pred_angles_list.append(smoothed_angle)
    
    M = cv2.getRotationMatrix2D((cols/2,rows/2), -smoothed_angle, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    cv2.imshow("steering wheel", dst)
    i += 1

cv2.destroyAllWindows()

# Calcular métricas
accuracy = accuracy_score(true_labels, pred_labels)
precision = precision_score(true_labels, pred_labels, average='weighted')
recall = recall_score(true_labels, pred_labels, average='weighted')
f1 = f1_score(true_labels, pred_labels, average='weighted')

mse = mean_squared_error(true_angles_list, pred_angles_list)
mae = mean_absolute_error(true_angles_list, pred_angles_list)
r2 = r2_score(true_angles_list, pred_angles_list)

print("\nMétricas de clasificación:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")

print("\nMétricas de regresión:")
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R2 Score: {r2}")

# Calcular el porcentaje de predicciones dentro de ciertos umbrales
thresholds = [5, 10, 15]  # grados
for threshold in thresholds:
    within_threshold = np.sum(np.abs(np.array(true_angles_list) - np.array(pred_angles_list)) < threshold)
    percentage = (within_threshold / len(true_angles_list)) * 100
    print(f"Porcentaje de predicciones dentro de {threshold}°: {percentage:.2f}%")


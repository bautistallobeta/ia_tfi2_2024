# Instrucciones para correr el trainer

## Requerimientos

- Haber instalado correctamente Conda
- Tener una carpeta `/train_dataset` con el dataset para entrenar (Nota: Es importante que las imágenes del dataset tengan de nombre `0.jpg`, `1.jpg`, ..., `999.jpg`). [Link a los datasets sugeridos](https://github.com/SullyChen/driving-datasets).
- **IMPORTANTE**: dentro de la misma carpeta, el dataset debe tener un archivo llamado `data.txt` donde tenga cada foto con el ángulo de giro del volante. Ejemplo:

```
  0.jpg -6.0
  1.jpg -4.0
  2.jpg -3.9
  .
  .
  .
  99.jpg 0.5
```

- Tener la arquitectura del modelo en formato '.py' en la raíz de la carpeta Trainer con el nombre `model.py`.

## Uso del entrenador

### Local

- Utilizando conda, navegar hasta la carpeta `/Trainer`
- Instalar los requerimientos del entrenador utilizando el comando `pip install -r requirements.txt`.
- correr el comando `python train.py`.
- una vez terminado el proceso, copiar el archivo `model.ckpt` y la arquitectura del modelo `model.py` y llevarlas al runner para testear tu modelo.

### En Google Colab

- Seguir las instrucciones del laboratorio.
  > 💡 **Tip**:
  > Aparte de cambiar la arquitectura de la red, podes cambiar también los hiperparámetros del entrenamiento en el archivo './train.py' tal y como hiciste en el TFI pasado. Buscá los hiperparámetros delimitados en el archivo.

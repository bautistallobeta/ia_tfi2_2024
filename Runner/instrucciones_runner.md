# Instrucciones para correr el runner
## Requerimientos
* Haber instalado correctamente Conda
* Tener una carpeta `/trained_model` dentro de la carpeta del Runner, y que dentro contenga el modelo en tensorflow (en formato checkpoint `.ckpt`).
* Tener una carpeta `/test_dataset` con el dataset para probar el simulador (Nota: Es importante que las imágenes del dataset tengan de nombre `0.jpg`, `1.jpg`, ..., `999.jpg` para que el simulador funcione correctamente), en caso de utilizar un conjunto de imágenes, o un video en formato MP4 llamado `test_video.mp4` en caso de utilizar un video.
* Tener la arquitectura del modelo en formato '.py' en la raíz de la carpeta Runner con el nombre `model.py`.
## Uso del simulador
* Utilizando conda, navegar hasta la carpeta `/Runner`
* Instalar los requerimientos del simulador utilizando el comando `pip install -r requirements.txt`
* Elegir la opción a utilizar:

  1. Correr el comando `python run_video_tf` para testear el modelo con un video.
  2. Correr el comando `python run_dataset_tf` para testear el modelo  con un conjunto de imágenes.

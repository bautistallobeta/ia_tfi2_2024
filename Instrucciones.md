# Instrucciones de uso
## El proyecto
El proyecto consiste de 2 carpetas:
* La carpeta `/Trainer` contiene los archivos para entrenar la red convolucional.
* La carpeta `/Runner` contiene los archivos para correr el "simulador" de conducción.

Si tu computadora posee una placa de video de al menos 4gb, podes entrenar tu modelo localmente. En caso de que no, podés subir la carpeta en Google Colab o un Kernel de Kaggle para entrenar, y luego descargar el modelo y pegarlo en la carpeta correspondiente.

En cuanto al simulador de conducción, si o si necesitamos correrlo local.

## Requisitos
* Descargá e instalá Miniconda desde aqui: [Miniconda](https://docs.anaconda.com/free/miniconda/).
* Seguí los pasos de instalación.
* Una vez instalado, buscá en tus aplicaciones por el siguiente programa: `Anaconda Prompt (miniconda)`.
* Creamos un entorno utilizando el comando `conda create -n nombre_del_entorno`.
* Luego de creado el entorno, podés utilizar `conda activate nombre_del_entorno` para activar ese entorno.
* No hace falta volver a crear el entorno después de dejarlo de usar. Si no recordás como se llama el entorno, podes buscarlo con `conda env list`.
* Una vez cargado el entorno (Te vas a dar cuenta ya que a la izquierda de la linea de comandos vas a ver el nombre de tu entorno entre paréntesis, algo asi: `(base) C:\Users\emusa>`).
* Por último, navegá utilizando `cd` hacia la carpeta donde quieras trabajar, ya sea el Trainer o el Runner.
* Seguí las instrucciones de cada carpeta.
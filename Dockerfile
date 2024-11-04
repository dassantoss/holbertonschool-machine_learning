# Usar la imagen base de Ubuntu 20.04
FROM ubuntu:20.04

# Evitar preguntas durante la instalación de paquetes
ARG DEBIAN_FRONTEND=noninteractive

# Instalar Python, pip y las dependencias necesarias
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-tk \
    xvfb \
    x11vnc \
    x11-xserver-utils \
    xterm \
    libx11-6 \
    libx11-dev \
    libxft2 \
    libxft-dev \
    libxext-dev \
    xauth \
    libgomp1 \
    cmake \
    ninja-build \
    patchelf \
    build-essential \
    cython3 \
    gfortran \
    libhdf5-dev \
    python3-dev \
    python3-h5py \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Establecer la versión de Python deseada y asegurar que es la predeterminada
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Instalar las versiones especificadas de Numpy, TensorFlow, Keras y demás dependencias
RUN pip3 install numpy==1.24.3 matplotlib Pillow==10.3.0 pycodestyle==2.11.1 pandas

# Instalar versiones anteriores de TensorFlow y Keras compatibles con keras-rl2
RUN pip3 install tensorflow==2.10.0 keras==2.10.0

# Instalar scikit-learn y scipy
RUN pip3 install scikit-learn scipy==1.10.1

# Actualizar h5py a la versión especificada
RUN pip3 install h5py==3.11.0

# Instalar Tensorflow Hub, Transformers y TensorFlow Datasets
RUN pip3 install tensorflow-hub==0.15.0 transformers==4.44.2 tensorflow-datasets==4.9.2

# Instalar la versión específica de Gensim
RUN pip3 install gensim==4.3.3

# Instalar Gymnasium y otras dependencias, excluyendo `atari` para evitar conflictos
RUN pip3 install gymnasium==0.29.1 gymnasium[other] \
    keras-rl2==1.0.4 autorom[accept-rom-license]

# Establecer variables de entorno para solucionar problemas de OpenMP y TLS block
ENV OMP_NUM_THREADS=1
ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
ENV SKLEARN_NO_OPENMP=TRUE

# Configurar el directorio de trabajo en /app
WORKDIR /app

# Copiar todo el contenido del proyecto al contenedor
COPY . /app

# Exponer la variable DISPLAY para conectar con el servidor X
ENV DISPLAY=:0

# Comando por defecto para iniciar el servidor X virtual
CMD ["Xvfb", ":99", "-screen", "0", "1024x768x24", "-ac", "+extension", "GLX", "+render", "-noreset"]

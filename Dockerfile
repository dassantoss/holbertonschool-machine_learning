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

# Instalar Numpy compatible con TensorFlow
RUN pip3 install numpy==1.24.3

# Instalar Matplotlib y Pillow
RUN pip3 install matplotlib Pillow

# Instalar pycodestyle
RUN pip3 install pycodestyle==2.11.1

# Instalar scikit-learn utilizando binarios precompilados
RUN pip3 install scikit-learn

# Instalar scipy versión 1.10.1
RUN pip3 install scipy==1.10.1

# Reinstalar h5py para asegurar compatibilidad con HDF5
RUN pip3 install --force-reinstall h5py

# Instalar TensorFlow con mayor tiempo de espera
RUN pip3 install --default-timeout=100 tensorflow

# Instalar la versión específica de Gensim
RUN pip3 install gensim==4.3.3

# Verificar e instalar la versión específica de Keras
RUN pip3 install keras==2.13.1

# Instalar pandas
RUN pip3 install pandas

# Instalar Tensorflow Hub
RUN pip3 install tensorflow-hub==0.15.0

# Instalar Transformers versión 4.44.2
RUN pip3 install transformers==4.44.2

# Instalar TensorFlow Datasets (TFDS) versión 4.9.2
RUN pip3 install tensorflow-datasets==4.9.2

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

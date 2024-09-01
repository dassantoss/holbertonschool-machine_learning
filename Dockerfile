FROM ubuntu:20.04

# Evitar preguntas durante la instalación de paquetes
ARG DEBIAN_FRONTEND=noninteractive

# Instalar Python y pip junto con dependencias necesarias para la GUI
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
    xauth

# Limpiar cache de apt-get
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Establecer la versión de Python deseada y asegurar que es la predeterminada
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Instalar Numpy compatible con TensorFlow
RUN pip3 install numpy==1.24.3

# Instalar Matplotlib y Pillow
RUN pip3 install matplotlib Pillow

# Instalar pycodestyle
RUN pip3 install pycodestyle==2.11.1

# Instalar scikit-learn versión 1.3.2
RUN pip3 install scikit-learn==1.3.2

# Instalar scipy versión 1.10.1
RUN pip3 install scipy==1.10.1

# Instalar HDF5 y sus dependencias
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    python3-dev \
    python3-h5py

# Reinstalar h5py para asegurar compatibilidad con HDF5
RUN pip3 install --force-reinstall h5py

# Instalar TensorFlow con mayor tiempo de espera
RUN pip3 install --default-timeout=100 tensorflow

# Configurar el directorio de trabajo en /app
WORKDIR /app

# Exponer la variable DISPLAY para conectar con el servidor X
ENV DISPLAY=:0

# Iniciar un servidor X virtual al correr el contenedor
CMD ["Xvfb", ":99", "-screen", "0", "1024x768x24", "-ac", "+extension", "GLX", "+render", "-noreset"]

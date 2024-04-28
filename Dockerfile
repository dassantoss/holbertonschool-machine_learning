FROM ubuntu:20.04

# Evitar preguntas durante la instalación de paquetes
ARG DEBIAN_FRONTEND=noninteractive

# Instalar Python y pip junto con dependencias necesarias para la GUI
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-tk \
    # Estas son para el soporte de GUI
    xvfb \
    x11vnc \
    x11-xserver-utils \
    xterm \
    # Instalar librerías básicas para Matplotlib
    libx11-6 \
    libx11-dev \
    libxft2 \
    libxft-dev \
    libxext-dev

# Limpiar cache de apt-get
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Instalar Matplotlib y Pillow sin especificar la versión
RUN pip3 install matplotlib Pillow

# Establecer la versión de Python deseada y asegurar que es la predeterminada
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Instalar Numpy
RUN pip3 install numpy

# Instalar pycodestyle
RUN pip3 install pycodestyle==2.11.1

# Instalar scikit-learn
RUN pip3 install -U scikit-learn

# Configurar el directorio de trabajo en /app
WORKDIR /app

# Configurar un display virtual para gráficos
ENV DISPLAY=192.168.1.99:0

# Iniciar un servidor X virtual al correr el contenedor
CMD ["Xvfb", ":99", "-screen", "0", "1024x768x24", "-ac", "+extension", "GLX", "+render", "-noreset"]

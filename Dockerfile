FROM ubuntu:20.04

# Instalar Python y pip
RUN apt-get update && apt-get install -y python3 python3-pip

# Establecer la versión de Python deseada y asegurar que es la predeterminada
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Instalar Numpy
RUN pip3 install numpy

# Instalar pycodestyle
RUN pip3 install pycodestyle==2.11.1

# Configurar el directorio de trabajo en /app
WORKDIR /app

# Mantener el contenedor corriendo
CMD ["tail", "-f", "/dev/null"]

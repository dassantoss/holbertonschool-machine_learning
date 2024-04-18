
#!/usr/bin/env python3
"""
Este módulo grafica la relación entre el número atómico y la masa atómica.
"""
import matplotlib.pyplot as plt


def graficar_relacion_atomico_masa():
    """
    Esta función crea un gráfico que muestra la relación entre el número atómico
    y la masa atómica de los primeros 20 elementos de la tabla periódica.
    """
    # Paso 1: Definir los números atómicos de los primeros 20 elementos
    numeros_atomicos = list(range(1, 21))

    # Paso 2: Definir las masas atómicas de los primeros 20 elementos (valores estándar).
    masas_atomicas = [
        1.008, 4.0026, 6.94, 9.0122, 10.81, 12.011, 14.007, 15.999,
        18.998, 20.180, 22.990, 24.305, 26.982, 28.085, 30.974, 32.06,
        35.45, 39.948, 39.0983, 40.078
    ]

    # Paso 3: Crear la figura y el eje con matplotlib
    plt.figure(figsize=(10, 5))

    # Paso 4: Dibujar puntos y conectarlos con una línea
    plt.plot(numeros_atomicos, masas_atomicas, marker='o', linestyle='-')

    # Paso 5: Agregar títulos y etiquetas
    plt.title('Relación entre el número atómico y la masa atómica')
    plt.xlabel('Número atómico')
    plt.ylabel('Masa atómica (u)')

    # Paso 6: Mostrar una cuadrícula para facilitar la lectura
    plt.grid(True)

    # Paso 7: Mostrar el gráfico
    plt.show()

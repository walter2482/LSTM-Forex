import os
import subprocess

# Verificar si Ta-Lib ya está instalado
try:
    import talib
    print("Ta-Lib ya está instalado.")
except ImportError:
    print("Instalando Ta-Lib...")
    
    # Descargar el archivo fuente de TA-Lib
    subprocess.run(["wget", "http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz"])

    # Descomprimir el archivo
    subprocess.run(["tar", "-xzvf", "ta-lib-0.4.0-src.tar.gz"])

    # Acceder al directorio de TA-Lib
    os.chdir("ta-lib")

    # Ejecutar el comando de configuración
    subprocess.run(["./configure", "--prefix=/usr"])

    # Compilar e instalar TA-Lib
    subprocess.run(["make"])
    subprocess.run(["make", "install"])

    # Instalar el paquete de Python Ta-Lib
    subprocess.run(["pip", "install", "Ta-Lib"])

# Verificar si pandas_ta ya está instalado
try:
    import pandas_ta
    print("pandas_ta ya está instalado.")
except ImportError:
    print("Instalando pandas_ta...")
    
    # Instalar el paquete de Python pandas_ta
    subprocess.run(["pip", "install", "pandas_ta"])

# Verificar si keras_tuner ya está instalado
try:
    import keras_tuner
    print("keras_tuner ya está instalado.")
except ImportError:
    print("Instalando keras_tuner...")
    
    # Instalar el paquete de Python keras_tuner
    subprocess.run(["pip", "install", "keras_tuner"])

# Instalar tensorflow-addons
subprocess.run(["pip", "install", "tensorflow-addons==0.16.1"])

# Instalar keras-self-attention
subprocess.run(["pip", "install", "keras-self-attention"])

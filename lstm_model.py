# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 16:22:54 2023

@author: Walter
"""


# Funcionalidad de manejo de fechas y tiempo
import datetime
import time
from datetime import date
from dateutil.relativedelta import relativedelta

# Funcionalidad de procesamiento de datos y cálculos
import numpy as np
import pandas as pd
import pandas_ta as ta
import talib

# Funcionalidad de visualización
import matplotlib.pyplot as plt

# Funcionalidad de manejo de archivos y directorios
import glob
import os
from subprocess import call

# Funcionalidad de machine learning y deep learning
import tensorflow as tf
import keras_tuner
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from keras.models import load_model

# Funcionalidad de seguimiento y notificaciones
from tqdm import tqdm

# Funcionalidad de manejo de advertencias
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


# Definir el callback para mostrar la pérdida y las métricas por época
class LossMetricsCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        print(f'Época {epoch + 1}: Pérdida = {logs["loss"]}, Val. Pérdida = {logs["val_loss"]}')


def clean_folder(folder: str) -> None:
    """
    Elimina todos los archivos con extensión .csv dentro de la carpeta especificada.

    Parámetros:
    - folder (str): Ruta de la carpeta en la que se eliminarán los archivos .csv.

    Retorna:
    None
    """
    files = glob.glob(folder + "*.csv")
    for f in files:
        os.remove(f)
    return None


def get_data(list_instrument=list, init_day=str, finish_day=str,
             folder_data=str, timeframe=str, format=str):
    """
    Obtiene datos históricos de instrumentos financieros utilizando la herramienta Dukascopy.

    Parámetros:
        - list_instrument (list): Lista de instrumentos financieros para los que se obtendrán los datos.
        - init_day (str): Fecha de inicio en formato 'YYYY-MM-DD'.
        - finish_day (str): Fecha de finalización en formato 'YYYY-MM-DD'.
        - folder_data (str): Ruta de la carpeta donde se guardarán los datos descargados.
        - timeframe (str): Marco de tiempo de los datos ('tick', '1min', '5min', '15min', '30min', '1hour', '1day', '1week', '1month').
        - format (str): Formato de los datos ('csv', 'json', 'bin', 'parquet').

    Retorna:
        None
    """
    for i in tqdm(range(len(list_instrument))):
        command = f"npx dukascopy-node -i {list_instrument[i]} -from {init_day} -to {finish_day} -t {time_frame} -f {format} -dir {folder_data} -v {volume} -r 10 -bs 1"
        call(command, shell=True)
        time.sleep(5)

    return None


def read_data(folder_data: str) -> dict:
    """
    Lee y procesa los datos de archivos CSV en una carpeta dada.

    Args:
        folder_data (str): Ruta de la carpeta que contiene los archivos CSV.

    Returns:
        dict: Diccionario de instrumentos con los datos procesados. Cada clave es el nombre del instrumento y el valor es un DataFrame de pandas.

    Raises:
        FileNotFoundError: Si la carpeta especificada no existe.

    """

    dict_instruments = {}  # Diccionario para almacenar los datos de los instrumentos
    os.chdir(folder_data)  # Cambiar el directorio de trabajo al de la carpeta especificada
    # Obtener la lista de archivos CSV en la carpeta
    files = [file for file in os.listdir(folder_data) if file != ".ipynb_checkpoints"]

    for file in files:
        # Obtener el nombre del instrumento a partir del nombre del archivo
        instrument_name = file.split('-')[0]
        # Leer el archivo CSV y cargarlo en un DataFrame de pandas
        df = pd.read_csv(folder_data + file)
        columns = {
            'close': f"close_{instrument_name}",
            'high': f"high_{instrument_name}",
            'low': f"low_{instrument_name}",
            'open': f"open_{instrument_name}",
            'volume': f"volume_{instrument_name}"
        }
        # Renombrar las columnas del DataFrame según el instrumento
        df.rename(columns=columns, inplace=True)
        # Convertir la columna de timestamps a formato de fecha y hora
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
        # Establecer la columna de timestamps como índice del DataFrame
        df.set_index("timestamp", inplace=True)
        # Agregar el DataFrame al diccionario de instrumentos
        dict_instruments[instrument_name] = df

    return dict_instruments


def concat_df(dict_instruments: dict, interest_instrument: str) -> pd.DataFrame:
    """
    Concatena los DataFrames de instrumentos contenidos en un diccionario.

    Args:
        dict_instruments (dict): Diccionario de instrumentos donde cada clave es el nombre del instrumento y el valor es un DataFrame de pandas.
        interest_instrument (str): Nombre del instrumento de interés.

    Returns:
        pd.DataFrame: DataFrame resultante después de la concatenación de los DataFrames de instrumentos.

    """

    df = pd.concat([dict_instruments[key].drop(
        [f"open_{key}"], axis=1) if interest_instrument not in key else dict_instruments[key] for key in dict_instruments], axis=1)
    df.dropna(inplace=True)  # Eliminar filas con valores nulos en el DataFrame resultante
    return df


def multirame_indicator(dataframe, indicator=str, n_indicators=int, base_period=int, ins_interest=str):
    """
    Calcula múltiples indicadores técnicos y los añade como columnas adicionales en el DataFrame.

    Args:
        dataframe (pd.DataFrame): DataFrame de entrada.
        indicator (str): Nombre del indicador técnico a calcular.
        n_indicators (int): Número de indicadores a calcular.
        base_period (int): Periodo base para el cálculo de los indicadores.
        ins_interest (str): Nombre del instrumento de interés.

    Returns:
        pd.DataFrame: DataFrame con los indicadores técnicos calculados añadidos como columnas adicionales.

    """

    for i in range(n_indicators):
        period = base_period + 5 * i
        if indicator == "ADX":
            dataframe[f'ADX_{i}'] = talib.ADX(
                dataframe[f"close_{ins_interest}"], dataframe[f"high_{ins_interest}"], dataframe[f"low_{ins_interest}"], timeperiod=period)
        elif indicator == "CCI":
            dataframe[f'CCI_{i}'] = talib.CCI(
                dataframe[f"close_{ins_interest}"], dataframe[f"high_{ins_interest}"], dataframe[f"low_{ins_interest}"], timeperiod=period)
        elif indicator == "ROC":
            dataframe[f'ROC_{i}'] = talib.ROCR100(
                dataframe[f"close_{ins_interest}"], timeperiod=period)
        elif indicator == "RSI":
            dataframe[f'RSI_{i}'] = talib.RSI(dataframe[f"close_{ins_interest}"], timeperiod=period)
        elif indicator == "MFI":
            dataframe[f'MFI_{i}'] = talib.MFI(dataframe[f"high_{ins_interest}"], dataframe[f"low_{ins_interest}"],
                                              dataframe[f"close_{ins_interest}"], dataframe[f"volume_{ins_interest}"], timeperiod=period)
        elif indicator == "WILLR":
            dataframe[f'WILLR_{i}'] = talib.WILLR(
                dataframe[f"high_{ins_interest}"], dataframe[f"low_{ins_interest}"], dataframe[f"close_{ins_interest}"], timeperiod=period)
        elif indicator == "AROON":
            dataframe[f'AROON_a_{i}'], dataframe[f'AROON_b_{i}'] = talib.AROON(
                dataframe[f"high_{ins_interest}"], dataframe[f"low_{ins_interest}"], timeperiod=period)
        elif indicator == "TEMA":
            dataframe[f'TEMA_{i}'] = talib.TEMA(
                dataframe[f"close_{ins_interest}"], timeperiod=period)
        elif indicator == "NATR":
            dataframe[f'NATR_{i}'] = talib.NATR(
                dataframe[f"high_{ins_interest}"], dataframe[f"low_{ins_interest}"], dataframe[f"close_{ins_interest}"], timeperiod=period)
        elif indicator == "TEMA":
            dataframe[f'TEMA_{i}'] = talib.TEMA(
                dataframe[f"close_{ins_interest}"], timeperiod=period)
    return dataframe


def correlations(dataframe: pd.DataFrame, interes_instrument: str, periods: int) -> pd.DataFrame:
    """
    Calcula las correlaciones entre el instrumento de interés y otros instrumentos en el DataFrame.

    Args:
        dataframe (pd.DataFrame): DataFrame de entrada.
        interes_instrument (str): Nombre del instrumento de interés.
        periods (int): Número de períodos para el cálculo de la correlación.

    Returns:
        pd.DataFrame: DataFrame con las correlaciones calculadas añadidas como columnas adicionales.

    """

    df = dataframe.copy()  # Copia el DataFrame para evitar modificar el original

    for column in df.columns:
        if column.startswith("close_") and column != f"close_{interes_instrument}":
            other_instrument = column.split("_")[1]
            correlation = talib.CORREL(df[f"close_{interes_instrument}"], df[column], periods)
            df[f"CORR_{interes_instrument}_{other_instrument}"] = correlation
    return df


def derivative_df(dataframe, interest_intrument=str):
    """
    Calcula las diferencias entre los precios de cierre de otros instrumentos y los agrega al DataFrame.

    Args:
        dataframe (pd.DataFrame): DataFrame de entrada.
        interest_instrument (str): Nombre del instrumento de interés.

    Returns:
        pd.DataFrame: DataFrame con las diferencias calculadas añadidas como columnas adicionales.

    """
    for column in dataframe:
        if ("close_" in column) and (column != "close_" + interest_intrument):
            other_instrument = column.split("_")[1]
            df[f"diff_{other_instrument}"] = df[f"close_{other_instrument}"].diff()
        else:
            None
    return df


def VOLATILITY_VOLUME_MOMENTUM(dataframe, interest_intrument=str,
                               mfi_period=int):
    """
    Calcula indicadores de volumen, momentum y volatilidad para instrumentos distintos al de interés
    y los agrega al DataFrame.

    Args:
        dataframe (pd.DataFrame): DataFrame de entrada.
        interest_instrument (str): Nombre del instrumento de interés.
        mfi_period (int): Período para el indicador MFI.

    Returns:
        pd.DataFrame: DataFrame con los indicadores calculados añadidos como columnas adicionales.

    """

    for column in dataframe:
        if ("close_" in column) and (column != ("close_" +
                                     interest_intrument)):

            other_instrument = column.split("_")[1]

            # Volume
            dataframe[f"AD_{other_instrument}"] = talib.AD(
                dataframe[f"high_{other_instrument}"],
                dataframe[f"low_{other_instrument}"],
                dataframe[f"close_{other_instrument}"],
                dataframe[f"volume_{other_instrument}"])

            # Other momentum
            dataframe[f'MFI_{other_instrument}'] = talib.MFI(
                dataframe[f"high_{other_instrument}"],
                dataframe[f"low_{other_instrument}"],
                dataframe[f"close_{other_instrument}"],
                dataframe[f"volume_{other_instrument}"],
                timeperiod=mfi_period)

            # Volatility
            dataframe[f'TRANGE_{other_instrument}'] = talib.TRANGE(
                dataframe[f"high_{other_instrument}"],
                dataframe[f"low_{other_instrument}"],
                dataframe[f"close_{other_instrument}"])
        else:
            None
    return dataframe


def drop_price(dataframe, interest_intrument=str):
    """
    Elimina las columnas de precio (close, low, high) de instrumentos distintos al de interés
    del DataFrame.

    Args:
        dataframe (pd.DataFrame): DataFrame de entrada.
        interest_instrument (str): Nombre del instrumento de interés.

    Returns:
        pd.DataFrame: DataFrame con las columnas de precio eliminadas.

    """
    for column in dataframe:
        if ("close_" in column) and (column != "close_" + interest_intrument):
            other_instrument = column.split("_")[1]
            dataframe = dataframe.drop(columns=[f"close_{other_instrument}"])

        if ("low_" in column) and (column != "low_" + interest_intrument):
            other_instrument = column.split("_")[1]
            dataframe = dataframe.drop(columns=[f"low_{other_instrument}"])
        if ("high_" in column) and (column != "high_" + interest_intrument):
            other_instrument = column.split("_")[1]
            dataframe = dataframe.drop(columns=[f"high_{other_instrument}"])
        else:
            None
    return dataframe


def architecture_model(hp):
    """
    Crea un modelo de arquitectura LSTM con hiperparámetros ajustables utilizando Keras Tuner.

    Args:
        hp (keras_tuner.HyperParameters): Hiperparámetros ajustables.

    Returns:
        tensorflow.keras.Model: Modelo de arquitectura LSTM con GRU en la capa de entrada.

    """
    model = keras.Sequential()

    model.add(
        GRU(
            hp.Int('input_units', min_value=32, max_value=512, step=32),
            return_sequences=True,
            activation="tanh",
            input_shape=(x_train.shape[1], x_train.shape[2])
        )
    )

    for i in range(hp.Int('n_layers', 1, 5)):
        model.add(
            Dropout(
                hp.Float(
                    f'dropout_rate_{i}',
                    min_value=0,
                    max_value=0.5,
                    step=0.1
                )
            )
        )

        model.add(
            LSTM(
                hp.Int(
                    f'lstm_{i}_units',
                    min_value=32,
                    max_value=512,
                    step=32
                ),
                return_sequences=True,
                activation="tanh"
            )
        )

    model.add(
        Dropout(
            hp.Float(
                'dropout_rate_last',
                min_value=0,
                max_value=0.5,
                step=0.1
            )
        )
    )

    model.add(
        LSTM(
            hp.Int('last_layer', min_value=16, max_value=512, step=32)
        )
    )

    model.add(Dense(y_train.shape[1], activation="tanh"))
    model.compile(loss='mae', optimizer='SGD', metrics=['mae', 'mse', 'mape'])

    return model



if __name__ == "__main__":

    # Get date today
    today = date.today()

    # Apply configuration of today
    finish_day = "now"

    # Year of len dataset
    years_data = 4

    # Calculation of first day
    init_day = today - relativedelta(years=years_data)

    # Folder path
    folder = "/home/walter/forex_data/"

    # Path folder destination of files
    folder_data = f"/home/walter/forex_data/data_{today}/"

    # Check if folder exist, if not create a new folder
    if not os.path.exists(folder_data):
        os.makedirs(folder_data)

    # Periods of dataframe
    time_frame = "h1"

    # Assing volumn to dataframe
    volume = "True"

    # Format of file dataset
    format_file = "csv"

    # Interest instrumen to predict
    interes_intrument = "eurusd"

    # Other instrument to add dataframe
    list_pairs = [
        "eurusd",
        "lightcmdusd",
        "usa500idxusd",
        "usdcad",
        "audusd"]

    # Name of model
    name_model = "modelo_entrenado.h5"

    """
    DATA PROCESSING

    # Clean folder of old files
    # clean_folder(folder_data)

    # Download datasets
    """

    get_data(list_instrument=list_pairs, init_day=init_day,
             finish_day=finish_day,
             folder_data=folder_data,
             timeframe=time_frame,
             format=format_file)

    # Read all files in folder
    dict_df = read_data(folder_data)

    # Create a unique dataframe with concatened all dataframes
    df = concat_df(dict_instruments=dict_df,
                   interest_instrument=interes_intrument)

    """
    PANEL INDICATORS INDICATORS
    """
    # Name of study instrument
    ins_interest = "eurusd"

    # Close-Prices of interest instrument
    close = df[f"close_{interes_intrument}"]

    # Low-Price of interest instrument
    low = df[f"low_{interes_intrument}"]

    # High-Prices of interest instrument
    high = df[f"high_{interes_intrument}"]

    # Open-Prices of interest instrument
    open_price = df[f"open_{interes_intrument}"]

    # Volume-Values of interest instrument
    volume = df[f"volume_{interes_intrument}"]

    # Limpiar datos vacios
    df = df[df[f"close_{interes_intrument}"].notna()]
    df = df[df[f"low_{interes_intrument}"].notna()]
    df = df[df[f"high_{interes_intrument}"].notna()]
    df = df[df[f"open_{interes_intrument}"].notna()]
    df = df[df[f"volume_{interes_intrument}"].notna()]

    """
    MOMENTUM INDICATORS
    """
    # Number of indicators by indicator
    n_indicators = 10

    # Obtain a base number of calcultations
    base_period = 10

    # List of indicators names to calculate
    mf_indicators = ["CCI", "ADX", "ROC", "RSI", "MFI", "WILLR"]
    datilla = df

    # For each indicator to calculate add columns with periods and base values
    for indicator in range(len(mf_indicators)):
        multirame_indicator(
            dataframe=df, indicator=mf_indicators[indicator],
            n_indicators=n_indicators, base_period=base_period,
            ins_interest=ins_interest)

    # Moving Average Convergene Divergence
    df["MACD_a"], df["MACD_b"], df["MACD_c"] = talib.MACD(
        df[f"close_{interes_intrument}"], fastperiod=12,
        slowperiod=26, signalperiod=9)

    # STOCH
    df["STOCH_FAST"], df["STOCH_SLOW"] = talib.STOCH(
        df[f"high_{interes_intrument}"],
        df[f"low_{interes_intrument}"],
        df[f"close_{interes_intrument}"],
        fastk_period=8, slowk_period=3,
        slowk_matype=3, slowd_period=13, slowd_matype=0)

    # STOCHRSI
    df["STOCHRSI_fast_k"], df["STOCHRSI_fast_d"] = talib.STOCHRSI(
        df[f"close_{interes_intrument}"], timeperiod=14, fastk_period=5,
        fastd_period=3, fastd_matype=0)

    """
    TREND INDICATORS
    """
    # TEMA MULTIFRAME INDICATORS
    multirame_indicator(dataframe=df, indicator="TEMA",
                        n_indicators=40, base_period=5,
                        ins_interest=ins_interest)

    # VORTEX
    vortex = ta.vortex(low=df[f"low_{interes_intrument}"],
                       close=df[f"close_{interes_intrument}"],
                       high=df[f"high_{interes_intrument}"])
    df[f'Vortexp_Close_{interes_intrument}'] = vortex.VTXP_14
    df[f'Vortexm_Close_{interes_intrument}'] = vortex.VTXM_14

    """
    VOLATILITY INDICATORS
    """
    # NORMALIZED AVERAGE TRUE RANGE
    multirame_indicator(dataframe=df, indicator="NATR",
                        n_indicators=6, base_period=5,
                        ins_interest=ins_interest)

    """
    DESCOMPOSITON INDICATORS
    """
    series_aberration = ta.aberration(
        low=df[f"low_{interes_intrument}"],
        close=df[f"close_{interes_intrument}"],
        high=df[f"high_{interes_intrument}"])
    df['ABER_ZG_5_15'] = series_aberration['ABER_ZG_5_15']
    df['ABER_SG_5_15'] = series_aberration['ABER_SG_5_15']
    df['ABER_XG_5_15'] = series_aberration['ABER_XG_5_15']
    df['ABER_ATR_5_15'] = series_aberration['ABER_ATR_5_15']
    df['HWM'] = ta.hwc(df[f"close_{interes_intrument}"])['HWM']
    df['HWU'] = ta.hwc(df[f"close_{interes_intrument}"])['HWU']
    df['HWL'] = ta.hwc(df[f"close_{interes_intrument}"])['HWL']

    """
    VOLUME INDICATORS
    """
    # Accumulation/Distribution Indicator (A/D)
    df["AD"] = talib.AD(df[f"high_{interes_intrument}"],
                        df[f"low_{interes_intrument}"],
                        df[f"close_{interes_intrument}"],
                        df[f"volume_{interes_intrument}"])

    # On-Balance Volume (OBV)
    df["OBV"] = talib.OBV(
        df[f"high_{interes_intrument}"], df[f"low_{interes_intrument}"])

    # Chaikin Money Flow indicator strategy
    df["Chaikyn_Money_Flow"] = ta.cmf(df[f"high_{interes_intrument}"],
                                      df[f"low_{interes_intrument}"],
                                      df[f"close_{interes_intrument}"],
                                      df[f"open_{interes_intrument}"])

    """
    DERIVATIVES
    """
    # Newton Zone
    df["first_derivative"] = df[f"close_{interes_intrument}"].diff()

    """
    TRANSFORMS
    """
    # Average Price
    df['AVGPRICE'] = talib.AVGPRICE(df[f"high_{interes_intrument}"],
                                    df[f"low_{interes_intrument}"],
                                    df[f"close_{interes_intrument}"],
                                    df[f"open_{interes_intrument}"])

    # Median Price
    df['MEDPRICE'] = talib.MEDPRICE(
        df[f"high_{interes_intrument}"], df[f"low_{interes_intrument}"])

    # Typical Price
    df['TYPPRICE'] = talib.TYPPRICE(
        df[f"high_{interes_intrument}"],
        df[f"low_{interes_intrument}"],
        df[f"close_{interes_intrument}"])

    # Weighted Close Price
    df['WCLPRICE'] = talib.WCLPRICE(
        df[f"high_{interes_intrument}"],
        df[f"low_{interes_intrument}"],
        df[f"close_{interes_intrument}"])

    # Rolling mid price
    df["MIDPRICE"] = talib.MIDPRICE(
        df[f"high_{interes_intrument}"],
        df[f"low_{interes_intrument}"], 14)

    """
    CYCLE INDICATORS
    """
    # Hilbert Transform - Dominant Cycle Period
    df['HT_DCPERIOD'] = talib.HT_DCPERIOD(df[f"close_{interes_intrument}"])

    # Hilbert Transform - Dominant Cycle Phase
    df['HT_DCPHASE'] = talib.HT_DCPHASE(df[f"close_{interes_intrument}"])

    # Hilbert Transform - Phasor Components
    df['HT_PHASOR_INPHASE'], df["HT_PHASOR_INPHASE_QUADRATURE"] = talib.HT_PHASOR(
        df[f"close_{interes_intrument}"])

    # Hilbert Transform - SineWave
    df['HT_SINE'], df["HT_SINE_LEADSINE"] = talib.HT_SINE(
        df[f"close_{interes_intrument}"])

    """
    STATISTIC FUNCTIONS
    """
    # Variance
    df['VAR'] = talib.VAR(df[f"close_{interes_intrument}"])

    # Standard Deviation
    df['STDDEV'] = talib.STDDEV(df[f"close_{interes_intrument}"])

    # Entropia
    df['ENTROPY'] = ta.entropy(df[f"close_{interes_intrument}"])

    # Volume indicator Kurtosis
    df['VOLUME_Kurtosis'] = ta.kurtosis(df[f"close_{interes_intrument}"])

    # Volume indicator Skewness
    df['VOLUME_Skewness'] = ta.skew(df[f"close_{interes_intrument}"])

    # Volume indicator Quantile
    df['VOLUME_Quantile'] = ta.quantile(df[f"close_{interes_intrument}"])

    # Volume indicator Z-Score
    df['VOLUME_Zscore'] = ta.zscore(df[f"close_{interes_intrument}"])

    """
    CORRELATIONS FUNCTIONS
    """
    # Correlation of other instrument to calculate
    """
    df = correlations(
        dataframe=df, interes_instrument=interes_intrument, periods=24)
    """

    """
    AUTOCORRELATION
    """
    # Autocorellation of interest instrument
    df[f"AUTOCORR_{interes_intrument}"] = close.rolling(
        5).apply(lambda x: x.autocorr(), raw=False)

    # Lines Psicologycal of Prices in interes instrument
    df[f"LINES_{interes_intrument}"] = ta.psl(df[f"close_{interes_intrument}"])

    """
    Invocation of all indicators
    """

    """
    DERIVATIVES OF OTHER INSTRUMENTS OR RETURNS OR ACELERATION
    """
    # Derivative of other instrument
    df = derivative_df(dataframe=df, interest_intrument=interes_intrument)

    """
    FOR EACH INSTRUMENT CREATE VOLUMNE-VOLATILIY AND MOMENTUM INDICATOR FOR SELF
    """
    data1 = df
    VOLATILITY_VOLUME_MOMENTUM(
        dataframe=df, interest_intrument=interes_intrument,
        mfi_period=12)
    data2 = df


    """
    DROP CLOSE PRICES OF THE OTHERS INSTRUMENTS
    """

    df = drop_price(dataframe=df, interest_intrument=interes_intrument)
    # df = df.drop(columns=[f"high_{interes_intrument}",
    #                      f"open_{interes_intrument}", f"low_{interes_intrument}"])

    """
    Modelacion en Keras: Parametros
    """

    #  Configuraciones
    windows_size = 96
    batch_size = 32
    epochs = 150
    test_size = 0.2

    data = df
    # Guardamos los datos
    data[f"close_{interes_intrument}"].dropna()

    # Transformacion de escalado
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Transformacion escalado para Ypredict
    scaler_y = MinMaxScaler()

    # Limpieza de datos
    data.dropna(inplace=True)

    # Shift para Ypredict para predecir los precios de cierre
    lable = data[f"close_{interes_intrument}"].shift(periods=-windows_size)

    # Normalizar el conjunto de datos
    data = scaler.fit_transform(data)

    # Scaler convirtio la data en npdarray por lo cual para operar y completar dataset convertimos a df nuevamente
    data = pd.DataFrame(data)


    # Preparación de los datos para Ypredict
    y = lable.dropna().astype(float)
    y = np.array(y, dtype=np.float64)
    y = scaler_y.fit_transform(y.reshape(-1, 1))

    # Preparacion de los datos para X
    x = [data.iloc[i - windows_size:i].values for i in range(windows_size, len(data) + 1)]
    x = x[:len(y)]
    x = np.array(x, dtype=np.float64)

    print(
        f"Las dimensiones del ndarray x son:{x.shape} y las del vector y:{y.shape}(estos son precios de cierre)")

    # Split los datos de entrenamiento-test-validacion
    x_train, x_val_test, y_train, y_val_test = train_test_split(
        x,
        y,
        test_size=test_size)

    # Separar los datos entre test y validacion
    x_val, x_test, y_val, y_test = train_test_split(
        x_val_test,
        y_val_test,
        test_size=test_size)

    print(f"La matriz x_train tiene como dimension: {x_train.shape}")
    print(f"La matriz y_train tiene como dimension: {y_train.shape}")
    print(f"La matriz x_val tiene como dimension: {x_val.shape}")
    print(f"La matriz x_test tiene como dimension: {x_test.shape}")
    print(f"La matriz y_val tiene como dimension: {y_val.shape}")
    print(f"La matriz y_test tiene como dimension: {y_test.shape}")

    # Numero de columnas igual a entrada en LSTM
    number_columns = len(df.columns)


    """
    Modelacion en Keras: Evolucion de hiperparmetros de la arquitectura
    """

    # Generar la busqueda
    tuner = keras_tuner.Hyperband(architecture_model,
                                  objective='val_loss',
                                  factor=5,
                                  directory='/home/walter/forex_data/model',
                                  project_name='model_lstm'
                                  )

    # Numero de revisiones
    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3
        )


    # Efectura la busqueda de mejor modelo
    tuner.search(x_train,
                 y_train,
                 epochs=epochs,
                 batch_size=batch_size,
                 validation_data=(x_val, y_val),
                 callbacks=[LossMetricsCallback()]
                 )

    # Obtenemos el mejor modelo
    best_model = tuner.get_best_models()[0]

    # Asignakos el modelo "best model"
    model = tuner.hypermodel.build(best_model)


    """
    Modelacion de Keras: Entrenamiento del modelo
    """
    """
    model = keras.Sequential()

    model.add(GRU(320, return_sequences=True, activation="tanh", input_shape=(x_train.shape[1], x_train.shape[2])))
    
    model.add(Dropout(0.2))
    model.add(LSTM(96, return_sequences=True, activation="tanh"))
    
    model.add(Dropout(0))
    model.add(LSTM(320, return_sequences=True, activation="tanh"))
    
    model.add(Dropout(0.4))
    model.add(LSTM(32, return_sequences=True, activation="tanh"))
    
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=True, activation="tanh"))
    
    model.add(Dropout(0))
    model.add(LSTM(80, return_sequences=True, activation="tanh"))
    
    model.add(Dense(y_train.shape[1], activation="tanh"))


    model.compile(loss='mae', optimizer='Adamax', metrics=['mae', 'mse', 'mape'])

    """ 

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=test_size)

    model.save(folder+name_model)


    """
    # Cargar el modelo entrenado
    model = load_model(folder+name_model)
    """
    # Visualizar la pérdida y la precisión por época
    fig, ax = plt.subplots(figsize=(20, 6))
    epochs = range(1, len(history.history['loss']) + 1)
    plt.plot(epochs, history.history['loss'], label='Pérdida de entrenamiento')
    plt.plot(epochs, history.history['val_loss'], label='Pérdida de validación')
    plt.xlabel('Época')
    plt.ylabel('Valor')
    plt.title('Pérdida y Precisión por Época')
    plt.legend()

    plt.tight_layout()  # Ajustar el espacio entre subplots
    plt.show()


    """
    Grafico de prediccion de precios test
    """


    from keras.models import load_model

    # Cargar el modelo entrenado
    model = load_model(folder+name_model)


    # Obtener las predicciones del modelo en el conjunto de prueba
    y_pred = model.predict(x_test)
    
    # Desescalar los valores predichos
    y_pred_reshaped = y_pred.reshape(-1, 1)
    y_pred_inverse = scaler_y.inverse_transform(y_pred_reshaped)
    y_pred_restored = y_pred_inverse.reshape(y_pred.shape)
    
    # Desescalar los valores reales
    y_test_inverse = scaler_y.inverse_transform(y_test)


    # Graficar los valores reales y los valores predichos
    fig, ax = plt.subplots(figsize=(20, 6))
    indices = np.arange(len(y_test_inverse))
    
    # Graficar los valores reales
    plt.plot(indices,
             y_test_inverse,
             label='Valores reales',
             color='blue')
    
    # Graficar los valores predichos
    plt.plot(indices,
             y_pred_restored[:, 0, 0],
             label='Valores predichos',
             color='orange')
    
    # Configurar el estilo de las líneas y los marcadores
    plt.plot(indices,
             y_test_inverse,
             'o', markersize=4,
             color='blue',
             alpha=0.5)

    plt.plot(indices,
             y_pred_restored[:, 0, 0],
             'o', markersize=4,
             color='orange',
             alpha=0.5)
    
    # Configurar los ejes y el título del gráfico
    plt.xlabel('Índice')
    plt.ylabel('Precios de cierre')
    plt.title('Comparación entre valores reales y valores predichos')
    
    # Añadir una cuadrícula
    plt.grid(True)
    
    # Añadir una leyenda
    plt.legend()
    
    # Ajustar el espacio entre subplots
    plt.tight_layout()
    plt.show()

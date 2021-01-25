from math import ceil
import numpy as np
import pandas as pd
import json
from category_encoders.cat_boost import CatBoostEncoder

def split_train_test(df, test_rate=0.2):
    
    total_rows = df.shape[0]
    test_rows = ceil(total_rows*test_rate)
    
    x_train = df.iloc[:-test_rows].drop('Stage', axis=1)
    y_train = df.iloc[:-test_rows]['Stage'].to_frame()
    x_test = df.iloc[-test_rows:].drop('Stage', axis=1)
    y_test = df.iloc[-test_rows:]['Stage'].to_frame()
    
    return x_train, y_train, x_test, y_test

def split_labels(df):
    
    y = df[['Stage']]
    x = df.drop(columns=['Stage'])
    
    return x, y

def normalizacion_numericas(x_train, x_test=None, modo='normalizacion', columnas=None):
    
    if columnas is None:
        columnas = x_train.select_dtypes(include=['int64', 'float64']).columns
        
    if modo == 'normalizacion':
        for col in columnas:
            if col == 'Opportunity_ID': continue
            if x_train[col].std() != 0:
                x_train[col] = ((x_train[col] - x_train[col].mean()) / x_train[col].std()).astype(float).round(2)
            else:
                x_train[col] = (x_train[col] - x_train[col].mean()).astype(float).round(2)
            #Poco eficiente cortar cada iteracion de un for con 2 if, pero se corre pocas veces.
            if x_test is None: continue
            if x_test[col].std() != 0:
                x_test[col]  = ((x_test[col] - x_test[col].mean()) / x_test[col].std()).astype(float).round(2)
            else:
                x_test[col]  = (x_test[col] - x_test[col].mean()).astype(float).round(2)
    
    if x_test is None: return x_train
    
    return x_train, x_test
    
    #Agregar mas modos

def conversion_fechas(x_train, x_test=None, columnas=None):
    
    if columnas is None:
        columnas = x_train.select_dtypes(include='datetime64').columns
    
    fecha_inicio = np.datetime64('2000-01-01')

    for col in columnas:
        x_train[col] = ((x_train[col] - fecha_inicio)) / np.timedelta64(1, 'D')
        if x_test is None: continue
        x_test[col]  = ((x_test[col] - fecha_inicio)) / np.timedelta64(1, 'D')
    
    if x_test is None: return x_train
    
    return x_train, x_test

def codificar_categoricas(x_train, y_train, x_test, modo='expanding_mean', columnas=None):
    
    if columnas is None:
        columnas = x_train.select_dtypes(include='category').columns
    
    if modo == 'expanding_mean':
        x_y_train = x_train.copy()
        x_y_train['Stage'] = y_train['Stage']
        
        auxiliar = dict()
        codificaciones = dict()
        
        for col in columnas:
            last_one = x_y_train.groupby(col).tail(1)
            for (idx, reg) in zip(last_one[col].index, last_one[col].values):
                auxiliar[reg] = (col, idx)
            cumulative_sum = x_y_train.groupby(col)["Stage"].cumsum() - x_y_train["Stage"]
            cumulative_count = x_y_train.groupby(col).cumcount()
            x_train[col] = cumulative_sum/cumulative_count
        #Llenamos los NaN generados por cumsum con ceros.
        x_train.fillna(0, inplace=True)
        #Guardamos la codificacion de cada categoria segun su nombre.
        for k, v in auxiliar.items():
            col = v[0]
            idx = v[1]
            codificaciones[(col, k)] = x_train.loc[idx, col]
        
        #Codifico a las categorias del set de test con la ultima codificacion del set de train.
        for col in columnas:
            x_test[col] = x_test[col].astype(object)
            for (idx, reg) in zip(x_test[col].index, x_test[col]):
                if ((col, reg) in codificaciones):
                    x_test.loc[idx, col] = codificaciones[(col, reg)]
                else:
                    #Codifico como cero, se puede mejorar
                    x_test.loc[idx, col] = 0
    
    elif modo == 'catboost':
        #Esto igual no deberia pasar, pero se filtra por las dudas.
        if 'Stage' in columnas : columnas.remove('Stage')
        #Creamos una instancia del encoder pasandole las columnas a codificar
        ohe = CatBoostEncoder(cols = columnas, return_df = True)
        #Entrenamos el encoder a partir del df de train y df de test
        ohe.fit(x_train, y_train)
        
        colum_transformadas_train = ohe.transform(x_train, y_train)
        colum_transformadas_test = ohe.transform(x_test)
        for columna in columnas:
            x_train[columna] = colum_transformadas_train[columna].copy()
            x_test[columna] = colum_transformadas_test[columna].copy()
    
    return x_train, x_test

def df_a_vector(x_df):
    
    return np.asarray(x_df.values).astype('float32')


def diagnostico_df(df, eliminar=False):
    a_eliminar = pd.Series(dtype='float64')
    incompatibles = False
    filas_antes = df.shape[0]
    
    for col in df.columns:
        nulos = df[col].isnull()
        nones = df[col].astype('str').str.contains('None')
        nat   = df[col].astype('str').str.contains('NaT')
        infinitos = ((df[col] == np.inf) | (df[col] == -np.inf))
        
        a_eliminar = nulos | nones | nat | infinitos
        suma = a_eliminar.sum()
        if suma > 0:
            print(f'Suma: {suma}, Columna: {col}')
            incompatibles = True
            if (eliminar):
                df.drop(df[a_eliminar].index, inplace=True)
            
    if (not incompatibles): print("Ninguna columna tiene datos incompatibles")
    if (eliminar and incompatibles):
        filas_despues = df.shape[0]
        print(f"Se eliminaron {filas_antes - filas_despues} filas incompatibles del dataframe")
        
        
def hyperparams_to_json(hp_dict, model_name):
    with open(model_name + '_best_hyperparam.json', 'w') as fd:
        print(f"Guardando hiperparametros en el archivo: '{model_name}_best_hyperparam.json'")
        json.dump(hp_dict, fd, sort_keys=False, indent=4)


def hyperparams_from_json(model_name):
    
    hp_dict = dict()
    #Dejo que la excepcion por fallo de apertura sea manejada por el caller
    with open(model_name + '_best_hyperparam.json', 'r') as fd:
        print(f"Cargando hiperparametros desde el archivo: '{model_name}_best_hyperparam.json'")
        hp_dict = json.load(fd)
    
    return hp_dict
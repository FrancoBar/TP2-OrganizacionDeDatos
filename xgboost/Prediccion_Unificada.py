#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from datetime import datetime
from itertools import combinations

PREDICCION_REAL = False
MAXIMIZAR_HIPERPARAMETROS = False
PARAMETROS = {"booster":"gbtree", "max_depth":3, "eta": 0.5, "objective": "binary:logistic", "nthread":2,"gamma" : 0}
RONDAS = 20


# In[2]:


#APERTURA DE ARCHIVO DE ARCHIVOS
entrenamiento_temp = pd.read_csv("../Train_TP2_Datos_2020-2C.csv")
entrenamiento_temp = entrenamiento_temp[( entrenamiento_temp['Stage'] == 'Closed Won') | ( entrenamiento_temp['Stage'] == 'Closed Lost')]
entrenamiento_temp = entrenamiento_temp.loc[(entrenamiento_temp["ASP_Currency"] == entrenamiento_temp["Total_Taxable_Amount_Currency"])]
#entrenamiento = entrenamiento.loc[entrenamiento["Total_Taxable_Amount"] > 0]

test = pd.read_csv("../Test_TP2_Datos_2020-2C.csv")


# In[3]:


#FORMATO FECHAS

#Respalda fecha, usada para separa entrenamiento y test
entrenamiento_temp['Fecha'] = pd.to_datetime(entrenamiento_temp['Opportunity_Created_Date'])
columnas_fecha = ['Month','Last_Modified_Date','Account_Created_Date','Opportunity_Created_Date','Quote_Expiry_Date','Planned_Delivery_Start_Date','Planned_Delivery_End_Date']

def formato_fechas(x):
    for columna in columnas_fecha:
        x[columna] = pd.to_datetime(x[columna])
        
formato_fechas(entrenamiento_temp)
if(PREDICCION_REAL): 
    formato_fechas(test)


# In[4]:


#DIVISION ENTRE SET DE ENTRENAMIENTO Y SET DE TEST

if(PREDICCION_REAL):
    entrenamiento = entrenamiento_temp
else:
    entrenamiento = entrenamiento_temp.loc[entrenamiento_temp['Fecha'].dt.year == 2015].copy()
    test          = entrenamiento_temp.loc[entrenamiento_temp['Fecha'].dt.year == 2016].copy()
    entrenamiento_label = (entrenamiento['Stage'] == 'Closed Won').astype(int)
    test_label          = (test['Stage'] == 'Closed Won').astype(int)

del entrenamiento_temp


# In[5]:


#LIMPIEZA

def limpiar(x):
    x = x.drop(columns=['ASP_(converted)_Currency','Quote_Type','Brand','Product_Type','Size','Product_Category_B','Price','Currency','Last_Activity','Actual_Delivery_Date','Prod_Category_A'])
    x = x.drop(columns=['ID','Opportunity_Name','Sales_Contract_No'])
    return x

entrenamiento = limpiar(entrenamiento)
test = limpiar(test)


# In[6]:


#NUEVOS FEATURES


# In[7]:


#Agrego feature: Duracion de la oportunidad
entrenamiento['Opportunity_Duration'] = (entrenamiento['Last_Modified_Date'] - entrenamiento['Opportunity_Created_Date']) / np.timedelta64(1, 'D')
test['Opportunity_Duration'] = (test['Last_Modified_Date'] - test['Opportunity_Created_Date']) / np.timedelta64(1, 'D')
#Agrego feature: Total_Amount_USD
entrenamiento["Total_Amount_USD"] = entrenamiento["Total_Amount"] * entrenamiento["ASP_(converted)"] / entrenamiento["ASP"]
test["Total_Amount_USD"] = test["Total_Amount"] * test["ASP_(converted)"] / test["ASP"]
#Agrego feature: Total_Taxable_Amount_USD
entrenamiento["Total_Taxable_Amount_USD"] = entrenamiento["Total_Taxable_Amount"] * entrenamiento["ASP_(converted)"] / entrenamiento["ASP"]
test["Total_Taxable_Amount_USD"] = test["Total_Taxable_Amount"] * test["ASP_(converted)"] / test["ASP"]
#Agrego feature: Total_Amount_sobre_Total_Taxable_Amount
entrenamiento["Total_Amount_sobre_Total_Taxable_Amount"] = entrenamiento["Total_Amount_USD"] / entrenamiento["Total_Taxable_Amount_USD"]
test["Total_Amount_sobre_Total_Taxable_Amount"] = test["Total_Amount_USD"] / test["Total_Taxable_Amount_USD"]

def mean_encoding(train,test,col_group,col_mean):
    
    codificaciones = dict()

    last_one = train.groupby(col_group).tail(1)
    for (idx, reg) in zip(last_one[col_group].index, last_one[col_group].values):
        codificaciones[reg] = (col_group + "_" + col_mean, idx)

    train[col_group + "_" + col_mean] = train.groupby(col_group)[col_mean].transform("mean")    

    #Llenamos los NaN generados por cumsum con ceros.
    train.fillna(0,inplace = True)

    #Guardamos la codificacion de cada categoria segun su nombre.
    for k, v in codificaciones.items():
        col = v[0]
        idx = v[1]
        codificaciones[k] = train.loc[idx, col]
    
    # Utilizo las ultimas codificaciones de cada categoria del train set para codificar el test set.
    # Para eso utilizo el diccionario de codificaciones.

    #columnas_categoricas = x_test.select_dtypes(include='category').columns

    test[col_group + "_" + col_mean] = test[col_group].astype(object)
    for (idx, reg) in zip(test[col_group + "_" + col_mean].index, test[col_group + "_" + col_mean]):
        if (reg in codificaciones):
            test.loc[idx, col_group + "_" + col_mean] = codificaciones[reg]
        else:
            #Codifico como cero, se puede mejorar
            test.loc[idx, col_group + "_" + col_mean] = 0
    test[col_group + "_" + col_mean] = test[col_group + "_" + col_mean].astype(float)

    


mean_encoding(entrenamiento,test,"Billing_Country","Opportunity_Duration")
mean_encoding(entrenamiento,test,"Account_Type","Opportunity_Duration")
mean_encoding(entrenamiento,test,"Region","ASP")
mean_encoding(entrenamiento,test,"Billing_Country","ASP")
mean_encoding(entrenamiento,test,"Billing_Country","Total_Amount_USD")
mean_encoding(entrenamiento,test,"Billing_Country","Bureaucratic_Code_0_Approved")
mean_encoding(entrenamiento,test,"Product_Family","Opportunity_Duration")
mean_encoding(entrenamiento,test,"Product_Family","Total_Amount_USD")
mean_encoding(entrenamiento,test,"Product_Family","Bureaucratic_Code_0_Approved")


# In[8]:


#FEATURE - Duracion por familia
df_zona = entrenamiento[['Stage','Region','Territory','Product_Family','Planned_Delivery_Start_Date']]
df_zona = df_zona[df_zona['Stage'] == 'Closed Won']
df_familia = df_zona.groupby(['Product_Family'])['Planned_Delivery_Start_Date'].agg(['max','min']).reset_index()
df_familia['Duracion'] = (df_familia['max'] - df_familia['min']).dt.days
df_familia.columns = ['Product_Family','Planed_Delivery_Fecha_Max','min','Duracion_Familia']
entrenamiento = entrenamiento.merge(df_familia[['Product_Family','Duracion_Familia','Planed_Delivery_Fecha_Max']],on='Product_Family',how='left')
entrenamiento['Vida_Util_Ventaja'] =  (entrenamiento['Planned_Delivery_Start_Date'] - entrenamiento['Planed_Delivery_Fecha_Max']).dt.days - entrenamiento['Duracion_Familia']
entrenamiento = entrenamiento.drop('Planed_Delivery_Fecha_Max',1)
test = test.merge(entrenamiento[['Product_Family','Duracion_Familia','Vida_Util_Ventaja']].drop_duplicates(subset=['Product_Family']),left_on='Product_Family',right_on='Product_Family',how='left')

#FEATURE - Duracion por region
df_zona = entrenamiento[['Stage','Region','Territory','Product_Family','Planned_Delivery_Start_Date']]
df_zona = df_zona[df_zona['Stage'] == 'Closed Won']
df_region = df_zona.groupby(['Region'])['Planned_Delivery_Start_Date'].agg(['max','min']).reset_index()
df_region['Duracion'] = (df_region['max'] - df_region['min']).dt.days
df_region.columns = ['Region','Region_Planed_Delivery_Fecha_Max','Region_min','Duracion_Region']
entrenamiento = entrenamiento.merge(df_region[['Region','Duracion_Region','Region_Planed_Delivery_Fecha_Max']],on='Region',how='left')
entrenamiento['Region_Vida_Util_Ventaja'] =  (entrenamiento['Planned_Delivery_Start_Date'] - entrenamiento['Region_Planed_Delivery_Fecha_Max']).dt.days - entrenamiento['Duracion_Region']
entrenamiento = entrenamiento.drop('Region_Planed_Delivery_Fecha_Max',1)
test = test.merge(entrenamiento[['Region','Duracion_Region','Region_Vida_Util_Ventaja']].drop_duplicates(subset=['Region']),left_on='Region',right_on='Region',how='left')

#FEATURE - Duracion por territorio
df_zona = entrenamiento[['Stage','Region','Territory','Product_Family','Planned_Delivery_Start_Date']]
df_zona = df_zona[df_zona['Stage'] == 'Closed Won']
df_territorio = df_zona.groupby(['Territory'])['Planned_Delivery_Start_Date'].agg(['max','min']).reset_index()
df_territorio['Duracion'] = (df_territorio['max'] - df_territorio['min']).dt.days
df_territorio.columns = ['Territory','Territory_Planed_Delivery_Fecha_Max','Territory_min','Duracion_Territory']
entrenamiento = entrenamiento.merge(df_territorio[['Territory','Duracion_Territory','Territory_Planed_Delivery_Fecha_Max']],on='Territory',how='left')
entrenamiento['Territory_Vida_Util_Ventaja'] =  (entrenamiento['Planned_Delivery_Start_Date'] - entrenamiento['Territory_Planed_Delivery_Fecha_Max']).dt.days - entrenamiento['Duracion_Territory']
entrenamiento = entrenamiento.drop('Territory_Planed_Delivery_Fecha_Max',1)
test = test.merge(entrenamiento[['Territory','Duracion_Territory','Territory_Vida_Util_Ventaja']].drop_duplicates(subset=['Territory']),left_on='Territory',right_on='Territory',how='left')


# In[9]:


#FECHAS A DIAS
def fecha_a_dias(x):
    for columna in columnas_fecha:
        x[columna] = x[columna].apply(lambda x : (pd.to_datetime(x) - pd.to_datetime('01/01/2000', format='%m/%d/%Y')).days)

fecha_a_dias(entrenamiento)
fecha_a_dias(test)


# In[10]:


#CATEGORICAS A NUMERICAS - PROMEDIO
"""Se debe pasar train y test ordenados """
def expansion_mean_encoding(columnas_categoricas,train,test,label):
    #Dividimos el dataset de entrenamiento en features y labels
    #Armo un df extra que me ayudara para codificar las categoricas.
    #x_y_train = filtrado.iloc[:-test_rows]
    #x_train = x_y_train.drop('Stage', axis=1)
    #y_train = x_y_train['Stage'].to_frame()
    #x_test = filtrado.iloc[-test_rows:].drop('Stage', axis=1)
    #y_test = filtrado.iloc[-test_rows:]['Stage'].to_frame()

    #En el set de train.
    #columnas_categoricas = x_train.select_dtypes(include='category').columns

    codificaciones = dict()

    for col in columnas_categoricas:
        last_one = train.groupby(col).tail(1)
        for (idx, reg) in zip(last_one[col].index, last_one[col].values):
            codificaciones[reg] = (col, idx)
        cumulative_sum = train.groupby(col)[label].cumsum() - train[label]
        cumulative_count = train.groupby(col).cumcount()
        train[col] = cumulative_sum/cumulative_count

    #Llenamos los NaN generados por cumsum con ceros.
    train.fillna(0,inplace = True)

    #Guardamos la codificacion de cada categoria segun su nombre.
    for k, v in codificaciones.items():
        col = v[0]
        idx = v[1]
        codificaciones[k] = train.loc[idx, col]
    
    # Utilizo las ultimas codificaciones de cada categoria del train set para codificar el test set.
    # Para eso utilizo el diccionario de codificaciones.

    #columnas_categoricas = x_test.select_dtypes(include='category').columns

    for col in columnas_categoricas:
        test[col] = test[col].astype(object)
        for (idx, reg) in zip(test[col].index, test[col]):
            if (reg in codificaciones):
                test.loc[idx, col] = codificaciones[reg]
            else:
                #Codifico como cero, se puede mejorar
                test.loc[idx, col] = 0
        test[col] = test[col].astype(float)
        
columnas_categoricas = list(entrenamiento.select_dtypes(include=['object']).columns)
if 'Stage' in columnas_categoricas : columnas_categoricas.remove('Stage')
entrenamiento["label"] = (entrenamiento['Stage'] == 'Closed Won').astype(int)
entrenamiento.sort_values("Fecha")
expansion_mean_encoding(columnas_categoricas,entrenamiento,test,"label")
entrenamiento = entrenamiento.drop(columns='label')


# In[11]:


#CATEGORICAS A NUMERICAS  - ORDINAL
def categoricas_a_numericas(x):
    ohe = skl.preprocessing.OrdinalEncoder()
    columnas_object = list(x.select_dtypes(include=['object']).columns)
    if 'Stage' in columnas_object : columnas_object.remove('Stage')
    for columna in columnas_object:
        copia = x[[columna]].copy().dropna()
        df_temp = pd.DataFrame(ohe.fit_transform(copia)).astype('int32')
        df_temp.columns = [columna]
        x[columna] = df_temp[columna]

categoricas_a_numericas(entrenamiento)
categoricas_a_numericas(test)


# In[12]:


#Filtrado de columnas - No remover Stage o Fecha
#entrenamiento = entrenamiento[['Total_Amount_Currency']]


# In[13]:


if(PREDICCION_REAL):
    objetivo = (entrenamiento['Stage'] == 'Closed Won').astype(int)
    entrenamiento = entrenamiento.drop(columns=['Stage','Fecha'])
    
    d_entrenamiento = xgb.DMatrix(entrenamiento.values, objetivo.values)
    d_prueba = xgb.DMatrix(test.values)
    
    bst = xgb.train(PARAMETROS, d_entrenamiento, RONDAS)
    preds = bst.predict(d_prueba)

    resultados = test[['Opportunity_ID']].copy()
    resultados['Target'] = pd.Series(preds)
    resultados = resultados.groupby('Opportunity_ID').mean()
    resultados = resultados.reset_index()
    resultados['Target'] = resultados['Target'].apply(lambda x: int(x >= 0.5))
    
    resultados.to_csv("prediccion.csv", index=False)
    resultados['Target'].value_counts()


# In[14]:


if(PREDICCION_REAL): sys.exit()


# In[15]:


set_entrenamiento = xgb.DMatrix(entrenamiento.drop(columns=['Stage','Fecha']),label = entrenamiento_label)
set_test  = xgb.DMatrix(test .drop(columns=['Stage','Fecha']),label = test_label)
evaluacion = [(set_test, 'eval'), (set_entrenamiento, 'train')]


# In[21]:


modelo = xgb.train(PARAMETROS, set_entrenamiento, RONDAS, evaluacion)

xgb.plot_importance(modelo)
plt.show()


# In[22]:


prediccion = modelo.predict(set_test)

prediccion = [1 if i > .5 else 0 for i in prediccion]

def metricas(objetivo, prediccion):
    matriz_conf = confusion_matrix(objetivo, prediccion)
    score = accuracy_score(objetivo, prediccion)
    reporte = classification_report(objetivo, prediccion)
    metricas = [matriz_conf, score, reporte]
    return(metricas)

metrics = metricas(test_label, prediccion)
print("Prediccion Test")
[print(i) for i in metrics]
print(skl.metrics.log_loss(test_label,prediccion))

prediccion = modelo.predict(set_entrenamiento)
prediccion = [1 if i > .5 else 0 for i in prediccion]
metricas = metricas(entrenamiento_label, prediccion)
print("Prediccion Train")
[print(i) for i in metricas]


# In[ ]:


if(not MAXIMIZAR_HIPERPARAMETROS): sys.exit()


# In[20]:


min_log_loss = 1000
d = 1000
e = 1000
r = 1000
for depth in range(2,14):
    for eta in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
        for rondas in [10,20,30,40]:
            parametros = {"booster":"gbtree", "max_depth":depth, "eta": eta, "objective": "binary:logistic", "nthread":2}
            modelo = xgb.train(parametros, set_entrenamiento, rondas, evaluacion)
            prediccion = modelo.predict(set_test)
            prediccion = [1 if i > .5 else 0 for i in prediccion]
            log_loss = skl.metrics.log_loss(test_label,prediccion)
            if (log_loss < min_log_loss):
                min_log_loss = log_loss
                d = depth
                e = eta
                r = rondas


print("log: ",min_log_loss)                
print("depth: ",d)      
print("eta: ",e)      
print("rondas: ",r)


# In[ ]:





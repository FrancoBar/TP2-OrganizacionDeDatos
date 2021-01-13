import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
entrenamiento = pd.read_csv("../Train_TP2_Datos_2020-2C.csv")
test = pd.read_csv("../Test_TP2_Datos_2020-2C.csv")

entrenamiento["Fecha"] = pd.to_datetime(entrenamiento["Planned_Delivery_Start_Date"])

entrenamiento = entrenamiento.loc[(entrenamiento["ASP_Currency"] == entrenamiento["Total_Taxable_Amount_Currency"])]
entrenamiento = entrenamiento.loc[entrenamiento["Total_Taxable_Amount"] > 0]
entrenamiento = entrenamiento.loc[(entrenamiento["Planned_Delivery_End_Date"] != "NaT")]

entrenamiento['Last_Modified_Date'] = pd.to_datetime(entrenamiento['Last_Modified_Date'])
entrenamiento['Opportunity_Created_Date'] = pd.to_datetime(entrenamiento['Opportunity_Created_Date'])


entrenamiento['Opportunity_Duration'] = (entrenamiento['Last_Modified_Date'] - entrenamiento['Opportunity_Created_Date']) / np.timedelta64(1, 'D')
entrenamiento["Total_Amount_USD"] = entrenamiento["Total_Amount"] * entrenamiento["ASP_(converted)"] / entrenamiento["ASP"]
entrenamiento["Total_Taxable_Amount_USD"] = entrenamiento["Total_Taxable_Amount"] * entrenamiento["ASP_(converted)"] / entrenamiento["ASP"]
entrenamiento["Total_Amount_sobre_Total_Taxable_Amount"] = entrenamiento["Total_Amount_USD"] / entrenamiento["Total_Taxable_Amount_USD"]
entrenamiento["Account_Created_Date_(Unix)"] = pd.to_datetime(entrenamiento["Account_Created_Date"]).transform(lambda x : datetime.timestamp(x))
entrenamiento["Opportunity_Created_Date_(Unix)"] = pd.to_datetime(entrenamiento["Opportunity_Created_Date"]).transform(lambda x : datetime.timestamp(x))
entrenamiento["Last_Modified_Date_(Unix)"] = pd.to_datetime(entrenamiento["Last_Modified_Date"]).transform(lambda x : datetime.timestamp(x))
entrenamiento["Planned_Delivery_Start_Date_(Unix)"] = pd.to_datetime(entrenamiento["Planned_Delivery_Start_Date"]).transform(lambda x : datetime.timestamp(x))
entrenamiento["Planned_Delivery_End_Date_(Unix)"] = pd.to_datetime(entrenamiento["Planned_Delivery_End_Date"]).transform(lambda x : datetime.timestamp(x))




test['Last_Modified_Date'] = pd.to_datetime(test['Last_Modified_Date'])
test['Opportunity_Created_Date'] = pd.to_datetime(test['Opportunity_Created_Date'])


test['Opportunity_Duration'] = (test['Last_Modified_Date'] - test['Opportunity_Created_Date']) / np.timedelta64(1, 'D')
test["Total_Amount_USD"] = test["Total_Amount"] * test["ASP_(converted)"] / test["ASP"]
test["Total_Taxable_Amount_USD"] = test["Total_Taxable_Amount"] * test["ASP_(converted)"] / test["ASP"]
test["Total_Amount_sobre_Total_Taxable_Amount"] = test["Total_Amount_USD"] / test["Total_Taxable_Amount_USD"]
test["Account_Created_Date_(Unix)"] = pd.to_datetime(test["Account_Created_Date"]).transform(lambda x : datetime.timestamp(x))
test["Opportunity_Created_Date_(Unix)"] = pd.to_datetime(test["Opportunity_Created_Date"]).transform(lambda x : datetime.timestamp(x))
test["Last_Modified_Date_(Unix)"] = pd.to_datetime(test["Last_Modified_Date"]).transform(lambda x : datetime.timestamp(x))
test["Planned_Delivery_Start_Date_(Unix)"] = pd.to_datetime(test["Planned_Delivery_Start_Date"]).transform(lambda x : datetime.timestamp(x))
#test["Planned_Delivery_End_Date_(Unix)"] = pd.to_datetime(test["Planned_Delivery_End_Date"]).transform(lambda x : datetime.timestamp(x))

#entrenamiento["Quote_Expiry_Date_(Unix)"] = pd.to_datetime(entrenamiento["Quote_Expiry_Date"]).transform(lambda x : datetime.timestamp(x)) #Se pierden 4000 casos

#print(test.info())


#train = entrenamiento.loc[entrenamiento["Fecha"].dt.year <= 2018]
#test_ = entrenamiento.loc[entrenamiento["Fecha"].dt.year > 2018]
train = entrenamiento




#train_datos = train[["Stage",'Opportunity_Duration',"Total_Amount_USD","Total_Taxable_Amount_USD","Total_Amount_sobre_Total_Taxable_Amount","Account_Created_Date_(Unix)","Opportunity_Created_Date_(Unix)","Last_Modified_Date_(Unix)","Planned_Delivery_Start_Date_(Unix)","Planned_Delivery_End_Date_(Unix)"]]
#train_label = (train_datos["Stage"] == "Closed Won").astype(int)
#test_datos = test[["Stage",'Opportunity_Duration',"Total_Amount_USD","Total_Taxable_Amount_USD","Total_Amount_sobre_Total_Taxable_Amount","Account_Created_Date_(Unix)","Opportunity_Created_Date_(Unix)","Last_Modified_Date_(Unix)","Planned_Delivery_Start_Date_(Unix)","Planned_Delivery_End_Date_(Unix)"]]
#test_datos = test[['Opportunity_Duration',"Total_Amount_USD","Total_Taxable_Amount_USD","Total_Amount_sobre_Total_Taxable_Amount","Account_Created_Date_(Unix)","Opportunity_Created_Date_(Unix)","Last_Modified_Date_(Unix)","Planned_Delivery_Start_Date_(Unix)","Planned_Delivery_End_Date_(Unix)"]]
#test_label = (test_datos["Stage"] == "Closed Won").astype(int)

train_datos = train[["Stage",'Opportunity_Duration',"Total_Amount_USD","Total_Taxable_Amount_USD","Total_Amount_sobre_Total_Taxable_Amount","Account_Created_Date_(Unix)","Opportunity_Created_Date_(Unix)","Last_Modified_Date_(Unix)","Planned_Delivery_Start_Date_(Unix)"]]
train_label = (train_datos["Stage"] == "Closed Won").astype(int)
test_datos = test[['Opportunity_Duration',"Total_Amount_USD","Total_Taxable_Amount_USD","Total_Amount_sobre_Total_Taxable_Amount","Account_Created_Date_(Unix)","Opportunity_Created_Date_(Unix)","Last_Modified_Date_(Unix)","Planned_Delivery_Start_Date_(Unix)"]]



set_train = xgb.DMatrix(train_datos.drop("Stage",1),label=train_label)
set_test = xgb.DMatrix(test_datos)

parametros = {"booster":"gbtree", "max_depth":4, "eta": 0.3, "objective": "binary:logistic", "nthread":2}
rondas = 30

evaluacion = [(set_test, "eval"), (set_train, "train")]


modelo = xgb.train(parametros, set_train, rondas)

xgb.plot_importance(modelo)
plt.show()

prediccion = modelo.predict(set_test)

test = pd.read_csv("../Test_TP2_Datos_2020-2C.csv")
resultados = test[['Opportunity_ID']].copy()
resultados['Target'] = pd.Series(prediccion)
resultados = resultados.groupby('Opportunity_ID').mean()
resultados = resultados.reset_index()
def temp(x):
    return int(x >= 0.5)
resultados['Target'] = resultados['Target'].apply(temp)
resultados.to_csv("prueba.csv", index=False)
#prediccion = [1 if i >= .5 else 0 for i in prediccion]

#print(prediccion)
#serie = pd.Series(prediccion)
#serie.to_csv("mi.csv",index=False)
"""
def metricas(objetivo, prediccion):
    matriz_conf = confusion_matrix(objetivo, prediccion)
    score = accuracy_score(objetivo, prediccion)
    reporte = classification_report(objetivo, prediccion)
    metricas = [matriz_conf, score, reporte]
    return(metricas)

metrics = metricas(test_label, prediccion)
print("Prediccion Test")
[print(i) for i in metrics]

prediccion = modelo.predict(set_train)
prediccion = [1 if i > .5 else 0 for i in prediccion]
metricas = metricas(train_label, prediccion)
print("Prediccion Train")
[print(i) for i in metricas]"""


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from itertools import combinations
entrenamiento = pd.read_csv("../Train_TP2_Datos_2020-2C.csv")
test = pd.read_csv("../Test_TP2_Datos_2020-2C.csv")

entrenamiento["Fecha"] = pd.to_datetime(entrenamiento["Opportunity_Created_Date"])

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
#entrenamiento["Quote_Expiry_Date_(Unix)"] = pd.to_datetime(entrenamiento["Quote_Expiry_Date"]).transform(lambda x : datetime.timestamp(x)) #Se pierden 4000 casos


entrenamiento["Billing_Country_Vida_Util"] = entrenamiento.groupby("Billing_Country")["Opportunity_Duration"].transform("mean")
entrenamiento["Account_Type_Vida_Util"] = entrenamiento.groupby("Account_Type")["Opportunity_Duration"].transform("mean")
entrenamiento["Region_ASP"] = entrenamiento.groupby("Region")["ASP"].transform("mean")
entrenamiento["Billing_Country_ASP"] = entrenamiento.groupby("Billing_Country")["ASP"].transform("mean")
entrenamiento["Billing_Country_Total_Amount"] = entrenamiento.groupby("Billing_Country")["Total_Amount_USD"].transform("mean")
entrenamiento["Billing_Country_Buro"] = entrenamiento.groupby("Billing_Country")["Bureaucratic_Code_0_Approved"].transform("mean")

entrenamiento["Product_Family_Vida_Util"] = entrenamiento.groupby("Product_Family")["Opportunity_Duration"].transform("mean")
entrenamiento["Product_Family_Total_Amount"] = entrenamiento.groupby("Product_Family")["Total_Amount"].transform("mean")
entrenamiento["Product_Family_Buro"] = entrenamiento.groupby("Product_Family")["Bureaucratic_Code_0_Approved"].transform("mean")


pruebas = ["Product_Family_Vida_Util","Product_Family_Total_Amount","Product_Family_Buro"]

no = ["Opportunity_Created_Date_(Unix)","Account_Created_Date_(Unix)","Planned_Delivery_End_Date_(Unix)","Planned_Delivery_Start_Date_(Unix)",'Opportunity_Duration',"Last_Modified_Date_(Unix)","Total_Amount_USD"
,"Total_Taxable_Amount_USD",
"ASP","ASP_(converted)","TRF","Account_Type_Vida_Util","Billing_Country_Vida_Util","Region_ASP","Billing_Country_Buro","Billing_Country_Total_Amount","Billing_Country_ASP","Submitted_for_Approval","Total_Amount_sobre_Total_Taxable_Amount"]

si = ["Pricing, Delivery_Terms_Approved"]
train = entrenamiento.loc[entrenamiento["Fecha"].dt.year == 2015]
test = entrenamiento.loc[entrenamiento["Fecha"].dt.year == 2016]

columnas = ["Pricing, Delivery_Terms_Approved"] + pruebas + no + ["Stage"]

#opcion_83_porciento = ['Pricing, Delivery_Terms_Approved', 'TRF', 'Account_Type_Vida_Util',"Stage"] "max_depth":3, "eta": 0.1 rondas= 10

train_datos = train[columnas]
train_label = (train_datos["Stage"] == "Closed Won").astype(int)
test_datos = test[columnas]
test_label = (test_datos["Stage"] == "Closed Won").astype(int)




set_train = xgb.DMatrix(train_datos.drop("Stage",1),label=train_label)
set_test = xgb.DMatrix(test_datos.drop("Stage",1),label = test_label)
evaluacion = [(set_test, "eval"), (set_train, "train")]
"""

min_log_loss = 1000
d = 1000
e = 1000
r = 1000
for depth in range(2,14):
    for eta in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
        for rondas in [10,20,30,40]:
            parametros = {"booster":"gbtree", "max_depth":depth, "eta": eta, "objective": "binary:logistic", "nthread":2}
            modelo = xgb.train(parametros, set_train, rondas, evaluacion)
            prediccion = modelo.predict(set_test)
            prediccion = [1 if i > .5 else 0 for i in prediccion]
            log_loss = sklearn.metrics.log_loss(test_label,prediccion)
            if (log_loss < min_log_loss):
                min_log_loss = log_loss
                d = depth
                e = eta
                r = rondas


print("log: ",min_log_loss)                
print("depth: ",d)      
print("eta: ",e)      
print("rondas: ",r)    
"""

"""
res = []
i = 0
for col in combinations(columnas,5):
    train_datos = train[list(col) + ["Stage"]]
    train_label = (train_datos["Stage"] == "Closed Won").astype(int)
    test_datos = test[list(col) + ["Stage"]]
    test_label = (test_datos["Stage"] == "Closed Won").astype(int)

    set_train = xgb.DMatrix(train_datos.drop("Stage",1),label=train_label)
    set_test = xgb.DMatrix(test_datos.drop("Stage",1),label = test_label)

    evaluacion = [(set_test, "eval"), (set_train, "train")]
    parametros = {"booster":"gbtree", "max_depth":6, "eta": 0.3, "objective": "binary:logistic", "nthread":2}
    rondas = 10
    modelo = xgb.train(parametros, set_train, rondas, evaluacion)
    prediccion = modelo.predict(set_test)
    prediccion = [1 if i > .5 else 0 for i in prediccion]
    log_loss = sklearn.metrics.log_loss(test_label,prediccion)
    res.append((col,log_loss))
    i+=1
    print(i)
res.sort(key=lambda x : -x[1])
for r in res:
    print(r)
        

 
"""

parametros = {"booster":"gbtree", "max_depth":3, "eta": 0.3, "objective": "binary:logistic", "nthread":2,"gamma" : 0}
rondas = 5



modelo = xgb.train(parametros, set_train, rondas, evaluacion)

xgb.plot_importance(modelo)
plt.show()

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
print(sklearn.metrics.log_loss(test_label,prediccion))

prediccion = modelo.predict(set_train)
prediccion = [1 if i > .5 else 0 for i in prediccion]
metricas = metricas(train_label, prediccion)
print("Prediccion Train")
[print(i) for i in metricas]

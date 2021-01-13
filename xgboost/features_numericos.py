import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot as plt
entrenamiento = pd.read_csv("../Train_TP2_Datos_2020-2C.csv")
test = pd.read_csv("../Test_TP2_Datos_2020-2C.csv")

entrenamiento["Fecha"] = pd.to_datetime(entrenamiento["Planned_Delivery_Start_Date"])
train = entrenamiento.loc[entrenamiento["Fecha"].dt.year <= 2018]
test = entrenamiento.loc[entrenamiento["Fecha"].dt.year > 2018]

train_datos = train[["Stage","Pricing, Delivery_Terms_Quote_Appr","Pricing, Delivery_Terms_Approved","Bureaucratic_Code_0_Approval","Bureaucratic_Code_0_Approved","Submitted_for_Approval","ASP","ASP_(converted)","TRF","Total_Amount","Total_Taxable_Amount"]]
train_label = (train_datos["Stage"] == "Closed Won").astype(int)
test_datos = test[["Stage","Pricing, Delivery_Terms_Quote_Appr","Pricing, Delivery_Terms_Approved","Bureaucratic_Code_0_Approval","Bureaucratic_Code_0_Approved","Submitted_for_Approval","ASP","ASP_(converted)","TRF","Total_Amount","Total_Taxable_Amount"]]
test_label = (test_datos["Stage"] == "Closed Won").astype(int)




set_train = xgb.DMatrix(train_datos.drop("Stage",1),label=train_label)
set_test = xgb.DMatrix(test_datos.drop("Stage",1),label = test_label)

parametros = {"booster":"gbtree", "max_depth":4, "eta": 0.3, "objective": "binary:logistic", "nthread":2}
rondas = 30

evaluacion = [(set_test, "eval"), (set_train, "train")]


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

prediccion = modelo.predict(set_train)
prediccion = [1 if i > .5 else 0 for i in prediccion]
metricas = metricas(train_label, prediccion)
print("Prediccion Train")
[print(i) for i in metricas]


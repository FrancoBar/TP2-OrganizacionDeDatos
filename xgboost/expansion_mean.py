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
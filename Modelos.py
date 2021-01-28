from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.regularizers import l1
from keras.activations import relu
from keras.utils.generic_utils import get_custom_objects
from keras import backend
from keras.utils import *
import xgboost as xgb

##########################################################

# Redefino función de activación 'relu'

def my_relu(x):
    return backend.relu(x, max_value=1.0)

get_custom_objects().update({'my_relu': Activation(my_relu)})

# Redefino función de activación 'swish'

class Swish(Activation):
    
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'

def swish(x):
    return (backend.sigmoid(x) * x)

get_custom_objects().update({'swish': Swish(swish)})

##########################################################

def get_neural_network_model(params, input_dim):
    
    alpha = params['alpha']
    model = Sequential()    
    model.add(Dense(int(params['first_layer']['neurons']), 
                    input_dim=input_dim, 
                    activation=params['first_layer']['activation'],
                    kernel_regularizer=l1(alpha)))
    
    model.add(Dropout(params['first_layer']['dropout']))
    
    for layer in params['hidden_layers']:
        if layer['config']['is_on']:
            model.add(Dense(int(layer['config']['neurons']), kernel_regularizer=l1(alpha), activation=layer['config']['activation']))
            model.add(Dropout(layer['config']['dropout']))
    
    model.add(Dense(1, activation=params['last_layer']['activation']))
    model.compile(loss='binary_crossentropy', optimizer=params['optimizer'])
    backend.set_value(model.optimizer.learning_rate, params['learning_rate'])
    
    return model


def get_xgboost_model(params):

    model = xgb.XGBClassifier(max_depth = int(params['max_depth']),
                              n_estimators = params['n_estimators'],
                              learning_rate = params['learning_rate'],
                              gamma = params['gamma'],
                              min_child_weight = params['min_child_weight'],
                              subsample = params['subsample'],
                              colsample_bytree = params['colsample_bytree'],
                              use_label_encoder=False)


    return model

# coding: utf-8

# In[3]:


#-*- coding: UTF-8 -*-
import sys
import math
import keras
import numpy as np
from keras.models import load_model
from keras import regularizers
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.utils.data_utils import get_file
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten,merge,LSTM,Dropout,Reshape,Flatten,Lambda
from keras.layers import Conv1D, MaxPooling1D, Embedding,RepeatVector,Masking
from keras.models import Model
import keras.backend as K
from keras.models import load_model


def save_model_to_file(model, struct_file, weights_file):
    # save model structure
    model_struct = model.to_json()
    open(struct_file, 'w').write(model_struct)

    # save model weights
    model.save_weights(weights_file)


low_data=np.loadtxt("/home/suny/google-winter-camp/lowlevel_matrix",delimiter= ' ')
high_data=np.loadtxt("/home/suny/google-winter-camp/pmi_matrix",delimiter=' ')
predict_data=np.loadtxt("/home/suny/google-winter-camp/predict_data",delimiter=' ')

#low_data=np.random.rand(1000,1000)
#high_data=np.random.rand(1000,1000)
#predict_data=np.random.rand(1000,1000)

input_dim=low_data.shape[0]


inputs1=Input(shape=(input_dim,))
encoded=Dense(512,activation='sigmoid')(inputs1)
decoded= Dense(input_dim,activation='sigmoid')(encoded)

encoder_decoder = Model(inputs=[inputs1], outputs=[decoded])
my_rmsprop=keras.optimizers.RMSprop(lr=0.002, rho=0.9, epsilon=1e-08, decay=0.0)
encoder_decoder.compile(optimizer=my_rmsprop, loss='mean_squared_error')
print (encoder_decoder.summary())

for i in range(50):
    print ("epoch",i)
    encoder_decoder.fit([low_data],[high_data],batch_size= 32, epochs=1)
    
encoder_decoder.save("/home/suny/google-winter-camp/model")

result = encoder_decoder.predict(predict_data)

np.savetxt('/home/suny/google-winter-camp/model_ans',result,delimiter=' ')


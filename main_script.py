import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from scipy.stats import entropy
from scipy import stats
from sklearn import metrics
from sklearn.preprocessing import StandardScaler,normalize,Normalizer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
import keras
from keras.layers.embeddings import Embedding

from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, SpatialDropout1D
from keras.layers import Conv1D, MaxPooling1D
from keras import backend as K
from keras.utils import np_utils
from keras.layers import LSTM


from IPython import embed
#Change Directory if needed

basepath = "./Csv/Outputs"
li2 = []
inputScenarios = []

#Read through every CSV file but only use the columns that have faulty data
print('Reading Started!')

with os.scandir(basepath) as entries:
    for entry in entries:
        if entry.is_file():
            df = pd.read_csv(entry,usecols=[16,17,18,20,21,22])
            # x_scaled = preprocessing.MinMaxScaler().fit_transform(df)
            li2.append(df)
            inputScenarios.append(int(entry.name.split('_')[1]))


print('Reading Done!')
#Concatenate to a datafrme and read from that dataframe

li2 = np.vstack(li2)
df1 = pd.DataFrame(li2).to_csv('./test.csv', index = False, header=True)           
df2 = pd.read_csv('./test.csv',chunksize=601)  
scenario = pd.DataFrame(inputScenarios)
scenario.columns=['scenario']

         
Mean=[]
SD=[]
Kurt=[]
Skew=[]
Var=[]

SEM=[]
RMS=[]
Crst=[]
MSD = []
#For the chunks of data in dataframe 2, loop through and append each statistical feature to a list
for chunk in df2:
    chunk.to_numpy()
    Mean.append(np.mean(chunk,axis=0))
    SD.append(np.std(chunk,axis=0))
    Var.append(np.var(chunk,axis=0))
    Kurt.append(chunk.kurtosis(axis=0))
    Skew.append(chunk.skew(axis=0))
    SEM.append(stats.sem(chunk,axis=0))
    MSD.append(stats.median_absolute_deviation(chunk,axis=0))
    RMS.append(np.sqrt(np.mean(chunk**2)))
    cf = np.max(np.abs(chunk))/np.sqrt(np.mean(np.square(chunk)))
    Crst.append(cf)

#Place the list of calculated values into a dataframe     
df_mean = pd.DataFrame(Mean)
df_sd = pd.DataFrame(SD)
df_kurt = pd.DataFrame(Kurt)
df_skew = pd.DataFrame(Skew)
df_var = pd.DataFrame(Var)
df_sem = pd.DataFrame(SEM)
df_rms = pd.DataFrame(RMS)
df_crst = pd.DataFrame(Crst)
df_MSD = pd.DataFrame(MSD)


data = pd.concat([df_mean,df_sd,df_kurt,df_skew,df_var,df_sem,df_rms,df_crst,df_MSD],axis=1)

df3 = pd.concat([data,scenario],axis=1)   

database = pd.DataFrame(df3)

dataframe = database[database.scenario != 0]


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#Define dataset 
X = dataframe.iloc[0:2359,0:54].values 
y = dataframe.iloc[0:2359,54]


encoder = LabelEncoder()
encoded_Y = encoder.fit_transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)



# from sklearn.linear_model import LogisticRegression
# from sklearn.multiclass import OneVsRestClassifier

#  # define model
# model = LogisticRegression()
# # define the ovr strategy
# ovr = OneVsRestClassifier(model)
# # fit model
# ovr.fit(X, y)
# # make predictions
# yhat = ovr.predict(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,dummy_y, test_size=0.3, stratify=y,random_state=0)


# X_train = X_train.reshape((X_train.shape[0], X_train.shape[1],1))
# X_test = X_test.reshape((X_test.shape[0], X_test.shape[1],1))


from sklearn import preprocessing

from keras.optimizers import SGD

model = Sequential()
model.add(Dense(1000, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.1))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(y_train.shape[1], activation='sigmoid'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer=sgd, metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_split=0.3, epochs=100, batch_size=500)
accuracy = model.evaluate(X_test, y_test, batch_size=500, verbose=0)

preds = model.predict(X_test)
preds[preds>=0.5] = 1
preds[preds<0.5] = 0

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

metrics.f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))

plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(r'C:\Users\fiyin\OneDrive\Documents\project\Model_Acc.svg') 
plt.show()            

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.savefig(r'C:\Users\fiyin\OneDrive\Documents\project\Model_Loss.svg') 
plt.show()


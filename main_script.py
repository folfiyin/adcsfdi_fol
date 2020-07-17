#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from scipy.stats import entropy
from scipy import stats
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.pipeline import Pipeline

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
            x_scaled = preprocessing.MinMaxScaler().fit_transform(df)
            li2.append(x_scaled)
            inputsScenarios.appned(int(entry.name.split('_')[1]))


print('Reading Done!')
#Concatenate to a datafrme and read from that dataframe

li2 = np.vstack(li2)
df1 = pd.DataFrame(li2,columns=['q1_faulty','q2_faulty','q3_faulty','o1_faulty','o2_faulty','o3_faulty']).to_csv('./test.csv', index = False, header=True)           
df2 = pd.read_csv('./test.csv',chunksize=601)  
      
      
scenario.remove('scenario')
Scenario = pd.DataFrame(scenario)
Scenario.columns=['scenario']


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
    Mean.append(np.mean(chunk))
    SD.append(np.std(chunk))
    Var.append(np.var(chunk))
    Kurt.append(chunk.kurtosis())
    Skew.append(chunk.skew())
    SEM.append(stats.sem(chunk))
    MSD.append(stats.median_absolute_deviation(chunk))
    RMS.append(np.sqrt(np.mean(chunk**2)))
    cf = np.max(np.abs(chunk))/np.sqrt(np.mean(np.square(chunk)))
    Crst.append(cf)

#Place the list of calculated values into a dataframe     
df_mean = pd.DataFrame(Mean)
df_mean.columns = ['MeanQ1', 'MeanQ2', 'MeanQ3','MeanO1','MeanO2','MeanO3']
df_sd = pd.DataFrame(SD)
df_sd.columns = ['SdQ1','SdQ2','SdQ3','SdO1','SdO2','SdO3']
df_kurt = pd.DataFrame(Kurt)
df_kurt.columns = ['KQ1','KQ2','KQ3','KO1','KO2','KO3']
df_skew = pd.DataFrame(Skew)
df_skew.columns = ['SkQ1','SkQ2','SkQ3','SkO1','SkO2','SkO3']
df_var = pd.DataFrame(Var)
df_var.columns = ['VarQ1','VarQ2','VarQ3','VarO1','VarO2','VarO3'] 
  
df_sem = pd.DataFrame(SEM)
df_sem.columns = ['SemQ1','SemQ2','SemQ3','SemO1','SemO2','SemO3']
df_rms = pd.DataFrame(RMS)
df_rms.columns = ['RMSQ1','RMSQ2','RMSQ3','RMSO1','RMSO2','RMSO3']
df_crst = pd.DataFrame(Crst)
df_crst.columns = ['CRSTQ1','CRSTQ2','CRSTQ3','CRSTO1','CRSTO2','CRSTO3']
df_MSD = pd.DataFrame(MSD)
df_MSD.columns = ['MSDQ1','MSDQ2','MSDQ3','MSDO1','MSDO2','MSDO3']
#Open the input csv file and take the scenario column only 

# [ idea x 6 ]
# [ . ]
# [ . ]
# [ N (num of datasets in ID order) ]
# N x (idea x 6 ) array

data = pd.concat([df_mean,df_sd,df_kurt,df_skew,df_var,df_sem,df_rms,df_crst,df_MSD],axis=1)
# data.drop(data.tail(4).index,inplace=True)

# # df_zscore = (data - data.mean())/data.std()

database = pd.concat([data,Scenario],axis=1)
# database.drop(database.tail(13).index,inplace=True)
# print(database)

from sklearn.model_selection import train_test_split,cross_val_score,KFold
# # from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# # from sklearn.preprocessing import StandardScaler


X = database.iloc[625:3124,0:55].values 
y = database.iloc[625:3124,54]

encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

# from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,dummy_y, test_size=0.3, stratify=y,random_state=0)

# # sc = StandardScaler()
# # X_train = sc.fit_transform(X_train)
# # X_test = sc.transform(X_test)

from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
# define dataset
# get a list of models to evaluate
# def get_models():
#   models = dict()
#   for i in range(2, 10):
#       rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=i)
#       model = DecisionTreeClassifier()
#       models[str(i)] = Pipeline(steps=[('s',rfe),('m',model)])
#   return models
 
# # evaluate a give model using cross-validation
# def evaluate_model(model):
#   cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=0)
#   scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
#   return scores
# models = get_models()
# # evaluate the models and store results
# results, names = list(), list()
# for name, model in models.items():
#   scores = evaluate_model(model)
#   results.append(scores)
#   names.append(name)
#   print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# # plot model performance for comparison
# plt.boxplot(results, labels=names, showmeans=True)
# plt.show()

import keras
from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, SpatialDropout1D
from keras.layers import Conv1D, MaxPooling1D
from keras import backend as K
from keras.utils import np_utils
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import LSTM
from sklearn.preprocessing import StandardScaler

# X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size=0.3)
num_classes = 4
 
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1],1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1],1))

model = Sequential()
model.add(Conv1D(filters=16, kernel_size=3, input_shape=(55, 1)))
# model.add(MaxPooling1D(pool_size=5))
model.add(SpatialDropout1D(0.5))
model.add(LSTM(30, dropout=0.3, recurrent_dropout=0.3))
model.add(Dense(10, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,validation_split=0.3, batch_size=50, epochs=100)
score = model.evaluate(X_test, y_test, batch_size=300, verbose=1)
y_pred = model.predict(X_test)

y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

metrics.f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))


# Making the Confusion Matrix and Classification report 
x_axis_labels = [1,2,3,4]
y_axis_labels = [1,2,3,4]
cm = confusion_matrix(y_test, y_pred,labels=np.unique(y_pred),normalize='true')
print(cm)
print('Classification Report\n',classification_report(y_test, y_pred))

import seaborn as sn
sn.set(font_scale=1) # for label size
sn.heatmap(cm, annot=True,cbar=False, xticklabels=x_axis_labels,yticklabels=y_axis_labels, annot_kws={"size": 16},fmt=".1f") # font size
plt.xlabel('Scenarios')
plt.ylabel('Scenarios')
# plt.show()
plt.savefig('./Confusion_matrix.svg')
plt.show()

plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('./Model_Acc.svg') 
plt.show()            

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.savefig('./Model_Loss.svg') 
plt.show()

def summarize_results(score):
    print(score)
    m, s = mean(score), std(score)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
    

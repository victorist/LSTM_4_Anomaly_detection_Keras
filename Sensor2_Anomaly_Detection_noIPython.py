# Bearing Failure Anomaly Detection
# In this workbook, we use an autoencoder neural network to 
# identify vibrational anomalies from sensor readings in a set of bearings. 
# The goal is to be able to predict future bearing failures before they happen. 
# The vibrational sensor readings are from the NASA Acoustics and Vibration Database. 
# Each data set consists of individual files that are 1-second vibration signal snapshots
#  recorded at 10 minute intervals. 
# Each file contains 20,480 sensor data points that were obtained by reading 
# the bearing sensors at a sampling rate of 20 kHz.
# This autoencoder neural network model is created using Long Short-Term Memory 
# (LSTM) recurrent neural network (RNN) cells within the Keras / TensorFlow framework.
# using Tensorfow version 2.x

# import libraries
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import joblib

import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt

from numpy.random import seed
# from tensorflow import set_random_seed
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf.random.set_seed(123) 

from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers
print('Библиотеки импортированы успешно.')
# # Data loading and pre-processing
# An assumption is that mechanical degradation in the bearings occurs gradually over time; 
# therefore, we use one datapoint every 10 minutes in the analysis. 
# Each 10 minute datapoint is aggregated by using the mean absolute value of 
# the vibration recordings over the 20,480 datapoints in each file. 
# We then merge together everything in a single dataframe.

save_dir = 'results/bearing_failure/'
print()
print('Директория с данныйми: ', save_dir)
# load, average and merge sensor samples
# data_dir = 'data/bearing_data'
data_dir = '/Users/vistratov/dev_data/data/bearing_data'
merged_data = pd.DataFrame()

for filename in os.listdir(data_dir):
    dataset = pd.read_csv(os.path.join(data_dir, filename), sep='\t')
    dataset_mean_abs = np.array(dataset.abs().mean())
    dataset_mean_abs = pd.DataFrame(dataset_mean_abs.reshape(1,4))
    dataset_mean_abs.index = [filename]
    merged_data = merged_data.append(dataset_mean_abs)
    
merged_data.columns = ['Bearing 1', 'Bearing 2', 'Bearing 3', 'Bearing 4']

# transform data file index to datetime and sort in chronological order
merged_data.index = pd.to_datetime(merged_data.index, format='%Y.%m.%d.%H.%M.%S')
merged_data = merged_data.sort_index()

merged_data.to_csv(save_dir + 'Averaged_BearingTest_Dataset.csv')
print("\nDataset seved in the"+ save_dir +"Averaged_BearingTest_Dataset.csv\n")
print("Dataset shape:", merged_data.shape)
print("\nDataframe:\n")
print(merged_data.head(-3))

# # Define train/test data
# Before setting up the models, we need to define train/test data. To do this, we perform a simple split where we train on the first part of the dataset (which should represent normal operating conditions) and test on the remaining parts of the dataset leading up to the bearing failure.

train = merged_data['2004-02-12 10:52:39': '2004-02-15 12:52:39']
test = merged_data['2004-02-15 12:52:39':]
print("\nTraining dataset shape:", train.shape)
print("Test dataset shape:", test.shape)


fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
ax.plot(train['Bearing 1'], label='Bearing 1', color='blue', animated = True, linewidth=1)
ax.plot(train['Bearing 2'], label='Bearing 2', color='red', animated = True, linewidth=1)
ax.plot(train['Bearing 3'], label='Bearing 3', color='green', animated = True, linewidth=1)
ax.plot(train['Bearing 4'], label='Bearing 4', color='black', animated = True, linewidth=1)
plt.legend(loc='lower left')
ax.set_title('Show 1 - Bearing Sensor Training Data', fontsize=16)
#plt.show()
plt.savefig(save_dir+'Show 1 - Bearing Sensor Training Data.pdf')


# Let’s get a different perspective of the data by transforming 
# the signal from the time domain to the frequency domain using a discrete Fourier transform.

# transforming data from the time domain to the frequency domain using fast Fourier transform
train_fft = np.fft.fft(train)
test_fft = np.fft.fft(test)

# frequencies of the healthy sensor signal
fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
ax.plot(train_fft[:,0].real, label='Bearing 1', color='blue', animated = True, linewidth=1)
ax.plot(train_fft[:,1].imag, label='Bearing 2', color='red', animated = True, linewidth=1)
ax.plot(train_fft[:,2].real, label='Bearing 3', color='green', animated = True, linewidth=1)
ax.plot(train_fft[:,3].real, label='Bearing 4', color='black', animated = True, linewidth=1)
plt.legend(loc='lower left')
ax.set_title('Show 2 - Bearing Sensor Training Frequency Data', fontsize=16)
# plt.show()
plt.savefig(save_dir+'Show 2 - Bearing Sensor Training Frequency Data.pdf')

# frequencies of the degrading sensor signal
fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
ax.plot(test_fft[:,0].real, label='Bearing 1', color='blue', animated = True, linewidth=1)
ax.plot(test_fft[:,1].imag, label='Bearing 2', color='red', animated = True, linewidth=1)
ax.plot(test_fft[:,2].real, label='Bearing 3', color='green', animated = True, linewidth=1)
ax.plot(test_fft[:,3].real, label='Bearing 4', color='black', animated = True, linewidth=1)
plt.legend(loc='lower left')
ax.set_title('Show 3 - Bearing Sensor Test Frequency Data', fontsize=16)
# plt.show()
plt.savefig(save_dir+'Show 3 - Bearing Sensor Test Frequency Data.pdf')

# normalize the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(train)
X_test = scaler.transform(test)
scaler_filename = "scaler_data"
joblib.dump(scaler, scaler_filename)

# reshape inputs for LSTM [samples, timesteps, features]
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
print("\nTraining data shape:", X_train.shape)
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
print("Test data shape:", X_test.shape)

# MODEL
# define the autoencoder network model

def autoencoder_model(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = LSTM(16, activation='relu', return_sequences=True, 
                kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(4, activation='relu', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(4, activation='relu', return_sequences=True)(L3)
    L5 = LSTM(16, activation='relu', return_sequences=True)(L4)
    output = TimeDistributed(Dense(X.shape[2]))(L5)    
    model = Model(inputs=inputs, outputs=output)
    return model

# create the autoencoder model
model = autoencoder_model(X_train)
model.compile(optimizer='adam', loss='mae')     
model.summary()

# fit the model to the data
print("\n>>> Training process started <<<\n")
nb_epochs = 100
batch_size = 5
history = model.fit(X_train, X_train, epochs=nb_epochs, batch_size=batch_size, validation_split=0.05).history
print("\n>>> Training process finished <<<\n")

# plot the training losses
fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
ax.plot(history['loss'], 'b', label='Train', linewidth=2)
ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
ax.set_title('Model loss', fontsize=16)
ax.set_ylabel('Loss (mae)')
ax.set_xlabel('Epoch')
ax.legend(loc='upper right')
# plt.show()
plt.savefig(save_dir+'Show 4 - The training losses.pdf')


# # Distribution of Loss Function
# By plotting the distribution of the calculated loss in the training set, 
# one can use this to identify a suitable threshold value for identifying an anomaly. 
# In doing this, one can make sure that this threshold is set above the “noise level” and 
# that any flagged anomalies should be statistically significant above the background noise.

# plot the loss distribution of the training set
X_pred = model.predict(X_train)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred = pd.DataFrame(X_pred, columns=train.columns)
X_pred.index = train.index

scored = pd.DataFrame(index=train.index)
Xtrain = X_train.reshape(X_train.shape[0], X_train.shape[2])
scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtrain), axis = 1)
plt.figure(figsize=(16,9), dpi=80)
plt.title('Show 5 - Loss Distribution', fontsize=16)
sns.distplot(scored['Loss_mae'], bins = 20, kde= True, color = 'blue');
plt.xlim([0.0,.5])
plt.savefig(save_dir+'Show 5 - Loss Distribution.pdf')


# From the above loss distribution, let's try a threshold value of 0.275 for flagging an anomaly. We can then calculate the loss in the test set to check when the output crosses the anomaly threshold.

# calculate the loss on the test set
X_pred = model.predict(X_test)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred = pd.DataFrame(X_pred, columns=test.columns)
X_pred.index = test.index

scored = pd.DataFrame(index=test.index)
Xtest = X_test.reshape(X_test.shape[0], X_test.shape[2])
scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtest), axis = 1)
scored['Threshold'] = 0.275
scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']
print("Scored:")
print(scored.head())
print(scored.tail())

# calculate the same metrics for the training set 
# and merge all data in a single dataframe for plotting
X_pred_train = model.predict(X_train)
X_pred_train = X_pred_train.reshape(X_pred_train.shape[0], X_pred_train.shape[2])
X_pred_train = pd.DataFrame(X_pred_train, columns=train.columns)
X_pred_train.index = train.index

scored_train = pd.DataFrame(index=train.index)
scored_train['Loss_mae'] = np.mean(np.abs(X_pred_train-Xtrain), axis = 1)
scored_train['Threshold'] = 0.275
scored_train['Anomaly'] = scored_train['Loss_mae'] > scored_train['Threshold']
scored = pd.concat([scored_train, scored])

# Having calculated the loss distribution and the anomaly threshold, 
# we can visualize the model output in the time leading up to the bearing failure.


# plot bearing failure time plot
scored.plot(logy=True,  figsize=(16,9), ylim=[1e-2,1e2], color=['blue','red'])
plt.title('Show 6 - Visualize the model output', fontsize=16)
plt.savefig(save_dir+'Show 6 - Visualize the model output.pdf')

# This analysis approach is able to flag the upcoming bearing malfunction well 
# in advance of the actual physical failure. It is important to define a suitable 
# threshold value for flagging anomalies while avoiding too many false positives during normal operating conditions.
# save all model information, including weights, in h5 format
model.save(save_dir+"Cloud_model.h5")
print("\nModel saved in the Cloud_model.h5")

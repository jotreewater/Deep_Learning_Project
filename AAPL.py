# User defined
STOCK = 'AAPL'
TEST_RATIOS = [0.75,0.80,0.85]
LOOK_BACKS = [1]
MODELS = [1,2]
SIZES = [50,100,150]
EPOCHS = 12

# Get stock data
import pandas_datareader as pdr
key = '26be6a052b3fc4d0995f25cc8e182c5b4f477eb9'
df = pdr.get_data_tiingo(STOCK, api_key=key, start='1/1/00')
df.to_csv(f'{STOCK}.csv')

# Get Time-stamp
import time
TIME = time.localtime()
TIME = f'{TIME.tm_hour}-{TIME.tm_min}-{TIME.tm_sec}'
print(TIME)

#name = f'{STOCK}-{TEST_RATIO}-{LOOK_BACK}-{MODEL}-{SIZE}-{TIME}'
#print(name)

# Read stocks from csv
import pandas
dataframe = pandas.read_csv(f'{STOCK}.csv', usecols=[2], engine='python')
dataframe = dataframe.append(dataframe.iloc[len(dataframe)-1], ignore_index = True) # Repeats final stock data point to predict one point into the future

# Plot Dataset
import matplotlib.pyplot as plt
dataset = dataframe.values
dataset = dataset.astype('float32')
plt.plot(dataset)
plt.show() # Demonstration Purposes

# Plot Scale
import numpy
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
dataset=scaler.fit_transform(numpy.array(dataset).reshape(-1,1))
plt.plot(dataset)
plt.show() # Demonstration Purposes

# Keras dataset formatting
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

# Model Resources
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# Start Loop
best_model = 'name'
test_score = 1
for TEST_RATIO in TEST_RATIOS:
    #print(f'test ratio:{TEST_RATIO}')
    train_size = int(len(dataset) * TEST_RATIO)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    train = train.reshape(len(train),1,1)
    test = test.reshape(len(test),1,1)
    for LOOK_BACK in LOOK_BACKS:
        #print(f'look back:{LOOK_BACK}')
        trainX, trainY = create_dataset(train, LOOK_BACK)
        testX, testY = create_dataset(test, LOOK_BACK)
        for MODEL in MODELS:
            #print(f'model: :{MODEL}')
            for SIZE in SIZES:
                #print(f'size: :{SIZE}')
                name = f'{STOCK}-{TEST_RATIO}-{LOOK_BACK}-{MODEL}-{SIZE}-{TIME}'
                print(name)
                model = Sequential()
                if MODEL == 1:
                    model.add(Bidirectional(LSTM(SIZE, activation='tanh')))
                    model.add(Dense(1))
                if MODEL == 2:
                    model.add(LSTM(SIZE, activation='tanh', return_sequences=True))
                    model.add(LSTM(SIZE, activation='tanh', return_sequences=True))
                    model.add(LSTM(SIZE, activation='tanh'))
                    model.add(Dense(1))
                model.compile(loss='mean_squared_error', optimizer='adam')
                tensorboard = TensorBoard(log_dir=f"{STOCK}/{name}")
                filepath = f'{name}'  # unique file name that will include the epoch and the validation acc for that epoch
                checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='mean_squared_error', verbose=1, save_best_only=True, mode='max')) # saves only the best ones
                # For Accuracy
                model.fit(trainX, trainY, epochs=EPOCHS, batch_size=64, verbose=1, callbacks=[tensorboard, checkpoint])
                # For Speed
                # model.fit(trainX, trainY, epochs=EPOCHS, batch_size=64, verbose=1, callbacks=[tensorboard])
                new_test_score = model.evaluate(testX, testY, verbose=0)
                if new_test_score < test_score:
                    best_model = name
                    look_back = LOOK_BACK
                    test_score = new_test_score

# Load best model
from tensorflow import keras
model = keras.models.load_model(f'models/{best_model}.model')
print(best_model)
print(test_score)

# Evaluate and plot best performing model
import math
print('Test Score: %.8f MSE (%.8f RMSE)' % (test_score, math.sqrt(test_score)))

# Show Model Results
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
trainpredict=model.predict(trainX)
testpredict=model.predict(testX)

# Transform data back into USD and reshape for plotting
trainpredict=scaler.inverse_transform(trainpredict)
testpredict=scaler.inverse_transform(testpredict)
dataset=scaler.inverse_transform(dataset)
trainpredict = trainpredict.reshape(len(trainpredict),1)
testpredict = testpredict.reshape(len(testpredict),1)

# Shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainpredict)+look_back, :] = trainpredict

# Shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainpredict)+(look_back*2)+1:len(dataset)-1, :] = testpredict

# Plot baseline and predictions
plt.plot(dataset)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

# Print most recent stock price
print(dataframe.iloc[len(dataframe)-1])

# Print tomorrow's predicted stock price
print(testpredict[len(testpredict)-1])




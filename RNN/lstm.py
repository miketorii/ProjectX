import numpy as np
import matplotlib.pyplot as plt

print('----start LSTM----')

n = 500
t = np.linspace(0, 20.0*np.pi, n)
X = np.sin(t)

window = 10

last = int(n/5.0)
Xtrain = X[:-last]
Xtest = X[-last-window:]

xin = []
next_X = []

for i in range(window, len(Xtrain)):
    xin.append(Xtrain[i-window:i])
    next_X.append(Xtrain[i])
    
xin, next_X = np.array(xin), np.array(next_X)
xin = xin.reshape(xin.shape[0], xin.shape[1], 1)

#import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

m = Sequential()
m.add(LSTM(units=50, return_sequences=True, input_shape=(xin.shape[1],1)))
m.add(Dropout(0.2))
m.add(LSTM(units=50))
m.add(Dropout(0.2))
m.add(Dense(units=1))
m.compile(optimizer='adam', loss='mean_squared_error')

history = m.fit(xin, next_X, epochs = 50, batch_size = 50, verbose = 0)

plt.figure()
plt.ylabel('loss'); plt.xlabel('epoch')
plt.semilogy(history.history['loss'])
plt.show()

xin = []
next_X1 = []

for i in range(window, len(Xtrain)):
    xin.append(Xtrain[i-window:i])
    next_X1.append(Xtrain[i])
    
xin, next_X1 = np.array(xin), np.array(next_X1)
xin = xin.reshape(xin.shape[0], xin.shape[1], 1)

X_pred = m.predict(xin)

plt.figure()
plt.plot(X_pred, ':', label='LSTM')
plt.plot(next_X1, '--', label='Actual')
plt.legend()
plt.show()

X_pred = Xtest.copy()

for i in range(window, len(X_pred)):
    xin = X_pred[i-window:i].reshape((1, window, 1))
    X_pred[i] = m.predict(xin)

plt.figure()
plt.plot(X_pred[window:], ':', label='LSTM')
plt.plot(next_X1, '--', label='Actual')
plt.legend()
plt.show()    
    
print('----end----')







           

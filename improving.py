from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

steps_per_cycle = 50
number_of_cycles = 500
random_factor = 0.4
random.seed(0)

n = 1000
limit_low = 0
limit_high = 0.48
my_data = np.random.normal(0, 0.5, n) \
          + np.abs(np.random.normal(0, 2, n) \
                   * np.sin(np.linspace(0, 3 * np.pi, n))) \
          + np.sin(np.linspace(0, 5 * np.pi, n)) ** 2 \
          + np.sin(np.linspace(1, 6 * np.pi, n)) ** 2

scaling = (limit_high - limit_low) / (max(my_data) - min(my_data))
my_data = my_data * scaling

Y = my_data + (limit_low - min(my_data))

df = pd.DataFrame({"Signal_strength": Y})


def _load_data(data, n_prev=100):
    docX, docY = [], []
    for i in range(len(data) - n_prev):
        docX.append(data.iloc[i:i + n_prev].as_matrix())
        docY.append(data.iloc[i + n_prev].as_matrix())
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY


length_of_sequences = 2
test_size = 0.25
ntr = int(len(df) * (1 - test_size))
df_train = df.iloc[:ntr]

df_test = df.iloc[ntr:]
(X_train, y_train) = _load_data(df_train, n_prev=length_of_sequences)
(X_test, y_test) = _load_data(df_test, n_prev=length_of_sequences)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

neurons = 512
dropout = 0.25
output_size = 1
activ_func = 'tanh'
loss = 'mse'
optimizer = "adam"

model = Sequential()
model.add(LSTM(neurons, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), activation=activ_func))
model.add(Dropout(dropout))
# model.add(LSTM(neurons, return_sequences=True,input_shape=X_train.shape[2], activation=activ_func))
# model.add(Dropout(dropout))
model.add(LSTM(neurons, activation=activ_func))
model.add(Dropout(dropout))
model.add(Dense(units=output_size))
model.add(Activation(activ_func))
model.compile(loss=loss, optimizer=optimizer, metrics=['mae'])
model.summary()

model.fit(X_train, y_train, batch_size=100, epochs=100, verbose=False)
y_pred = model.predict(X_test)
plt.figure(figsize=(19, 3))

plt.plot(y_test, label="true")
plt.plot(y_pred, label="predicted")
plt.legend()
plt.show()

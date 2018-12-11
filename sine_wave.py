import pandas as pd
import numpy as np
import math

import random
import matplotlib.pyplot as plt

steps_per_cycle = 50
number_of_cycles = 500
random_factor = 0.4
random.seed(0)
df = pd.DataFrame(np.arange(steps_per_cycle * number_of_cycles + 1), columns=["t"])

df["sin_t"] = df.t.apply(
    lambda x: math.sin(x * (2 * math.pi / steps_per_cycle) + random.uniform(-1.0, +1.0) * random_factor))
df["sin_t_clean"] = df.t.apply(lambda x: math.sin(x * (2 * math.pi / steps_per_cycle)))


def _load_data(data, n_prev=100):
    """
    """

    docX, docY = [], []
    for i in range(len(data) - n_prev):
        docX.append(data.iloc[i:i + n_prev].as_matrix())
        # print(len(docX))
        docY.append(data.iloc[i + n_prev].as_matrix())
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY


length_of_sequences = 2
test_size = 0.25
ntr = int(len(df) * (1 - test_size))
df_train = df[["sin_t"]].iloc[:ntr]
df_test = df[["sin_t"]].iloc[ntr:]
(X_train, y_train) = _load_data(df_train, n_prev=length_of_sequences)
(X_test, y_test) = _load_data(df_test, n_prev=length_of_sequences)


from keras.layers import Input
from keras.models import Model
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import SimpleRNN


def define_model(length_of_sequences, batch_size=None, stateful=False):
    in_out_neurons = 1
    hidden_neurons = 1
    inp = Input(batch_shape=(batch_size,
                             length_of_sequences,
                             in_out_neurons))

    rnn = SimpleRNN(hidden_neurons,
                    return_sequences=False,
                    stateful=stateful,
                    name="RNN")(inp)

    dens = Dense(in_out_neurons, name="dense")(rnn)
    model = Model(inputs=[inp], outputs=[dens])

    model.compile(loss="mean_squared_error", optimizer="rmsprop")

    return (model, (inp, rnn, dens))

model, (inp, rnn, dens) = define_model(length_of_sequences=X_train.shape[1])
model.summary()

hist = model.fit(X_train, y_train, batch_size=600, epochs=1000,
                 verbose=False, validation_split=0.05)
y_pred = model.predict(X_test)
plt.figure(figsize=(19, 3))
plt.plot(y_test, label="true")
plt.plot(y_pred, label="predicted")
plt.legend()
plt.show()

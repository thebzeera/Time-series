import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing

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
X = np.arange(1, 1001)

df = pd.DataFrame({"Time": X, "Signal_strength": Y})

train = df[0:750]
test = df[750:]

dd = np.asarray(train.Signal_strength)
y_hat = test.copy()
y_hat['naive'] = dd[len(dd) - 1]  # Naive method
plt.figure(figsize=(12, 8))
plt.plot(train.index, train['Signal_strength'], label='Train')
plt.plot(test.index, test['Signal_strength'], label='Test')
plt.plot(y_hat.index, y_hat['naive'], label='Naive Forecast')
plt.legend(loc='best')
plt.title("Naive Forecast")
plt.show()

rms = sqrt(mean_squared_error(test.Signal_strength, y_hat.naive))  # Root mean square
print(rms, ":Using naive")

y_hat_avg = test.copy()
y_hat_avg['avg_forecast'] = train['Signal_strength'].mean()  # Simple average
plt.figure(figsize=(12, 8))
plt.plot(train['Signal_strength'], label='Train')
plt.plot(test['Signal_strength'], label='Test')
plt.plot(y_hat_avg['avg_forecast'], label='Average Forecast')
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(test.Signal_strength, y_hat_avg.avg_forecast))
print(rms, ":Using simple average")

y_hat_avg = test.copy()
y_hat_avg['moving_avg_forecast'] = train['Signal_strength'].rolling(60).mean().iloc[-1]  # Moving Average
plt.figure(figsize=(16, 8))
plt.plot(train['Signal_strength'], label='Train')
plt.plot(test['Signal_strength'], label='Test')
plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast')
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(test.Signal_strength, y_hat_avg.moving_avg_forecast))
print(rms, ":Using Moving Average")

y_hat_avg = test.copy()
fit2 = SimpleExpSmoothing(np.asarray(train['Signal_strength'])).fit(smoothing_level=0.6,
                                                                    optimized=False)  # simple exponentional Smoothing
y_hat_avg['SES'] = fit2.forecast(len(test))
plt.figure(figsize=(16, 8))
plt.plot(train['Signal_strength'], label='Train')
plt.plot(test['Signal_strength'], label='Test')
plt.plot(y_hat_avg['SES'], label='SES')
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(test.Signal_strength, y_hat_avg.SES))
print(rms, ":Using Simple Exponential Smoothing")

y_hat_avg = test.copy()
fit1 = Holt(np.asarray(train['Signal_strength'])).fit(smoothing_level=0.3, smoothing_slope=0.1)
y_hat_avg['Holt_linear'] = fit1.forecast(len(test))
plt.figure(figsize=(16, 8))
plt.plot(train['Signal_strength'], label='Train')
plt.plot(test['Signal_strength'], label='Test')
plt.plot(y_hat_avg['Holt_linear'], label='Holt_linear')
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(test.Signal_strength, y_hat_avg.Holt_linear))
print(rms, ":using holt linear")

y_hat_avg = test.copy()
fit1 = ExponentialSmoothing(np.asarray(train['Signal_strength']), seasonal_periods=7, trend='add',
                            seasonal='add', ).fit()  # Holt_Winter
y_hat_avg['Holt_Winter'] = fit1.forecast(len(test))
plt.figure(figsize=(16, 8))
plt.plot(train['Signal_strength'], label='Train')
plt.plot(test['Signal_strength'], label='Test')
plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
plt.legend(loc='best')

rms = sqrt(mean_squared_error(test.Signal_strength, y_hat_avg.Holt_Winter))
print(rms, ":using holt winter")

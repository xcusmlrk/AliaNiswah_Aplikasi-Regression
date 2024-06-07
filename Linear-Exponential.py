import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

# Import data diambil dari kaggle, Jumlah latihan soal (NL) dan Nilai ujian siswa (NT)
NL = np.array([1, 2, 2, 2, 5, 6, 6, 6])
NT = np.array([91, 65, 45, 36, 66, 61, 63, 42])

# Exponential Regression
X = NL.reshape(-1, 1)
y = NT
log_y = np.log(y)
exponential_model = LinearRegression()
exponential_model.fit(X, log_y)
log_y_pred = exponential_model.predict(X)
y_pred_exponential = np.exp(log_y_pred)
rms_error_exponential = np.sqrt(np.mean((y - y_pred_exponential) ** 2))
print("Galat RMS (Exponential Regression):", rms_error_exponential)

def test_regression_exponential():
    assert isinstance(rms_error_exponential, float), "Invalid linear RMS error"
    assert len(y_pred_exponential) == len(NL), "Linear prediction does not match the number of data points"
    print("All Exponential Regression tests passed.")

test_regression_exponential()

# Linear Regression
def linear_regression(x, m, c):
    return m * x + c

popt, pcov = curve_fit(linear_regression, NL, NT)

m, c = popt

y_pred_linear = linear_regression(NL, m, c)
rms_error_linear = np.sqrt(np.mean((NT - y_pred_linear) ** 2))
print("Galat RMS (Linear Regression):", rms_error_linear)

def test_regression_liner():
    assert isinstance(rms_error_linear, float), "Invalid linear RMS error"
    assert len(y_pred_linear) == len(NL), "Linear prediction does not match the number of data points"
    print("All Regerssion Linear tests passed.")

test_regression_liner()

# Plotting the graph
plt.scatter(NL, NT, label='Original Data')

# Plot Exponential Regression
plt.plot(NL, y_pred_exponential, color='green', label='Exponential Regression')

# Plot Linear Regression
plt.plot(NL, y_pred_linear, color='red', label='Linear Regression')

plt.xlabel('Number of Exercises (NL)')
plt.ylabel('Test Scores (NT)')
plt.legend()
plt.title('Relationship between Number of Exercises and Test Scores\n')
plt.show()


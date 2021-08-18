import numpy as np
from scipy.signal import find_peaks
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

#### Read and plot data ####
data = np.loadtxt("data/twoTracesToShow.csv", skiprows=1, delimiter=",")
data = data.T
x = data[0]
y1 = data[1]
y2 = data[2]
_, axes = plt.subplots(2)
axes[0].plot(x, y1)
axes[1].plot(x, y2)
plt.show()

#### Find peaks ####
indices1 = find_peaks(y1)[0]  # gives the indices of where the peaks are
indices2 = find_peaks(y2)[0]
peaksX1, peaksY1 = x[indices1], y1[indices1]
peaksX2, peaksY2 = x[indices2], y2[indices2]
_, axes = plt.subplots(2)
axes[0].plot(x, y1)
axes[0].scatter(peaksX1, peaksY1, marker="x", color="red")
axes[1].plot(x, y2)
axes[1].scatter(peaksX2, peaksY2, marker="x", color="red")
plt.show()

#### Data with background ####
data = np.loadtxt("data/traceWithBackground.csv", skiprows=1, delimiter=",")
data = data.T
x = data[0]
y = data[1]
plt.plot(x, y)
plt.show()

x0, x1, x2 = np.polynomial.polynomial.polyfit(x, y, 2)
background = x0 + x1 * x + x2 * x ** 2
plt.plot(x, y)
plt.plot(x, background, linestyle="--")
plt.show()

y -= background
y -= np.min(y)
plt.plot(x, y)
plt.show()  # Not perfect, yet much much better with a simple fit.

#### Statistical analysis ####
data = np.loadtxt("data/populationsToCompare9p8.csv", skiprows=1, delimiter=",")
data = data.T
x1 = data[0]
x2 = data[1]
stats = ttest_ind(x1, x2, equal_var=False)
print(stats)

data = np.loadtxt("data/populationsToCompare.csv", skiprows=1, delimiter=",")
data = data.T
x1 = data[0]
x2 = data[1]
stats = ttest_ind(x1, x2, equal_var=False)
print(stats)

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd
from typing import Callable
from scipy.stats import ttest_ind


class TraceCalcique:

    def __init__(self, xmin: float, xmax: float, nbX: int, espacementMoyen: float, espacementEcartType: float,
                 nbPics: int = 10):
        self._espacementMoyen = espacementMoyen
        self._espacementEcartType = espacementEcartType
        self._nbPics = nbPics
        self._x = np.linspace(xmin, xmax, nbX)
        self._espacements = np.random.normal(espacementMoyen, espacementEcartType, nbPics - 1)
        espacementsCumsum = np.append(0, np.cumsum(self._espacements))
        self._positions = espacementsCumsum
        self._y = np.zeros_like(self._x)
        for x_ in self._positions:
            self._y[self._x >= x_] += np.exp(-(self._x - x_)[self._x >= x_])

    @property
    def x(self):
        return self._x.copy()

    @property
    def y(self):
        return self._y.copy()

    @property
    def espacements(self):
        return self._espacements.copy()

    @property
    def positions(self):
        return self._positions.copy()

    def espacementsEstimes(self):
        peaks = find_peaks(self._y)[0]
        xpeaks = self._x[peaks]
        ypeaks = self._y[peaks]
        return xpeaks.copy(), ypeaks.copy()

    def ajouterBckg(self, eq: Callable):
        self._y += eq(self._x)

    def enleverBckg(self, eq: Callable):
        self._y -= eq(self._x)
        self._y -= np.min(self._y)

    def sauvegarderTrace(self, nomFichier: str):
        data = np.vstack([self._x, self._y])
        data = data.T
        df = pd.DataFrame(data, columns=["x", "y"])
        df.to_csv(nomFichier, index=False)

    def afficherPics(self, axis: plt.Axes = None, show: bool = True, afficherEspacementsEstimes: bool = True):
        if axis is None:
            fig, axis = plt.subplots()
        axis.plot(self._x, self._y, label="Trace calcique")
        if afficherEspacementsEstimes:
            xpeaks, ypeaks = self.espacementsEstimes()
            axis.scatter(xpeaks, ypeaks, color="red", marker="x", label="Espacements estimés (avec find_peaks)")
        axis.legend()
        axis.set_xlabel("Temps [u.a.]")
        axis.set_ylabel("Amplitude [u.a.]")
        if show:
            plt.show()
        return axis


class TracesCalciques:

    def __init__(self, *tracesCalciques: TraceCalcique):
        self._traces = tracesCalciques

    def afficherTracesCalciques(self):
        nbTraces = len(self._traces)
        fig, axes = plt.subplots(nbTraces)
        for trace, axis in zip(self._traces, axes):
            trace.afficherPics(axis, False)
        plt.show()

    def sauvegarderTraces(self, nomFichier: str):
        allY = []
        x = []
        for trace in self._traces:
            allY.append(trace.y)
            x = trace.x
        data = np.vstack([x, *allY])
        data = data.T
        columns = ["x"] + [f"y{i + 1}" for i in range(len(allY))]
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(nomFichier, index=False)


class GeneratingLotsOfSpikes:

    def __init__(self, spacingMean: float, spacingStdDev: float, nbSpikes: int = 1000):
        self._spacingMean = spacingMean
        self._spacingStdDev = spacingStdDev
        self._nbSpikes = nbSpikes
        self._spacings = np.random.normal(self._spacingMean, self._spacingStdDev, self._nbSpikes)

    @property
    def spacings(self):
        return self._spacings.copy()

    def showSpacingsHistogram(self, nbBins: int = None, density: bool = True, axis: plt.Axes = None, show: bool = True,
                              **histKwargs):
        if axis is None:
            fig, axis = plt.subplots()
        if nbBins is None:
            nbBins = int(self._nbSpikes ** 0.5)
        axis.hist(self._spacings, nbBins, density=density, **histKwargs)
        if show:
            plt.show()


class GeneratingLotsOfSpikes_differentPopulations:

    def __init__(self, *spikes: GeneratingLotsOfSpikes):
        self._spikes = spikes

    def showPopulations(self, densities: bool = True):
        fig, axis = plt.subplots()
        for i, spike in enumerate(self._spikes):
            spike.showSpacingsHistogram(None, densities, axis, False, alpha=0.5, label=f"Pop {i + 1}")
        plt.legend()
        plt.show()

    def savePopulations(self, filename: str):
        cols = []
        allSpikes = []
        for i, spike in enumerate(self._spikes):
            cols.append(f"Pop {i + 1}")
            allSpikes.append(spike.spacings)
        data = np.vstack(allSpikes).T
        df = pd.DataFrame(data, columns=cols)
        df.to_csv(filename, index=False)


def computeTTest(sample1, sample2):
    return ttest_ind(sample1, sample2, equal_var=False)


tc = TraceCalcique(-5, 105, 1000, 10, 1.5)
tc2 = TraceCalcique(-5, 105, 1000, 15, 1.6)
traces = TracesCalciques(tc, tc2)
# traces.sauvegarderTraces("workshopData/twoTracesToShow.csv")

tcBckg = TraceCalcique(-5, 105, 1000, 28, 1.5, 4)
tcBckg.afficherPics()
tcBckg.ajouterBckg(lambda x: -0.0002 * x ** 2 + 0.001 * x + 100)
tcBckg.afficherPics()
# tcBckg.sauvegarderTrace("workshopData/traceWithBackground.csv")
x = tcBckg.x
y = tcBckg.y
x0, x1, x2 = np.polynomial.polynomial.polyfit(x, y, 2)
tcBckg.enleverBckg(lambda x: x0 + x1 * x + x2 * x ** 2)
tcBckg.afficherPics()
# exit()
spikes = GeneratingLotsOfSpikes(10, 1.5, 1000)
spikes2 = GeneratingLotsOfSpikes(12, 2, 1000)
spikes_pop = GeneratingLotsOfSpikes_differentPopulations(spikes, spikes2)
spikes_pop.showPopulations()
# spikes_pop.savePopulations("workshopData/populationsToCompare.csv")
print(computeTTest(spikes.spacings, spikes2.spacings))

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd


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


tc = TraceCalcique(-5, 105, 1000, 10, 1.5)
tc2 = TraceCalcique(-5, 105, 1000, 15, 1.6)
traces = TracesCalciques(tc, tc2)
traces.sauvegarderTraces("data/twoTracesToShow.csv")


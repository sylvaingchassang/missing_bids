import pandas as pd
import numpy as np
import lazy_property
import abc
from six import add_metaclass


class Deviations(object):

    def __init__(self, bids, deviations, beliefs, costs):
        self._bids = np.array(bids).reshape((-1, 1))
        self._deviations = np.array([0] + deviations).reshape(1, -1)
        self._beliefs = np.array(beliefs).reshape(
            (self._bids.shape[0], self._deviations.shape[1]))
        self._costs = np.array(costs).reshape((-1, 1))

    @lazy_property.LazyProperty
    def bids_and_deviations(self):
        return self._bids * (1 + self._deviations)

    @lazy_property.LazyProperty
    def profits(self):
        return np.multiply(self._beliefs, self.bids_and_deviations -
                           self._costs)

    @lazy_property.LazyProperty
    def equilibrium_profits(self):
        return self.profits[:, 0].reshape(-1, 1)

    @lazy_property.LazyProperty
    def deviation_profits(self):
        return self.profits[:, 1:].reshape(-1, self._deviations.shape[1]-1)

    @lazy_property.LazyProperty
    def is_competitive(self):
        return 1. * (np.isclose(self.deviation_temptation, 0))

    @lazy_property.LazyProperty
    def deviation_temptation(self):
        return (np.max(self.profits, axis=1).reshape(-1, 1)
                - self.equilibrium_profits)

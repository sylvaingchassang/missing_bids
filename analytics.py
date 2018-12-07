import pandas as pd
import numpy as np
import lazy_property
import abc
from six import add_metaclass


class Environments(object):

    def __init__(self, num_actions, constraints=[]):
        self._num_actions = num_actions
        self._constraints = constraints

    def generate_environments(self, num=1e6, seed=0):
        raw_environments = self._generate_raw_environments(num, seed)
        return self._apply_constraints(raw_environments)

    def _generate_raw_environments(self, num, seed):
        np.random.seed(seed)
        env = np.random.rand(int(num), self._num_actions + 1)
        env = self._descending_sort_beliefs(env)
        return env

    @staticmethod
    def _descending_sort_beliefs(env):
        env[:, :-1] = -np.sort(-env[:, :-1], axis=1)
        return env

    def _apply_constraints(self, env):
        env = env[np.apply_along_axis(self._aggregate_constraint, 1, env), :]
        return env

    def _aggregate_constraint(self, e):
        if len(self._constraints) > 0:
            return all([cstr(e) for cstr in self._constraints])
        else:
            return True


@add_metaclass(abc.ABCMeta)
class PlausibilityConstraint(object):

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def __call__(self, e):
        pass


class InformationConstraint(PlausibilityConstraint):

    def __init__(self, k, sample_demands):
        self._k = k
        self._sample_demands = sample_demands

    @staticmethod
    def _inverse_llr(D, x):
        numerator = (D/(1.-D)) * np.exp(x)
        return numerator/(1 + numerator)

    @lazy_property.LazyProperty
    def belief_bounds(self):
        return [[self._inverse_llr(d, -self._k),
                 self._inverse_llr(d, self._k)]
                for d in self._sample_demands]

    def __call__(self, e):
        beliefs = e[:-1]
        list_constraints = [
            bound[0] <= belief <= bound[1]
            for bound, belief in zip(self.belief_bounds, beliefs)
        ]
        return all(list_constraints)


class MarkupConstraint(PlausibilityConstraint):

    def __init__(self, max_markup=.5):
        self._min_cost_ratio = 1/(1. + max_markup)

    def __call__(self, e):
        return e[-1] >= self._min_cost_ratio


class DimensionlessCollusionMetrics(object):

    @property
    def is_non_competitive(self):
        pass

    @property
    def normalized_deviation_temptation(self):
        pass


class MinCollusionSolver(object):

    pass


@add_metaclass(abc.ABCMeta)
class SolverWrap(object):
    pass


class ScipySolverWrap(SolverWrap):
    pass


class CvxpySolverWrap(SolverWrap):
    pass




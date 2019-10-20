import abc
import lazy_property
import numpy as np
from six import add_metaclass


@add_metaclass(abc.ABCMeta)
class EnvironmentBase:
    def __init__(self, num_actions, constraints=None,
                 project_constraint=False, initial_guesses=None):
        self._num_actions = num_actions
        self._constraints = constraints
        self._project_constraint = project_constraint
        self._initial_guesses = \
            initial_guesses if initial_guesses is not None else np.array([])

    def generate_environments(self, num_points=1e6, seed=0):
        raw_environments = self._generate_raw_environments(num_points, seed)
        raw_environments = self._apply_constraints(raw_environments)
        if len(raw_environments) == 0:
            raise ValueError('Constraints are not satisfied by generated '
                             'environments')
        return self._append_initial_guesses(raw_environments).round(9)

    @abc.abstractmethod
    def _generate_raw_environments(self, num, seed):
        """return num * pbm dim array"""

    def _append_initial_guesses(self, environments):
        if self._initial_guesses.size == 0:
            return environments
        else:
            guesses = self._apply_constraints(self._initial_guesses, False)
            return np.concatenate((environments, guesses), axis=0)

    def _apply_constraints(self, env, proj=None):
        proj = self._project_constraint if proj is None else proj
        if proj:
            env = self._project_on_constraint(env)
        env = env[np.apply_along_axis(self._aggregate_constraint, 1, env), :]
        return env

    def _project_on_constraint(self, env):
        for constraint in self._constraints:
            env = constraint.project(env)
        return env

    def _aggregate_constraint(self, e):
        return not self._constraints or all(f(e) for f in self._constraints)


class Environment(EnvironmentBase):

    def _generate_raw_environments(self, num, seed):
        np.random.seed(seed)
        env = np.random.rand(int(num), self._num_actions + 1)
        env = self._descending_sort_beliefs(env)
        return env

    @staticmethod
    def _descending_sort_beliefs(env):
        env[:, :-1] = descending_sort(env[:, :-1])
        return env


def descending_sort(arr, axis=1):
    return -np.sort(-np.array(arr), axis=axis)


@add_metaclass(abc.ABCMeta)
class PlausibilityConstraint(object):

    @abc.abstractmethod
    def __call__(self, e):
        """"""

    @abc.abstractmethod
    def project(self, e):
        """"""


class InformationConstraint(PlausibilityConstraint):

    def __init__(self, k, sample_demands):
        self._k = k
        self._sample_demands = sample_demands

    @staticmethod
    def _inverse_llr(d, x):
        numerator = (d / (1. - d)) * np.exp(x)
        return numerator / (1 + numerator)

    @lazy_property.LazyProperty
    def belief_bounds(self):
        bounds = np.array([
            [self._inverse_llr(d, -self._k), self._inverse_llr(d, self._k)]
            for d in self._sample_demands
        ])
        return bounds

    def __call__(self, e):
        beliefs = e[:-1]
        return np.all(self.belief_bounds[:, 0] <= beliefs) and \
               np.all(self.belief_bounds[:, 1] >= beliefs)

    def project(self, e):
        diff_bounds = np.diag(
            self.belief_bounds[:, 1] - self.belief_bounds[:, 0])
        lower_bounds = self.belief_bounds[:, 0]
        e[:, :-1] = np.dot(e[:, :-1], diff_bounds) + lower_bounds
        return e


class MarkupConstraint(PlausibilityConstraint):
    def __init__(self, max_markup=.5, min_markup=.0):
        self._max_cost = self._cost_ratio(min_markup)
        self._min_cost = self._cost_ratio(max_markup)

    def __call__(self, env):
        return self._min_cost <= env[-1] <= self._max_cost

    @staticmethod
    def _cost_ratio(markup):
        return 1. / (1. + markup)

    def project(self, env):
        env[:, -1] = self._min_cost + env[:, -1] * (
            self._max_cost - self._min_cost)
        return env


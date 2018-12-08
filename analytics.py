import numpy as np
import lazy_property
import abc
from six import add_metaclass
from scipy.spatial import ConvexHull
import cvxpy


class Environment(object):

    def __init__(self, num_actions, constraints=None):
        self._num_actions = num_actions
        self._constraints = constraints

    def generate_environments(self, num_points=1e6, seed=0):
        raw_environments = self._generate_raw_environments(num_points, seed)
        return self._apply_constraints(raw_environments)

    def _generate_raw_environments(self, num, seed):
        np.random.seed(seed)
        env = np.random.rand(int(num), self._num_actions + 1)
        env = self._descending_sort_beliefs(env)
        return env

    @staticmethod
    def _descending_sort_beliefs(env):
        env[:, :-1] = descending_sort(env[:, :-1])
        return env

    def _apply_constraints(self, env):
        env = env[np.apply_along_axis(self._aggregate_constraint, 1, env), :]
        return env

    def _aggregate_constraint(self, e):
        if self._constraints is not None:
            return all([cstr(e) for cstr in self._constraints])
        else:
            return True


def descending_sort(arr, axis=1):
    return -np.sort(-np.array(arr), axis=axis)


@add_metaclass(abc.ABCMeta)
class PlausibilityConstraint(object):

    @abc.abstractmethod
    def __call__(self, e):
        pass


class InformationConstraint(PlausibilityConstraint):

    def __init__(self, k, sample_demands):
        self._k = k
        self._sample_demands = sample_demands

    @staticmethod
    def _inverse_llr(d, x):
        numerator = (d / (1. - d)) * np.exp(x)
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

    def __init__(self, deviations):
        self._deviations = _ordered_deviations(deviations)

    @lazy_property.LazyProperty
    def equilibrium_index(self):
        return np.where(self._deviations == 0)[0][0]

    @abc.abstractmethod
    def __call__(self, env):
        pass

    def _normalized_deviation_temptation(self, env):
        beliefs, cost = env[:-1], env[-1]
        payoffs = np.multiply(beliefs, 1 + self._deviations - cost)
        return payoffs[self.equilibrium_index] - np.max(payoffs)


def _ordered_deviations(deviations):
    return np.sort(list(set([0] + deviations)))


class IsNonCompetitive(DimensionlessCollusionMetrics):
    def __call__(self, env):
        return 1. - 1. * np.isclose(
            self._normalized_deviation_temptation(env), .0)


class NormalizedDeviationTemptation(DimensionlessCollusionMetrics):

    def __call__(self, env):
        return self._normalized_deviation_temptation(env)


class MinCollusionSolver(object):

    def __init__(self,
                 auction_data, deviations, tolerance, metric,
                 plausibility_constraints, num_points=1e6, seed=0):
        self.auction_data = auction_data
        self.metric = metric(deviations)
        self._deviations = _ordered_deviations(deviations)
        self._constraints = plausibility_constraints
        self._tolerance = tolerance
        self._seed = seed
        self._num_points = num_points

    @property
    def environment(self):
        return Environment(
            len(self._deviations), constraints=self._constraints)

    @lazy_property.LazyProperty
    def _epigraph_extreme_points(self):
        env_perf = self._env_with_perf
        return env_perf[ConvexHull(env_perf).vertices, :]

    @property
    def _env_with_perf(self):
        env = self.environment.generate_environments(
            num_points=self._num_points, seed=self._seed)
        return np.append(
            env, np.apply_along_axis(self.metric, 1, env).reshape(-1, 1), 1)

    @property
    def _belief_extreme_points(self):
        return self._epigraph_extreme_points[:, :-2]

    @property
    def _metric_extreme_points(self):
        return self._epigraph_extreme_points[:, -1]


class CvxpySolverWrap(object):

    def __init__(self, metrics, beliefs, demands, tolerance):
        self._metrics = metrics
        self._beliefs = beliefs
        self._demands = demands
        self._tolerance = tolerance

    def problem(self):
        pass


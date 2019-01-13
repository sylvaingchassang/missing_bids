import numpy as np
import lazy_property
import abc
from six import add_metaclass
from scipy.spatial import ConvexHull
import cvxpy
import pandas as pd


class Environment(object):

    def __init__(self,
                 num_actions,
                 constraints=None,
                 project=False,
                 initial_guesses=np.array([])):
        self._num_actions = num_actions
        self._constraints = constraints
        self._project = project
        self._initial_guesses = initial_guesses

    def generate_environments(self, num_points=1e6, seed=0):
        raw_environments = self._generate_raw_environments(num_points, seed)
        constrained_environments = self._apply_constraints(raw_environments)
        if self._initial_guesses.size == 0:
            return constrained_environments
        else:
            return np.concatenate((constrained_environments,
                                   self.initial_guesses), axis=0)

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
        if self._project:
            env = self._project_on_constraint(env)
        env = env[np.apply_along_axis(self._aggregate_constraint, 1, env), :]
        return env

    def _project_on_constraint(self, env):
        for constraint in self._constraints:
            env = constraint.project(env)
        return env

    def _aggregate_constraint(self, e):
        return not self._constraints or all(f(e) for f in self._constraints)


def descending_sort(arr, axis=1):
    return -np.sort(-np.array(arr), axis=axis)


@add_metaclass(abc.ABCMeta)
class PlausibilityConstraint(object):

    @abc.abstractmethod
    def __call__(self, e):
        """"""

    def project(self, e):
        return e


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

    def __init__(self, max_markup=.5):
        self._min_cost_ratio = 1/(1. + max_markup)

    def __call__(self, e):
        return e[-1] >= self._min_cost_ratio

    def project(self, e):
        e[:, -1] = self._min_cost_ratio + e[:, -1] * (1 - self._min_cost_ratio)
        return e

    @lazy_property.LazyProperty
    def belief_bounds(self):
        bounds = np.array([self._min_cost_ratio, 1])
        return bounds


class DimensionlessCollusionMetrics(object):

    def __init__(self, deviations):
        self._deviations = _ordered_deviations(deviations)

    @lazy_property.LazyProperty
    def equilibrium_index(self):
        return np.where(self._deviations == 0)[0][0]

    @abc.abstractmethod
    def __call__(self, env):
        """"""

    def _normalized_deviation_temptation(self, env):
        beliefs, cost = env[:-1], env[-1]
        payoffs = np.multiply(beliefs, 1 + self._deviations - cost)
        return np.max(payoffs) - payoffs[self.equilibrium_index]


def _ordered_deviations(deviations):
    return np.sort(list(set([0] + deviations)))


class IsNonCompetitive(DimensionlessCollusionMetrics):
    def __call__(self, env):
        return ~np.isclose(self._normalized_deviation_temptation(env), .0)


class NormalizedDeviationTemptation(DimensionlessCollusionMetrics):

    def __call__(self, env):
        return self._normalized_deviation_temptation(env)


class MinCollusionSolver(object):

    def __init__(self, data, deviations, tolerance, metric,
                 plausibility_constraints, num_points=1e6, seed=0,
                 project=False,
                 solver_type='',
                 initial_guesses=np.array([])):
        self.data = data
        self.metric = metric(deviations)
        self._deviations = _ordered_deviations(deviations)
        self._constraints = plausibility_constraints
        self._tolerance = tolerance
        self._seed = seed
        self._num_points = num_points
        self._project = project
        self._solver_type = solver_type
        self._initial_guesses = initial_guesses

    @property
    def environment(self):
        return Environment(
            len(self._deviations),
            constraints=self._constraints,
            project=self._project,
            initial_guesses=self._initial_guesses
        )

    @lazy_property.LazyProperty
    def epigraph_extreme_points(self):
        env_perf = self._env_with_perf
        return env_perf[ConvexHull(env_perf).vertices, :]

    @lazy_property.LazyProperty
    def _env_with_perf(self):
        env = self.environment.generate_environments(
            num_points=self._num_points, seed=self._seed)
        return np.append(
            env, np.apply_along_axis(self.metric, 1, env).reshape(-1, 1), 1)

    @property
    def belief_extreme_points(self):
        return self.epigraph_extreme_points[:, :-2]

    @property
    def metric_extreme_points(self):
        return self.epigraph_extreme_points[:, -1]

    @property
    def demands(self):
        return np.array([
            self.data.get_counterfactual_demand(rho)
            for rho in self._deviations
        ])

    @lazy_property.LazyProperty
    def problem(self):
        return ConvexProblem(
            metrics=self.metric_extreme_points,
            beliefs=self.belief_extreme_points,
            demands=self.demands,
            tolerance=self._tolerance,
            solver_type=self._solver_type
        )

    @lazy_property.LazyProperty
    def solution(self):
        return self.problem.solution

    @property
    def is_solvable(self):
        return not np.isinf(self.solution)

    @lazy_property.LazyProperty
    def argmin(self):
        if self.is_solvable:
            df = pd.DataFrame(
                self.epigraph_extreme_points,
                columns=[str(d) for d in self._deviations] + ['cost', 'metric']
            )
            df["prob"] = self.problem.variable.value
            return df
        else:
            raise Exception('Constraints cannot be satisfied')


class ConvexProblem(object):

    def __init__(self, metrics, beliefs, demands, tolerance, solver_type=''):
        self._metrics = np.array(metrics).reshape(-1, 1)
        self._beliefs = np.array(beliefs)
        self._demands = np.array(demands).reshape(-1, 1)
        self._tolerance = tolerance
        self._solver_type = solver_type

    @lazy_property.LazyProperty
    def variable(self):
        return cvxpy.Variable((len(self._metrics), 1))

    @lazy_property.LazyProperty
    def constraints(self):
        return [
            self.variable >= 0,
            cvxpy.sum(self.variable) == 1,
            cvxpy.sum_squares(
                cvxpy.matmul(self._beliefs.T, self.variable) - self._demands
            ) <= self._tolerance  # IC
        ]

    @lazy_property.LazyProperty
    def objective(self):
        return cvxpy.Minimize(
            cvxpy.sum(cvxpy.multiply(self.variable, self._metrics))
        )

    @lazy_property.LazyProperty
    def problem(self):
        return cvxpy.Problem(self.objective, self.constraints)

    @lazy_property.LazyProperty
    def solution(self):
        installed_solvers = cvxpy.installed_solvers()
        if self._solver_type in installed_solvers:
            print('Using {} solver'.format(self._solver_type))
            return self.problem.solve(solver=self._solver_type, verbose=False)
        else:
            return self.problem.solve()



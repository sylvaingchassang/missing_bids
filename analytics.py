import numpy as np
import lazy_property
import abc
from six import add_metaclass
from scipy.spatial import ConvexHull
import cvxpy
import pandas as pd


class Environment:

    def __init__(self, num_actions, constraints=None,
                 project_constraint=False, initial_guesses=np.array([])):
        self._num_actions = num_actions
        self._constraints = constraints
        self._project_constraint = project_constraint
        self._initial_guesses = initial_guesses

    def generate_environments(self, num_points=1e6, seed=0):
        raw_environments = self._generate_raw_environments(num_points, seed)
        raw_environments = self._append_initial_guesses(raw_environments)
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

    def _append_initial_guesses(self, environments):
        return environments if self._initial_guesses.size == 0 \
            else np.concatenate((environments, self._initial_guesses), axis=0)

    def _apply_constraints(self, env):
        if self._project_constraint:
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
    def __init__(self, max_markup=.5):
        self._min_cost_ratio = 1 / (1. + max_markup)

    def __call__(self, e):
        return e[-1] >= self._min_cost_ratio

    def project(self, e):
        e[:, -1] = self._min_cost_ratio + e[:, -1] * (1 - self._min_cost_ratio)
        return e

    @lazy_property.LazyProperty
    def belief_bounds(self):
        bounds = np.array([self._min_cost_ratio, 1])
        return bounds


class DimensionlessCollusionMetrics:
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
        return 1. - 1. * np.isclose(
            self._normalized_deviation_temptation(env), .0)


class NormalizedDeviationTemptation(DimensionlessCollusionMetrics):
    def __call__(self, env):
        return self._normalized_deviation_temptation(env)


class MinCollusionSolver:
    def __init__(self, data, deviations, tolerance, metric,
                 plausibility_constraints, num_points=1e6, seed=0,
                 project=False):
        self.data = data
        self.metric = metric(deviations)
        self._deviations = _ordered_deviations(deviations)
        self._constraints = plausibility_constraints
        self._tolerance = tolerance
        self._seed = seed
        self._num_points = num_points
        self._project = project
        self._initial_guesses = np.array([])

    @property
    def environment(self):
        return Environment(
            len(self._deviations),
            constraints=self._constraints,
            project_constraint=self._project,
            initial_guesses=self._initial_guesses
        )

    @property
    def epigraph_extreme_points(self):
        env_perf = self._env_with_perf
        return env_perf[ConvexHull(env_perf).vertices, :]

    @property
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

    @property
    def problem(self):
        return ConvexProblem(
            metrics=self.metric_extreme_points,
            beliefs=self.belief_extreme_points,
            demands=self.demands,
            tolerance=self._tolerance,
        )

    @property
    def result(self):
        return MinCollusionResult(
            self.problem, self.epigraph_extreme_points, self._deviations
        )

    def set_initial_guesses(self, guesses):
        self._initial_guesses = guesses

    def set_seed(self, seed):
        self._seed = seed


class ConvexProblem:
    def __init__(self, metrics, beliefs, demands, tolerance):
        self._metrics = np.array(metrics).reshape(-1, 1)
        self._beliefs = np.array(beliefs)
        self._demands = np.array(demands).reshape(-1, 1)
        self._tolerance = tolerance

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
        return self.problem.solve()


class MinCollusionResult:
    def __init__(self, problem, epigraph_extreme_points, deviations):
        self._solution = problem.solution
        self._epigraph_extreme_points = epigraph_extreme_points
        self._deviations = deviations
        self._variable = problem.variable.value

    @property
    def solution(self):
        return self._solution

    @property
    def is_solvable(self):
        return not np.isinf(self.solution)

    @property
    def argmin(self):
        if self.is_solvable:
            df = pd.DataFrame(
                self._epigraph_extreme_points,
                columns=[str(d) for d in self._deviations] + ['cost', 'metric']
            )
            df["prob"] = self._variable
            return df
        else:
            raise Exception('Constraints cannot be satisfied')


class MinCollusionIterativeSolver(MinCollusionSolver):
    _solution_threshold = 0.005

    def __init__(self, data, deviations, tolerance, metric,
                 plausibility_constraints, num_points=1e6, seed=0,
                 project=False, number_iterations=1):
        super(MinCollusionIterativeSolver, self).__init__(
            data, deviations, tolerance, metric, plausibility_constraints,
            num_points=num_points, seed=seed, project=project
        )
        self._number_iterations = number_iterations

    @lazy_property.LazyProperty
    def solution(self):
        best_solutions = None
        min_share_of_collusive_hist = []

        for seed_delta in range(self._number_iterations):
            result = self.result.solution
            min_share_of_collusive_hist.append(result)
            sorted_solution = self.result.argmin.sort_values(
                "prob", ascending=False)
            best_sol_idx = np.where(np.cumsum(sorted_solution.prob)
                                    > 1 - self._solution_threshold)[0][0]
            sorted_solution.drop(['prob', 'metric'], axis=1, inplace=True)

            if best_solutions is not None:
                best_solutions = \
                    pd.concat([best_solutions,
                               sorted_solution.loc[:best_sol_idx + 1]])
            else:
                best_solutions = sorted_solution.loc[:best_sol_idx + 1]

            self.set_initial_guesses(best_solutions.values)
            self.set_seed(self._seed + seed_delta + 1)

        return min_share_of_collusive_hist

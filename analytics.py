import numpy as np
import lazy_property
import abc
from scipy.spatial import ConvexHull
import cvxpy
import pandas as pd

from . import environments
from . import auction_data


class DimensionlessCollusionMetrics:
    def __init__(self, deviations):
        self._deviations = ordered_deviations(deviations)

    @lazy_property.LazyProperty
    def equilibrium_index(self):
        return np.where(self._deviations == 0)[0][0]

    @abc.abstractmethod
    def __call__(self, env):
        """"""

    def _normalized_deviation_temptation(self, env):
        payoffs = self._get_payoffs(env)
        return np.max(payoffs) - payoffs[self.equilibrium_index]

    def _get_payoffs(self, env):
        beliefs, cost = env[:-1], env[-1]
        return np.multiply(beliefs, 1 + self._deviations - cost)


def ordered_deviations(deviations):
    return np.sort(list(set([0] + deviations)))


class IsNonCompetitive(DimensionlessCollusionMetrics):
    def __call__(self, env):
        return 1. - 1. * np.isclose(
            self._normalized_deviation_temptation(env), .0)


class EfficientIsNonCompetitive(DimensionlessCollusionMetrics):
    min_markup = 0
    max_markup = .5

    def __init__(self, deviations):
        super().__init__(deviations)

    def __call__(self, env):
        beliefs = env[:-1]
        cost_bounds = self._get_cost_bounds(beliefs)
        consistent = self.is_consistent(cost_bounds)
        return 1 - 1. * consistent

    def _get_cost_bounds(self, beliefs):
        d0 = beliefs[self.equilibrium_index]
        return [self._cost_bound(d0, dn, rho) for dn, rho in
                zip(beliefs, self._deviations)]

    def cost_lower_bound(self, cost_bounds):
        dev_bound = max(cost_bounds[:self.equilibrium_index]) if \
            self.equilibrium_index > 0 else 0
        return max(dev_bound, 1 / (1 + self.max_markup))

    def cost_upper_bound(self, cost_bounds):
        dev_bound = min(cost_bounds[self.equilibrium_index + 1:]) if \
            self.equilibrium_index + 1 < len(cost_bounds) else 1
        return min(dev_bound, 1 / (1 + self.min_markup))

    def is_consistent(self, cost_bounds):
        return self.cost_lower_bound(cost_bounds) <= \
               self.cost_upper_bound(cost_bounds)

    def _cost_bound(self, d0, dn, rho):
        is_exception, return_value = self._is_exception(d0, dn, rho)
        if is_exception:
            return return_value
        return (d0 - (1 + rho) * dn)/(d0 - dn)

    @staticmethod
    def _is_exception(d0, dn, rho):
        if -1e-8 <= rho < 0:
            return True, 0
        elif 0 < rho < 1e-8:
            return True, 1
        elif np.isclose(dn, d0):
            return True, np.NAN
        elif rho < 0 and dn < d0:
            return True, 0
        else:
            return False, None


class NormalizedDeviationTemptation(DimensionlessCollusionMetrics):
    def __call__(self, env):
        return self._normalized_deviation_temptation(env)


class DeviationTemptationOverProfits(DimensionlessCollusionMetrics):
    def __call__(self, env):
        cost = env[-1]
        eq_belief = env[self.equilibrium_index]
        if cost < 1:
            return self._normalized_deviation_temptation(env) / (
                eq_belief * (1 - cost))
        else:
            return 0


class ConvexProblem:
    def __init__(self, metrics, beliefs, demands, tolerance,
                 moment_matrix, moment_weights):
        self._metrics = np.array(metrics).reshape(-1, 1)
        self._beliefs = np.array(beliefs)
        self._demands = np.array(demands).reshape(-1, 1)
        self._tolerance = tolerance
        self._moment_matrix = moment_matrix
        self._moment_weights = moment_weights

    @lazy_property.LazyProperty
    def variable(self):
        return cvxpy.Variable((len(self._metrics), 1))

    @lazy_property.LazyProperty
    def constraints(self):
        return self._is_distribution + self._moment_constraint

    @property
    def _is_distribution(self):
        return [self.variable >= 0, cvxpy.sum(self.variable) == 1]

    @property
    def _moment_constraint(self):
        delta = cvxpy.matmul(self._beliefs.T, self.variable) - \
                self._demands
        moment = 1e2 * cvxpy.matmul(self._moment_matrix, delta)
        return [cvxpy.matmul(
            self._moment_weights, cvxpy.square(moment)) <= 1e4 *
                self._tolerance]

    @lazy_property.LazyProperty
    def objective(self):
        return cvxpy.Minimize(
            cvxpy.sum(cvxpy.multiply(self.variable, self._metrics)))

    @lazy_property.LazyProperty
    def problem(self):
        return cvxpy.Problem(self.objective, self.constraints)

    @lazy_property.LazyProperty
    def solution(self):
        return self.problem.solve()


class MinCollusionSolver:
    _precision = .0001
    _environment_cls = environments.Environment
    _pbm_cls = ConvexProblem

    def __init__(self, data, deviations, metric, plausibility_constraints,
                 tolerance=None, num_points=1e6, seed=0, project=False,
                 filter_ties=None, moment_matrix=None, moment_weights=None,
                 confidence_level=.95):
        self._data = data
        self.metric = metric(deviations)
        self._deviations = ordered_deviations(deviations)
        self._constraints = plausibility_constraints
        self._tolerance = None if tolerance is None else np.array(tolerance)
        self._seed = seed
        self._num_points = num_points
        self._project = project
        self._filter_ties = filter_ties
        self._initial_guesses = self._environments_from_demand(21)
        self._moment_matrix = moment_matrix if moment_matrix is not None \
            else self.default_moment_matrix
        self._moment_weights = moment_weights if moment_weights is not None \
            else self.default_moment_weights
        self._confidence_level = confidence_level

    def _environments_from_demand(self, n):
        return np.array([
            list(self.demands) + [c] for c in np.linspace(0, 1, n)])

    @property
    def default_moment_matrix(self):
        return auction_data.moment_matrix(len(self._deviations), 'level')

    @property
    def default_moment_weights(self):
        return np.ones_like(self._deviations)

    @property
    def environment(self):
        return self._environment_cls(
            len(self._deviations),
            constraints=self._constraints,
            project_constraint=self._project,
            initial_guesses=self._initial_guesses
        )

    @property
    def epigraph_extreme_points(self):
        env_perf = self._env_with_perf
        interior_env_perf = self._get_interior_dimensions(env_perf)
        return env_perf[ConvexHull(interior_env_perf).vertices, :]

    @property
    def _env_with_perf(self):
        env = self.environment.generate_environments(
            num_points=self._num_points, seed=self._seed)
        perf = np.apply_along_axis(self.metric, 1, env).reshape(-1, 1)
        return np.append(env, perf, 1)

    def _get_interior_dimensions(self, env_perf):
        variability = np.std(env_perf, axis=0)
        full_dimensions = variability > self._precision
        return env_perf[:, full_dimensions]

    @staticmethod
    def belief_extreme_points(epigraph):
        return epigraph[:, :-2]

    @staticmethod
    def metric_extreme_points(epigraph):
        return epigraph[:, -1]

    @lazy_property.LazyProperty
    def demands(self):
        return self.filtered_data.assemble_target_moments(self.deviations)

    @property
    def filtered_data(self) -> auction_data.AuctionData:
        if self._filter_ties is not None:
            return self._filter_ties(self._data)
        return self._data

    @property
    def share_of_ties(self):
        return 1. - self.filtered_data.df_bids.shape[0] / \
               self._data.df_bids.shape[0]

    @property
    def problem(self):
        epigraph = self.epigraph_extreme_points
        return self._pbm_cls(
            metrics=self.metric_extreme_points(epigraph),
            beliefs=self.belief_extreme_points(epigraph),
            demands=self.demands,
            tolerance=self.tolerance,
            moment_matrix=self._moment_matrix,
            moment_weights=self._moment_weights
        )

    @property
    def tolerance(self):
        if self._tolerance is None:
            self._tolerance = self._compute_tolerance()
        tol = np.maximum(self._tolerance, 1e-8).reshape(-1, 1)
        return tol if len(tol) > 1 else float(tol)

    def _compute_tolerance(self):
        distances = self._moment_distances
        return np.percentile(distances, 100 * self._confidence_level,
                             axis=distances.ndim-1)

    @property
    def _moment_distances(self):
        bootstrap_demand_sample = self.filtered_data.bootstrap_demand_sample(
            self._deviations, num_samples=100)
        target_demands = np.array(self.demands).reshape(1, -1)
        delta = np.add(bootstrap_demand_sample, -target_demands)
        moments_delta = np.dot(self._moment_matrix, delta.T)
        return np.dot(self._moment_weights, np.square(moments_delta))

    @property
    def joint_confidence(self):
        return np.mean(np.min(
            self._moment_distances <= self.tolerance, axis=0))

    @property
    def result(self):
        return MinCollusionResult(
            self.problem, self.epigraph_extreme_points, self._deviations,
            self.argmin_columns)

    @property
    def argmin_columns(self):
        return [str(d) for d in self._deviations] + ['cost', 'metric']

    def set_initial_guesses(self, guesses):
        self._initial_guesses = guesses

    def set_seed(self, seed):
        self._seed = seed

    @property
    def deviations(self):
        return self._deviations

    @property
    def seed(self):
        return self._seed


class MinCollusionResult:
    def __init__(self, problem, epigraph_extreme_points, deviations,
                 argmin_cols):
        self._solution = problem.solution
        self._epigraph_extreme_points = epigraph_extreme_points
        self._deviations = deviations
        self._variable = problem.variable.value
        self._solver_data = []
        self._argmin_cols = argmin_cols

    @property
    def solution(self):
        return self._solution

    @property
    def is_solvable(self):
        return not np.isinf(self.solution)

    @property
    def argmin(self):
        df = pd.DataFrame(
                self._sorted_argmin_array(),
                columns=["prob"] + self._argmin_cols)
        return df.reset_index(drop=True)

    def argmin_array_quantile(self, quantile=1):
        sorted_argmin = self._sorted_argmin_array()
        sel = np.cumsum(sorted_argmin[:, 0]) <= quantile
        sel[sum(sel)] = True
        return sorted_argmin[sel]

    def _sorted_argmin_array(self):
        if self.is_solvable:
            argmin = np.concatenate((self._variable,
                                     self._epigraph_extreme_points), axis=1)
            return argmin[(-argmin[:, 0]).argsort(axis=0)]
        else:
            raise Exception('Constraints cannot be satisfied')

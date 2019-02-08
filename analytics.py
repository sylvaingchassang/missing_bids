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


class DeviationTemptationOverProfits(DimensionlessCollusionMetrics):
    def __call__(self, env):
        cost = env[-1]
        eq_belief = env[self.equilibrium_index]
        return self._normalized_deviation_temptation(env) / (
            eq_belief * (1 - cost))


class MinCollusionSolver:
    _precision = .0001

    def __init__(self, data, deviations, metric, plausibility_constraints,
                 tolerance=None, num_points=1e6, seed=0, project=False,
                 filter_ties=None, moment_matrix=None, moment_weights=None,
                 confidence_level=.95):
        self._data = data
        self.metric = metric(deviations)
        self._deviations = _ordered_deviations(deviations)
        self._constraints = plausibility_constraints
        self._tolerance = None if tolerance is None else np.array(tolerance)
        self._seed = seed
        self._num_points = num_points
        self._project = project
        self._filter_ties = filter_ties
        self._initial_guesses = self._environments_from_demand(21)
        self._moment_matrix = moment_matrix if moment_matrix is not None \
            else auction_data.moment_matrix(len(self._deviations), 'level')
        self._moment_weights = moment_weights if moment_weights is not None \
            else np.ones_like(self._deviations)
        self._confidence_level = confidence_level

    def _environments_from_demand(self, n):
        return np.array([
            list(self.demands) + [c] for c in np.linspace(0, 1, n)])

    @property
    def environment(self):
        return environments.Environment(
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
        return np.append(
            env, np.apply_along_axis(self.metric, 1, env).reshape(-1, 1), 1)

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

    @property
    def demands(self):
        return np.array([
            self.filtered_data.get_counterfactual_demand(rho)
            for rho in self._deviations])

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
        return ConvexProblem(
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
        return np.maximum(self._tolerance, 1e-7).reshape(-1, 1)

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
            self.problem, self.epigraph_extreme_points, self._deviations
        )

    def set_initial_guesses(self, guesses):
        self._initial_guesses = guesses

    def set_seed(self, seed):
        self._seed = seed


class ConvexProblem:
    def __init__(self, metrics, beliefs, demands, tolerance,
                 moment_matrix=None, moment_weights=None):
        self._metrics = np.array(metrics).reshape(-1, 1)
        self._beliefs = np.array(beliefs)
        self._demands = np.array(demands).reshape(-1, 1)
        self._tolerance = tolerance
        self._moment_matrix = moment_matrix if moment_matrix is not None else \
            auction_data.moment_matrix(len(demands), 'level')
        self._moment_weights = np.ones_like(demands) if \
            moment_weights is None else moment_weights

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


class MinCollusionResult:
    def __init__(self, problem, epigraph_extreme_points, deviations):
        self._solution = problem.solution
        self._epigraph_extreme_points = epigraph_extreme_points
        self._deviations = deviations
        self._variable = problem.variable.value
        self._solver_data = []

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
            df = df.sort_values("prob", ascending=False)
            return df.reset_index(drop=True)
        else:
            raise Exception('Constraints cannot be satisfied')


class MinCollusionIterativeSolver(MinCollusionSolver):
    _solution_threshold = 0.01

    def __init__(self, data, deviations, metric, plausibility_constraints,
                 tolerance=None, num_points=1e6, seed=0, project=False,
                 filter_ties=None, number_iterations=1, moment_matrix=None,
                 moment_weights=None, confidence_level=.95):
        super(MinCollusionIterativeSolver, self).__init__(
            data, deviations, metric, plausibility_constraints,
            tolerance=tolerance, num_points=num_points, seed=seed,
            project=project, filter_ties=filter_ties,
            moment_matrix=moment_matrix, moment_weights=moment_weights,
            confidence_level=confidence_level)
        self._number_iterations = number_iterations

    @property
    def _interim_result(self):
        return MinCollusionResult(
            self.problem, self.epigraph_extreme_points, self._deviations)

    @property
    def result(self):
        selected_guesses = None
        list_solutions = []

        for seed_delta in range(self._number_iterations):
            interim_result = self._interim_result
            list_solutions.append(interim_result.solution)
            selected_guesses = self._get_new_guesses(
                interim_result, selected_guesses)
            self._set_guesses_and_seed(selected_guesses, seed_delta)
        interim_result._solver_data = {'iterated_solutions': list_solutions}
        return interim_result

    def _get_new_guesses(self, interim_result, selected_guesses):
        argmin = interim_result.argmin
        best_sol_idx = np.where(np.cumsum(argmin.prob)
                                > 1 - self._solution_threshold)[0][0]
        argmin.drop(['prob', 'metric'], axis=1, inplace=True)
        selected_argmin = argmin.loc[:best_sol_idx + 1]

        return pd.concat([selected_guesses, selected_argmin]) if \
            selected_guesses is not None else selected_argmin

    def _set_guesses_and_seed(self, selected_guesses, seed_delta):
        self.set_initial_guesses(selected_guesses.values)
        self.set_seed(self._seed + seed_delta + 1)

from numpy.testing import TestCase, assert_array_equal, \
    assert_array_almost_equal, assert_almost_equal
import numpy as np
import os
from parameterized import parameterized

from .. import environments
from ..auction_data import AuctionData, moment_matrix, FilterTies
from ..analytics import (
    IsNonCompetitive, NormalizedDeviationTemptation, MinCollusionSolver,
    ConvexProblem, DeviationTemptationOverProfits, EfficientIsNonCompetitive)
from ..solvers import MinCollusionIterativeSolver


class TestCollusionMetrics(TestCase):
    def setUp(self):
        self.env = [.5, .4, .3, .8]

    @parameterized.expand([
        [[-.02, .02], 1],
        [[-.2, .0, .02], 0]
    ])
    def test_is_non_competitive(self, deviations, expected):
        metric = IsNonCompetitive(deviations)
        assert metric(self.env) == expected

    @parameterized.expand([
        [[.0, .001], [.2, .1995, .5], 1],
        [[.0, .001], [.2, .199, .5], 0],
        [[-1e-9, .0, .001], [.5, .2, .199, .5], 0],
        [[-.01, .0], [.5, .2, .5], 1],
        [[-.01, .0], [.4, .2, .5], 0],
        [[-.01, .0, 1e-9], [.4, .2, .199, .5], 0],
        [[-.01, .0, .001], [.4, .2, .199, .5], 1]
    ])
    def test_efficient_is_non_competitive(self, deviations, env, expected):
        metric = EfficientIsNonCompetitive(deviations)
        metric.min_markup, metric.max_markup = .02, .5
        assert metric(env) == expected

    @parameterized.expand([
        [[-.02, .02], 0.01],
        [[-.2, .0, .02], 0]
    ])
    def test_deviation_temptation(self, deviations, expected):
        metric = NormalizedDeviationTemptation(deviations)
        assert np.isclose(metric(self.env), expected)

    @parameterized.expand([
        [[-.02, .02], 0.125],
        [[-.2, .0, .02], 0]
    ])
    def test_temptation_over_profits(self, deviations, expected):
        metric = DeviationTemptationOverProfits(deviations)
        assert np.isclose(metric(self.env), expected)


class TestMinCollusionSolver(TestCase):
    def setUp(self):
        path = os.path.join(
            os.path.dirname(__file__), 'reference_data', 'tsuchiura_data.csv')
        self.data = AuctionData(bidding_data_or_path=path)
        self.constraints = [environments.MarkupConstraint(.6),
                            environments.InformationConstraint(.5, [.65, .48])]
        (self.solver, self.filtered_solver, self.solver_fail,
         self.solver_project, self.solver_moments) = \
            [self._get_solver(args) for args in
             [[.0125, False], [.0125, False, FilterTies()], [.01, False],
              [.0125, True], [.0125, False, None, moment_matrix(2)]]]
        (self.iter_solver1, self.iter_solver2, self.iter_moment) = [
            self._get_solver(args, MinCollusionIterativeSolver) for args in
            [[.0125, False, None, None, n] for n in [1, 2]] +
            [[.007, False, None, moment_matrix(2), 2]]
        ]

    def _get_solver(self, args, solver=None, enhanced_guesses=False):
        solver = MinCollusionSolver if solver is None else solver
        tol, proj, _filt, mom_mat, num = args + [None] * (5 - len(args))
        arg_dict = dict(
            deviations=[-.02], metric=IsNonCompetitive, project=proj,
            tolerance=tol, plausibility_constraints=self.constraints, seed=0,
            num_points=10000, filter_ties=_filt, moment_matrix=mom_mat)
        if solver == MinCollusionIterativeSolver:
            arg_dict.update({'num_evaluations': num})
        else:
            arg_dict.update({'enhanced_guesses': enhanced_guesses})
        return solver(self.data, **arg_dict)

    def test_deviations(self):
        assert_array_equal(self.solver._deviations, [-.02, 0])

    def test_demand(self):
        assert_array_almost_equal(self.solver.demands, [0.693839, 0.25017])

    def test_guesses(self):
        solver = self._get_solver([.0125, False], enhanced_guesses=True)
        assert_array_almost_equal(
            solver._environments_from_demand(2),
            [[0.693839, 0.25017, 0.],
             [0.693839, 0.25017, 1.],
             [1., 1., 0.], [0, 0, 1], [1, 0, 1]]
        )
        assert_almost_equal(solver.result.solution, 0.303379, decimal=5)

    def test_environment(self):
        assert_array_almost_equal(
            [[0.544883, 0.423655, 0.645894]],
            self.solver.environment.generate_environments(3, 0))

    def test_generate_env_perf(self):
        assert_array_almost_equal(
            self.solver._env_with_perf[:3],
            [[0.544883, 0.423655, 0.645894, 1.],
             [0.725254, 0.501324, 0.956084, 0.],
             [0.590873, 0.574325, 0.653201, 0.]])

    def test_extreme_points(self):
        assert_array_almost_equal(
            self.solver.epigraph_extreme_points[:3],
            [[0.590873, 0.574325, 0.653201, 0.],
             [0.530537, 0.372679, 0.922111, 1.],
             [0.706872, 0.367475, 0.649534, 1.]])

    def test_belief_extreme_points(self):
        epigraph = self.solver.epigraph_extreme_points
        assert_array_almost_equal(
            self.solver.belief_extreme_points(epigraph)[:3],
            [[0.590873, 0.574325], [0.530537, 0.372679], [0.706872, 0.367475]])

    def test_metric_extreme_points(self):
        epigraph = self.solver.epigraph_extreme_points
        assert_array_almost_equal(
            self.solver.metric_extreme_points(epigraph)[:3], [0., 1., 1.])

    def test_solution(self):
        assert_almost_equal(self.solver.result.solution, 0.303379, decimal=5)

    def test_solvable(self):
        assert self.solver.result.is_solvable

    def test_not_solvable(self):
        assert not self.solver_fail.result.is_solvable
        with self.assertRaises(Exception) as context:
            _ = self.solver_fail.result.argmin
        assert 'Constraints cannot be' in str(context.exception)

    def test_argmin_distribution(self):
        assert is_distribution(self.solver.result.argmin['prob'])

    def test_argmin(self):
        cols = ['prob', '-0.02', '0.0', 'cost', 'metric']
        df = self.solver.result.argmin[cols]
        assert_array_almost_equal(
            df.iloc[:2],
            [[0.5919, 0.6822, 0.3629, 0.9626, 0.],
             [0.303379, 0.718859, 0.360351, 0.693249, 1.]], decimal=4)

    def test_argmin_array(self):
        argmin = self.solver.result._sorted_argmin_array()
        assert_array_almost_equal(
            argmin[:2],
            [[0.5919, 0.6822, 0.3629, 0.9626, 0.],
             [0.303379, 0.718859, 0.360351, 0.693249, 1.]], decimal=4)

    def test_argmin_quantile(self):
        argmin = self.solver.result.argmin_array_quantile(.9)
        assert argmin.shape == (3, 5)
        assert_array_almost_equal(
            argmin, [[0.5919, 0.6822, 0.3629, 0.9626, 0.],
                     [0.303379, 0.718859, 0.360351, 0.693249, 1.],
                     [0.1046, 0.6258, 0.3596, 0.9917, 0.]], decimal=4)

    def test_constraint_project_environments(self):
        assert_array_equal(self.solver._env_with_perf.shape, [384, 4])
        assert_array_equal(
            self.solver_project._env_with_perf.shape, [10000, 4])

    def test_constraint_project_extreme_points(self):
        assert_array_equal(self.solver.epigraph_extreme_points.shape, [80, 4])
        assert_array_equal(
            self.solver_project.epigraph_extreme_points.shape, [193, 4])

    def test_constraint_project_solution(self):
        assert_almost_equal(self.solver_project.result.solution, .0)

    def test_iter(self):
        assert_almost_equal(self.iter_solver1.result.solution, 0.30337910)
        assert_almost_equal(
            self.iter_solver2.result._solver_data['iterated_solutions'],
            [.3033789, 0], decimal=5)

    def test_filter(self):
        assert self.solver.share_of_ties == 0
        assert_almost_equal(self.filtered_solver.share_of_ties, 0.01038121)
        assert_almost_equal(
            self.filtered_solver.result.solution, 0.2250626, decimal=5)

    def test_moments(self):
        cols = ['prob', '-0.02', '0.0']
        assert_almost_equal(
            self.solver_moments.result.solution, 0.180774, decimal=5)
        assert_array_almost_equal(
            self.solver_moments.result.argmin[cols].iloc[:2],
            [[0.81923, 0.68217, 0.3629], [0.18077, 0.75245, 0.36346]],
            decimal=5)

    def test_moments_iterated(self):
        cols = ['prob', '-0.02', '0.0']
        assert_almost_equal(
            self.iter_moment.result.solution, 0.61708711, decimal=5)
        assert_array_almost_equal(
            self.iter_moment.result.argmin[cols].iloc[:2],
            [[0.55727, 0.72525, 0.36095], [0.37046, 0.75192, 0.37151]],
            decimal=5)

    def test_compute_tolerance(self):
        assert_almost_equal(
            self.solver_moments._compute_tolerance(), 0.00027320)

    def test_tolerance(self):
        args = [None, False, None, moment_matrix(2), None]
        solver = self._get_solver(args)
        assert_almost_equal(solver.tolerance, 0.00027320)

    def test_get_interior(self):
        array = np.array([[1, 2], [1.00001, 1]])
        assert_array_almost_equal(
            self.solver._get_interior_dimensions(array),
            [[2], [1]]
        )

    def test_degenerate_constraints(self):
        constraints = [environments.MarkupConstraint(.6),
                       environments.InformationConstraint(.001, [.65, .48])]
        solver = MinCollusionIterativeSolver(
            data=self.data, deviations=[-.02], metric=IsNonCompetitive,
            plausibility_constraints=constraints, num_points=100.0,
            tolerance=.3, project=True)
        assert solver.result.is_solvable

    def test_multidimensional_moment_constraint(self):
        constraints = [environments.MarkupConstraint(.6),
                       environments.InformationConstraint(.1, [.69, .25])]
        solver = MinCollusionSolver(
            data=self.data, deviations=[-.02], metric=IsNonCompetitive,
            plausibility_constraints=constraints, num_points=100.0,
            project=True, moment_matrix=moment_matrix(2),
            moment_weights=np.identity(2), confidence_level=.95)
        assert_almost_equal(solver.tolerance, [[0.0001724], [0.0001399]])
        assert_almost_equal(solver.result.solution, 0)
        assert_almost_equal(solver.joint_confidence, .9)


class TestConvexSolver(TestCase):
    def setUp(self):
        self.metrics = [1., 0, 1, 1, 0]
        self.demands = [.5, .4]
        self.beliefs = np.array(
            [[.6, .5], [.45, .4], [.7, .6], [.4, .3], [.4, .2]])
        tolerance = .0005
        self.cvx = ConvexProblem(
            self.metrics, self.beliefs, self.demands, tolerance,
            moment_matrix=moment_matrix(len(self.demands), 'level'),
            moment_weights=np.ones_like(self.demands)
        )
        self.res = self.cvx.solution
        self.argmin = self.cvx.variable.value

    def test_minimal_value(self):
        assert_almost_equal(self.res, 0.1347556)

    def test_solution(self):
        assert_array_almost_equal(
            self.argmin, [[0], [.75759], [.13475], [0], [.10764]], decimal=5)

    def test_solution_is_distribution(self):
        assert is_distribution(self.argmin)

    def test_aggregate_subjective_demand_close_to_target(self):
        subjective_demand = np.dot(self.cvx._beliefs.T, self.argmin)
        diff = subjective_demand - self.cvx._demands
        assert all(np.abs(diff) <= np.sqrt(self.cvx._tolerance))
        assert_almost_equal(np.sum(np.square(diff)), self.cvx._tolerance)

    @parameterized.expand([
        [[1, 0], 0.199974275, [1, 2], [[0.8000232], [0.1999726]]],
        [[0, 1], 0, [1, 4], [[0.6666672], [0.3333328]]]
    ])
    def test_moments_weights(self, weights, solution, selection, argmin):
        pbm = ConvexProblem(
            self.metrics, self.beliefs, self.demands, tolerance=0,
            moment_weights=weights, moment_matrix=moment_matrix(2))
        assert_almost_equal(pbm.solution, solution, decimal=4)
        assert_almost_equal(pbm.variable.value[selection, :], argmin,
                            decimal=4)


def is_distribution(arr):
    return all([all(arr >= -1e-9), np.isclose(np.sum(arr), 1.)])

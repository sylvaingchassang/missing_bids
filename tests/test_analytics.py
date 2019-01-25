from numpy.testing import TestCase, assert_array_equal, \
    assert_array_almost_equal, assert_almost_equal
import numpy as np
import os
from parameterized import parameterized

from .. import environments
from .. import auction_data
from .. import analytics


class TestCollusionMetrics(TestCase):
    def setUp(self):
        self.env = [.5, .4, .3, .8]

    @parameterized.expand([
        [[-.02, .02], True],
        [[-.2, .0, .02], False]
    ])
    def test_is_non_competitive(self, deviations, expected):
        metric = analytics.IsNonCompetitive(deviations)
        assert metric(self.env) == expected

    @parameterized.expand([
        [[-.02, .02], 0.01],
        [[-.2, .0, .02], 0]
    ])
    def test_deviation_temptation(self, deviations, expected):
        metric = analytics.NormalizedDeviationTemptation(deviations)
        assert np.isclose(metric(self.env), expected)


class TestMinCollusionSolver(TestCase):
    def setUp(self):
        path = os.path.join(
            os.path.dirname(__file__), 'reference_data', 'tsuchiura_data.csv')
        data = auction_data.AuctionData(bidding_data_or_path=path)
        constraints = [environments.MarkupConstraint(.6),
                       environments.InformationConstraint(.5, [.65, .48])]
        self.solver, self.solver_fail, self.solver_project = [
            analytics.MinCollusionSolver(
                data, deviations=[-.02], metric=analytics.IsNonCompetitive,
                tolerance=tol, plausibility_constraints=constraints,
                num_points=10000, seed=0, project=project)
            for tol, project in [(.0125, False), (.01, False), (.0125, True)]]

        self.iter_solver1, self.iter_solver2 = [
            analytics.MinCollusionIterativeSolver(
                data, deviations=[-.02], metric=analytics.IsNonCompetitive,
                tolerance=0.0125, plausibility_constraints=constraints,
                num_points=10000, seed=0, project=False,
                number_iterations=num) for num in [1, 2]]

        self.filtered_solver = analytics.MinCollusionSolver(
            data, deviations=[-.02], metric=analytics.IsNonCompetitive,
            tolerance=0.0125, plausibility_constraints=constraints,
            num_points=10000, seed=0, project=False,
            filter_ties=auction_data.FilterTies())

    def test_deviations(self):
        assert_array_equal(self.solver._deviations, [-.02, 0])

    def test_demand(self):
        assert_array_almost_equal(self.solver.demands, [0.693839, 0.25017])

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
            [[0.590873, 0.574325],
             [0.530537, 0.372679],
             [0.706872, 0.367475]])

    def test_metric_extreme_points(self):
        epigraph = self.solver.epigraph_extreme_points
        assert_array_almost_equal(
            self.solver.metric_extreme_points(epigraph)[:3], [0., 1., 1.])

    def test_solution(self):
        assert_almost_equal(self.solver.result.solution, 0.30337910)

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
            [[0.59193, 0.68217, 0.362903, 0.962636, 0.],
             [0.303379, 0.718859, 0.360351, 0.693249, 1.]])

    def test_constraint_project_environments(self):
        assert_array_equal(
            self.solver._env_with_perf.shape,
            [384, 4])
        assert_array_equal(
            self.solver_project._env_with_perf.shape,
            [10000, 4])

    def test_constraint_project_extreme_points(self):
        assert_array_equal(
            self.solver.epigraph_extreme_points.shape, [80, 4])
        assert_array_equal(
            self.solver_project.epigraph_extreme_points.shape, [193, 4])

    def test_constraint_project_solution(self):
        assert_almost_equal(self.solver_project.result.solution, .0)

    def test_iter(self):
        assert_almost_equal(self.iter_solver1.result.solution, 0.30337910)
        assert_almost_equal(
            self.iter_solver2.result._solver_data['iterated_solutions'],
            [.303378989, -1.9551292724687417e-10])

    def test_filter(self):
        assert self.solver.share_of_ties == 0
        assert_almost_equal(self.filtered_solver.share_of_ties, 0.02756977)
        assert_almost_equal(self.filtered_solver.result.solution, 0.3421390)


class TestConvexSolver(TestCase):
    def setUp(self):
        metrics = [1., 0, 1, 1, 0]
        demands = [.5, .4]
        beliefs = np.array(
            [[.6, .5], [.45, .4], [.7, .6], [.4, .3], [.4, .2]])
        tolerance = .0005
        self.cvx = analytics.ConvexProblem(
            metrics, beliefs, demands, tolerance)
        self.res = self.cvx.solution
        self.argmin = self.cvx.variable.value

    def test_minimal_value(self):
        assert_almost_equal(self.res, 0.1347556)

    def test_solution(self):
        assert_array_almost_equal(
            self.argmin,
            [[4.35054877e-09], [7.57598639e-01], [1.34755686e-01],
             [1.52098246e-09], [1.07645669e-01]])

    def test_solution_is_distribution(self):
        assert is_distribution(self.argmin)

    def test_aggregate_subjective_demand_close_to_target(self):
        subjective_demand = np.dot(self.cvx._beliefs.T, self.argmin)
        diff = subjective_demand - self.cvx._demands
        assert all(np.abs(diff) <= np.sqrt(self.cvx._tolerance))
        assert_almost_equal(np.sum(np.square(diff)), self.cvx._tolerance)


def is_distribution(arr):
    return all([all(arr >= 0), np.isclose(np.sum(arr), 1.)])

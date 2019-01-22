from numpy.testing import TestCase, assert_array_equal, \
    assert_array_almost_equal, assert_almost_equal
import pandas as pd
import numpy as np
import os
from parameterized import parameterized
from .. import auction_data
from .. import analytics


class TestAuctionData(TestCase):
    def setUp(self):
        path = os.path.join(
            os.path.dirname(__file__), 'reference_data', 'tsuchiura_data.csv')
        self.auctions = auction_data.AuctionData(
            bidding_data_or_path=path
        )

    def test_bids(self):
        df = self.auctions.df_bids
        assert df.shape == (5876, 7)
        assert self.auctions.df_auctions.shape == (1469, 2)
        cols = ["norm_bid", "most_competitive", "lowest", "second_lowest"]
        expected = pd.DataFrame(
            [[0.973545, 0.97619, 0.973545, 0.97619],
             [0.978836, 0.973545, 0.973545, 0.97619],
             [0.986772, 0.973545, 0.973545, 0.97619],
             [0.976190, 0.973545,  0.973545, 0.97619],
             [0.978836, 0.973545,  0.973545, 0.97619]],
            columns=cols
        )
        assert_array_almost_equal(expected, df[df["pid"] == 15][cols])

    def test_read_frame(self):
        data = auction_data.AuctionData(self.auctions.raw_data)
        assert_array_almost_equal(
            data.raw_data.norm_bid, self.auctions.raw_data.norm_bid)

    def test_auction_data(self):
        assert_array_almost_equal(
            self.auctions.df_auctions.lowest.values[:10],
            np.array([0.89655173, 0.94766617, 0.94867122, 0.69997638,
                      0.9385258, 0.74189192, 0.7299363, 0.94310075,
                      0.96039605, 0.97354496])
        )

    def test_counterfactual_demand(self):
        dmd = self.auctions.get_counterfactual_demand(.05)
        assert_almost_equal(dmd, 0.02067733151)

    def test_demand_function(self):
        dmd = self.auctions.demand_function(start=-.01, stop=.01, num=4)
        assert_array_almost_equal(dmd, [[0.495575], [0.293397], [0.21341],
                                        [0.105599]])


class TestEnvironments(TestCase):
    def setUp(self):
        self.constraints = [analytics.MarkupConstraint(.6),
                            analytics.InformationConstraint(.5, [.65, .48])]
        self.env_no_cons = analytics.Environment(num_actions=2)
        self.env_with_initial_guesses = \
            analytics.Environment(num_actions=2,
                                  initial_guesses=np.array([[0.7, 0.5, 0.6]]))

    def test_generate_raw_environments(self):
        assert_array_almost_equal(
            self.env_no_cons._generate_raw_environments(3, seed=0),
            [[0.715189, 0.548814, 0.602763],
             [0.544883, 0.423655, 0.645894],
             [0.891773, 0.437587, 0.963663]]
        )

    def test_generate_environments_with_initial_guesses(self):
        assert_array_almost_equal(
            self.env_with_initial_guesses.generate_environments(
                num_points=3, seed=0),
            [[0.715189, 0.548814, 0.602763],
             [0.544883, 0.423655, 0.645894],
             [0.891773, 0.437587, 0.963663],
             [0.7, 0.5, 0.6]]
        )

    def test_generate_environments_no_cons(self):
        assert_array_almost_equal(
            self.env_no_cons._generate_raw_environments(3, seed=0),
            self.env_no_cons.generate_environments(3, seed=0),
        )

    @parameterized.expand([
        [[0], [[0.544883, 0.423655, 0.645894],
               [0.891773, 0.437587, 0.963663]]],
        [[1], [[0.715189, 0.548814, 0.602763],
               [0.544883, 0.423655, 0.645894]]],
        [[0, 1], [[0.544883, 0.423655, 0.645894]]]
    ])
    def test_generate_environments_cons(self, cons_id, expected):
        env = analytics.Environment(
            num_actions=2, constraints=[self.constraints[i] for i in cons_id])
        assert_array_almost_equal(
            env.generate_environments(3, seed=0),
            expected)

    def test_generate_environment_with_projection(self):
        env = analytics.Environment(
            num_actions=2, constraints=self.constraints,
            project_constraint=True)
        assert_array_almost_equal(
            env.generate_environments(3, seed=1),
            [[0.691139, 0.460906, 0.625043],
             [0.597473, 0.394812, 0.659627],
             [0.60716, 0.404473, 0.773788]])


class TestConstraints(TestCase):
    def setUp(self):
        self.mkp = analytics.MarkupConstraint(2.)
        self.info = analytics.InformationConstraint(.01, [.5, .4, .3])
        self.ref_environments = np.array([[.8, .4, .3, .1],
                                          [.9, .3, .1, .8]])

    def test_markup_constraint(self):
        assert not self.mkp([.5, .6, .33])
        assert self.mkp([.5, .6, .34])
        assert_array_almost_equal(
            self.mkp.project(self.ref_environments),
            [[0.8, 0.4, 0.3, 0.4],
             [0.9, 0.3, 0.1, 0.866667]])

    def test_info_bounds(self):
        assert_array_almost_equal(
            self.info.belief_bounds,
            [[0.4975, 0.5025], [0.397602, 0.402402], [0.297904, 0.302104]])
        assert_array_almost_equal(
            self.info.project(self.ref_environments),
            [[0.5015, 0.399522, 0.299164, 0.1],
             [0.502, 0.399042, 0.298324, 0.8]])

    def test_info(self):
        assert self.info([.5, .4, .3, .5])
        assert not self.info([.5, .4, .35, .5])
        assert not self.info([.45, .4, .3, .5])


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
        constraints = [analytics.MarkupConstraint(.6),
                       analytics.InformationConstraint(.5, [.65, .48])]
        self.solver, self.solver_fail, self.solver_project = [
            analytics.MinCollusionSolver(
                data, deviations=[-.02], metric=analytics.IsNonCompetitive,
                tolerance=tol, plausibility_constraints=constraints,
                num_points=10000, seed=0, project=project)
            for tol, project in [(.0125, False), (.01, False), (.0125, True)]]

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
        assert_array_almost_equal(
            self.solver.belief_extreme_points[:3],
            [[0.590873, 0.574325],
             [0.530537, 0.372679],
             [0.706872, 0.367475]])

    def test_metric_extreme_points(self):
        assert_array_almost_equal(
            self.solver.metric_extreme_points[:3], [0., 1., 1.])

    def test_solution(self):
        assert_almost_equal(self.solver.solution, 0.30337910)

    def test_solvable(self):
        assert self.solver.is_solvable

    def test_not_solvable(self):
        assert not self.solver_fail.is_solvable
        with self.assertRaises(Exception) as context:
            _ = self.solver_fail.argmin
        assert 'Constraints cannot be' in str(context.exception)

    def test_argmin_distribution(self):
        assert is_distribution(self.solver.argmin['prob'])

    def test_argmin(self):
        cols = ['prob', '-0.02', '0.0', 'cost', 'metric']
        df = self.solver.argmin[cols]
        assert_array_almost_equal(
            df.iloc[[12, 15]],
            [[.303378989, .718859179, .360350558, .693249354, 1.0],
             [.104691195, .625773600, .359648881, .991686340, 0.0]])

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
        assert_almost_equal(self.solver_project.solution, .0)


class TestConvexSolver(TestCase):
    def setUp(self):
        metrics = [1., 0, 1, 1, 0]
        demands = [.5, .4]
        beliefs = np.array(
            [[.6, .5], [.45, .4], [.7, .6], [.4, .3], [.4, .2]])
        tolerance = .0005
        self.cvx = analytics.ConvexProblem(metrics, beliefs, demands, tolerance)
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


class TestMinCollusionIterativeSolver(TestCase):
    def setUp(self):
        path = os.path.join(
            os.path.dirname(__file__), 'reference_data', 'tsuchiura_data.csv')
        data = auction_data.AuctionData(bidding_data_or_path=path)
        constraints = [analytics.MarkupConstraint(.6),
                       analytics.InformationConstraint(.5, [.65, .48])]
        self.solver_single = analytics.MinCollusionIterativeSolver(
            data, deviations=[-.02], metric=analytics.IsNonCompetitive,
            tolerance=0.0125, plausibility_constraints=constraints,
            num_points=10000, first_seed=0, project=False,
            number_iterations=1, show_graph=False)

        self.solver_multiple = analytics.MinCollusionIterativeSolver(
            data, deviations=[-.02], metric=analytics.IsNonCompetitive,
            tolerance=0.0125, plausibility_constraints=constraints,
            num_points=10000, first_seed=0, project=False,
            number_iterations=2, show_graph=False)

    def test_solution_single(self):
        assert_almost_equal(self.solver_single.solution, 0.30337910)

    def test_solution_multiple(self):
        print(self.solver_multiple.solution)
        assert_almost_equal(
            self.solver_multiple.solution,
            [.303378989, -1.9551292724687417e-10]
        )

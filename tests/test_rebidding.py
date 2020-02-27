import os
import numpy as np
from numpy.testing import TestCase, assert_array_almost_equal, \
    assert_almost_equal
from parameterized import parameterized

from rebidding import (
    MultistageAuctionData, MultistageIsNonCompetitive,
    RefinedMultistageData, RefinedMultistageIsNonCompetitive,
    RefinedMultistageEnvironment, refined_moment_matrix,
    RefinedMultistageSolver, IteratedRefinedMultistageSolver,
    ParallelRefinedMultistageSolver, EfficientMultistageIsNonCompetitive)
from auction_data import _read_bids, FilterTies
from environments import MarkupConstraint
from .test_analytics import is_distribution


def _load_multistage_data():
    path = os.path.join(
        os.path.dirname(__file__), 'reference_data', 'tsuchiura_data.csv')
    raw_data = _read_bids(path)
    raw_data['reserveprice'] *= .985
    raw_data['norm_bid'] = raw_data['bid'] / raw_data['reserveprice']
    return raw_data


class TestMultistageAuctionData(TestCase):
    def setUp(self) -> None:
        self.auctions = MultistageAuctionData(_load_multistage_data())

    def test_second_round(self):
        assert_almost_equal(
            self.auctions.share_second_round, 0.11912865)

    def test_raise_error(self):
        self.assertRaises(NotImplementedError,
                          self.auctions.get_share_marginal,
                          self.auctions.df_bids, .1)

    def test_share_marginal(self):
        assert_almost_equal(
            self.auctions.get_share_marginal(
                self.auctions.df_bids, -.02), 0.08492171)

    def test_share_marginal_cont(self):
        assert_almost_equal(
            self.auctions.share_marginal_cont(self.auctions.df_bids, -.02),
            0.08492171)

    def test_share_marginal_info(self):
        assert_almost_equal(
            self.auctions.share_marginal_info(self.auctions.df_bids, -.01),
            0.0238257)

    def test_get_counterfactual_demand(self):
        assert_array_almost_equal(
            [self.auctions.get_counterfactual_demand(r) for r in [-.05, .05]],
            [0.775868, 0.02067733151])


class TestRefinedMultistageData(TestCase):
    def setUp(self) -> None:
        self.data = RefinedMultistageData(_load_multistage_data())

    @parameterized.expand((
        [-.01, [0.4954901, 0.0597345, 0.0238257]],
        [.01, 0.10534377127297481]
    ))
    def test_get_counterfactual_demand(self, rho, expected):
        assert_almost_equal(
            self.data.get_counterfactual_demand(rho), expected
        )

    def test_assemble_target_moments(self):
        assert_array_almost_equal(
            self.data.assemble_target_moments([-.01, 0, .005]),
            [0.49549, 0.059735, 0.023826, 0.25017, 0.18065])
        assert_array_almost_equal(
            self.data.assemble_target_moments(
                [-.01, 0, .005], self.data.df_bids),
            [0.49549, 0.059735, 0.023826, 0.25017, 0.18065])

    def test_filter(self):
        filter_ties = FilterTies(.0001)
        assert np.sum(filter_ties.get_ties(self.data)) == 61
        assert filter_ties(self.data).df_bids.shape == (5815, 7)
        assert isinstance(filter_ties(self.data), RefinedMultistageData)

    def test_bootstrap(self):
        demand_sample = self.data.bootstrap_demand_sample([-.01, 0, .005], 3)
        assert demand_sample.shape == (3, 5)
        assert_array_almost_equal(
            demand_sample.round(2),
            [[0.5, 0.06, 0.02, 0.25, 0.17],
             [0.49, 0.06, 0.02, 0.24, 0.18],
             [0.49, 0.06, 0.02, 0.24, 0.17]]
        )


class TestMultistageIsNonCompetitive(TestCase):

    def setUp(self):
        self.env = np.array([.5, .4, .3, .8])

    @parameterized.expand([
        [[-.03, .02], [[0.085, 0.08, 0.066], [-.0075]]],
        [[-.02, .02], [[0.09, 0.08, 0.066], [-.005]]],
        [[-.2, .0, .02], [[0., 0.08, 0.066], [-.05]]]
    ])
    def test_payoff_penalty(self, deviations, expected):
        MultistageIsNonCompetitive.max_win_prob = .75
        metric = MultistageIsNonCompetitive(deviations)
        assert_array_almost_equal(
            metric._get_payoffs(self.env), expected[0])
        assert_array_almost_equal(
            metric._get_penalty(self.env), expected[1])

    @parameterized.expand([
        [[-.03, .02], 0],
        [[-.02, .02], 1],
        [[-.2, .0, .02], 0],
        [[.01, .02], 0]
    ])
    def test_ic(self, deviations, expected):
        MultistageIsNonCompetitive.max_win_prob = .75
        metric = MultistageIsNonCompetitive(deviations)
        assert_array_almost_equal(metric(self.env), expected)


class TestRefinedMultistageIsNonCompetitive(TestCase):

    def setUp(self):
        self.env = np.array([.6, .1, .05, .3, .15, .95])
        self.metric_type = RefinedMultistageIsNonCompetitive

    @parameterized.expand([
        [[-.01, .01], [0.018375, 0.015, 0.009]],
        [[-.01, 0, .01], [0.018375, 0.015, 0.009]],
        [[-.05, .02], [-0.003125, 0.015, 0.0105]],
        [[-.05, .1], [-0.003125, 0.015, 0.0225]]
    ])
    def test_payoffs(self, deviations, expected):
        metric = self.metric_type(deviations)
        assert_array_almost_equal(metric._get_payoffs(self.env), expected)

    @parameterized.expand([
        [[-.01, .01], 1],
        [[-.01, 0, .01], 1],
        [[-.05, .02], 0],
        [[-.05, .1], 1]
    ])
    def test_ic(self, deviations, expected):
        metric = self.metric_type(deviations)
        assert_array_almost_equal(metric(self.env), expected)

    def test_raise_error(self):
        self.assertRaises(
            ValueError, self.metric_type, [-.1, -.01, 0, .1])
        self.assertRaises(
            ValueError, self.metric_type, [-.1, .01, 0, .1])


class TestEfficientMultistageIsNonCompetitive(TestCase):

    @parameterized.expand([
        [[.0, ]]
    ])
    def test_penalized_payoff_bounds(self, deviations, beliefs, expected):
        metric = EfficientMultistageIsNonCompetitive(deviations, .02, .5)
        assert_array_almost_equal(metric._get_cost_bounds(beliefs), expected)


class TestRefinedMultistageEnvironment(TestCase):

    def setUp(self):
        self.env = RefinedMultistageEnvironment(num_actions=2)

    def test_private_generate_raw_environments(self):
        assert_array_almost_equal(
            self.env._generate_raw_environments(3, 1).round(2),
            [[0.72, 0.16, 0.06, 0.42, 0., 0.67],
             [0.3, 0.07, 0.61, 0.15, 0.09, 0.42],
             [0.4, 0.04, 0.02, 0.35, 0.19, 0.56]]
        )


def test_refined_moment_matrix():
    assert_array_almost_equal(
        refined_moment_matrix(),
        np.array([
            [1, 0, 0, 0, 0],
            [1, -1, 0, -1, 0],
            [0, -1, 1, 0, 0],
            [-1, 0, 0, 1, 0],
            [0, 0, 0, -1, 1]
        ]))
    assert_array_almost_equal(
        refined_moment_matrix(False), np.identity(5))


class TestRefinedSolvers(TestCase):
    def setUp(self) -> None:
        filter_ties = FilterTies(.0001)
        markup_constraint = MarkupConstraint(.5, .02)
        self.data = filter_ties(RefinedMultistageData(_load_multistage_data()))
        args = (self.data, [-.02, 0, .002], 
                RefinedMultistageIsNonCompetitive, [markup_constraint])
        kwargs = dict(
            num_points=1e3, seed=0, project=False,
            filter_ties=filter_ties, moment_matrix=None, moment_weights=None,
            confidence_level=.95)
        self.solver = RefinedMultistageSolver(*args, **kwargs)
        self.parallel_solver = ParallelRefinedMultistageSolver(*args, **kwargs)
        kwargs['num_evaluations'] = 10
        self.iter_solver = IteratedRefinedMultistageSolver(*args, **kwargs)

    def test_moment_matrix(self):
        assert_array_almost_equal(
            self.solver._moment_matrix, refined_moment_matrix())
        assert_array_almost_equal(
            self.solver._moment_weights, 5 * [1])

    def test_tolerance(self):
        assert_almost_equal(
            self.solver.tolerance, 0.0003502449)

    def test_generate_env_perf(self):
        assert_array_almost_equal(
            self.solver._env_with_perf[:3].round(2),
            [[0.83, 0.12, 0.06, 0.09, 0.02, 0.76, 1.],
             [0.77, 0.04, 0.04, 0.46, 0.26, 0.85, 1.],
             [0.62, 0.03, 0.08, 0.57, 0.02, 0.79, 0.]])

    def test_demand(self):
        assert_array_almost_equal(
            self.solver.demands,
            [0.693981, 0.085297, 0., 0.250559, 0.239123])

    def test_solution(self):
        assert_almost_equal(
            self.solver.result.solution, 0.751241, decimal=5)

    def test_argmin_distribution(self):
        assert is_distribution(self.solver.result.argmin['prob'])

    def test_argmin(self):
        cols = ['prob'] + self.solver.argmin_columns
        df = self.solver.result.argmin[cols]
        assert_array_almost_equal(
            df.iloc[:2],
            [[0.2, 0.7, 0.1, 0., 0.3, 0.2, 0.9, 1.],
             [0.2, 0.7, 0.1, 0., 0.3, 0.2, 0.8, 1.]], decimal=1)

    def test_iter(self):
        assert_almost_equal(
            self.iter_solver.result.solution, 0.271439, decimal=5)

    def test_iter_argmin(self):
        cols = ['prob'] + self.iter_solver.solver.argmin_columns
        df = self.iter_solver.result.argmin[cols]
        assert_array_almost_equal(
            df.iloc[:2],
            [[.59, .51, .09, .025, .13, .086, .98, 0.],
             [.27, 1.0, .065, 0, .3, .29, .86, 1.]], decimal=1)

    def test_parallel_solution(self):
        assert_almost_equal(self.parallel_solver.result.solution, 0.30190327)

import os
import numpy as np
from rebidding import (MultistageAuctionData, MultistageIsNonCompetitive,
    RefinedMultistageData, RefinedMultistageIsNonCompetitive,
    RefinedMultistageEnvironment)
from auction_data import _read_bids
from numpy.testing import TestCase, assert_array_almost_equal, \
    assert_almost_equal
from parameterized import parameterized


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

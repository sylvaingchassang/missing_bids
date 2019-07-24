import os
import numpy as np
from rebidding import MultistageAuctionData, MultistageIsNonCompetitive
from auction_data import _read_bids
from numpy.testing import TestCase, assert_array_almost_equal, \
    assert_almost_equal
from parameterized import parameterized


class TestMultistageAuctionData(TestCase):
    def setUp(self) -> None:
        path = os.path.join(
            os.path.dirname(__file__), 'reference_data', 'tsuchiura_data.csv')
        raw_data = _read_bids(path)
        raw_data['reserveprice'] *= .985
        raw_data['norm_bid'] = raw_data['bid'] / raw_data['reserveprice']
        self.auctions = MultistageAuctionData(raw_data)

    def test_second_round(self):
        assert_almost_equal(
            self.auctions.share_second_round, 0.11912865)

    def test_raise_error(self):
        self.assertRaises(NotImplementedError,
                          self.auctions.get_share_marginal, .1)

    def test_share_marginal(self):
        assert_almost_equal(
            self.auctions.get_share_marginal(-.02), 0.08492171)

    def test_get_counterfactual_demand(self):
        assert_array_almost_equal(
            [self.auctions.get_counterfactual_demand(r) for r in [-.05, .05]],
            [0.775868, 0.02067733151])


class TestMultistageIsNonCompetitive(TestCase):

    def setUp(self):
        self.env = np.array([.5, .4, .3, .8])

    @parameterized.expand([
        [[-.03, .02], [[0.085, 0.08, 0.066], [-.0075]]],
        [[-.02, .02], [[0.09, 0.08, 0.066], [-.005]]],
        [[-.2, .0, .02], [[0., 0.08, 0.066], [-.05]]]
    ])
    def test_payoff_penalty(self, deviations, expected):
        metric = MultistageIsNonCompetitive(deviations, .75)
        assert_array_almost_equal(
            metric._get_payoffs(self.env), expected[0])
        assert_array_almost_equal(
            metric._get_penalty(self.env), expected[1])

    @parameterized.expand([
        [[-.03, .02], 0],
        [[-.02, .02], 1],
        [[-.2, .0, .02], 0]
    ])
    def test_ic(self, deviations, expected):
        metric = MultistageIsNonCompetitive(deviations, .75)
        assert_array_almost_equal(metric(self.env), expected)

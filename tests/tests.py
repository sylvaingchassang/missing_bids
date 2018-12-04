from numpy.testing import TestCase, assert_array_equal, \
    assert_array_almost_equal
import numpy as np
import auction_data
import os
import analytics


class TestAuctionData(TestCase):
    def setUp(self):
        self.auctions = auction_data.AuctionData(
            bidding_data=os.path.join('reference_data', 'tsuchiura_data.csv'))

    def test_bid_data(self):
        assert self.auctions.df_bids.shape == (5876, 7)
        assert self.auctions.df_auctions.shape == (1469, 2)
        assert_array_equal(
            self.auctions._df_bids.pid.values[:10],
            np.array([15, 15, 15, 15, 15, 16, 16, 16, 16, 16])
        )

    def test_auction_data(self):
        assert_array_almost_equal(
            self.auctions.df_auctions.lowest.values[:10],
            np.array([0.89655173, 0.94766617, 0.94867122, 0.69997638,
                      0.9385258, 0.74189192, 0.7299363, 0.94310075,
                      0.96039605, 0.97354496])
        )

    def test_most_competitive(self):
        assert_array_almost_equal(
            self.auctions.df_bids.most_competitive.values[10:20],
            np.array(
                [0.79662162, 0.74189192, 0.74189192, 0.74189192, 0.74189192,
                 0.74189192, 0.74189192, 0.74189192, 0.74189192, 0.74189192])
        )

    def test_counterfactual_demand(self):
        dmd = self.auctions.get_counterfactual_demand(.05)
        assert_array_almost_equal(dmd, 0.02067733151)

    def test_demand_function(self):
        dmd = self.auctions.demand_function(start=-.01, stop=.01, num=4)
        assert_array_almost_equal(dmd, [[0.495575], [0.293397], [0.21341],
                                        [0.105599]])


class TestDeviations(TestCase):

    def setUp(self):
        self.dev = analytics.Deviations(1, [-.02, .01], [.3, .5, .2], .5)
        self.dev_array = analytics.Deviations(
            [1, 2], [-.02, .01], [[.3, .5, .2], [.4, .41, .1]], [.5, .9])

    def test_bids(self):
        assert_array_almost_equal(
            self.dev.bids_and_deviations, [[1, 0.98, 1.01]])
        assert_array_almost_equal(
            self.dev_array.bids_and_deviations,
            [[1, 0.98, 1.01], [2, 1.96, 2.02]]
        )

    def test_profits(self):
        assert_array_almost_equal(self.dev.profits, [[0.15, 0.24, 0.102]])
        assert_array_almost_equal(
            self.dev_array.profits,
            [[0.15, 0.24, 0.102], [0.44, 0.4346, 0.112]])

    def test_equilibrium_profits(self):
        assert_array_almost_equal(self.dev.equilibrium_profits, [[0.15]])
        assert_array_almost_equal(
            self.dev_array.equilibrium_profits, [[0.15], [0.44]])

    def test_deviation_profits(self):
        assert_array_almost_equal(self.dev.deviation_profits, [[0.24, 0.102]])
        assert_array_almost_equal(
            self.dev_array.deviation_profits,
            [[0.24, 0.102], [0.4346, 0.112]])

    def test_is_competitive(self):
        assert_array_almost_equal(self.dev.is_competitive, [[0]])
        assert_array_almost_equal(self.dev_array.is_competitive, [[0], [1]])

    def test_deviation_temptation(self):
        assert_array_almost_equal(self.dev.deviation_temptation, [[.09]])
        assert_array_almost_equal(self.dev_array.deviation_temptation,
                                  [[.09], [0]])

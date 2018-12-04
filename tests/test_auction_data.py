from numpy.testing import TestCase, assert_array_equal, \
    assert_array_almost_equal
import numpy as np
import auction_data
import os


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

from numpy.testing import TestCase, assert_array_equal, \
    assert_array_almost_equal
import numpy as np
import auction_data
import os


class TestAuctionData(TestCase):
    def setUp(self):
        self.auctions = auction_data.AuctionData(
            bidding_data=os.path.join('reference_data', 'tsuchiura_data.csv'))

        self.auctions.compute_demand_moments()
        self.auctions.categorize_histories()

    def test_bid_data(self):
        assert self.auctions._df_bids.shape == (5850, 74)
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
            self.auctions._df_bids.most_competitive.values[10:20],
            np.array(
                [0.79662162, 0.74189192, 0.74189192, 0.74189192, 0.74189192,
                 0.74189192, 0.74189192, 0.74189192, 0.74189192, 0.74189192]
            )
        )

    def test_categorize_histories(self):
        assert self.auctions.enum_categories == \
               {(0, 0, 0): 817,
                (0, 1, 0): 3576,
                (1, 0, 0): 1441,
                (1, 0, 1): 16}

    def test_get_demand(self):
        p_c = (0.0, 0.10000000000000001, 0.90000000000000002, 0.5)
        assert_array_almost_equal(
            self.auctions.get_demand(p_c),
            [0.026, 0.550154, 0.001368]
        )
        assert_array_almost_equal(
            self.auctions.get_competitive_share(p_c), 0.5761538461538462)

    def test_counterfactual_demand(self):
        dmd = self.auctions.get_counterfactual_demand(.05, .05)
        assert_array_almost_equal(
            dmd.demand.iloc[[1, 200, 400, 600, 800, 999]].values,
            [0.86, 0.792821, 0.492137, 0.105299, 0.031624, 0.020855]
        )

    def test_dist_bid_gap(self):
        assert_array_almost_equal(
            np.percentile(self.auctions.get_bid_gaps(), [10, 50, 90]),
            [0.0013519, 0.00573612, 0.02761007]
        )

    def test_tied_winners(self):
        assert_array_almost_equal(
            [np.mean(self.auctions._df_bids.most_competitive ==
                     self.auctions._df_bids.norm_bid),
             np.mean(self.auctions.df_auctions.lowest ==
                     self.auctions.df_auctions.second_lowest),
             np.mean(self.auctions._df_bids.lowest ==
                     self.auctions._df_bids.second_lowest)],
            [0., 0.008169, 0.005983]
        )

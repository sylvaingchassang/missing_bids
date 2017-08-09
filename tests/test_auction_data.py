from numpy.testing import TestCase, assert_array_equal, \
    assert_array_almost_equal
import numpy as np
import auction_data
import os


class TestAuctionData(TestCase):
    def setUp(self):
        self.auctions = auction_data.AuctionData(
            reference_file=os.path.join('reference_data', 'tsuchiura_data.csv')
        )
        self.auctions.compute_demand_moments()
        self.auctions.categorize_histories()

    def test_bid_data(self):
        assert_array_equal(
            self.auctions.df_bids.pid.values[:10],
            np.array([15, 15, 15, 15, 15, 16, 16, 16, 16, 16])
        )

    def test_auction_data(self):
        assert_array_almost_equal(
            self.auctions.df_auctions.lowest.values[:10],
            np.array([0.89655173, 0.94766617, 0.94867122, 0.69997638,
                      0.9385258,  0.74189192, 0.7299363, 0.94310075,
                      0.96039605, 0.97354496])
        )

    def test_most_competitive(self):
        assert_array_almost_equal(
            self.auctions.df_bids.most_competitive.values[10:20],
            np.array(
                [0.79662162, 0.74189192, 0.74189192, 0.74189192, 0.74189192,
                 0.74189192, 0.74189192, 0.74189192, 0.74189192, 0.74189192]
            )
        )

    def test_categorize_histories(self):
        assert self.auctions.enum_categories == \
               {(0, 0, 0): 817,
                (0, 1, 0): 3602,
                (1, 0, 0): 1441,
                (1, 0, 1): 16}

    def test_get_demand(self):
        p_c = (0.0, 0.10000000000000001, 0.90000000000000002, 0.5)
        assert_array_almost_equal(
            self.auctions.get_demand(p_c),
            (0.025884955752212391, 0.55170183798502381, 0.0013614703880190605)
        )
        assert_array_almost_equal(
            self.auctions.get_competitive_share(p_c), 0.577586793737)

    def test_counterfactual_demand(self):
        dmd = self.auctions.counterfactual_demand(.05, .05)
        assert_array_almost_equal(
            dmd.iloc[[1, 200, 400, 600, 800, 999]].values,
            np.array([[0.86061947], [0.79373724], [0.49438393],
                      [0.10483322], [0.031484], [0.02076242]])
        )

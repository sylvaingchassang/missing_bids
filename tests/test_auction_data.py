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
        self.auctions.generate_auction_data()
        self.auctions.add_most_competitive()
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
               {(0, 0, 0): 980,
                (0, 1, 0): 3576,
                (1, 0, 0): 1441,
                (1, 0, 1): 16}



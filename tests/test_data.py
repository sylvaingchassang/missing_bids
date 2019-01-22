from numpy.testing import TestCase, assert_array_almost_equal, \
    assert_almost_equal
import pandas as pd
import numpy as np
import os
from .. import auction_data


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

    def test_filter(self):
        filter_ties = auction_data.FilterTies(.0001)
        assert np.sum(filter_ties.get_ties(self.auctions)) == 61
        assert filter_ties(self.auctions).df_bids.shape == (5714, 7)

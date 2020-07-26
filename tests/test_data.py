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

    def test_bids_and_data_share_index(self):
        assert_array_almost_equal(
            self.auctions.df_bids.norm_bid,
            self.auctions.data.loc[self.auctions.df_bids.index].norm_bid
        )

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
        assert_array_almost_equal(
            [self.auctions.get_counterfactual_demand(r) for r in [-.05, .05]],
            [0.86096, 0.02067733151])

    def test_demand_function(self):
        dmd = self.auctions.demand_function(start=-.01, stop=.01, num=4)
        assert_array_almost_equal(dmd, [[0.495575], [0.293397], [0.21341],
                                        [0.105599]])

    def test_filter(self):
        filter_ties = auction_data.FilterTies(.0001)
        assert np.sum(filter_ties.get_ties(self.auctions)) == 61
        assert filter_ties(self.auctions).df_bids.shape == (5815, 7)
        assert isinstance(filter_ties(self.auctions), auction_data.AuctionData)

    def test_bootstrap(self):
        assert_array_almost_equal(
            self.auctions.bootstrap_demand_sample([-.01, 0, .01], 4),
            [[0.4952349, 0.2451498, 0.1017699],
             [0.4902995, 0.2423417, 0.1021103],
             [0.4913206, 0.2441287, 0.0949626],
             [0.4989789, 0.2570626, 0.1050885]]
        )

    def test_single_bootstrap(self):
        assert_array_almost_equal(
            self.auctions._single_bootstrap((20, 0, [-.01, 0, .01])),
            [0.25, 0.2, 0.15])

    def test_moment_matrix(self):
        assert_array_almost_equal(
            auction_data.moment_matrix(3),
            [[1,  0,  0], [-1,  1,  0], [0, -1,  1]])

    def test_moment_distance(self):
        candidate = [.3, .2, .1]
        target = [.25, .14, .03]
        assert_almost_equal(
            auction_data.moment_distance(candidate, target, [1, 2, 3]), 0.003)

    def test_from_clean_bids(self):
        df_bids = self.auctions.df_bids
        df_bids = df_bids.loc[self.auctions.data.bidder == 'muramatsu']
        auction_from_bids = auction_data.AuctionData.from_clean_bids(df_bids)
        assert_array_almost_equal(auction_from_bids.df_bids, df_bids)

    def test_filtering_correct(self):
        filter_ties = auction_data.FilterTies(tolerance=.0001)
        assert_almost_equal(
            filter_ties.get_ties(self.auctions).mean(), 0.01038121)
        filtered_data = filter_ties(self.auctions)
        assert_almost_equal(filter_ties.get_ties(filtered_data).mean(), 0)

    def test_assemble_target_moments(self):
        assert_array_almost_equal(
            self.auctions.assemble_target_moments([-.01, .02]),
            [0.495575, 0.047651]
        )


def test_extend_or_append():
    a = []
    b = auction_data.extend_or_append(a, 1)
    b = auction_data.extend_or_append(b, [2, 3])
    assert b == [1, 2, 3]
    assert a is b


class TestAuctionDataPIDMean(TestCase):
    def setUp(self):
        path = os.path.join(
            os.path.dirname(__file__), 'reference_data', 'tsuchiura_data.csv')
        self.auctions = auction_data.AuctionDataPIDMean(
            bidding_data_or_path=path
        )
        self.deviations = [-.02, 0, .005]

    def test_counterfactual_demand(self):
        assert_array_almost_equal(
            [self.auctions.get_counterfactual_demand(r) for r in [-.05, .05]],
            [0.896378, 0.023846])

    def test_standard_deviation(self):
        assert_almost_equal(self.auctions.standard_deviation(
            self.deviations, (.4, .2, .4)), 0.34073885)

    def test_win_vector(self):
        df_bids = self.auctions._win_vector(
            self.auctions.df_bids, self.deviations)
        assert_almost_equal(
            df_bids[['pid'] + self.deviations].head(3),
            [[15, 1., 1., 0.],
             [15, 1., 0., 0.],
             [15, 1., 0., 0.]])

    def test_demand_vector(self):
        assert_array_almost_equal(
            self.auctions.demand_vector(self.deviations),
            [0.757466, 0.29353, 0.211717])

    def test_num_auctions(self):
        assert self.auctions.num_auctions == 1469

    def test_confidence_threshold(self):
        assert_almost_equal(
            self.auctions.confidence_threshold([-1, 1, 0], self.deviations),
            -0.4425795)

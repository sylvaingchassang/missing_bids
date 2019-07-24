import os
from rebidding import MultistageAuctionData
from auction_data import _read_bids
from numpy.testing import TestCase, assert_array_equal, assert_almost_equal


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

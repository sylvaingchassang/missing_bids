from numpy.testing import TestCase, assert_array_equal, \
    assert_array_almost_equal
import numpy as np
import auction_data, ic_sets
import os


class TestICSets(TestCase):
    def setUp(self):
        self.auctions = auction_data.AuctionData(
            reference_file=os.path.join('reference_data', 'tsuchiura_data.csv')
        )
        self.ic_set = ic_sets.ICSets(
            rho_m=.05, rho_p=.001, auction_data=self.auctions,
            k=.1, m=.5, t=.05
        )
        self.p_c = (0.10000000000000001, 0.90000000000000002,
                    0.70000000000000007, 0.30000000000000004)

        self.set_a = ([0.16648095792449691, 0.26648095792449694],
                [0.3662980209545984, 0.46629802095459838],
                [-0.049201729585897226, 0.050798270414102779])

        self.set_b = ([0.20000025362773816, 0.23392262538665495],
                [0.39222022052922134, 0.44078195431462858],
                [0.00072235981479677454, 0.00088215118557172516])

        self.box = [[0.20000025362773816, 0.23392262538665495],
              [0.39222022052922134, 0.44078195431462858],
              [0.00072235981479677454, 0.00088215118557172516]]

        self.extreme_points = np.array([[0.20000025, 0.44078195, 0.00072236],
                                        [0.20000025, 0.44078195, 0.00088215],
                                        [0.23392263, 0.39222022, 0.00072236]])

    def test_init(self):
        ic = self.ic_set
        assert (ic.rho_m, ic.rho_p, ic.k, ic.m, ic.t) == \
               (.05, .001, .1, .5, .05)

    def test_shape_parameters(self):
        assert self.ic_set.lower_slope_pp == 0.002991026919242274
        assert self.ic_set.tangent_binding_pm == 0.0196078431372549

    def test_boxes(self):
        assert self.ic_set.compute_set_a(self.p_c) == self.set_a
        assert self.ic_set.compute_set_b(self.p_c) == self.set_b
        assert self.ic_set.intersect_box(self.set_a, self.set_b) == self.box
        assert_array_almost_equal(
            self.ic_set.get_box_extreme_points(self.box)[2:5],
            self.extreme_points
        )


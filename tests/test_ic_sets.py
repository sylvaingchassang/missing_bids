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

        self.extreme_points_box = \
            np.array([[0.20000025, 0.44078195, 0.00072236],
                      [0.20000025, 0.44078195, 0.00088215],
                      [0.23392263, 0.39222022, 0.00072236]])

        self.triangle = \
            np.array([[1, 1, 1], [1.2, 1.5, 1.6], [.9, .4, .5]])

        self.set_z = np.array(
            [[2.00000254e-01, 0.00000000e+00, 3.92157360e-03],
             [2.00000254e-01, 0.00000000e+00, 2.00000254e-01],
             [2.33922625e-01, 0.00000000e+00, 4.58671814e-03],
             [2.33922625e-01, 0.00000000e+00, 2.33922625e-01],
             [2.00000254e-01, 7.99999746e-01, 3.92157360e-03],
             [2.00000254e-01, 7.99999746e-01, 2.00000254e-01],
             [2.33922625e-01, 7.66077375e-01, 4.58671814e-03],
             [2.33922625e-01, 7.66077375e-01, 2.33922625e-01],
             [2.00000254e-01, 0.00000000e+00, 5.98206142e-04],
             [2.00000254e-01, 3.52941624e-02, 5.98206142e-04],
             [2.33922625e-01, 0.00000000e+00, 6.99668870e-04],
             [2.33922625e-01, 4.12804633e-02, 6.99668870e-04]]
        )

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
            self.extreme_points_box
        )

    def test_value_pm(self):
        assert_array_almost_equal(self.ic_set.value_pm(.25),
                                  0.04411764705882354)
        assert_array_almost_equal(self.ic_set.value_pm(.25, .002),
                                  0.16891891891891894)

    def test_set_z(self):
        extreme_points_z = self.ic_set.extreme_points_set_z(*self.set_b[0])
        assert_array_almost_equal(extreme_points_z, self.set_z)

    def test_perpendicular(self):
        x, y, z = self.triangle
        u = self.ic_set.get_perpendicular(*self.triangle)
        assert_array_almost_equal(np.dot(u, x - y), 0)
        assert_array_almost_equal(np.dot(u, x - z), 0)

        x2 = self.triangle.flatten()
        u = self.ic_set.get_perpendicular(x2)
        assert_array_almost_equal(np.dot(u, x - y), 0)
        assert_array_almost_equal(np.dot(u, x - z), 0)

    def test_choose_3(self):
        assert_array_equal(
            self.ic_set.choose_3_in(6),
            np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 1, 5], [0, 2, 3],
                      [0, 2, 4], [0, 2, 5], [0, 3, 4], [0, 3, 5], [0, 4, 5],
                      [1, 2, 3], [1, 2, 4], [1, 2, 5], [1, 3, 4], [1, 3, 5],
                      [1, 4, 5], [2, 3, 4], [2, 3, 5], [2, 4, 5], [3, 4, 5]])
        )

    def test_triplets(self):
        assert_array_almost_equal(
            self.ic_set.get_triplets(self.set_z)[[0, 100, 200]],
            np.array([[2.00000254e-01, 0.00000000e+00, 3.92157360e-03,
                       2.00000254e-01, 0.00000000e+00, 2.00000254e-01,
                       2.33922625e-01, 0.00000000e+00, 4.58671814e-03],
                      [2.33922625e-01, 0.00000000e+00, 4.58671814e-03,
                       2.33922625e-01, 0.00000000e+00, 2.33922625e-01,
                       2.00000254e-01, 7.99999746e-01, 3.92157360e-03],
                      [2.33922625e-01, 7.66077375e-01, 4.58671814e-03,
                       2.33922625e-01, 7.66077375e-01, 2.33922625e-01,
                       2.00000254e-01, 0.00000000e+00, 5.98206142e-04]])
        )

    def test_is_separable(self):
        pass

    def test_is_rationalizable(self):
        pass

    def test_share_competitive(self):
        assert self.ic_set.assess_share_competitive(3) == (1, (1, 1, 1, 1))

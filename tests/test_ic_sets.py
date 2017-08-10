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

        self.set_a = ([[0.171528,  0.271528],
                       [0.379101,  0.479101],
                       [-0.049183,  0.050817]])

        self.set_b = ([[0.204764,  0.239252],
                       [0.404797,  0.453753],
                       [0.000739,  0.000903]])

        self.box = [[0.204764,  0.239252],
                    [0.404797, 0.453753],
                    [0.000739, 0.000903]]

        self.extreme_points_box = \
            np.array([[0.204764,  0.453753,  0.000739],
                      [0.204764,  0.453753,  0.000903],
                      [0.239252,  0.404797,  0.000739]])

        self.triangle = \
            np.array([[1, 1, 1], [1.2, 1.5, 1.6], [.9, .4, .5]])

        self.set_z = np.array(
            [[0.204764, 0., 0.00401498],
             [0.204764, 0., 0.204764],
             [0.239252, 0., 0.00469122],
             [0.239252, 0., 0.239252],
             [0.204764, 0.795236, 0.00401498],
             [0.204764, 0.795236, 0.204764],
             [0.239252, 0.760748, 0.00469122],
             [0.239252, 0.760748, 0.239252],
             [0.204764, 0., 0.00061245],
             [0.204764, 0.03613482, 0.00061245],
             [0.239252, 0., 0.00071561],
             [0.239252, 0.04222094, 0.00071561]]
        )

    def test_init(self):
        ic = self.ic_set
        assert (ic.rho_m, ic.rho_p, ic.k, ic.m, ic.t) == \
               (.05, .001, .1, .5, .05)

    def test_shape_parameters(self):
        assert self.ic_set.lower_slope_pp == 0.002991026919242274
        assert self.ic_set.tangent_binding_pm == 0.0196078431372549

    def test_boxes(self):
        assert_array_almost_equal(self.ic_set.compute_set_a(self.p_c),
                                  self.set_a)
        assert_array_almost_equal(self.ic_set.compute_set_b(self.p_c),
                                  self.set_b)
        assert_array_almost_equal(
            self.ic_set.intersect_box(self.set_a, self.set_b),
            self.box)
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
            np.array([[2.04764000e-01, 0.00000000e+00, 4.01498000e-03,
                       2.04764000e-01, 0.00000000e+00, 2.04764000e-01,
                       2.39252000e-01, 0.00000000e+00, 4.69122000e-03],
                      [2.39252000e-01, 0.00000000e+00, 4.69122000e-03,
                       2.39252000e-01, 0.00000000e+00, 2.39252000e-01,
                       2.04764000e-01, 7.95236000e-01, 4.01498000e-03],
                      [2.39252000e-01, 7.60748000e-01, 4.69122000e-03,
                       2.39252000e-01, 7.60748000e-01, 2.39252000e-01,
                       2.04764000e-01, 0.00000000e+00, 6.12450000e-04]])
        )

    def test_is_separable(self):
        pyramid = np.array(
            [[0, 0, 0], [0, 1, 0], [.5, 0, .5], [.5, 0, .4]]
        )

        cube = np.array(
            [[.4, .21, .35], [.4, .21, .45],
             [.4, .31, .35], [.4, .31, .45],
             [.45, .21, .35], [.45, .21, .45],
             [.45, .31, .35], [.45, .31, .45]]
        )

        assert self.ic_set.is_separable(pyramid, cube) == True
        cube = np.array(
            [[.4, .19, .35], [.4, .19, .45],
             [.4, .31, .35], [.4, .31, .45],
             [.45, .19, .35], [.45, .19, .45],
             [.45, .31, .35], [.45, .31, .45]]
        )
        assert self.ic_set.is_separable(pyramid, cube) == False

    def test_is_rationalizable(self):
        p_c = (0, 0, 0, 0)
        assert self.ic_set.is_rationalizable(p_c) == True
        p_c = (1, 1, 1, 1)
        assert self.ic_set.is_rationalizable(p_c) == False
        p_c = (.5, .5, .5, .5)
        assert self.ic_set.is_rationalizable(p_c) == True

    def test_share_competitive(self):
        assert_array_almost_equal(
            self.ic_set.assess_share_competitive(3)[0],
            0.8773825731790333
        )

    def test_lower_bound_collusive(self):
        df_bids = self.auctions.df_bids.copy(deep=True)
        self.ic_set.auction_data.set_bid_data(
            df_bids.loc[df_bids.minprice.isnull()])

        lbc = self.ic_set.lower_bound_collusive(.0007)
        assert_array_almost_equal(
            [lbc[key] for key in ['tied_winner', 'deviate_up']],
            [0.010417, 0.985614]
        )
        self.ic_set.auction_data.set_bid_data(df_bids)
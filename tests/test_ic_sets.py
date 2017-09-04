from numpy.testing import TestCase, assert_array_equal, \
    assert_array_almost_equal, assert_almost_equal
import numpy as np
import auction_data
import ic_sets
import os
import itertools
import pandas as pd
import matplotlib.pyplot as plt


class TestICSets(TestCase):
    def setUp(self):
        self.auctions = auction_data.AuctionData(
            bids_path=os.path.join('reference_data', 'bids_data.csv'),
            auction_path=os.path.join('reference_data', 'auction_data.csv')
        )

        self.ic_set = ic_sets.ICSets(
            rho_m=.05, rho_p=.001, auction_data=self.auctions,
            k=.1, m=.5, t=.05
        )
        self.p_c = (0.10000000000000001, 0.90000000000000002,
                    0.70000000000000007, 0.30000000000000004)

        self.set_a = [[0.172513, 0.272513],
                      [0.377897, 0.477897],
                      [-0.049179, 0.050821]]

        self.set_b = [[0.205693, 0.240291],
                      [0.403613, 0.452535],
                      [0.000742, 0.000907]]

        self.box = [[0.205693, 0.240291],
                    [0.403613, 0.452535],
                    [0.000742, 0.000907]]

        self.extreme_points_box = \
            [[0.205693, 0.452535, 0.000742],
             [0.205693, 0.452535, 0.000907],
             [0.240291, 0.403613, 0.000742]]

        self.triangle = \
            np.array([[1, 1, 1], [1.2, 1.5, 1.6], [.9, .4, .5]])

        self.set_z = np.array(
            [[2.05693000e-01, 0.00000000e+00, 6.15233300e-04],
             [2.05693000e-01, 3.62987647e-02, 6.15233300e-04],
             [2.05693000e-01, 0.00000000e+00, 3.21656893e-03],
             [2.05693000e-01, 7.94307000e-01, 3.21656893e-03],
             [2.05693000e-01, 0.00000000e+00, 2.05693000e-01],
             [2.05693000e-01, 7.94307000e-01, 2.05693000e-01],
             [2.40291000e-01, 0.00000000e+00, 7.18716849e-04],
             [2.40291000e-01, 4.24042941e-02, 7.18716849e-04],
             [2.40291000e-01, 0.00000000e+00, 3.59638065e-03],
             [2.40291000e-01, 7.59709000e-01, 3.59638065e-03],
             [2.40291000e-01, 0.00000000e+00, 2.40291000e-01],
             [2.40291000e-01, 7.59709000e-01, 2.40291000e-01]])

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
            np.array([[2.05693000e-01, 0.00000000e+00, 6.15233300e-04,
                       2.05693000e-01, 3.62987647e-02, 6.15233300e-04,
                       2.05693000e-01, 0.00000000e+00, 3.21656893e-03],
                      [2.05693000e-01, 0.00000000e+00, 3.21656893e-03,
                       2.05693000e-01, 7.94307000e-01, 3.21656893e-03,
                       2.05693000e-01, 0.00000000e+00, 2.05693000e-01],
                      [2.40291000e-01, 0.00000000e+00, 7.18716849e-04,
                       2.40291000e-01, 4.24042941e-02, 7.18716849e-04,
                       2.40291000e-01, 0.00000000e+00, 3.59638065e-03]])
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
        assert self.ic_set.is_rationalizable(p_c) == False

    def test_share_competitive(self):
        test_dict = self.ic_set.assess_share_competitive(3)
        assert_array_almost_equal(
            [test_dict['non_comp_tied_bids'], test_dict['non_comp_IC']],
            [0.00442477876106, 0.122617426821]
        )

    def test_lower_bound_collusive(self):
        df_bids = self.auctions.all_bids
        self.ic_set.auction_data.set_bid_data(
            df_bids.loc[df_bids.minprice.isnull()])

        lbc = self.ic_set.lower_bound_collusive(.0007)
        assert_array_almost_equal(
            [lbc[key] for key in ['tied_winner', 'deviate_up']],
            [0.0049342105263157892, 0.99506299771369899]
        )
        self.ic_set.auction_data.set_bid_data(df_bids)

    def test_intersection_binding_pp_lower_pp(self):
        assert_array_almost_equal(
            self.ic_set.binding_pp(self.ic_set.intersect_d),
            self.ic_set.lower_slope_pp * self.ic_set.intersect_d)

    def test_is_rationalizable_iid(self):
        steps = np.linspace(0, 1, 5)
        list_p_c = list(itertools.product(steps, repeat=4))

        ic_solver = ic_sets.ICSets(rho_p=.001, rho_m=.001,
                                   auction_data=self.auctions,
                                   k=0, t=.0, m=.5)

        for test_pc in list_p_c:
            is_rationalizable_iid = ic_solver.is_rationalizable_iid(test_pc)
            is_rationalizable = ic_solver.is_rationalizable(test_pc)

            assert is_rationalizable == is_rationalizable_iid

    def test_2d_bounds(self):

        for i in list(np.linspace(0, 5, num = 3)):
            ic_solver = ic_sets.ICSets(rho_p=.001, rho_m=.01,
                                       auction_data=self.auctions,
                                       k=i, t=0.025, m=0.5)
            full_solution = ic_solver.assess_share_competitive(num_steps=5)
            solution_2d = \
                ic_solver.assess_share_competitive_2d_bound(num_steps=5)
            upward_only = \
                ic_solver.assess_share_competitive_upward_only(num_steps=5)
            assert ic_solver.almost_less(1 - full_solution['non_comp_IC'],
                                         1 - solution_2d['non_comp_IC']) and \
                   ic_solver.almost_less(1 - full_solution['non_comp_IC'],
                                         1 - upward_only['non_comp_IC'])



import numpy as np
from auction_data import AuctionData
import itertools

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from shapely.geometry import Polygon


class ICSets(object):
    def __init__(self, rho_p, rho_m, auction_data, m, t, k):
        self._rho_p = None
        self._rho_m = None
        self._auction_data = None
        self._m = None
        self._t = None
        self._k = None
        self.set_deviations(rho_p=rho_p, rho_m=rho_m)
        self.set_data(auction_data)
        self.set_tolerance_parameters(m, t, k)

    def set_deviations(self, rho_p, rho_m):
        self._rho_p = rho_p
        self._rho_m = rho_m
        if self._auction_data is not None:
            self.set_data()

    def set_data(self, auction_data=None):
        if auction_data is None:
            auction_data = self._auction_data
        assert isinstance(auction_data, AuctionData)
        auction_data.compute_demand_moments(rho_m=self.rho_m, rho_p=self.rho_p)
        auction_data.categorize_histories()
        self._auction_data = auction_data

    def set_tolerance_parameters(self, m=None, t=None, k=None):
        if m is not None:
            self._m = m
        if t is not None:
            self._t = t
        if k is not None:
            self._k = k
        if self._auction_data is not None:
            self.set_data()

    @property
    def rho_p(self):
        return self._rho_p

    @property
    def rho_m(self):
        return self._rho_m

    @property
    def auction_data(self):
        return self._auction_data

    @property
    def m(self):
        return self._m

    @property
    def t(self):
        return self._t

    @property
    def k(self):
        return self._k

    @property
    def lower_slope_pp(self):
        return self.rho_p / (1. + self.rho_p - 1. / (1. + self.m))

    @property
    def intersect_d(self):
        # value of d such that binding_pp(d) = lower_slope_pp * d
        return (1-self.rho_m - 1/(1+self.m)) / (1 - 1/(1+self.m))

    def binding_pp(self, d):
        # min value of pp at which value_pm(d, pp) == 1 - d
        return self._invert_pp(d, 1 - d)

    def _invert_pp(self, d, pm):
        # value of pp at which value_pm(d, pp) = pm
        if pm > 1 - d:
            raise ValueError('pm > 1 - d --- no inversion to pp possible')
        elif self.value_pm(d, self.lower_slope_pp * d) > pm:
            return self.lower_slope_pp * d
        return self.rho_p * d * pm / (self.rho_m * d +
                                      (self.rho_p + self.rho_m) * pm)

    def invert_pp(self, d, pm):
        return max(self._invert_pp(d, pm), self.lower_slope_pp * d)

    @property
    def tangent_binding_pm(self):
        return self.rho_p / (self.rho_p + self.rho_m)

    def value_pm(self, d, pp=None):
        if pp is None:
            pp = self.lower_slope_pp * d
        if pp >= (self.rho_p / (self.rho_p + self.rho_m)) * d:
            return 1 - d
        else:
            # noinspection PyTypeChecker
            return min(
                1 - d,
                (self.rho_m * d) / (self.rho_p * (-1 + d / pp) - self.rho_m)
            )

    def extreme_points_set_z(self, u_d, o_d, u_pm=0):
        o_d = min(o_d, 1-u_pm)
        if u_d > o_d:
            return None
        basis_d_pp = []
        for d in [u_d, o_d]:
            basis_d_pp += [(d, self.invert_pp(d, u_pm)),
                           (d, self.binding_pp(d)), (d, d)]
        list_points = []
        for b in basis_d_pp:
            d, pp = b
            list_points += [(d, u_pm, pp), (d, self.value_pm(d, pp), pp)]
            assert self.almost_less(u_pm, self.value_pm(d, pp))

        return np.array(list_points)

    def plot_z(self, u_d, o_d, u_pm=0, ax=None):
        z = self.extreme_points_set_z(u_d, o_d, u_pm)
        if ax is None:
            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, projection='3d')
        for i in range(6):
            ind = [i, i+6]
            ax.plot(xs=z[ind, 2], ys=z[ind, 0], zs=z[ind, 1], c='k')
        for i in range(2):
            ind = [i * 6 + j for j in [0, 3, 2, 4,1, 2, 4, 5, 3, 0]]
            ax.plot(xs=z[ind, 2], ys=z[ind, 0], zs=z[ind, 1], c='b')
        plt.xlabel('pp', fontsize=20)
        plt.ylabel('d', fontsize=20)
        ax.view_init(ax.elev, ax.azim+90)
        return ax

    def plot_box(self, box, ax=None):
        b = self.get_box_extreme_points(box)
        if ax is None:
            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xs=b[:, 2], ys=b[:, 0], zs=b[:, 1], c='r')
        ind = [0,  2, 1, 3, 5, 7, 4, 6, 0, 1, 5, 4, 0, 2, 3, 7, 6, 2]
        ax.plot(xs=b[ind, 2], ys=b[ind, 0], zs=b[ind, 1], c='r')
        plt.xlabel('pp', fontsize=20)
        plt.ylabel('d', fontsize=20)
        ax.view_init(ax.elev, ax.azim+90)
        return ax

    def compute_set_a(self, p_c):
        # set a describes empirical moment constraints
        d, pm, pp = self.auction_data.get_demand(p_c)
        t = self.t
        range_d = [d - t, d + t]
        range_pm = [pm - t, pm + t]
        range_pp = [pp - t, pp + t]
        return range_d, range_pm, range_pp

    @staticmethod
    def almost_less(c, d):
        return np.logical_or(c < d, np.isclose(c, d))

    @staticmethod
    def strictly_less(c, d):
        return np.logical_and(c < d, ~np.isclose(c, d))

    def intersect_range(self, range_a, range_b):
        c, d = (max(range_a[0], range_b[0]),
                min(range_a[1], range_b[1]))
        if self.almost_less(c, d):
            return [c, d]
        else:
            return None

    def intersect_box(self, set_a, set_b):
        """Note: a box in three dimensions is described by three ranges"""
        return [self.intersect_range(r_a, r_b) for
                (r_a, r_b) in zip(set_a, set_b)]

    def underline_b(self, x):
        return self.overline_b(x, -self.k)

    def overline_b(self, x, k=None):
        # set b describes information constraints
        k = self.k if k is None else k
        l = float(x) / (1. - x)
        return l * np.exp(k) / (1. + l * np.exp(k))

    def compute_set_b(self, p_c):
        d, pm, pp = self.auction_data.get_demand(p_c)
        return tuple(
            [self.underline_b(X), self.overline_b(X)]
            for X in [d, pm, pp]
        )

    @staticmethod
    def get_box_extreme_points(box):
        return np.array(list(itertools.product(*box)))

    @staticmethod
    def get_perpendicular(x, y=None, z=None):
        if y is None:
            y = x[3:6]
            z = x[6:9]
            x = x[0:3]
        u = x - y
        v = x - z
        c = np.cross(u, v)
        if np.sum(np.abs(c)) == 0:
            return np.array([1., 1., 1.])
        else:
            return c

    @staticmethod
    def choose_3_in(n):
        list_sel = []
        if n < 3:
            return np.array([[0, min(n-1, 1), 0]])
        for a in xrange(n - 2):
            for b in xrange(a + 1, n - 1):
                for c in xrange(b + 1, n):
                    list_sel.append([a, b, c])
        return np.array(list_sel)

    def get_triplets(self, array_points):
        n = len(array_points)
        array_sel = self.choose_3_in(n)
        return np.concatenate(
            tuple(array_points[array_sel[:, i]] for i in range(3)),
            axis=1
        )

    def is_separable(self, extreme_points_box, extreme_points_z, seed=888):
        all_extreme_points = np.concatenate(
            (extreme_points_z, extreme_points_box), axis=0
        )
        all_extreme_points = np.unique(all_extreme_points, axis=0)

        all_triangles = self.get_triplets(all_extreme_points)
        perpendiculars = np.apply_along_axis(
            self.get_perpendicular, axis=1, arr=all_triangles
        )

        np.random.seed(seed)
        alt_directions = np.random.randn(100, 3)
        perpendiculars = np.concatenate((perpendiculars, alt_directions),
                                        axis=0)

        score_box = np.dot(extreme_points_box, perpendiculars.T)
        score_set_z = np.dot(extreme_points_z, perpendiculars.T)

        max_score_box, max_score_z = (np.max(score_box, axis=0),
                                      np.max(score_set_z, axis=0))
        min_score_box, min_score_z = (np.min(score_box, axis=0),
                                      np.min(score_set_z, axis=0))

        box_below = self.almost_less(max_score_box, min_score_z)
        is_box_below = np.any(box_below)
        z_below = self.almost_less(max_score_z, min_score_box)
        is_z_below = np.any(z_below)

        separates = (is_box_below or is_z_below)
        # we need to check that the ordering is not degenerate
        if is_box_below:
            min_below, min_above, max_below, max_above = (
                score[box_below] for score in [min_score_box, min_score_z,
                                               max_score_box, max_score_z])
        else:
            min_below, min_above, max_below, max_above = (
                score[z_below] for score in [min_score_z, min_score_box,
                                             max_score_z, max_score_box])

        valid = (np.any(self.strictly_less(min_below, min_above)) &
                 np.any(self.strictly_less(max_below, max_above)))

        return separates & valid

    def is_rationalizable(self, p_c, option=None):
        set_a = self.compute_set_a(p_c)  # empirical moment constraints
        set_b = self.compute_set_b(p_c)  # information constraints
        box = self.intersect_box(set_a, set_b)
        if None in box:
            return False

        extreme_points_box = self.get_box_extreme_points(box)
        u_d, o_d = set_b[0]
        u_pm, _ = set_b[1]
        extreme_points_z = self.extreme_points_set_z(
            u_d=u_d, o_d=o_d, u_pm=u_pm)

        if option is not None:
            ax = self.plot_z(u_d, o_d, u_pm)
            self.plot_box(box, ax=ax)

        return ~self.is_separable(extreme_points_box, extreme_points_z)

    def is_rationalizable_iid(self, p_c):
        # compute pp, pm, d
        d, pm, pp = self.auction_data.get_demand(p_c)

        # always rationalizable if empirical demand is 0
        if np.isclose(d, 0):
            return True

        # check that there exists c such that bids are IC
        left_hand_side = 1 / (1 + self.m) if pm <= 0 else \
            max(1 - self.rho_m - self.rho_m * (d / pm), 1 / (1 + self.m))

        right_hand_side = - 1 if pp <= 0 else \
            1 + self.rho_p - self.rho_p * (d / pp)

        return left_hand_side <= right_hand_side

    def p_bar_minus(self, x, b_upper_pp):
        num = self.rho_m * x
        if b_upper_pp == 0:
            return 1 - x
        denom = - self.rho_m - self.rho_p + self.rho_p * x / b_upper_pp
        if denom <= 0:
            return 1 - x
        else:
            return np.min([1 - x, num / denom])

    def inv_p_bar_minus(self, y, b_upper_pp):
        # Get inverse of getPbarm()
        if np.isclose(y, self.p_bar_minus(1 - y, b_upper_pp)):
            return 1 - y
        else:
            return y * (self.rho_m + self.rho_p) / \
                    (self.rho_p * y / b_upper_pp - self.rho_m)

    def p_bar_minus_kinks(self, b_upper_pp):
        a = self.rho_p / b_upper_pp
        b = - self.rho_p - self.rho_p / b_upper_pp
        c = self.rho_p + self.rho_m
        if (b*b - 4*a*c >= 0):
            return [(-b - np.sqrt(b*b - 4*a*c)) / (2*a),
                    (-b + np.sqrt(b*b - 4*a*c)) / (2*a)]
        else:
            return None

    # For Z and I(q) to be non-empty we need both conditions 1 and 2:
    def z_i_nonempty_cond_1(self, set_b):
        return self.lower_slope_pp * set_b[0][0] <= set_b[2][1]

    def z_i_nonempty_cond_2(self, set_b):
        num = self.rho_m * set_b[0][1]
        denom = - self.rho_m - self.rho_p + \
                self.rho_p * set_b[0][1] / set_b[2][1]
        if denom <= 0:
            return True
        else:
            denom = np.max([0, denom])
            return set_b[1][0] <= num/denom

    def check_kinks_conv_z_i_2d_bound(self, set_b):
        # If both the upper and rightmost vertices are on the straight
        # part or curved part of P-bar-minus, then it's a triangle.
        # Otherwise it will have a 4th vertex at the kink point.
        upper_on_straight = np.isclose(
                self.p_bar_minus(set_b[0][0], set_b[2][1]), 1 - set_b[0][0])
        right_on_straight= np.isclose(
                self.inv_p_bar_minus(set_b[1][0], set_b[2][1]),
                1 - set_b[1][0])
        return {'upper_on_straight' : upper_on_straight,
                'right_on_straight' : right_on_straight}

    def is_triangle_conv_z_i_2d_bound(self, kink_checks):
        # If both True or both False, it's a triangle:
        return len(set(kink_checks.values())) == 1

    def add_kink_points_2d_bound(self, set_z_hat, set_b, kink_checks):
        # If conv(Z intersection I) is not a triangle we need to add the
        # relevant kink point:
        kinks = self.p_bar_minus_kinks(set_b[2][1])
        if kink_checks['upper_on_straight'] and \
                not kink_checks['right_on_straight']:
            return set_z_hat[0:2] + \
                [(kinks[0],
                  self.p_bar_minus(kinks[0], set_b[2][1]))] + \
                set_z_hat[2:4]
        elif not kink_checks['upper_on_straight'] and \
                kink_checks['right_on_straight']:
            return set_z_hat[0:2] + \
                [(kinks[1],
                  self.p_bar_minus(kinks[1], set_b[2][1]))] + \
                set_z_hat[2:4]

    def get_conv_z_i_2d_bound(self, p_c):
        d, pm, pp = self.auction_data.get_demand(p_c)
        set_b = self.compute_set_b(p_c)

        if self.z_i_nonempty_cond_1(set_b) and self.z_i_nonempty_cond_2(set_b):
            # Vertices of the set (triangle):
            set_z_hat = [(set_b[0][0], set_b[1][0]),
                    (self.inv_p_bar_minus(set_b[1][0], set_b[2][1]),
                     set_b[1][0]),
                    (set_b[0][0], self.p_bar_minus(set_b[0][0], set_b[2][1]))]
            set_z_hat = set_z_hat + [set_z_hat[0]]  # Close the triangle

            # Check if we need to add a vertex at a kink point of p-bar-minus:
            kink_checks = self.check_kinks_conv_z_i_2d_bound(set_b)
            if self.is_triangle_conv_z_i_2d_bound(kink_checks):
                return set_z_hat
            else:
                return self.add_kink_points_2d_bound(set_z_hat, set_b,
                        kink_checks)
        else:
            return None

    def is_rationalizable_2d_bound(self, p_c):
        set_z_i = Polygon(self.get_conv_z_i_2d_bound(p_c))
        if set_z_i is None:
            return False
        set_a = self.compute_set_a(p_c)
        set_a = Polygon([(set_a[0][0], set_a[1][0]),
                (set_a[0][1], set_a[1][0]),
                (set_a[0][1], set_a[1][1]),
                (set_a[0][0], set_a[1][1]),
                (set_a[0][0], set_a[1][0])])
        return set_z_i.intersects(set_a)

    def lower_bound_collusive(self, rho_p):
        tied_winners = self.auction_data.tied_winners
        lower_bound_collusive = {'tied_winner': tied_winners}

        revenue = self.auction_data.get_counterfactual_demand(
            rho_p, .0).revenue
        revenue = revenue.loc[revenue.index > 0].sort_index()
        elasticity = (revenue.iloc[1:] - revenue.iloc[0]) / (
            revenue.iloc[1] * revenue.index[:-1])
        lower_bound_collusive['deviate_up'] = \
            (1 - tied_winners) * elasticity[rho_p]

        return lower_bound_collusive

    def _assess_share_competitive(self, num_steps, is_rationalizable):
        steps = np.linspace(0, 1, num_steps)
        list_p_c = list(itertools.product(steps, repeat=4))
        max_competitive = 0
        arg_max = list_p_c[0]
        for i, p_c in enumerate(list_p_c):
            if is_rationalizable(p_c):
                share_comp = self.auction_data.get_competitive_share(p_c)
                if share_comp > max_competitive:
                    max_competitive = share_comp
                    arg_max = p_c
        tied_winners = self.auction_data.tied_winners
        return {'non_comp_tied_bids': tied_winners,
                'non_comp_IC': (1 - tied_winners) * (1 - max_competitive),
                'arg_max': arg_max}

    def assess_share_competitive(self, num_steps=11):
        return self._assess_share_competitive(num_steps,
                                              self.is_rationalizable)

    def assess_share_competitive_iid(self, num_steps=11):
        return self._assess_share_competitive(num_steps,
                                              self.is_rationalizable_iid)

    @property
    def q_star(self):
        num = (self.auction_data.enum_categories[(1, 0, 1)] *
                (1 - self.lower_slope_pp)) + \
                self.t * (1 - self.lower_slope_pp)
        denom = self.lower_slope_pp * \
                self.auction_data.enum_categories[(1, 0, 0)]
        if denom == 0:
            return 1
        else:
            return np.min([1, num/denom])

    def assess_share_competitive_2d_bound(self, num_steps=11):
        steps = np.linspace(0, 1, num_steps)
        list_p_c = list(itertools.product(steps, repeat=2))
        list_p_c = [(1., x[0], x[1], 1.) for x in list_p_c]

        # Drop if lower than q_star:
        list_p_c = [x for x in list_p_c if x[1] <= self.q_star]

        max_competitive = 0
        arg_max = list_p_c[0]
        for i, p_c in enumerate(list_p_c):
            if self.is_rationalizable_2d_bound(p_c):
                share_comp = self.auction_data.get_competitive_share(p_c)
                if share_comp > max_competitive:
                    max_competitive = share_comp
                    arg_max = p_c
        tied_winners = self.auction_data.tied_winners
        return {'non_comp_tied_bids': tied_winners,
                'non_comp_IC': (1 - tied_winners) * (1 - max_competitive),
                'arg_max': arg_max}

    def assess_share_competitive_upward_only(self, num_steps=11):
        max_competitive = (1. / (self.auction_data.num_histories)) * \
                (self.q_star * self.auction_data.enum_categories[(1, 0, 0)] +
                 self.auction_data.enum_categories[(0, 0, 0)] +
                 self.auction_data.enum_categories[(0, 1, 0)] +
                 self.auction_data.enum_categories[(1, 0, 1)])
        tied_winners = self.auction_data.tied_winners
        return {'non_comp_tied_bids': tied_winners,
                'non_comp_IC': (1 - tied_winners) * (1 - max_competitive),
                'arg_max': (1, self.q_star, 1, 1)}


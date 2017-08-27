import numpy as np
from auction_data import AuctionData
import itertools


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

    def set_data(self, auction_data=None):
        assert isinstance(auction_data, AuctionData)
        # auction_data.set_bid_data(auction_data.df_bids)
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
        return (1-self.rho_m - 1/(1+self.m)) / (1 - 1/(1+self.m))

    def binding_pp(self, d):
        return self.rho_p * d / (self.rho_m + self.rho_p +
                                 self.rho_m * d/(1-d))

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

    def extreme_points_set_z(self, u_d, o_d):
        list_points_unconstrained, list_points_constrained = (
            [(u_d, 0, a * u_d), (u_d, 0, u_d), (o_d, 0, a * o_d),
             (o_d, 0, o_d), (u_d, 1 - u_d, a * u_d), (u_d, 1 - u_d, u_d),
             (o_d, 1 - o_d, a * o_d), (o_d, 1 - o_d, o_d)]
            for a in [self.lower_slope_pp, self.tangent_binding_pm]
        )

        if self.tangent_binding_pm <= self.lower_slope_pp:
            list_points = list_points_unconstrained
        else:
            list_points = list_points_constrained
            list_points += [
                (u_d, 0, self.lower_slope_pp * u_d),
                (u_d, self.value_pm(u_d), self.lower_slope_pp * u_d),
                (o_d, 0, self.lower_slope_pp * o_d),
                (o_d, self.value_pm(o_d), self.lower_slope_pp * o_d)
            ]

            if self.intersect_d <= o_d:
                list_points.append(
                    (self.intersect_d, self.value_pm(self.intersect_d),
                     self.lower_slope_pp * self.intersect_d)
                )
        return np.array(list_points)

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

    def is_rationalizable(self, p_c):
        set_a = self.compute_set_a(p_c)  # empirical moment constraints
        set_b = self.compute_set_b(p_c)  # information constraints
        box = self.intersect_box(set_a, set_b)
        if None in box:
            return False

        extreme_points_box = self.get_box_extreme_points(box)
        extreme_points_z = self.extreme_points_set_z(*set_b[0])

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

    def lower_bound_collusive(self, rho_p):
        num_tied = self.auction_data.df_tied_bids.shape[0]
        num_untied = self.auction_data.df_bids.shape[0]
        tied_winner = num_tied / float(num_tied + num_untied)
        lower_bound_collusive = {'tied_winner': tied_winner}

        revenue = self.auction_data.get_counterfactual_demand(
            rho_p, .0).revenue
        revenue = revenue.loc[revenue.index > 0].sort_index()
        elasticity = (revenue.iloc[1:] - revenue.iloc[0]) / (
            revenue.iloc[1] * revenue.index[:-1])
        lower_bound_collusive['deviate_up'] = \
            (1 - tied_winner) * elasticity[rho_p]

        return lower_bound_collusive

    def assess_share_competitive(self, num_steps=11):
        steps = np.linspace(0, 1, num_steps)
        list_p_c = list(itertools.product(steps, repeat=4))
        max_competitive = 0
        arg_max = list_p_c[0]
        for i, p_c in enumerate(list_p_c):
            if self.is_rationalizable(p_c):
                share_comp = self.auction_data.get_competitive_share(p_c)
                if share_comp > max_competitive:
                    max_competitive = share_comp
                    arg_max = p_c
        return max_competitive, arg_max

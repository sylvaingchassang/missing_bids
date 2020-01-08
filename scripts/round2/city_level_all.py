from scripts.figures_import_helper import *
# %matplotlib inline

# before/after industry comparisons, using national data

# +
filename = 'tsuchiura_data.csv'
tsuchiura_data = auction_data.AuctionData(os.path.join(path_data, filename))
tsuchiura_data = auction_data.AuctionData(
    tsuchiura_data.df_bids.loc[tsuchiura_data.data.minprice.isnull()])
plot_delta(tsuchiura_data)

filename = 'municipal_pub_reserve_no_pricefloor.csv'
other_data = auction_data.AuctionData(os.path.join(path_data, filename))
plot_delta(other_data)

# +
s1 = set(other_data.df_bids.pid)
s2 = set(tsuchiura_data.df_bids.pid)

assert len(s2.intersection(s1)) == 0
# -

all_bids = pd.concat((other_data.df_bids, tsuchiura_data.df_bids), axis=0)
data = auction_data.AuctionData.from_clean_bids(all_bids)
plot_delta(data, filename='R2/city_auctions_delta')

list_devs = [up_deviations, down_deviations, all_deviations]
list_solutions = []
for devs in list_devs:
    solutions, ties = compute_solution_parallel(
        data, devs)
    list_solutions.append(1 - ties - solutions * (1-ties))

print('saving plot\n')
pretty_plot(
    'R2/city auctions',
    list_solutions,
    [r"deviations {}".format(devs) for devs in list_devs],
    xlabel='minimum markup',
    mark=np.array(['k.:', 'k.--', 'k.-']),
    xticks=r2_min_mkps
)

print('saving data\n')
save2frame(list_solutions,
           ['min_m={}'.format(m) for m in r2_min_mkps],
           'R2/city_auctions')


# varying maximum markup

list_max_markups = [.5, 1, 1.5]
devs = all_deviations
list_solutions = []

for max_markup in list_max_markups:
    def this_constraint():
        return [[environments.MarkupConstraint(
            max_markup=max_markup, min_markup=min_markup)]
            for min_markup in r2_min_mkps]
    this_solver = ComputeMinimizationSolution(
        solver_cls=solvers.ParallelSolver, constraint_func=this_constraint)
    solutions, ties = compute_solution_parallel(data, devs)
    list_solutions.append(1 - ties - solutions * (1-ties))

print('saving plot\n')
pretty_plot(
    'R2/city auctions -- varying max markup',
    list_solutions,
    [r"max markup {}".format(max_markup) for max_markup in list_max_markups],
    xlabel='minimum markup',
    mark=np.array(['k.:', 'k.--', 'k.-']),
    xticks=r2_min_mkps
)

print('saving data\n')
save2frame(list_solutions,
           ['min_m={}'.format(m) for m in r2_min_mkps],
           'R2/city_auctions_max_markup')

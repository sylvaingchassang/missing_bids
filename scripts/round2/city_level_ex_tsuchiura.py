from scripts.figures_import_helper import *
# %matplotlib inline

# before/after industry comparisons, using national data

# +
filename = 'municipal_pub_reserve_no_pricefloor.csv'
data = auction_data.AuctionData(os.path.join(path_data, filename))
plot_delta(data, filename='R2/city_auctions_delta_ex_tsuchiura')

list_devs = [[.0, .001], [-.025, .0], [-.025, .0, .001]]
list_solutions = []
for devs in list_devs:    
    solutions, ties = compute_solution_parallel(
        data, devs)
    list_solutions.append(1 - ties - solutions * (1-ties))

print('saving plot\n')
pretty_plot(
    'R2/city auctions (ex Tsuchiura)',
    list_solutions,
    [r"deviations {}".format(devs) for devs in list_devs],
    xlabel='minimum markup',
    mark=np.array(['k.:', 'k.--', 'k.-']),
    xticks=r2_min_mkps
)

print('saving data\n')
save2frame(list_solutions,
           ['min_m={}'.format(m) for m in r2_min_mkps],
           'R2/city_auctions_ex_tsuchiura')

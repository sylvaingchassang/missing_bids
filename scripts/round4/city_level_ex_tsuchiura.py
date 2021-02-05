from scripts.round3.figures_import_helper_r3 import *
# %matplotlib inline

# before/after industry comparisons, using national data

# +
filename = 'municipal_pub_reserve_no_pricefloor.csv'
data = asymptotics.PIDMeanAuctionData(os.path.join(path_data, filename))
plot_delta(data, filename='R3/city_auctions_delta_ex_tsuchiura')

list_devs = [up_deviations, down_deviations, all_deviations]
list_solutions = []
for devs in list_devs:    
    solutions, ties = compute_asymptotic_solution(
        data, devs)
    list_solutions.append(1 - ties - solutions * (1 - ties))

print('saving plot\n')
pretty_plot(
    'R3/city auctions (ex Tsuchiura)',
    list_solutions,
    [r"deviations {}".format(dev_repr(devs)) for devs in list_devs],
    xlabel='minimum markup',
    mark=np.array(['k.:', 'k.--', 'k.-']),
    xticks=r3_min_markups
)

print('saving data\n')
save2frame(list_solutions,
           ['min_m={}'.format(m) for m in r3_min_markups],
           'R3/city_auctions_ex_tsuchiura')

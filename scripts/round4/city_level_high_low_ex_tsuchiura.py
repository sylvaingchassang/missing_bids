from scripts.round3.figures_import_helper_r3 import *
# %matplotlib inline

# before/after industry comparisons, using national data

# +
filename = 'municipal_pub_reserve_no_pricefloor.csv'
data = asymptotics.AsymptoticAuctionData(os.path.join(path_data, filename))

data_low = asymptotics.AsymptoticAuctionData.from_clean_bids(
    data.df_bids.loc[data.df_bids.norm_bid < .9])
data_high = asymptotics.AsymptoticAuctionData.from_clean_bids(
    data.df_bids.loc[data.df_bids.norm_bid > .9])
                                                     
plot_delta(data_low, filename='R4/city_delta_low_ex_tsuchiura')
plot_delta(data_high, filename='R4/city_delta_high_ex_tsuchiura')
# -

deviations = all_deviations
list_solutions = []
for data in [data_low, data_high]:    
    solutions, ties = compute_asymptotic_solution(
        data, deviations)
    list_solutions.append(1 - ties - solutions * (1-ties))

print('saving plot\n')
pretty_plot(
    'R4/city auctions -- high and low bids (ex Tsuchiura)',
    list_solutions,
    ['normalized bid $ < .9$', 'normalized bid $> .9$'],
    xlabel='minimum markup',
    xticks=r3_min_markups
)

print('saving data\n')
save2frame(list_solutions,
           ['min_m={}'.format(m) for m in r3_min_markups],
           'R4/city_auctions_high_low_ex_tsuchiura')



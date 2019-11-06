from scripts.figures_import_helper import *
# %matplotlib inline

# before/after industry comparisons, using national data

# +
filename = 'municipal_pub_reserve_no_pricefloor.csv'
data = auction_data.AuctionData(os.path.join(path_data, filename))

data_low = auction_data.AuctionData.from_clean_bids(
    data.df_bids.loc[data.df_bids.norm_bid < .9])
data_high = auction_data.AuctionData.from_clean_bids(
    data.df_bids.loc[data.df_bids.norm_bid > .9])
                                                     
plot_delta(data_low, filename='R2/city_delta_low_ex_tsuchiura')
plot_delta(data_high, filename='R2/city_delta_high_ex_tsuchiura')
# -

deviations = [-.025, 0, .001]
list_solutions = []
for data in [data_low, data_high]:    
    solutions, ties = compute_solution_parallel(
        data, deviations)
    list_solutions.append(1 - ties - solutions * (1-ties))

print('saving plot\n')
pretty_plot(
    'R2/city auctions -- high and low bids (ex Tsuchiura)',
    list_solutions,
    ['normalized bid $ < .9$', 'normalized bid $> .9$'],
    xlabel='minimum markup',
    xticks=r2_min_mkps
)

print('saving data\n')
save2frame(list_solutions,
           ['min_m={}'.format(m) for m in r2_min_mkps],
           'R2/city_auctions_high_low_ex_tsuchiura')



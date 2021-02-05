from scripts.round3.figures_import_helper_r3 import *
# %matplotlib inline

# before/after industry comparisons, using national data

print('='*20, '\n city level high low')


# +
filename = 'tsuchiura_data.csv'
tsuchiura_data = asymptotics.AsymptoticAuctionData(
    os.path.join(path_data, filename))
tsuchiura_data = asymptotics.AsymptoticAuctionData(
    tsuchiura_data.df_bids.loc[tsuchiura_data.data.minprice.isnull()])
plot_delta(tsuchiura_data)

filename = 'municipal_pub_reserve_no_pricefloor.csv'
other_data = asymptotics.AsymptoticAuctionData(
    os.path.join(path_data, filename))

# +
s1 = set(other_data.df_bids.pid)
s2 = set(tsuchiura_data.df_bids.pid)

assert len(s2.intersection(s1)) == 0

# +
all_bids = pd.concat((other_data.df_bids, tsuchiura_data.df_bids), axis=0)
data_low = asymptotics.AsymptoticAuctionData.from_clean_bids(
    all_bids.loc[all_bids.norm_bid < .9])
data_high = asymptotics.AsymptoticAuctionData.from_clean_bids(
    all_bids.loc[all_bids.norm_bid > .9])
                                                     
plot_delta(data_low, filename='R4/city_delta_low')
plot_delta(data_high, filename='R4/city_delta_high')
# -

deviations = all_deviations
list_solutions = []
for data in [data_low, data_high]:    
    solutions, ties = compute_asymptotic_solution(
        data, deviations)
    list_solutions.append(1 - ties - solutions * (1-ties))

print('saving plot\n')
pretty_plot(
    'R4/city auctions -- high and low bids',
    list_solutions,
    ['normalized bid $ < .9$', 'normalized bid $> .9$'],
    xlabel='minimum markup',
    xticks=r3_min_markups
)

print('saving data\n')
save2frame(list_solutions,
           ['min_m={}'.format(m) for m in r3_min_markups],
           'R4/city_auctions_high_low')



from scripts.round3.figures_import_helper_r3 import *


# bids close and far from reserve price, using national data

print('='*20 + '\n' + 'National sample (high/low bids)')
print('collecting and processing data')
national_data = asymptotics.MultistageAsymptoticAuctionData(
    os.path.join(path_data, 'sample_with_firm_rank.csv'))

national_data_above = \
    asymptotics.MultistageAsymptoticAuctionData.from_clean_bids(
        national_data.df_bids.loc[
            national_data.data.norm_bid.between(0.8, .95)])
national_data_below = \
    asymptotics.MultistageAsymptoticAuctionData.from_clean_bids(
        national_data.df_bids.loc[national_data.data.norm_bid < 0.8])

plot_delta(national_data_above, filename='R4/national_data_above_deltas')
plot_delta(national_data_below, filename='R4/national_data_below_deltas')


print('computing problem solutions')
deviations = all_deviations
solutions_above, _ = compute_asymptotic_multistage_solution(
    national_data_above, deviations)
solutions_below, _ = compute_asymptotic_multistage_solution(
     national_data_below, deviations)

print('saving plot\n')
pretty_plot('R4/high vs low normalized bids (national auctions)',
            [1 - solutions_above, 1 - solutions_below],
            ["normalized bid within [.8, .95]", "normalized bid below .8"],
            xlabel='minimum markup',
            xticks=r3_min_markups)

print('saving data\n')
save2frame([1 - solutions_above, 1 - solutions_below],
           ['min_m={}'.format(m) for m in r3_min_markups],
           'R4/high vs low normalized bids (national auctions)')

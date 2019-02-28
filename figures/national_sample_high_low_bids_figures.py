from figures.figures_import_helper import *


# bids close and far from reserve price, using national data

print('='*20 + '\n' + 'National sample (high/low bids)')
print('collecting and processing data')
national_data = auction_data.AuctionData(
    os.path.join(path_data, 'sample_with_firm_rank.csv'))

national_data_above = auction_data.AuctionData.from_clean_bids(
    national_data.df_bids.loc[national_data.data.norm_bid > 0.8])
national_data_below = auction_data.AuctionData.from_clean_bids(
    national_data.df_bids.loc[national_data.data.norm_bid < 0.8])

print('computing problem solutions')
deviations = [-.025, .0, .001]
solutions_above, _ = compute_minimization_solution(
    national_data_above, deviations)
solutions_below, _ = compute_minimization_solution(
     national_data_below, deviations)

print('saving plot\n')
pretty_plot(
        'high vs low normalized bids',
        [solutions_above, solutions_below],
        ["normalized bid above .8", "normalized bid below .8"]
   )

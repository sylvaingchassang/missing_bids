from scripts.figures_import_helper import *

# before/after industry comparisons, using national data

list_data_sets = [
    ('Bridges', 'bc_collusion.csv', [-.025, .0, .001]),
    ('Electric', 'ec_collusion.csv', [-.025, .0, .001]),
    ('Pre-Stressed Concrete', 'pc_collusion.csv', [-.025, .0, .001]),
    ('Floods', 'fc_collusion.csv', [-.03, .0, .001])
]

for industry, file, deviations in list_data_sets:
    print('='*20 + '\n' + industry)
    print('collecting and processing data')
    data = auction_data.AuctionData(os.path.join(path_data, file))
    data_before = auction_data.AuctionData(
        data.df_bids.loc[data.data.before == 1])
    data_after = auction_data.AuctionData(
        data.df_bids.loc[data.data.before.isnull()])

    print('computing before/after problem solution')
    solutions_before, _ = compute_minimization_solution_unfiltered(
        data_before, deviations)
    solutions_after, _ = compute_minimization_solution_unfiltered(
        data_after, deviations)
    del data
    del data_before
    del data_after

    print('saving plot\n')
    pretty_plot(
        os.path.join('R1', industry),
        np.array([1 - solutions_before, 1 - solutions_after]),
        np.array(["before investigation", "after investigation"])
    )
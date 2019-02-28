from scripts.figures_import_helper import *
import pandas as pd

print('='*20 + '\n' + 'National sample (individual firms)')
print('collecting and processing data')
national_data = auction_data.AuctionData(
    os.path.join(path_data, 'sample_with_firm_rank.csv'))

deviations = [-.025, .0, .001]
filter_ties = auction_data.FilterTies(tolerance=.0001)
filtered_data = filter_ties(national_data)
share_competitive = []

print('solving minimization problem for each firm')
for rank in range(10):
    print('firm {}'.format(rank + 1))
    filtered_data_firm = auction_data.AuctionData.from_clean_bids(
        filtered_data.df_bids.loc[filtered_data.data.rank2 == rank + 1])
    demand_firm = [filtered_data_firm.get_counterfactual_demand(rho) for rho in
                   deviations]
    constraints = [
        environments.MarkupConstraint(max_markup=.5, min_markup=.02),
        environments.InformationConstraint(k=1, sample_demands=demand_firm)]

    min_collusion_solver = analytics.MinCollusionIterativeSolver(
        data=filtered_data_firm,
        deviations=deviations,
        metric=analytics.IsNonCompetitive,
        plausibility_constraints=constraints,
        num_points=num_points,
        seed=0,
        project=False,
        filter_ties=None,
        number_iterations=number_iterations_individual_firms,
        confidence_level=.95,
        moment_matrix=auction_data.moment_matrix(deviations, 'slope'),
        moment_weights=np.array([0, 0, 1])
    )

    share_competitive.append(
        [rank + 1, 1 - min_collusion_solver.result.solution])

pd.DataFrame(data=share_competitive, columns=['rank', 'share_comp']).to_csv(
    os.path.join(path_figures, 'individual_firms.csv'), index=False)

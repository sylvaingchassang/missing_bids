from scripts.round3.figures_import_helper_r3 import *
from datetime import datetime

print('='*20 + '\n' + 'National sample (individual firms)')
print('collecting and processing data')
national_data = asymptotics.MultistageAsymptoticAuctionData(
    os.path.join(path_data, 'sample_with_firm_rank.csv'))

deviations = all_deviations
filter_ties = auction_data.FilterTies(tolerance=.0001)
filtered_data = filter_ties(national_data)
share_competitive = []

print('solving minimization problem for each firm')
for rank in range(30):
    print('firm {}'.format(rank + 1))
    filtered_data_firm = \
        asymptotics.MultistageAsymptoticAuctionData.from_clean_bids(
            filtered_data.df_bids.loc[national_data.data.rank2 == rank + 1])

    metric = rebidding.EfficientMultistageIsNonCompetitive
    metric.min_markup, metric.max_markup = .025, .5
    min_collusion_solver = asymptotics.ParallelAsymptoticMultistageSolver(
        data=filtered_data_firm,
        deviations=deviations,
        metric=metric,
        plausibility_constraints=[environments.EmptyConstraint()],
        num_points=NUM_POINTS,
        seed=0,
        project=False,
        filter_ties=None,
        num_evaluations=NUM_EVAL,
        confidence_level=.95,
        moment_matrix=multistage_moment_matrix,
        enhanced_guesses=True
    )

    share_competitive.append(
        [rank + 1, 1 - min_collusion_solver.result.solution])

save2frame(share_competitive,
           ['rank', 'share_comp'], 'R4/national_individual_firms')

print("**** END TIME {} ****".format(
      datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
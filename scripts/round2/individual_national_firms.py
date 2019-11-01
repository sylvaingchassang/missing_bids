from scripts.figures_import_helper import *
# %matplotlib inline

print('='*20 + '\n' + 'National sample (individual firms)')
print('collecting and processing data')
national_data = rebidding.RefinedMultistageData(
    os.path.join(path_data, 'sample_with_firm_rank.csv'))

deviations = [-.025, .0, .001]
filter_ties = auction_data.FilterTies(tolerance=.0001)
filtered_data = filter_ties(national_data)
share_competitive = []

print('solving minimization problem for each firm')
for rank in range(30):
    print('firm {}'.format(rank + 1))
    filtered_data_firm = rebidding.RefinedMultistageData.from_clean_bids(
        filtered_data.df_bids.loc[filtered_data.data.rank2 == rank + 1])
    constraints = [
        environments.MarkupConstraint(max_markup=.5, min_markup=.02)]

    min_collusion_solver = rebidding.ParallelRefinedMultistageSolver(
        data=filtered_data_firm,
        deviations=deviations,
        metric=rebidding.RefinedMultistageIsNonCompetitive,
        plausibility_constraints=constraints,
        num_points=NUM_POINTS,
        seed=0,
        project=False,
        filter_ties=None,
        num_evaluations=NUM_EVAL,
        confidence_level=1 - .05 / len(deviations),
        moment_matrix=rebidding.refined_moment_matrix(),
        moment_weights=np.identity(5)
    )

    share_competitive.append(
        [rank + 1, 1 - min_collusion_solver.result.solution])

save2frame(share_competitive,
           ['rank', 'share_comp'], 'R2/national_individual_firms')

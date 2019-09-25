import environments
import numpy as np
import rebidding as rb
from itertools import product

m_array = [0, .05, .1, .15, .2]

deviations = [-.02, .0, .001]

multistage_pc_data = rb.RefinedMultistageData(
    '~/Desktop/missing_bids_troubleshooting/pc_collusion.csv')

multistage_pc_data_after = rb.RefinedMultistageData(
    multistage_pc_data.df_bids.loc[multistage_pc_data.data.before.isnull()])

multistage_demands_pc_after = \
    multistage_pc_data_after.assemble_target_moments(deviations)

for m_0, seed in product(m_array, [0, 1, 2]):
    constraints = [
        environments.MarkupConstraint(max_markup=.5, min_markup=m_0)]

    min_collusion_solver = rb.IteratedRefinedMultistageSolver(
        data=multistage_pc_data_after,
        deviations=deviations,
        metric=rb.RefinedMultistageIsNonCompetitive,
        plausibility_constraints=constraints,
        num_points=1000.0,
        seed=seed,
        project=True,
        filter_ties=None,
        number_iterations=25,
        confidence_level=.95,
        moment_matrix=rb.refined_moment_matrix(),
        moment_weights=np.identity(5)
    )
    min_collusion_solver.max_best_sol_index = 100
    share = min_collusion_solver.result.solution
    print('min markup {}, seed {} | share non collusive: {}'.format(
        m_0, seed, 1 - share))
from rlberry.experiment import load_experiment_results
from rlberry.stats import plot_episode_rewards
import matplotlib.pyplot as plt

import matplotlib
from pylab import rcParams
rcParams['figure.figsize'] = 8, 5
rcParams['mathtext.default'] = 'regular'
rcParams['font.size'] = 14
matplotlib.rcParams.update({'errorbar.capsize': 0})
# ------------------------------------------
# Load results
# ------------------------------------------
EXPERIMENT_NAMES = ['twinrooms_exp', 'twinrooms_exp_unif_discr.yaml']

output_data = load_experiment_results('results', EXPERIMENT_NAMES)

PLOT_TITLES = {
    'rs_kernelucbvi_symmetric': 'Kernel-UCBVI + expert knowledge',
    'rs_kernelucbvi': 'Kernel-UCBVI',
    'rs_greedykernelucbvi': 'Greedy-Kernel-UCBVI',
    'adaptiveql': 'AdaptiveQL',
    'ucbvi': 'UCBVI',
    'optql': 'OptQL',
}


# Get list of AgentStats
_stats_list = list(output_data['stats'].values())
stats_list = []
for stats in _stats_list:
    if stats.agent_name in PLOT_TITLES:
        stats.agent_name = PLOT_TITLES[stats.agent_name]
        stats_list.append(stats)
        print("n agents = ", len(stats.fitted_agents))

# -------------------------------
# Plot and save
# -------------------------------
plot_episode_rewards(stats_list, cumulative=False, show=False)

matplotlib.rcParams['text.usetex'] = True
plot_episode_rewards(stats_list, cumulative=True, show=False, grid=False)
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.xlabel("episode", labelpad=0)


# show save all figs
figs = [plt.figure(n) for n in plt.get_fignums()]
for ii, fig in enumerate(figs):
    fname = output_data['experiment_dirs'][0] / 'fig_{}.pdf'.format(ii)
    fig.savefig(fname, format='pdf', bbox_inches='tight')

plt.show()

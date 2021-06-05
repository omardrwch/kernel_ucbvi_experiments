"""
To run the experiment:
$ python run.py config.yaml

To see more options:
$ python run.py config.yaml -h
"""

from rlberry.experiment import experiment_generator

for agent_stats in experiment_generator():
    print(agent_stats.agent_class)
    print(agent_stats.init_kwargs)
    agent_stats.fit()
    agent_stats.save()
    agent_stats.save_results()


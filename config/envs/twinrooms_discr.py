from rlberry.envs.benchmarks.generalization.twinrooms import TwinRooms
from rlberry.wrappers.discretize_state import DiscretizeStateWrapper

def constructor(n_bins, noise_room1, noise_room2):
    env = TwinRooms(noise_room1, noise_room2)
    env = DiscretizeStateWrapper(env, n_bins)
    return env


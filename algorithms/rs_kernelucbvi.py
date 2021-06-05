import logging

import numpy as np

import gym.spaces as spaces
from rlberry.agents import Agent
from rlberry.agents.dynprog.utils import backward_induction
from rlberry.agents.dynprog.utils import backward_induction_in_place
from rlberry.utils.writers import PeriodicWriter
from algorithms.rs_utils import update_model, update_value_and_get_action, map_to_representative


logger = logging.getLogger(__name__)


class RSKernelUCBVIAgent(Agent):
    """
    Implements KernelUCBVI with representative states.

    Parameters
    ----------
    env : Model
        Online model with continuous (Box) state space and discrete actions
    n_episodes : int
        number of episodes
    gamma : double
        Discount factor in [0, 1]. If gamma is 1.0, the problem is set to
        be finite-horizon.
    horizon : int
        Horizon of the objective function. If None and gamma<1, set to
        1/(1-gamma).
    lp_metric: int
        The metric on the state space is the one induced by the p-norm,
        where p = lp_metric. Default = 2, for the Euclidean metric.
    kernel_type : string
        See rlberry.agents.kernel_based.kernels.kernel_func for
        possible kernel types.
    scaling: numpy.ndarray
        Must have the same size as state array, used to scale the states
        before computing the metric.
        If None, set to:
        - (env.observation_space.high - env.observation_space.low) if high
            and low are bounded
        - np.ones(env.observation_space.shape[0]) if high or low
        are unbounded
    bandwidth : double
        Kernel bandwidth.
    min_dist : double
        Minimum distance between two representative states
    max_repr : int
        Maximum number of representative states.
        If None, it is set to  (sqrt(d)/min_dist)**d, where d
        is the dimension of the state space
    bonus_scale_factor : double
        Constant by which to multiply the exploration bonus,
        controls the level of exploration.
    beta : double
        Regularization constant.
    bonus_type : string
        Type of exploration bonus. Currently, only "simplified_bernstein"
            is implemented.
    real_time_dp: bool, default: False
        If True, use real-time dynamic programming
    use_twinroom_symmetry: bool, default: False
        If True, use the fact that TwinRooms is symmetric when computing distances.
    """

    name = "RS-KernelUCBVI"

    def __init__(self,
                 env,
                 n_episodes=1000,
                 gamma=1.0,
                 horizon=None,
                 lp_metric=2,
                 kernel_type="epanechnikov",
                 scaling=None,
                 bandwidth=0.05,
                 min_dist=0.1,
                 max_repr=1000,
                 bonus_scale_factor=1.0,
                 beta=0.01,
                 bonus_type="simplified_bernstein",
                 real_time_dp=False,
                 use_twinroom_symmetry=False,
                 **kwargs):
        # init base class
        Agent.__init__(self, env, **kwargs)

        self.n_episodes = n_episodes
        self.gamma = gamma
        self.horizon = horizon
        self.lp_metric = lp_metric
        self.kernel_type = kernel_type
        self.bandwidth = bandwidth
        self.min_dist = min_dist
        self.bonus_scale_factor = bonus_scale_factor
        self.beta = beta
        self.bonus_type = bonus_type
        self.real_time_dp = real_time_dp
        self.use_twinroom_symmetry = use_twinroom_symmetry

        if real_time_dp:
            self.name = "GreedyKernelUCBVI"

        # check environment
        assert isinstance(self.env.observation_space, spaces.Box)
        assert isinstance(self.env.action_space, spaces.Discrete)

        # other checks
        assert gamma >= 0 and gamma <= 1.0
        if self.horizon is None:
            assert gamma < 1.0, \
                "If no horizon is given, gamma must be smaller than 1."
            self.horizon = int(np.ceil(1.0 / (1.0 - gamma)))

        # state dimension
        self.state_dim = self.env.observation_space.shape[0]

        # compute scaling, if it is None
        if scaling is None:
            # if high and low are bounded
            if (self.env.observation_space.high == np.inf).sum() == 0 \
                    and (self.env.observation_space.low == -np.inf).sum() == 0:
                scaling = self.env.observation_space.high \
                    - self.env.observation_space.low
                # if high or low are unbounded
            else:
                scaling = np.ones(self.state_dim)
        else:
            assert scaling.ndim == 1
            assert scaling.shape[0] == self.state_dim
        self.scaling = scaling

        # maximum value
        r_range = self.env.reward_range[1] - self.env.reward_range[0]
        if r_range == np.inf or r_range == 0.0:
            logger.warning("{}: Reward range is  zero or infinity. ".format(self.name)
                           + "Setting it to 1.")
            r_range = 1.0

        self.v_max = np.zeros(self.horizon)
        self.v_max[-1] = r_range
        for hh in reversed(range(self.horizon-1)):
            self.v_max[hh] = r_range + self.gamma*self.v_max[hh+1]

        # number of representative states and number of actions
        if max_repr is None:
            max_repr = int(np.ceil((1.0 * np.sqrt(self.state_dim)
                                    / self.min_dist) ** self.state_dim))
        self.max_repr = max_repr

        # current number of representative states
        self.M = None
        self.A = self.env.action_space.n

        # declaring variables
        self.episode = None  # current episode
        self.representative_states = None  # coordinates of all repr states
        self.N_sa = None   # sum of weights at (s, a)
        self.B_sa = None   # bonus at (s, a)
        self.R_hat = None  # reward  estimate
        self.P_hat = None  # transitions estimate
        self.Q = None  # Q function
        self.V = None  # V function

        self.Q_policy = None  # Q function for recommended policy

        # initialize
        self.reset()

    def reset(self, **kwargs):
        self.M = 0
        H = self.horizon
        S = self.max_repr
        A = self.A
        dim = self.state_dim
    
        self.representative_states = np.zeros((S, dim))
        self.N_sa = np.zeros((S, A))
        self.B_sa = self.v_max[0] * np.ones((S, A))

        self.R_hat = np.zeros((S, A))
        self.P_hat = np.zeros((S, A, S))

        self.V = self.v_max[0] * np.ones((H, S))
        self.Q = np.zeros((H, S, A))
        self.Q_policy = None

        self.episode = 0

        # default writer
        self.writer = PeriodicWriter(self.name,
                                     log_every=5*logger.getEffectiveLevel())

    def policy(self, state, hh=0, **kwargs):
        return self._get_action(state, hh)

    def fit(self, **kwargs):
        info = {}
        self._rewards = np.zeros(self.n_episodes)
        self._cumul_rewards = np.zeros(self.n_episodes)
        for _ in range(self.n_episodes):
            self._run_episode()

        # compute Q function for the recommended policy
        self.Q_policy, _ = backward_induction(self.R_hat[:self.M, :],
                                              self.P_hat[:self.M, :, :self.M],
                                              self.horizon, self.gamma)

        info["n_episodes"] = self.n_episodes
        info["episode_rewards"] = self._rewards
        return info

    def _map_to_repr(self, state, accept_new_repr=True):
        repr_state = map_to_representative(state,
                                           self.lp_metric,
                                           self.representative_states,
                                           self.M,
                                           self.min_dist,
                                           self.scaling,
                                           accept_new_repr,
                                           self.use_twinroom_symmetry)
        # check if new representative state
        if repr_state == self.M:
            self.M += 1
        return repr_state

    def _update(self, state, action, next_state, reward):
        repr_state = self._map_to_repr(state)
        repr_next_state = self._map_to_repr(next_state)

        update_model(repr_state, action, repr_next_state, reward,
                     self.M,
                     self.representative_states,
                     self.lp_metric,
                     self.scaling,
                     self.bandwidth,
                     self.bonus_scale_factor,
                     self.beta,
                     self.v_max[0],
                     self.bonus_type,
                     self.kernel_type,
                     self.N_sa,
                     self.B_sa,
                     self.P_hat,
                     self.R_hat,
                     self.use_twinroom_symmetry)

    def _get_action(self, state, hh=0):
        """ Sampling policy. """
        repr_state = self._map_to_repr(state, accept_new_repr=False)
        if not self.real_time_dp:
            assert self.Q is not None
            return self.Q[hh, repr_state, :].argmax()
        else:
            if self.M > 0:
                update_fn = update_value_and_get_action
                return update_fn(
                    repr_state,
                    hh,
                    self.V[:, :self.M],
                    self.R_hat[:self.M, :],
                    self.P_hat[:self.M, :, :self.M],
                    self.B_sa[:self.M, :],
                    self.gamma,
                    self.v_max,
                    )
            else:
                return self.env.action_space.sample()

    def _run_episode(self):
        # interact for H steps
        episode_rewards = 0
        state = self.env.reset()
        for hh in range(self.horizon):
            action = self._get_action(state, hh)
            next_state, reward, done, _ = self.env.step(action)
            self._update(state, action, next_state, reward)
            state = next_state
            episode_rewards += reward

            if done:
                break

        # run backward induction
        if not self.real_time_dp:
            backward_induction_in_place(
                                self.Q[:, :self.M, :], self.V[:, :self.M],
                                self.R_hat[:self.M, :]+self.B_sa[:self.M, :],
                                self.P_hat[:self.M, :, :self.M],
                                self.horizon, self.gamma, self.v_max[0])

        ep = self.episode
        self._rewards[ep] = episode_rewards
        self._cumul_rewards[ep] = episode_rewards \
            + self._cumul_rewards[max(0, ep - 1)]

        self.episode += 1
        #
        if self.writer is not None:
            avg_reward = self._cumul_rewards[ep]/max(1, ep)

            self.writer.add_scalar("episode", self.episode, None)
            self.writer.add_scalar("ep reward", episode_rewards)
            self.writer.add_scalar("avg reward", avg_reward)
            self.writer.add_scalar("representative states", self.M)

        # return sum of rewards collected in the episode
        return episode_rewards

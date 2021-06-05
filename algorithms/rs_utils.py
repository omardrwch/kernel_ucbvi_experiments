import numpy as np
from rlberry.utils.jit_setup import numba_jit
from algorithms.metrics_and_kernels import metric_lp, kernel_func

@numba_jit
def update_model(repr_state,
                 action,
                 repr_next_state,
                 reward,
                 n_representatives,
                 repr_states,
                 lp_metric,
                 scaling,
                 bandwidth,
                 bonus_scale_factor,
                 beta,
                 v_max,
                 bonus_type,
                 kernel_type,
                 N_sa,
                 B_sa,
                 P_hat,
                 R_hat,
                 use_twinroom_symmetry):
    """
    Model update function, lots of arguments so we can use JIT.
    """
    # aux var for transition update
    dirac_next_s = np.zeros(n_representatives)
    dirac_next_s[repr_next_state] = 1.0

    for u_repr_state in range(n_representatives):
        # compute weight
        dist = metric_lp(repr_states[repr_state, :],
                         repr_states[u_repr_state, :],
                         lp_metric,
                         scaling,
                         use_twinroom_symmetry)
        weight = kernel_func(dist/bandwidth, kernel_type=kernel_type)

        # aux variables
        prev_N_sa = beta + N_sa[u_repr_state, action]  # regularization beta
        current_N_sa = prev_N_sa + weight

        # update weights
        N_sa[u_repr_state, action] += weight

        # update transitions
        P_hat[u_repr_state, action, :n_representatives] =\
            dirac_next_s*weight / current_N_sa + \
            (prev_N_sa/current_N_sa) * \
            P_hat[u_repr_state, action, :n_representatives]

        # update rewards
        R_hat[u_repr_state, action] = weight*reward/current_N_sa + \
            (prev_N_sa/current_N_sa)*R_hat[u_repr_state, action]

        # update bonus
        B_sa[u_repr_state, action] = compute_bonus(N_sa[u_repr_state, action],
                                                   beta, bonus_scale_factor,
                                                   v_max, bonus_type)


@numba_jit
def compute_bonus(sum_weights, beta, bonus_scale_factor, v_max, bonus_type):
    n = beta + sum_weights
    if bonus_type == "simplified_bernstein":
        return bonus_scale_factor * np.sqrt(1.0/n) + (1+beta)*(v_max)/n
    else:
        raise NotImplementedError("Error: unknown bonus type.")


# @numba_jit
def update_value_and_get_action(state,
                                hh,
                                V,
                                R_hat,
                                P_hat,
                                B_sa,
                                gamma,
                                v_max):
    """
    state : int
    hh : int
    V : np.ndarray
        shape (H, S)
    R_hat : np.ndarray
        shape (S, A)
    P_hat : np.ndarray
        shape (S, A, S)
    B_sa : np.ndarray
        shape (S, A)
    gamma : double
    v_max : np.ndarray
        shape (H,)
    """
    H = V.shape[0]
    S, A = R_hat.shape[-2:]
    best_action = 0
    max_val = 0
    previous_value = V[hh, state]

    for aa in range(A):
        q_aa = R_hat[state, aa] + B_sa[state, aa]

        if hh < H-1:
            for sn in range(S):
                q_aa += gamma*P_hat[state, aa, sn]*V[hh+1, sn]

        if aa == 0 or q_aa > max_val:
            max_val = q_aa
            best_action = aa

    V[hh, state] = max_val
    V[hh, state] = min(v_max[hh], V[hh, state])
    V[hh, state] = min(previous_value, V[hh, state])

    return best_action


@numba_jit
def map_to_representative(state,
                          lp_metric,
                          representative_states,
                          n_representatives,
                          min_dist,
                          scaling,
                          accept_new_repr,
                          use_twinroom_symmetry):
    """Map state to representative state. """
    dist_to_closest = np.inf
    argmin = -1
    for ii in range(n_representatives):
        dist = metric_lp(state, representative_states[ii, :],
                         lp_metric,
                         scaling,
                         use_twinroom_symmetry)
        if dist < dist_to_closest:
            dist_to_closest = dist
            argmin = ii

    max_representatives = representative_states.shape[0]
    if (dist_to_closest > min_dist) \
        and (n_representatives < max_representatives) \
            and accept_new_repr:
        new_index = n_representatives
        representative_states[new_index, :] = state
        return new_index
    return argmin

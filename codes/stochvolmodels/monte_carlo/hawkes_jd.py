
# built in
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import matplotlib.pyplot as plt
import pandas as pd

# Parallel
# import ray
# ray.init()


# stochvolmodels pricers
from stochvolmodels.pricers.model_pricer import ModelParams


class MC_simulator:
    def __init__(self, params, T0, T_max, n_steps, n_paths, with_compensators, seed0):
        self.params = params
        self.with_compensators = with_compensators
        self.T0 = T0
        self.T_max = T_max
        self.n_steps = n_steps
        self.n_paths = n_paths
        self.branching_ratio_p = ((params.mean_exp_J_p-1)*params.beta11 + (params.mean_exp_J_m-1)*params.beta12) / params.kappa_p
        self.branching_ratio_m = ((params.mean_exp_J_p-1)*params.beta21 + (params.mean_exp_J_m-1)*params.beta22) / params.kappa_m
        self.seed0 = seed0
        # print('Positive jumps branching ratio:', self.branching_ratio_p,
        #       'Negative jumps branching ratio:', self.branching_ratio_m)
        

    def genPath_seed0(self): # for testing
        return genPaths_fn(self.T0, self.T_max, self.n_steps, self.n_paths, self.params, self.with_compensators, self.seed0)

    # def genPaths(self):
    #     seed_arr = np.array(range(self.seed0, self.seed0+self.n_paths))
    #     results = []
    #     for seed in seed_arr:
    #         results.append(genPaths_parallel.remote(self.T0, self.T_max, self.n_steps, self.n_paths, self.params, self.with_compensators, seed))
    #     return results


def genPaths_fn(T0, T_max, n_steps, n_paths, params, with_compensators, seed):
    dt = T_max/n_steps
    T_k_arr, lambda_p_k_left_arr, lambda_p_k_right_arr, lambda_m_k_left_arr, lambda_m_k_right_arr, J_arr = exact_simulation(T0, T_max, params, seed)
    info, jumps_info = simulate_log_returns(T_k_arr, lambda_p_k_left_arr, lambda_p_k_right_arr, lambda_m_k_left_arr, lambda_m_k_right_arr, J_arr, T_max, n_steps, dt, with_compensators, params, seed)
    return info, jumps_info

# @ray.remote(num_returns=2)
# def genPaths_parallel(T0, T_max, n_steps, n_paths, params, with_compensators, seed):
#     return genPaths_fn(T0, T_max, n_steps, n_paths, params, with_compensators, seed)


def D_k_plus_1(lambda_T_k_right, theta, kappa):
    U1 = np.random.uniform()
    return 1 + kappa*np.log(U1)/(lambda_T_k_right - theta)


def S_k_plus_1(lambda_T_k_right, theta, kappa):
    _D_k_plus_1 = D_k_plus_1(lambda_T_k_right, theta, kappa)

    U2 = np.random.uniform()
    S1 = -1/kappa*np.log(_D_k_plus_1)
    S2 = -1/theta*np.log(U2)

    if _D_k_plus_1 > 0:
        return np.min([S1, S2])
    else:
        return S2


def exact_simulation(T0, T_max, params, seed):
    
    np.random.seed(seed)

    T_k_arr = [T0]
    lambda_p_k_right_arr = [params.lambda_p]
    lambda_p_k_left_arr  = [params.lambda_p]

    lambda_m_k_right_arr = [params.lambda_m]
    lambda_m_k_left_arr  = [params.lambda_m]
    J_arr = [0]

    while True:
        _T_k = T_k_arr[-1]
        _lambda_p_T_k_left = lambda_p_k_left_arr[-1]
        _lambda_m_T_k_left = lambda_m_k_left_arr[-1]
        _lambda_p_T_k_right = lambda_p_k_right_arr[-1]
        _lambda_m_T_k_right = lambda_m_k_right_arr[-1]

        _T_k = T_k_arr[-1]
        _S_k_plus_1_p = S_k_plus_1(_lambda_p_T_k_right, params.theta_p, params.kappa_p)
        _S_k_plus_1_m = S_k_plus_1(_lambda_m_T_k_right, params.theta_m, params.kappa_m)

        if _S_k_plus_1_p < _S_k_plus_1_m: # positive jump
            _W_k_plus_1 = _S_k_plus_1_p
            _T_k_plus_1 = _T_k + _W_k_plus_1
            if _T_k_plus_1 > T_max:
                break
            J = np.random.exponential(params.eta_p) + params.nu_p
            _lambda_p_T_k_plus_1_left  = (_lambda_p_T_k_right - params.theta_p)*np.exp(-params.kappa_p*(_T_k_plus_1 - _T_k)) + params.theta_p
            _lambda_p_T_k_plus_1_right = _lambda_p_T_k_plus_1_left + params.beta11*J

            _lambda_m_T_k_plus_1_left  = (_lambda_m_T_k_right - params.theta_m)*np.exp(-params.kappa_m*(_T_k_plus_1 - _T_k)) + params.theta_m
            _lambda_m_T_k_plus_1_right = _lambda_m_T_k_plus_1_left + params.beta21*J

        else:
            _W_k_plus_1 = _S_k_plus_1_m # negative jump
            _T_k_plus_1 = _T_k + _W_k_plus_1
            if _T_k_plus_1 > T_max:
                break
            J = -np.random.exponential(params.eta_m) + params.nu_m
            _lambda_p_T_k_plus_1_left = (_lambda_p_T_k_right - params.theta_p)*np.exp(-params.kappa_p*(_T_k_plus_1 - _T_k)) + params.theta_p
            _lambda_p_T_k_plus_1_right = _lambda_p_T_k_plus_1_left + params.beta12*J

            _lambda_m_T_k_plus_1_left = (_lambda_m_T_k_right - params.theta_m)*np.exp(-params.kappa_m*(_T_k_plus_1 - _T_k)) + params.theta_m
            _lambda_m_T_k_plus_1_right = _lambda_m_T_k_plus_1_left + params.beta22*J

        # record lambda_p
        lambda_p_k_left_arr.append(_lambda_p_T_k_plus_1_left)
        lambda_p_k_right_arr.append(_lambda_p_T_k_plus_1_right)

        # record lambda_m
        lambda_m_k_left_arr.append(_lambda_m_T_k_plus_1_left)
        lambda_m_k_right_arr.append(_lambda_m_T_k_plus_1_right)

        # record jump size
        J_arr.append(J)

        # record jump time
        T_k_arr.append(_T_k_plus_1)
        # print(_T_k_plus_1)

    return T_k_arr, lambda_p_k_left_arr, lambda_p_k_right_arr, lambda_m_k_left_arr, lambda_m_k_right_arr, J_arr


def simulate_log_returns(T_k_arr, lambda_p_k_left_arr, lambda_p_k_right_arr, lambda_m_k_left_arr, lambda_m_k_right_arr, J_arr,
                         T_max, n_steps, dt, with_compensators, params, seed):
   
    np.random.seed(seed)
    
    # Pad lambdas on non jump times
    jumps_info = pd.DataFrame([T_k_arr,
                            lambda_p_k_left_arr, lambda_p_k_right_arr,
                            lambda_m_k_left_arr, lambda_m_k_right_arr, J_arr]).T
    jumps_info.columns = ['t', 
                        'lambda_p_left', 'lambda_p_right', 
                        'lambda_m_left', 'lambda_m_right', 'jump_size']

    T_arr = np.linspace(0,T_max+dt,n_steps+1)[1:]

    non_jumps_info = pd.DataFrame([T_arr, np.zeros(len(T_arr))]).T
    non_jumps_info.columns = ['t', 'jump_size']

    # Concatenate simulation times and jump times
    info = pd.concat([non_jumps_info, jumps_info])
    info = info.sort_values('t').reset_index(drop=True)

    for i in range(1, len(info)):
        _T = info.t.iloc[i-1]
        _lambda_p_T_left = info.lambda_p_left.iloc[i-1]
        _lambda_m_T_left = info.lambda_m_left.iloc[i-1]
        _lambda_p_T_right = info.lambda_p_right.iloc[i-1]
        _lambda_m_T_right = info.lambda_m_right.iloc[i-1]

        if np.isnan(_lambda_p_T_right):
            _lambda_p_T_right = _lambda_p_T_left
            info.lambda_p_right.iloc[i-1] = _lambda_p_T_left

        if np.isnan(_lambda_m_T_right):
            _lambda_m_T_right = _lambda_m_T_left
            info.lambda_m_right.iloc[i-1] = _lambda_m_T_left

        T = info.t.iloc[i]
        info.lambda_p_left.iloc[i] = (_lambda_p_T_right - params.theta_p)*np.exp(-params.kappa_p*(T - _T)) + params.theta_p
        info.lambda_m_left.iloc[i] = (_lambda_m_T_right - params.theta_m)*np.exp(-params.kappa_m*(T - _T)) + params.theta_m


    # Get compensators
    if with_compensators:
        comp_p = (params.mean_exp_J_p-1)*( params.theta_p*(info.t-info.t.shift(1)) + (info.lambda_p_right-params.theta_p)/params.kappa_p*( 1 - np.exp(-params.kappa_p*(info.t-info.t.shift(1)) )  )  )
        comp_m = (params.mean_exp_J_m-1)*( params.theta_m*(info.t-info.t.shift(1)) + (info.lambda_m_right-params.theta_m)/params.kappa_m*( 1 - np.exp(-params.kappa_m*(info.t-info.t.shift(1)) )  )  )
    else:
        comp_p = comp_m = 0

    info.loc[:,'comp_p'] = comp_p
    info.loc[:,'comp_m'] = comp_m

    info.loc[:,'brownian']  = np.random.normal(size=len(info))*params.sigma*np.sqrt(info.t-info.t.shift(1))
    info.loc[:,'drift'] = (params.mu-params.sigma**2/2)*(info.t-info.t.shift(1))
    info.loc[:,'dX'] = info.drift - info.comp_p - info.comp_m + info.brownian + info.jump_size 
    info.loc[:, 'X'] = info.dX.cumsum()

    jumps_info = info.loc[info.jump_size != 0,:].reset_index(drop=True)
    info       = info.loc[info.jump_size == 0,:].reset_index(drop=True)

    info = info.iloc[:-1,:]

    return info, jumps_info
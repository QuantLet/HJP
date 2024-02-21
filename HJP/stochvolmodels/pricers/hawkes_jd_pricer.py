
# built in
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, fsolve, root
from numba.typed import List
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass, asdict
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
from enum import Enum
import copy

import tensorflow.experimental.numpy as tnp
import tensorflow as tf

# stochvolmodels pricers
from stochvolmodels.pricers.core import mgf_pricer as mgfp
from stochvolmodels.pricers.core.config import VariableType
from stochvolmodels.pricers.core.mc_payoffs import compute_mc_vars_payoff
from stochvolmodels.pricers.model_pricer import ModelPricer, ModelParams
from stochvolmodels.utils.funcs import to_flat_np_array, set_time_grid, timer, set_seed, transform_to_tfcomplex128, transform_from_tfcomplex128_to_np

# data
from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.data.test_option_chain import get_btc_test_chain_data

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MAX_PHI = 500

@dataclass
class HawkesJDParams(ModelParams):
    """
    parameters of 2-factor Hawkes Jump Diffusion
    annualized params, close for BTC on daily frequency
    """
    mu:    float = 0.0
    sigma: float = 0.45

    # jumps sizes
    nu_p  : float = 0.03
    eta_p : float = 0.06
    nu_m  : float = -0.03
    eta_m : float = 0.06

    # positive jumps intensity
    lambda_p: float = 20
    theta_p : float = 20
    kappa_p : float = 10
    beta11  : float = 1
    beta12  : float = -1

    # minus jumps intensity
    lambda_m: float = 20
    theta_m : float = 20
    kappa_m : float = 10
    beta21  : float = 1
    beta22  : float = -1

    # helpers
    lambda_p_0: float = 20
    lambda_m_0: float = 20

    def __post_init__(self):
        self.compensator_p = np.exp(self.nu_p)/(1.0-self.eta_p) - 1.0
        self.compensator_m = np.exp(self.nu_m)/(1.0+self.eta_m) - 1.0
        self.mean_exp_J_p  = np.exp(self.nu_p)/(1-self.eta_p)
        self.mean_exp_J_m  = np.exp(self.nu_m)/(1+self.eta_m)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HawkesJDPricer(ModelPricer):

    def __init__(self, M=2**12, x_max=6, n_steps_per_ttm=200):
        # Compute and store coefficients for inverse discrete Fourier transform in advance
        self.n_steps_per_ttm = n_steps_per_ttm
        self.Delta_x = 2*x_max/(M-1)      # Delta x    : step size of x grid
        self.Delta_o = 2*tnp.pi/M/self.Delta_x # Delta omega: step size of omega grid

        m_grid = tnp.linspace(1, M, M, dtype=tnp.complex128)
        k_grid = tnp.linspace(1, M, M, dtype=tnp.complex128)
        m_grid_mesh, k_grid_mesh = tnp.meshgrid(m_grid, k_grid, indexing='ij')
        self.x_grid = -M/2*self.Delta_x + (k_grid-1)*self.Delta_x

        omega_grid = (m_grid-1)*self.Delta_o
        self.omega_grid = tnp.array([1j])*omega_grid

        part1 = tf.constant(0.5, dtype=tf.complex128)*tf.cast(m_grid_mesh==1, dtype=tf.complex128)
        part2 = tf.constant(1,   dtype=tf.complex128)*tf.cast(m_grid_mesh!=1, dtype=tf.complex128)
        part3 = tnp.exp(-1j*2*tnp.pi/M*(m_grid_mesh-1)*(k_grid_mesh-1))*(-1)**(m_grid_mesh-1)
        self.invFourier_coefs = (part1+part2)*part3*2/M/self.Delta_x
        self.M = M
        self.x_max = x_max

        # t_grid = tnp.linspace(0, 1, 2000, dtype=tnp.complex128)

    # @timer
    def price_chain(self,
                    option_chain: OptionChain,
                    params: HawkesJDParams,
                    is_spot_measure: bool = True,
                    **kwargs
                    ):
        """
        implementation of generic method price_chain using log sv wrapper
        """
        model_prices = hawkesjd_chain_pricer(model_params=params,
                                            ttms=option_chain.ttms,
                                            forwards=option_chain.forwards,
                                            discfactors=option_chain.discfactors,
                                            strikes_ttms=option_chain.strikes_ttms,
                                            optiontypes_ttms=option_chain.optiontypes_ttms,
                                            is_spot_measure=is_spot_measure,
                                            n_steps_per_ttm=self.n_steps_per_ttm,
                                            omega_grid=self.omega_grid, 
                                            x_grid=self.x_grid, 
                                            Delta_x=self.Delta_x, 
                                            invFourier_coefs=self.invFourier_coefs,
                                            **kwargs)
        return model_prices


    @timer
    def model_mc_price_chain(self,
                             option_chain: OptionChain,
                             params: HawkesJDParams,
                             nb_path: int = 100000,
                             **kwargs
                             ) -> (List[np.ndarray], List[np.ndarray]):
        return hawkesjd_mc_chain_pricer(ttms=option_chain.ttms,
                                        forwards=option_chain.forwards,
                                        discfactors=option_chain.discfactors,
                                        strikes_ttms=option_chain.strikes_ttms,
                                        optiontypes_ttms=option_chain.optiontypes_ttms,
                                        nb_path=nb_path,
                                        **params.to_dict())

    # need to overwrite the base
    def compute_chain_prices_with_vols(self,
                                       option_chain: OptionChain,
                                       params: HawkesJDParams,
                                    #    model_prices,
                                       **kwargs
                                       ):
        """
        price chain and compute model vols
        """
        model_prices = self.price_chain(option_chain=option_chain, params=params, **kwargs)
 
        model_ivols = option_chain.compute_model_ivols_from_chain_data(model_prices=model_prices, forwards=option_chain.forwards)
        return model_prices, model_ivols


    @timer
    def calibrate_model_params_to_chain(self,
                                        option_chain: OptionChain,
                                        params0: HawkesJDParams,
                                        is_vega_weighted: bool = True,
                                        is_unit_ttm_vega: bool = False,
                                        method = 'SLSQP',
                                        bounds = ((0, 0), (0.01, 5),
                                                (0.001, 5), (0.001, None), (None, -0.001), (0.001, None),
                                                (.01, None), (.01, None), (.01, None), (.01, None), (None, -.01),
                                                (.01, None), (.01, None), (.01, None), (.01, None), (None, -.01)),
                                        verbose = False,
                                        **kwargs
                                        ) -> HawkesJDParams:
        """
        implementation of model calibration interface
        bounds: mu, sigma,
                nu_p, eta_p, nu_m, eta_m, 
                lambda_p, theta_p, kappa_p, beta11, beta12,
                lambda_m, theta_m, kappa_m, beta21, beta22, 
        """
        x, y = option_chain.get_chain_data_as_xy()
        market_vols = to_flat_np_array(y)  # market mid quotes
        if is_vega_weighted:
            vegas_ttms = option_chain.get_chain_vegas(is_unit_ttm_vega=is_unit_ttm_vega)
            vegas_ttms = [vegas_ttm/sum(vegas_ttm) for vegas_ttm in vegas_ttms]
            weights = to_flat_np_array(vegas_ttms)
        else:
            weights = np.ones_like(market_vols)

        p0 = np.array([params0.mu, params0.sigma,
                       params0.nu_p, params0.eta_p, params0.nu_m, params0.eta_m, 
                       params0.lambda_p, params0.theta_p, params0.kappa_p, params0.beta11, params0.beta12,
                       params0.lambda_m, params0.theta_m, params0.kappa_m, params0.beta21, params0.beta22])

        def objective(pars: np.ndarray, args: np.ndarray) -> float:
            params = unpack_pars(pars=pars)
            model_vols = self.compute_model_ivols_for_chain(option_chain=option_chain, params=params)
            resid = np.nansum(weights * np.abs(to_flat_np_array(model_vols) - market_vols)/market_vols)
            if verbose:
                print(pars, resid)
            return resid

        # def jump_cond(pars: np.ndarray) -> float:
        #     params = unpack_pars(pars=pars)
        #     return params.jump1_cond + params.jump2_cond

        # constraints = ({'type': 'ineq', 'fun': jump_cond})
        constraints = None
        options = {'disp': True, 'ftol': 1e-8}

        if constraints is not None:
            res = minimize(objective, p0, args=None, method=method, constraints=constraints, bounds=bounds, options=options)
        else:
            res = minimize(objective, p0, args=None, method=method, bounds=bounds, options=options)

        fit_params = unpack_pars(pars=res.x)
        return fit_params


    @timer
    def calibrate_measure_change_params_to_chain(self,
                                        option_chain: OptionChain,
                                        params0: HawkesJDParams,
                                        is_vega_weighted: bool = True,
                                        is_unit_ttm_vega: bool = False,
                                        method = 'powell',
                                        options = {'disp': True, 'ftol': 1e-8}, 
                                        errors_weights = (1,0,0,0,0), 
                                        bounds = ((0.01, 3.00),(-20,20), (-20, 20)),
                                        p0 = (0.5,0,0),
                                        verbose=False, 
                                        **kwargs
                                        ):
        """
        implementation of model calibration interface
        p0: sigma, chi_p, chi_m
        """
        x, y = option_chain.get_chain_data_as_xy()
        market_vols = to_flat_np_array(y)  # market mid quotes
        if is_vega_weighted:
            vegas_ttms = option_chain.get_chain_vegas(is_unit_ttm_vega=is_unit_ttm_vega)
            vegas_ttms = [vegas_ttm/sum(vegas_ttm) for vegas_ttm in vegas_ttms]
            weights = to_flat_np_array(vegas_ttms)
        else:
            weights = np.ones_like(market_vols)

        params0 = transform_to_tfcomplex128(params0)
        
        eta_p0 = params0.eta_p.numpy().real
        eta_m0 = params0.eta_m.numpy().real

        w0, w1, w2, w3, w4 = errors_weights 
        
        def objective(pars: np.ndarray) -> float:
            _, chi_p, chi_m = pars
            if chi_p + 1/eta_p0 < 0:
                return 5000
            if chi_m - 1/eta_m0 > 0:
                return 5000
            _, params = unpack_and_transform_pars_for_measure_change(measure_transform_params=pars, params=params0)
            model_vols = self.compute_model_ivols_for_chain(option_chain=option_chain, params=params)
            resid   =  w0*np.nanmean(weights * np.abs(to_flat_np_array(model_vols) - market_vols))
            penalty =  w1*np.abs(chi_p) + w2*np.abs(chi_m) + w3*np.abs(chi_p+chi_m) + w4*np.abs(chi_p-chi_m) 
            
            if verbose:
                print(pars, resid, penalty)
            return resid + penalty

        # def jump_cond(pars: np.ndarray) -> float:
        #     params = unpack_and_transform_pars(pars=pars)
        #     return params.jump1_cond + params.jump2_cond

        # constraints = ({'type': 'ineq', 'fun': jump_cond})
        constraints = None

        if constraints is not None:
            res = minimize(objective, p0, method=method, constraints=constraints, bounds=bounds, options=options)
        else:
            res = minimize(objective, p0, method=method, bounds=bounds, options=options)

        fit_params = unpack_and_transform_pars_for_measure_change(measure_transform_params=res.x, params=params0)[1]
        return res.x, fit_params


    @timer
    def calibrate_measure_change_params_and_lambda_ts_to_chain(self,
                                        option_chain: OptionChain,
                                        params0: HawkesJDParams,
                                        is_vega_weighted: bool = True,
                                        is_unit_ttm_vega: bool = False,
                                        bounds = ((0.01, 3.00),(-20,20), (-20, 20), (0, None), (0, None)),
                                        p0 = (0.5,0,0, 100, 100),
                                        **kwargs
                                        ):
        """
        implementation of model calibration interface
        p0: sigma, chi_p, chi_m, lambda_p, lambda_m
        """

        x, y = option_chain.get_chain_data_as_xy()
        market_vols = to_flat_np_array(y)  # market mid quotes
        if is_vega_weighted:
            vegas_ttms = option_chain.get_chain_vegas(is_unit_ttm_vega=is_unit_ttm_vega)
            vegas_ttms = [vegas_ttm/sum(vegas_ttm) for vegas_ttm in vegas_ttms]
            weights = to_flat_np_array(vegas_ttms)
        else:
            weights = np.ones_like(market_vols)


        params0 = transform_to_tfcomplex128(params0)

        def objective(pars: np.ndarray, args: np.ndarray) -> float:

            m_pars = pars[:3]
            lambda_p = pars[3]
            lambda_m = pars[4]

            success, params = unpack_and_transform_pars_for_measure_change(measure_transform_params=m_pars, params=params0)
                
            if lambda_p < tf.math.real(params.theta_p):
                lambda_p = params.theta_p

            if lambda_m < tf.math.real(params.theta_m):
                lambda_m = params.theta_m

            params.lambda_p = tf.cast(lambda_p, tf.complex128)
            params.lambda_m = tf.cast(lambda_m, tf.complex128)


            if success==False:
                return 5000
            model_vols = self.compute_model_ivols_for_chain(option_chain=option_chain, params=params)
            resid = np.nansum(weights * np.abs(to_flat_np_array(model_vols) - market_vols))
            return resid

        # def jump_cond(pars: np.ndarray) -> float:
        #     params = unpack_and_transform_pars(pars=pars)
        #     return params.jump1_cond + params.jump2_cond

        # constraints = ({'type': 'ineq', 'fun': jump_cond})
        constraints = None

        options = {'disp': True, 'ftol': 1e-8}

        if constraints is not None:
            res = minimize(objective, p0, args=None, method='SLSQP', constraints=constraints, bounds=bounds, options=options)
        else:
            res = minimize(objective, p0, args=None, method='SLSQP', bounds=bounds, options=options)

        m_pars = res.x[:3]
        fit_params = unpack_and_transform_pars_for_measure_change(measure_transform_params=m_pars, params=params0)[1]
        fit_params.lambda_p = tf.cast(res.x[3], tf.complex128)
        fit_params.lambda_m = tf.cast(res.x[4], tf.complex128)
        return res.x, fit_params



def set_vol_scaler(sigma0: float, ttm: float) -> float:
    return np.clip(sigma0, 0.2, 0.5) * np.sqrt(np.minimum(ttm, 1.0 / 12.0))  # lower bound is two w


def payoff(forward, x_grid, strikes_ttm, optiontypes_ttm):
    optiontypes_ttm = pd.Series(optiontypes_ttm)

    Call_payoffs = np.clip( (forward * np.exp(x_grid)[:,tnp.newaxis]) - strikes_ttm[tnp.newaxis,:] , 0, None)
    Put_payoffs = np.clip( strikes_ttm[tnp.newaxis,:] - (forward * np.exp(x_grid)[:,tnp.newaxis])  , 0, None)

    payoffs_ttm = Call_payoffs*np.array(optiontypes_ttm=='C') + Put_payoffs*np.array(optiontypes_ttm=='P')
    return payoffs_ttm


def hawkesjd_chain_pricer(model_params: HawkesJDParams,
                          ttms: np.ndarray,
                          forwards: np.ndarray,
                          discfactors: np.ndarray,
                          strikes_ttms: List[np.ndarray],
                          optiontypes_ttms: List[np.ndarray],
                          x_grid: np.ndarray, 
                          Delta_x: float,
                          omega_grid: np.ndarray,
                          invFourier_coefs: np.ndarray, 
                          n_steps_per_ttm: int = 200,
                          is_stiff_solver: bool = False,
                          is_spot_measure: bool = True,
                          variable_type: VariableType = VariableType.LOG_RETURN,
                          vol_scaler: float = None
                          ):

    # Compute pdf_grid
    pdf_grid_ttms = []
    A=C=D=None
    ttm0 = 0

    for ttm in ttms:
        A,C,D,mgf = compute_hawkes_mgf_grid(ttm-ttm0, omega_grid, model_params, A, C, D, n_steps_per_ttm)
        pdf_grid = compute_pdf_grid(mgf, invFourier_coefs)
        pdf_grid_ttms.append(pdf_grid)
        ttm0 = ttm

    # outputs as list containing numpy arrays
    model_prices_ttms = []
    for ttm, forward, strikes_ttm, optiontypes_ttm, discfactor, pdf_grid in zip(ttms, forwards, strikes_ttms,
                                                                        optiontypes_ttms, discfactors, pdf_grid_ttms):

        model_prices_ttm = np.exp(-0.001*ttm)*tf.reduce_sum(payoff(forward, x_grid, strikes_ttm, optiontypes_ttm) * pdf_grid[:, tnp.newaxis], axis=0) * Delta_x
        model_prices_ttms.append(model_prices_ttm.numpy())

    return model_prices_ttms


@tf.function
def L_p(x, nu_p, eta_p):
    return tnp.exp(-nu_p*x)/(1+eta_p*x)

@tf.function
def L_m(x, nu_m, eta_m):
    return tnp.exp(-nu_m*x)/(1-eta_m*x)

@tf.function
def partial_t_A(omega, mu, sigma, kappa_p, kappa_m, theta_p, theta_m, arg_C, arg_D):
    part1 = -(mu-sigma**2/2)*omega
    part2 = -sigma**2/2*omega**2
    part3 = -kappa_p*theta_p*arg_C
    part4 = -kappa_m*theta_m*arg_D
    return part1+part2+part3+part4

@tf.function
def partial_t_C(omega, beta11, beta21, compensator_p, kappa_p, nu_p, eta_p, arg_C, arg_D):
    part1 = -L_p(-omega-arg_C*beta11-arg_D*beta21, nu_p, eta_p)+1
    part2 = compensator_p*omega + kappa_p*arg_C
    return part1+part2

@tf.function
def partial_t_D(omega, beta12, beta22, compensator_m, kappa_m, nu_m, eta_m, arg_C, arg_D):
    part1 = -L_m(-omega-arg_C*beta12 -arg_D*beta22, nu_m, eta_m)+1
    part2 = compensator_m*omega + kappa_m*arg_D
    return part1+part2


@tf.function
def compute_mgf_X_ACD_grid(omega_grid, t_grid, 
                            mu, sigma,
                            nu_p, eta_p, nu_m, eta_m, 
                            lambda_p, theta_p, kappa_p, beta11, beta12,
                            lambda_m, theta_m, kappa_m, beta21, beta22, 
                            compensator_p, compensator_m,
                            A0=None, C0=None, D0=None):

    A_grid = tf.TensorArray(tf.complex128, size=len(t_grid))
    C_grid = tf.TensorArray(tf.complex128, size=len(t_grid))
    D_grid = tf.TensorArray(tf.complex128, size=len(t_grid))

    if (A0==None) or (C0==None) or (D0==None):
        A_grid = A_grid.write(0, tf.zeros(shape=omega_grid.shape, dtype=tf.complex128))
        C_grid = C_grid.write(0, tf.zeros(shape=omega_grid.shape, dtype=tf.complex128))
        D_grid = D_grid.write(0, tf.zeros(shape=omega_grid.shape, dtype=tf.complex128))
    else:
        A_grid = A_grid.write(0, A0)
        C_grid = C_grid.write(0, C0)
        D_grid = D_grid.write(0, D0)

    dt = tf.cast(t_grid[1]-t_grid[0], tf.complex128)

    A = A_grid.read(0)
    C = C_grid.read(0)
    D = D_grid.read(0)

    for i in tf.range(1, len(t_grid)):

        A = A - partial_t_A(omega_grid, mu, sigma, kappa_p, kappa_m, theta_p, theta_m, C, D)*dt
        C = C - partial_t_C(omega_grid, beta11, beta21, compensator_p, kappa_p, nu_p, eta_p, C, D)*dt
        D = D - partial_t_D(omega_grid, beta12, beta22, compensator_m, kappa_m, nu_m, eta_m, C, D)*dt
        
        A_grid = A_grid.write(i, A)
        C_grid = C_grid.write(i, C)
        D_grid = D_grid.write(i, D)

    A_grid = A_grid.stack()
    C_grid = C_grid.stack()
    D_grid = D_grid.stack()

    return A_grid, C_grid, D_grid


def compute_hawkes_mgf_grid(ttm: float,
                              omega_grid: tnp.array,
                              model_params: HawkesJDParams,
                              A0=None, C0=None, D0=None, n_steps_per_ttm=200
                              ):

    mu    = model_params.mu
    sigma = model_params.sigma

    nu_p  = model_params.nu_p
    eta_p = model_params.eta_p
    nu_m  = model_params.nu_m
    eta_m = model_params.eta_m

    lambda_p = model_params.lambda_p
    kappa_p = model_params.kappa_p
    theta_p = model_params.theta_p
    kappa_m = model_params.kappa_m
    theta_m = model_params.theta_m

    lambda_m = model_params.lambda_m
    beta11 = model_params.beta11
    beta21 = model_params.beta21
    beta12 = model_params.beta12
    beta22 = model_params.beta22

    compensator_p = model_params.compensator_p
    compensator_m = model_params.compensator_m

    t_grid = tnp.linspace(0, ttm, n_steps_per_ttm, dtype=tnp.complex128)
    
    A_grid, C_grid, D_grid = compute_mgf_X_ACD_grid(omega_grid, t_grid, 
                                                    mu, sigma,
                                                    nu_p, eta_p, nu_m, eta_m, 
                                                    lambda_p, theta_p, kappa_p, beta11, beta12,
                                                    lambda_m, theta_m, kappa_m, beta21, beta22, 
                                                    compensator_p, compensator_m, 
                                                    A0, C0, D0)

    return  A_grid[-1], C_grid[-1], D_grid[-1], tnp.exp(A_grid+C_grid*model_params.lambda_p+D_grid*model_params.lambda_m)[-1,:]
    
@tf.function
def compute_pdf_grid(mgf_grid, invFourier_coefs):
    pdf_grid = mgf_grid[:,tnp.newaxis]*invFourier_coefs
    pdf_grid = tnp.sum(pdf_grid, axis=0)
    pdf_grid = tnp.real(pdf_grid)
    return pdf_grid


def unpack_pars(pars: np.ndarray) -> HawkesJDParams:
    mu, sigma, nu_p, eta_p, nu_m, eta_m, lambda_p, theta_p, kappa_p, beta11, beta12, lambda_m, theta_m, kappa_m, beta21, beta22 = pars
    params = HawkesJDParams(mu=mu, sigma=sigma,
                                    nu_p=nu_p, eta_p=eta_p, nu_m=nu_m, eta_m=eta_m, 
                                    lambda_p=lambda_p, theta_p=theta_p, kappa_p=kappa_p, beta11=beta11, beta12=beta12,
                                    lambda_m=lambda_m, theta_m=theta_m, kappa_m=kappa_m, beta21=beta21, beta22=beta22)
    params = transform_to_tfcomplex128(params)
    return params


def unpack_and_transform_pars_for_measure_change(measure_transform_params, params):
    sigma, chi_p, chi_m = measure_transform_params
    params = transform_from_tfcomplex128_to_np(copy.copy(params))
    
    def L_p(x):
        return np.exp(-params.nu_p*x)/(1+params.eta_p*x)

    def L_m(x):
        return np.exp(-params.nu_m*x)/(1-params.eta_m*x)
    
    _L_p = L_p(-chi_p)
    _L_m = L_m(-chi_m)

    params.sigma = sigma

    params.theta_p  *= _L_p
    params.lambda_p *= _L_p
    params.beta11   *= _L_p
    params.beta12   *= _L_p

    params.theta_m  *= _L_m
    params.lambda_m *= _L_m
    params.beta21   *= _L_m
    params.beta22   *= _L_m

    params.eta_p = params.eta_p/(1-params.eta_p*chi_p)
    params.eta_m = params.eta_m/(1+params.eta_m*chi_m)

    params.compensator_p = np.exp(params.nu_p)/(1.0-params.eta_p) - 1.0
    params.compensator_m = np.exp(params.nu_m)/(1.0+params.eta_m) - 1.0

    params.mean_exp_J_p = np.exp(params.nu_p)/(1.0-params.eta_p) 
    params.mean_exp_J_m = np.exp(params.nu_m)/(1.0+params.eta_m) 

    params = transform_to_tfcomplex128(params)
    
    
    return (_L_p, _L_m, chi_p, chi_m), params
    

class UnitTests(Enum):
    OPTION_PRICER = 1
    CHAIN_PRICER = 2
    SLICE_PRICER = 3
    MC_COMPARISION = 4
    CALIBRATOR = 5

if __name__ == '__main__':

    unit_test = UnitTests.MC_COMPARISION

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

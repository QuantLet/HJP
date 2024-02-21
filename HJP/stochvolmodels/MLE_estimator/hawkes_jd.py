
# built in
import numpy as np
from scipy.optimize import minimize
import pandas as pd
import scipy.stats as ss
from numpy import linalg as LA


dt = 1/365
DAYS_PER_YEAR = 365
HOURS_PER_YEAR = 365 * 24
SECONDS_PER_YEAR = 365 * 24 * 60 * 60  # minute, seconds

class Hawkes_MLE_estimator:
    def __init__(self, price, POTw1=1, POTw2=1, POTw3=1):
        # price: a pandas Series with index 'timestamp' and name 'spot'. 
        # POTw1 to 3: Weights of skewness, kurtosis, and percentage of diffusion observations in loss function function of POT
        self.price = price
        self.POTw1 = POTw1
        self.POTw2 = POTw2
        self.POTw3 = POTw3
        
        self.returns = np.log(self.price/self.price.shift(1))[1:]
    
        self.nu_m = None
        self.nu_p = None
        
    def run_Peak_over_Thresholds(self, nu_m0,  nu_p0, plug_in_nus=False, verbose=False):
        # If plug_in_nus is False:
        # 1. Calibrates nu_p, nu_m, eta_p, and eta_m to daily returns 
        # 2. Generates jumps_info as class object for Hawkes MLE
        
        # If plug_in_nus is True:
        # 1. Plugs in nu_m0 and nu_p0 and calibrates eta_p, and eta_m to daily returns 
        # 2. Generates jumps_info as class object for Hawkes MLE
        
        def objective(thresholds):
            threshold_l, threshold_h = thresholds
            if threshold_l > threshold_h:
                return 5000
            
            _id =   self.returns >= threshold_l
            _id *=  self.returns <= threshold_h


            diffusion = self.returns.loc[_id]
            
            diffusion_ratio = len(diffusion)/len(self.price) 

            s = ss.skew(diffusion)
            k = ss.kurtosis(diffusion)
            loss = self.POTw1 * np.abs(s) + self.POTw2 * np.abs(k) 
            loss += self.POTw3*diffusion_ratio

            if verbose:
                print('%.4f %.4f %.4f %.4f %.4f'%(thresholds[0], thresholds[1], s, k, diffusion_ratio))
            return loss
        
        if verbose: print('nu_m, nu_p, skew, excess kurt, diff ratio')
        
        if plug_in_nus == False:
            self.POT_result = minimize(objective, (nu_m0, nu_p0), method='powell')
            self.nu_m, self.nu_p = self.POT_result.x
        else: 
            self.nu_m, self.nu_p = nu_m0, nu_p0

        p,n,d = infer_jump_times(self.returns, self.nu_p, self.nu_m)
        self.positive_jumps_path = p
        self.negative_jumps_path = n
        self.diffusion = d
        
        self.eta_p =   np.mean(self.positive_jumps_path)-self.nu_p
        self.eta_m = -(np.mean(self.negative_jumps_path)-self.nu_m)

        # if verbose:
        print('# diffusion obs:',len(self.diffusion))
        print('# pos jumps obs:',len(self.positive_jumps_path))
        print('# neg jumps obs:',len(self.negative_jumps_path))
        print('Skewness of diffusion obs:',ss.skew(self.diffusion) )
        print('Excess kurtosis of diffusion obs:',ss.kurtosis(self.diffusion) )
    
        T = (self.price.index[-1] - self.price.index[0]).total_seconds() / SECONDS_PER_YEAR

        positive_t = (self.positive_jumps_path.index - self.price.index[0])
        positive_t = positive_t.total_seconds()/SECONDS_PER_YEAR

        negative_t = (self.negative_jumps_path.index - self.price.index[0])
        negative_t = negative_t.total_seconds()/SECONDS_PER_YEAR

        _positive_jumps_path = self.positive_jumps_path.copy()
        _negative_jumps_path = self.negative_jumps_path.copy()

        _positive_jumps_path.index = positive_t
        _negative_jumps_path.index = negative_t
        jumps_info = pd.concat([pd.Series({0:0}), _positive_jumps_path, _negative_jumps_path])
        jumps_info = jumps_info.sort_index()

        # Pad the last row if the last observation is not a jump
        if (jumps_info.index[-1] < T):
            jumps_info = pd.concat([jumps_info, pd.Series({T:0})])

        jumps_info = jumps_info.reset_index()
        jumps_info.columns = ['t', 'jump_size']
        
        jumps_info.loc[:,'jump_type'] = 0
        _id = jumps_info.jump_size > 0 
        jumps_info.loc[_id,'jump_type'] = 1
        
        _id = jumps_info.jump_size == 0
        jumps_info.jump_type.loc[_id] = 9

        jumps_info.loc[:, 'lambda_p_left'] = 0
        jumps_info.loc[:, 'lambda_m_left'] = 0
        jumps_info.loc[:, 'lambda_p_right'] = 0
        jumps_info.loc[:, 'lambda_m_right'] = 0
        
        self.jumps_info = jumps_info   
                    
    def calibrate_mu_and_sigma(self, verbose=False):
        try:
            self.diffusion
        except:
            print('Please run run_Peak_over_Thresholds first.')
            
        def f_diffusion(mu, sigma):
            return ss.norm.pdf(self.diffusion, loc=(mu-sigma**2/2)*dt, scale=sigma*np.sqrt(dt))

        def loss(pars):
            mu, sigma = pars
            l = -np.log(f_diffusion(mu, sigma)).sum()
            # print(pars, -l)
            return l

        self.diffusion_results = minimize(loss, (0,.5), bounds = ((None, None), (0.1, None)))
        self.mu, self.sigma = self.diffusion_results.x
        
        if verbose:
            print(self.diffusion_results)
            
    def calibrate_Hawkes_params(self,
                                pars0 = (.1,2,.1,2,
                                         .1,-.1,.1,-.1),
                                bounds = ((0,None),(0,None),(0,None),(0,None),
                                          (0,None),(None,0),(0,None),(None,0)),
                                penalty = 0, 
                                method = 'SLSQP', verbose=False):
        try:
            self.eta_p + self.nu_p
        except:
            print('Please run class function run_Peak_over_Thresholds first.')
        
        try:
            self.jumps_info
        except:
            print('Please run class function run_Peak_over_Thresholds first.')
        
        
        def objective(pars):
            theta_p, kappa_p, theta_m, kappa_m, beta11, beta12, beta21, beta22 = pars
            
            # Check spectral radius: https://www.math.fsu.edu/~ychen/research/multiHawkes.pdf
            # M is the kernel
            M = np.array( [[beta11, beta12],[beta21, beta22]] )
            M /= np.array( [[kappa_p, kappa_p], [kappa_m, kappa_m]] )
            mean_p =  self.eta_p + self.nu_p
            mean_m = -self.eta_m + self.nu_m
            M *= np.array( [[mean_p, mean_m], [mean_p, mean_m]] )

            try:
                self.eigenvalues, eigenvectors = LA.eig(M)
            except:
                return 5000

            # Spectral radius of kernel
            rho = np.max(np.abs(self.eigenvalues))
            
            if rho>=1:
                return 5000
            
            self.M=M
            
            l = 0
            l += likelihood(theta_p, kappa_p, theta_m, kappa_m, beta11, beta12, beta21, beta22, self.jumps_info)
            l -= penalty * np.mean(np.abs(pars - pars0)/pars0)
            
            if verbose: print(pars, l)
            
            return -l

        self.Hawkes_results = minimize(objective, pars0, method=method, bounds=bounds)

def likelihood(theta_p, kappa_p, theta_m, kappa_m, beta11, beta12, beta21, beta22, jumps_info):
    lambda_p_left_arr = [theta_p]
    lambda_m_left_arr = [theta_m]

    lambda_p_right_arr = [theta_p]
    lambda_m_right_arr = [theta_m]

    for i in range(1, len(jumps_info)):
        _T = jumps_info.t.iloc[i-1]
        _lambda_p_T_right = lambda_p_right_arr[-1]
        _lambda_m_T_right = lambda_m_right_arr[-1]

        T = jumps_info.t.iloc[i]
        jump_size = jumps_info.jump_size.iloc[i]
        lambda_p_T_left  = (_lambda_p_T_right - theta_p)*np.exp(-kappa_p*(T - _T)) + theta_p
        lambda_m_T_left  = (_lambda_m_T_right - theta_m)*np.exp(-kappa_m*(T - _T)) + theta_m

        if jump_size > 0:
            lambda_p_T_right = lambda_p_T_left + beta11*jump_size
            lambda_m_T_right = lambda_m_T_left + beta21*jump_size
        else:
            lambda_p_T_right = lambda_p_T_left + beta12*jump_size
            lambda_m_T_right = lambda_m_T_left + beta22*jump_size

        lambda_p_left_arr.append(lambda_p_T_left)
        lambda_m_left_arr.append(lambda_m_T_left)

        lambda_p_right_arr.append(lambda_p_T_right)
        lambda_m_right_arr.append(lambda_m_T_right)
    
    jumps_info.loc[:,'lambda_p_left'] = lambda_p_left_arr
    jumps_info.loc[:,'lambda_m_left'] = lambda_m_left_arr

    jumps_info.loc[:,'lambda_p_right'] = lambda_p_right_arr
    jumps_info.loc[:,'lambda_m_right'] = lambda_m_right_arr

    # Get compensators, integrals of lambdas from last jump time to current jump time
    comp_p = ( theta_p*(jumps_info.t-jumps_info.t.shift(1)) + (jumps_info.lambda_p_right.shift(1)-theta_p)/kappa_p*( 1 - np.exp(-kappa_p*(jumps_info.t-jumps_info.t.shift(1)) )  )  )
    comp_m = ( theta_m*(jumps_info.t-jumps_info.t.shift(1)) + (jumps_info.lambda_m_right.shift(1)-theta_m)/kappa_m*( 1 - np.exp(-kappa_m*(jumps_info.t-jumps_info.t.shift(1)) )  )  )

    jumps_info.loc[:,'comp_p'] = comp_p
    jumps_info.loc[:,'comp_m'] = comp_m

    # Intensity of positive jumps and negative jumps 
    part1a = jumps_info.lambda_p_left * (jumps_info.jump_type==1)
    part1b = jumps_info.lambda_m_left * (jumps_info.jump_type==0)

    part1 = (part1a + part1b)

    # Take only jump times 
    _id = (jumps_info.jump_type==1) + (jumps_info.jump_type==0) 
    part1 = np.log(part1.loc[_id]).sum()
    
    # Compensators
    part2 = ((jumps_info.comp_p + jumps_info.comp_m).iloc[1:].sum())
    
    return part1-part2


def infer_jump_times(returns, nu_p, nu_m):
    negative_jumps_id = returns <= nu_m
    positive_jumps_id = returns >= nu_p

    positive_jumps_path = returns.loc[positive_jumps_id]
    negative_jumps_path = returns.loc[negative_jumps_id]

    _id = returns >= nu_m
    _id *= returns <= nu_p
    diffusion = returns.loc[_id]

    return positive_jumps_path, negative_jumps_path, diffusion
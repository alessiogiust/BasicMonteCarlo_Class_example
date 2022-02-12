from numpy.core.fromnumeric import shape, size
import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


class BasicMonteCarlo():

    def __init__(self, S0, rf, fixed_var=None, K=None, kappa=None, theta=None, v0=None, rho=None, xi=None, lam=None, mj=None, vj=None):
        self.S0 = S0
        self.rf = rf
        self.fixed_var = fixed_var # fixed variance
        self.K = K
        self.kappa = kappa   # rate of mean reversion for variance in Heston
        self.theta = theta   # long run average variance for heston
        self.v0 = v0  # starting variance for Heston
        self.rho = rho  # correlation between Brownian Motion in Heston
        self.xi = xi   # volatility of volatility in Heston
        self.lam = lam   # number of jumps per year in Jump diffusion
        self.mj = mj   # mean of jump size in Jump diffusion
        self.vj = vj   # standard deviation of jump size in Jump diffusion
    
    def store_dataframe(self, sim_paths_output, steps, target_var="price", avg=False, max_val=False, min_val=False):
        dfpaths = pd.DataFrame(sim_paths_output, columns=[f"Step_{i+1}" for i in range(steps)])
        if target_var=="price":
            dfpaths.insert(0, "InitialValue", self.S0)
        else:
            dfpaths.insert(0, "InitialValue", self.v0)
        if avg==True:
            dfpaths["Average"] = dfpaths.loc[:, "Step_1":f"Step_{steps}"].mean(axis=1)
        if max_val==True:
            dfpaths["Max"] = dfpaths.loc[:, "Step_1":f"Step_{steps}"].max(axis=1)
        if min_val==True:
            dfpaths["Min"] = dfpaths.loc[:, "Step_1":f"Step_{steps}"].min(axis=1)
        return dfpaths
    
    def get_price_paths_summary(self, sim_paths_output, steps, target_var="price"):
        dfpaths = self.store_dataframe(sim_paths_output, steps=steps, target_var=target_var)
        summary = pd.DataFrame(columns=["Values"])
        summary.loc["N_Steps"] = steps
        summary.loc["Avg_FinalPrice"] = dfpaths.loc[:, f"Step_{steps}"].mean()  # average over last value of each path
        summary.loc["Max_FinalPrice"] = dfpaths.loc[:, f"Step_{steps}"].max()
        summary.loc["Min_FinalPrice"] = dfpaths.loc[:, f"Step_{steps}"].min()
        summary.loc["25_percentile"] = np.percentile(dfpaths.loc[:, f"Step_{steps}"].values, 25)
        summary.loc["50_percentile"] = np.percentile(dfpaths.loc[:, f"Step_{steps}"].values, 50)
        summary.loc["75_percentile"] = np.percentile(dfpaths.loc[:, f"Step_{steps}"].values, 75)
        return summary

    def get_returns_summary(self, sim_paths_output, steps, target_var="price"):
        dfpaths = self.store_dataframe(sim_paths_output, steps=steps, target_var=target_var)
        # Summary for the final returns after N steps
        dfretfinal = np.log(dfpaths.loc[:, f"Step_{steps}"]/dfpaths.loc[:, "InitialValue"])
        summaryfinal = pd.DataFrame(columns=["Values"])
        summaryfinal.loc["N_Steps"] = steps
        summaryfinal.loc["Avg_FinalReturn"] = dfretfinal.mean() 
        summaryfinal.loc["Max_FinalReturn"] = dfretfinal.max()
        summaryfinal.loc["Min_FinalReturn"] = dfretfinal.min()
        summaryfinal.loc["25_percentile"] = np.percentile(dfretfinal.values, 25)
        summaryfinal.loc["50_percentile"] = np.percentile(dfretfinal.values, 50)
        summaryfinal.loc["75_percentile"] = np.percentile(dfretfinal.values, 75)
        # Summary for the return series over N steps
        dfret = np.log(dfpaths.pct_change(axis=1)+1).dropna(axis=1)
        summarypaths = pd.DataFrame(columns=["Values"])
        summarypaths.loc["N_Steps"] = steps
        summarypaths.loc["Avg_Return"] = np.mean(dfret.mean())  # average over last value of each path
        summarypaths.loc["Max_Return"] = np.max(dfret.max())
        summarypaths.loc["Min_Return"] = np.min(dfret.min())
        summarypaths.loc["25_percentile"] = np.mean(dfret.apply(lambda x: np.percentile(x, 25), axis=1))
        summarypaths.loc["50_percentile"] = np.mean(dfret.apply(lambda x: np.percentile(x, 50), axis=1))
        summarypaths.loc["75_percentile"] = np.mean(dfret.apply(lambda x: np.percentile(x, 75), axis=1))
        summarypaths.loc["95_hist_VaR"] = np.mean(dfret.apply(lambda x: np.percentile(x, 5), axis=1))
        summarypaths.loc["99_hist_VaR"] = np.mean(dfret.apply(lambda x: np.percentile(x, 1), axis=1))
        # add skewness, kurtosis, other info...
        return summaryfinal, summarypaths

    def EUoptionprice_MC(self, sim_paths_output, T, steps, option_type="call"):
        """
        Price European options with Monte Carlo simulation
        """
        S_fin = np.array(sim_paths_output[:, steps-1])
        K_vector = np.full_like(S_fin, self.K)
        if option_type == "call":
            return np.exp(-self.rf*T)*np.mean(np.maximum(S_fin - K_vector, 0.0))
        elif option_type == "put":
            return np.exp(-self.rf*T)*np.mean(np.maximum(K_vector - S_fin, 0.0))

    def EUoptionprice_BSMformula(self, T, option_type="call"):
        """
        Price European options with Black-Scholes-Merton formula
        """
        d1 = ((np.log(self.S0/self.K) + (self.rf + 0.5*self.fixed_var)*T)) / (np.sqrt(self.fixed_var*T))
        d2 = d1 - np.sqrt(self.fixed_var * T)
        if option_type == "call":
            return st.norm.cdf(d1) * self.S0 - st.norm.cdf(d2) * self.K * np.exp(-self.rf * T)
        elif option_type == "put":
            return st.norm.cdf(-d2) * self.K * np.exp(-self.rf * T) - st.norm.cdf(-d1) * self.S0
        
    def EUGreeks_BSMformula(self, T, option_type="call"):
        """
        Compute European options Greeks with Black-Scholes-Merton formula
        """
        greeks = pd.DataFrame(columns=["BSMGreek_values"])
        greeks.loc["Option_type"] = option_type
        d1 = ((np.log(self.S0/self.K) + (self.rf + 0.5*self.fixed_var)*T)) / (np.sqrt(self.fixed_var*T))
        d2 = d1 - np.sqrt(self.fixed_var * T)
        if option_type == "call":
            delta = st.norm.cdf(d1)
            theta = (-self.S0*st.norm.pdf(d1)*np.sqrt(self.fixed_var))/(2*np.sqrt(T)) - self.rf*self.K*np.exp(-self.rf * T)*st.norm.cdf(d2)
            rho = self.K*T*np.exp(-self.rf * T)*st.norm.cdf(d2)
        elif option_type == "put":
            delta = st.norm.cdf(d1) - 1
            theta = (-self.S0*st.norm.pdf(d1)*np.sqrt(self.fixed_var))/(2*np.sqrt(T)) + self.rf*self.K*np.exp(-self.rf * T)*st.norm.cdf(-d2)
            rho = -self.K*T*np.exp(-self.rf * T)*st.norm.cdf(-d2)
        gamma = (st.norm.pdf(d1))/(self.S0*np.sqrt(self.fixed_var*T))
        vega = self.S0 * st.norm.pdf(d1) * np.sqrt(T)
        greeks.loc["Delta"] = delta
        greeks.loc["Gamma"] = gamma
        greeks.loc["Theta"] = theta
        greeks.loc["Vega"] = vega
        greeks.loc["Rho"] = rho
        return greeks

    def get_EUoptionpricing_summary(self, sim_paths_output, T, steps, Npaths=500):
        """
        set Npaths value if you changed the default value from the function sim_??_paths in the subclasses
        """
        summEU = pd.DataFrame(columns=["Values"])
        summEU.loc["BSM_Call_price"] = self.EUoptionprice_BSMformula(T=T)
        summEU.loc["BSM_Put_price"] = self.EUoptionprice_BSMformula(T=T, option_type="put")
        summEU.loc["MonteCarlo_Npaths"] = Npaths
        summEU.loc["MonteCarlo_Steps"] = steps
        summEU.loc["MonteCarlo_Call_price"] = self.EUoptionprice_MC(sim_paths_output=sim_paths_output, T=T, steps=steps)
        summEU.loc["MonteCarlo_Put_price"] = self.EUoptionprice_MC(sim_paths_output=sim_paths_output, T=T, steps=steps, option_type="put")
        summGreeksBSM = pd.DataFrame()
        s1 = self.EUGreeks_BSMformula(T=T)
        s2 = self.EUGreeks_BSMformula(T=T, option_type="put")
        summGreeksBSM = s1.merge(s2, right_index=True, left_index=True)
        summGreeksBSM.columns = ["BSM_values_Call", "BSM_values_Put"]
        summGreeksBSM.drop("Option_type", inplace=True)
        return summEU, summGreeksBSM

    def plot_paths(self, dfpaths, steps, n_paths_plot=5):
        """
        Plot Monte Carlo paths and results summary
        """
        df = dfpaths.transpose()  # invert df structure
        dfretfinal = np.log(dfpaths.loc[:, f"Step_{steps}"]/dfpaths.loc[:, "InitialValue"])
        fig, axd = plt.subplot_mosaic([['top', 'top'], ['left', 'right']], constrained_layout=True, figsize=(10, 6))
        axd['top'].plot(df.iloc[:, 0:n_paths_plot]) 
        axd['top'].set_xticklabels([])    # xaxis.set_visible(False)
        axd['top'].set_title(f"First {n_paths_plot} stock price paths")
        axd['left'].hist(dfpaths.loc[:, f"Step_{steps}"], bins=40) 
        axd['left'].set_title("Final stock prices histogram")
        axd['right'].hist(dfretfinal, bins=40)  
        axd['right'].set_title("Final logarithmic returns histogram")
        plt.show()


class GBMSimulation(BasicMonteCarlo):

    def __init__(self, S0, rf, fixed_var=None, K=None, kappa=None, theta=None, v0=None, rho=None, xi=None, lam=None, mj=None, vj=None):
        super().__init__(S0, rf, fixed_var, K, kappa, theta, v0, rho, xi, lam, mj, vj)

    def sim_gbm_paths(self, T, steps, Npaths=500, delta=False):
        dt = T/steps
        size = (Npaths, steps)
        prices = np.zeros(size)
        # delta = True used to compute Monte Carlo delta
        if delta==False:
            S_t = self.S0
        else:
            S_t = (self.S0)*1.05  # S0 + 5%
        for t in range(steps):
            Z = np.random.normal(0.0, 1.0, Npaths) * np.sqrt(dt)
            S_t = S_t*(np.exp((self.rf - 0.5*self.fixed_var)*dt + np.sqrt(self.fixed_var) * Z))
            prices[:, t] = S_t
        return prices
    
    def delta_MC(self, T, steps, Npaths=500, option_type="call"):
        # repeat process for all the other first order greeks
        p = self.sim_gbm_paths(T=T, steps=steps, Npaths=Npaths)
        Vp = self.EUoptionprice_MC(p, T=T, steps=steps, option_type=option_type)
        p_dS = self.sim_gbm_paths(T=T, steps=steps, Npaths=Npaths, delta=True)
        Vp_dS = self.EUoptionprice_MC(p_dS, T=T, steps=steps, option_type=option_type)
        dS = (self.S0)*1.05 - self.S0
        deltamc = (Vp_dS - Vp) / dS
        return deltamc


class HestonModelSimulation(BasicMonteCarlo):

    def __init__(self, S0, rf, fixed_var=None, K=None, kappa=None, theta=None, v0=None, rho=None, xi=None, lam=None, mj=None, vj=None):
        super().__init__(S0, rf, fixed_var, K, kappa, theta, v0, rho, xi, lam, mj, vj)

    def sim_heston_paths(self, T, steps, Npaths=500, return_vol=False, delta=False):
        dt = T/steps
        size = (Npaths, steps)
        prices = np.zeros(size)
        sigs = np.zeros(size)
        if delta==False:
            S_t = self.S0
        else:
            S_t = (self.S0)*1.05
        v_t = self.v0
        for t in range(steps):
            WT = np.random.multivariate_normal(np.array([0,0]), cov = np.array([[1, self.rho],[self.rho,1]]), size=Npaths) * np.sqrt(dt) 
            S_t = S_t * (np.exp((self.rf- 0.5*v_t)*dt + np.sqrt(v_t) * WT[:, 0])) 
            v_t = np.abs(v_t + self.kappa*(self.theta - v_t)*dt + self.xi * np.sqrt(v_t) * WT[:, 1])
            prices[:, t] = S_t
            sigs[:, t] = v_t
        if return_vol:
            return prices, sigs
        return prices
    
    def delta_MC(self, T, steps, Npaths=500, option_type="call"):
        p = self.sim_heston_paths(T=T, steps=steps, Npaths=Npaths)
        Vp = self.EUoptionprice_MC(p, T=T, steps=steps, option_type=option_type)
        p_dS = self.sim_heston_paths(T=T, steps=steps, Npaths=Npaths, delta=True)
        Vp_dS = self.EUoptionprice_MC(p_dS, T=T, steps=steps, option_type=option_type)
        dS = (self.S0)*1.05 - self.S0
        deltamc = (Vp_dS - Vp) / dS
        return deltamc


class JumpDiffusionSimulation(BasicMonteCarlo):

    def __init__(self, S0, rf, fixed_var=None, K=None, kappa=None, theta=None, v0=None, rho=None, xi=None, lam=None, mj=None, vj=None):
        super().__init__(S0, rf, fixed_var, K, kappa, theta, v0, rho, xi, lam, mj, vj)

    def sim_jumpdiffussion_paths(self, T, steps, Npaths=500, delta=False):
        size = (steps, Npaths)
        dt = T/steps 
        poi_rv = np.multiply(np.random.poisson(self.lam*dt, size=size), np.random.normal(self.mj, self.vj, size=size)).cumsum(axis=0)
        geo = np.cumsum(((self.rf - self.fixed_var/2 - self.lam*(self.mj + self.fixed_var*0.5))*dt + np.sqrt(dt*self.fixed_var) * np.random.normal(size=size)), axis=0)
        if delta==False:
            S = self.S0
        else:
            S = (self.S0)*1.05
        pt = np.exp(geo + poi_rv) * S
        pt = pt.transpose()
        return pt
    
    def delta_MC(self, T, steps, Npaths=500, option_type="call"):
        p = self.sim_jumpdiffussion_paths(T=T, steps=steps, Npaths=Npaths)
        Vp = self.EUoptionprice_MC(p, T=T, steps=steps, option_type=option_type)
        p_dS = self.sim_jumpdiffussion_paths(T=T, steps=steps, Npaths=Npaths, delta=True)
        Vp_dS = self.EUoptionprice_MC(p_dS, T=T, steps=steps, option_type=option_type)
        dS = (self.S0)*1.05 - self.S0
        deltamc = (Vp_dS - Vp) / dS
        return deltamc



# ######## GBM EXAMPLE
# steps = 150  # number of steps
# gbm = GBMSimulation(S0=65, rf=0.002, fixed_var=0.15, K=60)
# price_path = gbm.sim_gbm_paths(T=1, steps=steps, Npaths=50)  
# # # df_price_paths = gbm.store_dataframe(price_path, steps)
# # summ_pr = gbm.get_price_paths_summary(price_path, steps)
# # summ_ret, summ_ret_series = gbm.get_returns_summary(price_path, steps)
# # print(summ_ret_series)

# # d_MC_gbm = gbm.delta_MC(T=1, steps=steps, Npaths=10000, option_type="call")
# # print(d_MC_gbm)

# ######## HESTON EXAMPLE
# hms = HestonModelSimulation(S0=100, rf=0.002, fixed_var=0.15, K=105, kappa=4, theta=0.15, v0=0.15, rho=0.9, xi=0.9)
# price_path = hms.sim_heston_paths(T=1, steps=steps, Npaths=500, return_vol=False)  
# df_price_paths = hms.store_dataframe(price_path, steps)
# # summ_pr = hms.get_price_paths_summary(price_path, steps)
# # print(summ_pr)
# # hms.plot_paths(df_price_paths, steps)
# # print(f"MonteCarlo-Heston price: {hms.EUoptionprice_MC(price_path, T=1, steps=steps, option_type='call')}")
# # print(f"BSM price: {hms.EUoptionprice_BSMformula(T=1, option_type='call')}")

# # df1, df2 = hms.get_EUoptionpricing_summary(price_path, T=1, steps=steps, Npaths=500)
# # print(df1)
# # print(df2)

# ######## JUMP DIFFUSION EXAMPLE
# jdiff = JumpDiffusionSimulation(S0=100, rf=0.002, fixed_var=0.15, K=105, lam=1, mj=0, vj=0.3)
# price_path = jdiff.sim_jumpdiffussion_paths(T=1, steps=steps, Npaths=500)
# df_price_paths = jdiff.store_dataframe(price_path, steps)
# # jdiff.plot_paths(df_price_paths, steps)


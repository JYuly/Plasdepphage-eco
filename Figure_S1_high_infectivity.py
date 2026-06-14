from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from scipy.signal import find_peaks
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors

from numba import njit

# Define the model WITH delays for lysis and for pili expression after plasmid acquisition
@njit
def model_w_delays(t, y, alpha, gamma_hi, gamma_lo, delta, qc, CP, b, delta_P, K, cost, pl, T_lysis, T_pili, N):
    y = y.reshape(5+2*N) #for some reason without this y = [y0,y1,...,yn] becomes y = [[y0],[y1],...,[yn]]
    k_hi = CP*gamma_hi
    k_lo = CP*gamma_lo
    B0 = y[0]
    BP_hi = y[1]
    BP_lo = y[2]
    P = y[3]
    Bc = y[4]
    i_Bi = slice(5,5+N)
    i_BPnp = slice(5 + N, 5+ 2*N)
    Bi = y[5:5+N]
    BPnp = y[5+N:5+2*N] #np for "no pili"
  
    C = 1 - (B0 + BP_hi + BP_lo + Bc + sum(Bi) + sum(BPnp))/K
    dB0dt =  alpha*C*(B0 + pl*(1+cost)*(BP_hi + BP_lo)) - gamma_hi*B0*BP_hi - gamma_lo*B0*BP_lo - delta*B0 
    dBP_hidt = alpha*(1 + cost)*C*(1 - pl)*BP_hi + (N/T_pili)*BPnp[-1]  - qc*BP_hi - k_hi*BP_hi*P - delta*BP_hi
    dBP_lodt = alpha*(1 + cost)*C*(1 - pl)*BP_lo + qc*BP_hi - k_lo*BP_lo*P - delta*BP_lo
   

    dBcdt = alpha*C*Bc- delta*Bc
    
    #calculate vector-valued terms for lysis and pili expression intermediate states
    C_hill_hi = 10**50 #option to add hill function for phage absorption (huge number eliminates saturation)
    C_hill_lo = 10**50
    dPdt = b*(N/T_lysis)*Bi[-1] - delta_P*P - k_hi*sum(y[5:5+N])*P/(1 + P/C_hill_hi)
    dBidt = np.zeros(len(Bi))
    dBidt[0] = k_hi*BP_hi*P/(1 + P/C_hill_hi) + k_lo*BP_lo*P/(1 + P/C_hill_lo) - (N/T_lysis)*Bi[0]- delta*Bi[0]
    for i in range(1, len(Bi)):
        dBidt[i] = (N/T_lysis)*(Bi[i-1] - Bi[i]) - delta*Bi[i]
        
    dBPnpdt = np.zeros(len(BPnp))
    dBPnpdt[0] = (gamma_hi*BP_hi + gamma_lo*BP_lo)*B0 -(N/T_pili)*BPnp[0] - delta*BPnp[0] 
    for i in range(1, len(BPnp)):
        dBPnpdt[i] = (N/T_pili)*(BPnp[i-1] - BPnp[i]) - delta*BPnp[i]

    result = np.empty_like(y)
    scalar_vars = [dB0dt, dBP_hidt, dBP_lodt, dPdt, dBcdt]
    i_scalars = slice(0,5)
    result[i_scalars] = scalar_vars
    result[i_Bi] = dBidt
    result[i_BPnp] = dBPnpdt
    return result





#parameters
alpha = 1.5        # Growth rate of the bacteria w/o the plasmid
delta = 0.1     #Death rate of bacteria
b = 10**2     # Burst size of phage
delta_P = 0.1      # Rate of background death of phage
K = 10**8   #carrying capacity
cost = -0.5
pl = 0.001     # Probability of losing the plasmid
gamma_hi = 10**(-8) #Maximum conjugation rate

T_lysis = 0.5
T_pili = 0.5
N = 5

i_scalars = slice(0,5)
i_Bi = slice(5,5+N)
i_BPnp = slice(5 + N, 5+ 2*N)

#Define exctinction events (params are needed because solve_ivp always calls the event functions using them)
def Bp_extinction_event(t, y, alpha, gamma_hi, gamma_lo, delta, qc, CP, b, delta_P, K, cost, pl,T_lysis, T_pili, N):
    return y[1] + y[2] + np.sum(y[i_BPnp]) - 1
Bp_extinction_event.terminal = True
Bp_extinction_event.direction = -1

def P_extinction_event(t, y,  alpha, gamma_hi, gamma_lo, delta, qc, CP, b, delta_P, K, cost, pl,T_lysis, T_pili, N):
    return y[3] + np.sum(y[i_Bi]) - 1
P_extinction_event.terminal = True
P_extinction_event.direction = -1


extinction_events = [Bp_extinction_event, P_extinction_event]


#Low infectivity (10**(-1))
CP = 10**(1) #constant of proportionality between the congugation constant and the phage predation constant (due to pilus expression)
k_hi3 = gamma_hi*CP     # Phage predation constants
tf = 5000


Tend = 500 #time period to collect final results

def scan(gamma_lo, qc):
    args = [alpha, gamma_hi, gamma_lo, delta, qc, CP, b, delta_P, K, cost, pl,T_lysis, T_pili, N]
    # Initial condition (chosen to be close-ish to steady state to facilitate convergence)
    k_hi = gamma_hi*CP     # Phage predation constants
    k_lo = gamma_lo*CP     # Phage predation constants
    B00 = K
    BP_hi0 =  K
    BP_lo0 = 0
    P0 = 0#gamma_hi*B0/k_hi
    Bc0 = 0 #no competitor
    scalar_vars0 = [B00, BP_hi0, BP_lo0, P0, Bc0]
    y0 = np.empty(5+2*N)
    y0[i_scalars] = scalar_vars0
    y0[i_Bi] = 0
    y0[i_BPnp] = 0

    t_data = []
    y_data = []
    
    t0 = 0
    tEq = 1000
    solEq = solve_ivp(
        model_w_delays,
        [t0, tEq],
        y0,
        args = args,
        dense_output=False,
        method='LSODA', 
        max_step=1.0,
        )
     
    y0 = solEq.y[:, -1].copy()

    # If plasmid-carrying bacteria are extinct at this point, set them to 0
    if y0[1] + y0[2] + np.sum(y0[i_BPnp]) <= 1:
        y0[1] = 0.0
        y0[2] = 0.0
        y0[i_BPnp] = 0.0
        return [0, 0, 0, 0] #Plasmid is extinct, so predation rate and plasmid fraction are both 0


    y0[3] = 10 #add phage after coming to equilibrium
    t0 = 0
    

    sol = solve_ivp(
    model_w_delays,
    [t0, tf],
    y0,
    events=extinction_events,
    args = args,
    dense_output=False,
    method='LSODA', 
    max_step=0.1
    )


    # Store results
    t_data.extend(sol.t)
    y_data.extend(sol.y.T)
    # Check which event triggered
    triggered = [len(ev) > 0 for ev in sol.t_events]
    y0 = sol.y[:, -1].copy()


    if triggered[0]:  #plasmid extinction
        y0[1] = 0.0
        y0[2] = 0.0
        y0[i_BPnp] = 0.0
        return [0, 0, 0, 0] #Plasmid is extinct, so predation rate and plasmid fraction are both 0
    if triggered[1]:  #phage extinction
        y0[3] = 0.0
        y0[i_Bi] = 0.0
        return [0, (y0[1] + y0[2])/(y0[0] + y0[1] + y0[2]), 0, (y0[1] + y0[2])/(y0[0] + y0[1] + y0[2])] #Phage is extinct, so predation rate is 0 and plasmid fraction is determined by equilibrium without phage


    t = np.array(t_data)
    y = np.array(y_data)
    
    mask = t >= t[-1] - Tend
    predation_rate = y[mask, 3]*(k_hi*y[mask, 1] + k_lo*y[mask, 2])
    plasmid_containing = y[mask,1] + y[mask,2] #+np.sum(y[mask, 5:2*N]) #sum goes over both sets of intermediate states because both the infected and pre-pilus expressing cells have plasmids
    plasmid_fraction = plasmid_containing/(plasmid_containing + y[mask, 0]) #plasmid containing fraction

    pred_max_val = np.max(predation_rate)
    plas_frac_max_val = np.max(plasmid_fraction)

    #average values over the end period
    plas_frac_ave = np.average(plasmid_fraction)
    pred_ave = np.average(predation_rate)
    
    return [pred_max_val/(K*delta), plas_frac_max_val, pred_ave/(K*delta), plas_frac_ave] #normalize predation rate by max possible death rate of bacteria (when population is at carrying capacity and not limited by phage) to get relative predation rate

#Scan through values of Gcs and kc
Npoints = 100
gamma_lo_range = np.logspace(-15, -8, num = Npoints)
qc_range = np.logspace(-2.15, 0.75, num = Npoints)
# Create 2D meshgrid arrays
X, Y = np.meshgrid(gamma_lo_range, qc_range)
# Initialize an empty array for the Z data
pred_max = np.empty_like(X)
plas_max = np.empty_like(X)
pred_ave = np.empty_like(X)
plas_ave = np.empty_like(X)

# Add data to grid plot
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        pred_max[i,j], plas_max[i,j], pred_ave[i,j], plas_ave[i,j] = scan(X[i, j], Y[i, j])



print("Done")

# save computed data arrays
np.savez('phase_diagram_data_high_infectivity.npz', pred_max=pred_max, plas_max=plas_max, pred_ave=pred_ave, plas_ave=plas_ave, X = X, Y = Y)
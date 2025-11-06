# create_and_fit.py
# Self-contained script to:
# 1) generate synthetic monthly CSVs (2004-01 -> 2021-12; twitch starts 2012)
# 2) fit a logistic-Bass hybrid ODE model to the synthetic steam data
# 3) save fitted outputs and a plot
#
# Dependencies: numpy, pandas, scipy, matplotlib, python-dateutil

import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

outdir = "output_data"
os.makedirs(outdir, exist_ok=True)

# 1) Time axis: monthly from 2004-01 to 2021-12
start = datetime(2004,1,1)
end = datetime(2021,12,1)
dates = []
t = start
while t <= end:
    dates.append(t)
    t += relativedelta(months=1)
n_months = len(dates)
time_months = np.arange(n_months)

# ---------- Synthetic world population (carrying capacity proxy) ----------
pop_start = 6.4e9
pop_end = 7.8e9
pop = np.linspace(pop_start, pop_end, n_months)  # in people
df_pop = pd.DataFrame({"date": [d.strftime("%Y-%m-%d") for d in dates], "world_population": pop})
pop_csv = os.path.join(outdir, "world_population_monthly.csv")
df_pop.to_csv(pop_csv, index=False)

# ---------- Synthetic internet penetration (yearly-like smooth trend) ----------
rng = np.random.default_rng(42)
int_start = 0.15
int_end = 0.65
internet = np.linspace(int_start, int_end, n_months)
internet += 0.01 * np.sin(2*np.pi*time_months/12) + rng.normal(scale=0.005, size=n_months)
internet = np.clip(internet, 0, 1)
df_inet = pd.DataFrame({"date": [d.strftime("%Y-%m-%d") for d in dates], "internet_penetration": internet})
inet_csv = os.path.join(outdir, "internet_penetration_monthly.csv")
df_inet.to_csv(inet_csv, index=False)

# ---------- Synthetic Twitch viewership (starts 2012) ----------
twitch_start = datetime(2012,1,1)
dates_twitch = []
t = twitch_start
while t <= end:
    dates_twitch.append(t)
    t += relativedelta(months=1)
n_twitch = len(dates_twitch)
tidx = np.arange(n_twitch)
twitch = 0.2 * np.exp(0.35 * (tidx/12))  # exponential-ish growth
twitch = np.minimum(twitch, 60.0)         # cap for realism
twitch += rng.normal(scale=1.0, size=n_twitch)
twitch = np.clip(twitch, 0, None)
df_twitch = pd.DataFrame({"date": [d.strftime("%Y-%m-%d") for d in dates_twitch], "twitch_viewers_millions": twitch})
twitch_csv = os.path.join(outdir, "twitch_viewership_monthly.csv")
df_twitch.to_csv(twitch_csv, index=False)

# Build twitch_full aligned to the full date range (0 before 2012)
twitch_full = np.zeros(n_months)
for i,dt in enumerate(dates):
    if dt >= twitch_start:
        tidx = (dt.year - twitch_start.year)*12 + (dt.month - twitch_start.month)
        if 0 <= tidx < n_twitch:
            twitch_full[i] = twitch[tidx]
        else:
            twitch_full[i] = twitch[-1]
    else:
        twitch_full[i] = 0.0

# Normalize helpers (for drivers)
pop_millions = pop / 1e6
internet_norm = (internet - internet.min()) / (internet.max()-internet.min())
twitch_norm = (twitch_full - twitch_full.min()) / (twitch_full.max()-twitch_full.min())

# ---------- Synthesize Steam users using a "true" hybrid model (then add noise) ----------
true_params = {"p0":0.002, "a":0.01, "q0":0.02, "b":0.05, "k_frac":0.02}
p_t = true_params["p0"] + true_params["a"] * internet_norm
q_t = true_params["q0"] + true_params["b"] * twitch_norm
K_t = pop_millions * true_params["k_frac"]  # carrying capacity in millions

def simulate_hybrid_series(p_t_series, q_t_series, K_t_series, N0):
    # ODE: dN/dt = (p(t) + q(t) * N/M) * N * (1 - N/K(t))
    # treat time in months; series are monthly values; map t->floor index
    def rhs(t, N):
        idx = int(np.clip(np.floor(t), 0, len(p_t_series)-1))
        p = p_t_series[idx]
        q = q_t_series[idx]
        K = K_t_series[idx]
        M = K
        return (p + q * (N / M)) * N * (1 - N / K)
    sol = solve_ivp(rhs, (0, len(p_t_series)-1), [N0], t_eval=np.arange(len(p_t_series)), max_step=1.0)
    return sol.y[0]

N0 = 5.0  # 5 million in 2004
steam_clean = simulate_hybrid_series(p_t, q_t, K_t, N0)
steam_obs = steam_clean + rng.normal(scale=1.0, size=steam_clean.size)  # observation noise in millions
steam_obs = np.clip(steam_obs, 0, None)
df_steam = pd.DataFrame({"date": [d.strftime("%Y-%m-%d") for d in dates], "steam_users_millions": steam_obs})
steam_csv = os.path.join(outdir, "steam_users_monthly.csv")
df_steam.to_csv(steam_csv, index=False)

# Save the "true" debug series too
df_gen = pd.DataFrame({
    "date": [d.strftime("%Y-%m-%d") for d in dates],
    "N_clean_millions": steam_clean,
    "p_t": p_t,
    "q_t": q_t,
    "K_t_millions": K_t,
    "internet_norm": internet_norm,
    "twitch_norm": twitch_norm
})
gen_csv = os.path.join(outdir, "generated_series_debug.csv")
df_gen.to_csv(gen_csv, index=False)

# ---------- Fit the hybrid model to the synthetic observed steam data ----------
y_obs = steam_obs
internet_drive = internet_norm.copy()
twitch_drive = twitch_norm.copy()
pop_m = pop_millions.copy()

def residuals(x):
    p0, a, q0, b, k_frac = x
    if k_frac <= 0:
        return 1e6 * np.ones_like(y_obs)
    p_t_fit = p0 + a * internet_drive
    q_t_fit = q0 + b * twitch_drive
    K_t_fit = pop_m * k_frac
    N_sim = simulate_hybrid_series(p_t_fit, q_t_fit, K_t_fit, N0)
    return N_sim - y_obs

# initial guess and bounds
x0 = np.array([0.001, 0.005, 0.01, 0.01, 0.01])
lb = [0.0, 0.0, 0.0, 0.0, 1e-4]
ub = [0.01, 0.05, 0.2, 0.2, 0.1]

res = least_squares(residuals, x0, bounds=(lb, ub), verbose=2, xtol=1e-8, ftol=1e-8, max_nfev=200)
fitted_params = res.x
fitted_params_dict = {"p0":fitted_params[0], "a":fitted_params[1], "q0":fitted_params[2], "b":fitted_params[3], "k_frac":fitted_params[4]}

# Simulate with fitted params
p_t_fit = fitted_params_dict["p0"] + fitted_params_dict["a"] * internet_drive
q_t_fit = fitted_params_dict["q0"] + fitted_params_dict["b"] * twitch_drive
K_t_fit = pop_m * fitted_params_dict["k_frac"]
N_fitted = simulate_hybrid_series(p_t_fit, q_t_fit, K_t_fit, N0)

# Save fitted series and parameters
df_fit = pd.DataFrame({"date":[d.strftime("%Y-%m-%d") for d in dates], "steam_obs_millions": y_obs, "steam_fitted_millions": N_fitted})
fit_csv = os.path.join(outdir, "steam_fitted_monthly.csv")
df_fit.to_csv(fit_csv, index=False)

params_csv = os.path.join(outdir, "fitted_parameters.csv")
pd.DataFrame([fitted_params_dict]).to_csv(params_csv, index=False)

# drivers CSV
df_drivers = pd.DataFrame({
    "date": [d.strftime("%Y-%m-%d") for d in dates],
    "internet_penetration": internet,
    "internet_norm": internet_norm,
    "twitch_full_millions": twitch_full,
    "twitch_norm": twitch_norm,
    "world_population_millions": pop_millions
})
drivers_csv = os.path.join(outdir, "drivers_monthly.csv")
df_drivers.to_csv(drivers_csv, index=False)

# Plot observed vs fitted
plt.figure(figsize=(10,5))
plt.plot(pd.to_datetime(df_fit["date"]), df_fit["steam_obs_millions"], label="Observed Steam users (millions)")
plt.plot(pd.to_datetime(df_fit["date"]), df_fit["steam_fitted_millions"], label="Fitted model prediction (millions)")
plt.xlabel("Date")
plt.ylabel("Steam users (millions)")
plt.title("Observed vs Fitted Steam Users (Monthly) - Synthetic Data")
plt.legend()
plt.tight_layout()
plot_png = os.path.join(outdir, "observed_vs_fitted.png")
plt.savefig(plot_png)
plt.close()

print("Files saved to folder:", outdir)
print("Key files:")
print(" -", pop_csv)
print(" -", inet_csv)
print(" -", twitch_csv)
print(" -", steam_csv)
print(" -", fit_csv)
print(" -", params_csv)

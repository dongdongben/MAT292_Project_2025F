"""
run_fitting_scenarios.py

Corrected version with solver/DE fixes:
 - simulate_hybrid returns fixed-length arrays and penalizes failed solves
 - differential_evolution bounds shaped correctly as list of (low,high) tuples
 - safe numeric coercion and NaN handling
 - fallback from DE to local search if DE fails
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, differential_evolution
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import json
import warnings



# -------------------- User config --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(BASE_DIR, "output_results")
os.makedirs(OUTDIR, exist_ok=True)

# CSV names (must exist in same folder)
FILES = {
    "population": os.path.join(BASE_DIR, "world_population_04_01.csv"),
    "internet": os.path.join(BASE_DIR, "internet_penetration_04_01.csv"),
    "twitch": os.path.join(BASE_DIR, "twitch_views_04_01.csv"),
    "steam": os.path.join(BASE_DIR, "steam_users_04_01.csv"),
    "cpi": os.path.join(BASE_DIR, "tech_cpi_04_01.csv"),
    "ppi": os.path.join(BASE_DIR, "tech_ppi_04_01.csv")
}


# Extend year for scenario projections
FIT_END_YEAR = 2025
EXTEND_TO_YEAR = 2030

# How many months of history to use when fitting local trend for extrapolation
FIT_WINDOW_MONTHS = 60  # here we use last 3 years trend for extrapolation

# Initial guess / bounds for parameters p0,a1,a2,q0,b1,k1,k2 for fitting
PARAM_BOUNDS = {
    "lower": [1e-4, 0.0, 0.0, 1e-4, 0.0, 0.0, 0.0],
    "upper": [1.0, 20.0, 20.0, 2.0, 10.0, 50.0, 10]
}

# -----------------------------------------------------


def load_monthly_csv(filepath, col_name_hint):
    df = pd.read_csv(filepath)
    # find date column
    date_col_candidates = [c for c in df.columns if "date" in c.lower()]
    date_col = date_col_candidates[0]
    # value column
    val_cols = [c for c in df.columns if c != date_col]
    val_col = val_cols[0]
    df = df[[date_col, val_col]].rename(columns={date_col: "date", val_col: col_name_hint})

    # --- CLEAN DATE STRINGS ---
    df["date"] = (
        df["date"]
        .astype(str)
        .str.strip()
        .str.replace(r"[^\x00-\x7F]+", "", regex=True)
    )

    # Try to parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if df["date"].isna().any():
        bad = df[df["date"].isna()]
        raise ValueError(
            f"Unparsable dates found in {filepath}. Example bad rows:\n{bad.head()}"
        )

    df = df.sort_values("date").reset_index(drop=True)
    return df



def create_master_timeline(dfs):
    start = min(df["date"].min() for df in dfs.values())
    end   = max(df["date"].max() for df in dfs.values())
    # extend to EXTEND_TO_YEAR
    end_extended = datetime(EXTEND_TO_YEAR, 12, 1) #DEV, end vs end_extended
    dates = pd.date_range(start=start, end=end_extended, freq="MS")  # month starts
    return pd.DataFrame({"date": dates})



def merge_and_fill(master, df, col, fill_strategy="zero_before"):
    merged = master.merge(df, on="date", how="left")
    # linear interpolation used for internal missing values
    merged[col] = merged[col].interpolate(method='linear', limit_direction='backward')
    # handle months before earliest available depending on strategy
    if df["date"].min() > merged["date"].min():
        if fill_strategy == "zero_before":
            merged.loc[merged["date"] < df["date"].min(), col] = 0.0
        elif fill_strategy == "hold_before":
            first_val = merged.loc[merged["date"] >= df["date"].min(), col].iloc[0]
            merged.loc[merged["date"] < df["date"].min(), col] = first_val
    return merged



def extrapolate_driver(series, months_out, method):
    series = series.dropna().copy()
    series.index = pd.to_datetime(series.index)
    series = series.sort_index()

    if months_out <= 0:
        return series

    if method == "linear":
        return extrapolate_future(series, months_out, "linear")

    if method == "exp":
        return extrapolate_future(series, months_out, "exp")

    if method == "logistic":
        # Capping growth
        K = series.max() * 1.1
        t = np.arange(len(series))
        y = series.values

        # prevent invalid values
        if np.any(y <= 0):
            return extrapolate_future(series, months_out, "linear")

        fit_len = min(36, len(y))
        logit = np.log(y[-fit_len:] / (K - y[-fit_len:] + 1e-9))
        r = np.polyfit(t[-fit_len:], logit, 1)[0]

        future_t = np.arange(len(series), len(series) + months_out)
        y_future = K / (1 + np.exp(-r * (future_t - future_t[0])))

        future_index = pd.date_range(
            start=series.index[-1] + relativedelta(months=1),
            periods=months_out,
            freq="MS"
        )

        return pd.concat([series, pd.Series(y_future, index=future_index)])

    raise ValueError(f"Unknown extrapolation method: {method}")



def extrapolate_future(series, months_out, method, fit_window=FIT_WINDOW_MONTHS):
    """
    series: pandas Series indexed by datetime (monthly) with no NaNs
    months_out: number of months to extend beyond last index
    """
    # Ensure index is datetime
    series = series.copy()
    series.index = pd.to_datetime(series.index)
    series = series.sort_index()
    n = len(series)
    if n == 0:
        # if there is nothing to extrapolate - return zeros for months_out
        future_index = pd.date_range(start=series.index[-1] + relativedelta(months=1), periods=months_out, freq='MS') if n>0 else []
        return pd.Series([], index=series.index).append(pd.Series([0.0]*months_out, index=future_index))

    t = np.arange(n).reshape(-1,1)
    y = series.values.reshape(-1,1)

    fit_len = min(n, fit_window)
    t_fit = t[-fit_len:]
    y_fit = y[-fit_len:]

    # try exponential if requested and valid
    if method == "exp":
        mask = y_fit[:,0] > 0
        if mask.sum() >= max(3, fit_len//2):
            lr = LinearRegression()
            lr.fit(t_fit[mask], np.log(y_fit[mask]))
            future_t = np.arange(n, n+months_out).reshape(-1,1)
            y_future = np.exp(lr.predict(future_t)).ravel()
            future_index = pd.date_range(start=series.index[-1] + relativedelta(months=1), periods=months_out, freq='MS')
            ext = pd.Series(y_future, index=future_index)
            return pd.concat([series, ext])
        # if exponential does not work, fallback to linear
    # linear
    lr = LinearRegression()
    lr.fit(t_fit, y_fit)
    future_t = np.arange(n, n+months_out).reshape(-1,1)
    y_future = lr.predict(future_t).ravel()
    future_index = pd.date_range(start=series.index[-1] + relativedelta(months=1), periods=months_out, freq='MS')
    ext = pd.Series(y_future, index=future_index)
    return pd.concat([series, ext])



def split_historical_future(df, cutoff_year):
    hist = df[df["date"].dt.year <= cutoff_year].copy()
    fut  = df[df["date"].dt.year >  cutoff_year].copy()
    return hist, fut



def build_future_parameters(fitted_params, hist_df, future_df):
    p0, a1, a2, q0, b1, k1, k2 = fitted_params

    cpi = future_df["cpi_norm"].values
    ppi = future_df["ppi_norm"].values
    twitch = future_df["twitch_norm"].values
    pop = future_df["pop_norm"].values
    internet = future_df["internet_norm"].values

    p_t = p0 + a1 * cpi + a2 * ppi
    q_t = q0 + b1 * twitch
    K_t = k1 * pop + k2 * internet

    return p_t, q_t, K_t



# ODE simulation (N in same units as y_obs)
def simulate_hybrid(p_t, q_t, K_t, N0, t_len):
    """
    Simulate the hybrid ODE over t_len monthly steps.
    """
    def rhs(t, N):
        idx = int(np.clip(np.floor(t), 0, t_len-1))
        p = float(p_t[idx])
        q = float(q_t[idx])
        K = float(K_t[idx])
        if K <= 0:
            return 0.0
        return float((p + q * (N / K)) * N * (1 - N / K))

    # ensure arrays are numeric numpy arrays
    p_t = np.asarray(p_t, dtype=float)
    q_t = np.asarray(q_t, dtype=float)
    K_t = np.asarray(K_t, dtype=float)

    try:
        sol = solve_ivp(
            rhs,
            (0, t_len-1),
            [float(N0)],
            t_eval=np.arange(t_len),
            max_step=1.0,
            rtol=1e-3, #DEV, -6
            atol=1e-5, #-8
            method="RK45"   #Using RK45 method
        )
    except Exception as e:
        # If solver crashes, return large array
        warnings.warn(f"ODE solver exception: {e}; returning penalty array.")
        return np.ones(t_len) * 1e12

    # Validate solution shape and status
    if sol.status < 0 or sol.y.shape[1] != t_len:
        # solver failed or returned fewer points; return large array to reject these params
        warnings.warn("ODE solver did not return expected number of points; returning penalty array.")
        return np.ones(t_len) * 1e12

    # success
    return sol.y[0]



# Residual function for least-squares
def residuals_for_params(x, drivers, y_obs, N0):
    # x: [p0, a1, a2, q0, b1, k1, k2]
    p0, a1, a2, q0, b1, k1, k2 = x
    internet = np.asarray(drivers["internet_norm"], dtype=float)
    pop = np.asarray(drivers["pop_norm"], dtype=float)
    twitch = np.asarray(drivers["twitch_norm"], dtype=float)
    cpi = np.asarray(drivers.get("cpi_norm", np.zeros_like(internet)), dtype=float)
    ppi = np.asarray(drivers.get("ppi_norm", np.zeros_like(internet)), dtype=float)
    # compute time-varying coefficients
    p_t = p0 + a1 * cpi + a2 * ppi
    q_t = q0 + b1 * twitch
    K_t = k1 * pop + k2 * internet

    p_t = np.maximum(p_t, 1e-9)
    q_t = np.maximum(q_t, 0.0)
    K_t = np.maximum(K_t, np.max(y_obs)*1.05)  # ensure K at least slightly above observed max

    # if K_t.max() > 1e8 or K_t.max() < 1.2*np.max(y_obs):
    #     return np.ones_like(y_obs) * 1e6
    if np.any(np.isnan(K_t)) or np.any(np.isinf(K_t)) or np.any(K_t <= 0):
        # return a penalty residual vector same length as y_obs
        return np.ones_like(y_obs) * 1e6
    sim = simulate_hybrid(p_t, q_t, K_t, N0, len(y_obs))
    # if sim is a penalty (very large values), propagate that
    if sim.shape[0] != len(y_obs):
        return np.ones_like(y_obs) * 1e6
    return sim - y_obs



# Auto-fitting combining global and local
def fit_parameters_auto(drivers, y_obs, N0):
    # Build bounds list of (low, high) pairs required
    lb = PARAM_BOUNDS["lower"]
    ub = PARAM_BOUNDS["upper"]
    if len(lb) != len(ub):
        raise ValueError("PARAM_BOUNDS lower/upper length mismatch")
    bounds = [(float(lb[i]), float(ub[i])) for i in range(len(lb))]

    # start with differential evolution for global search (wrapped in try/except)
    def loss(x):
        r = residuals_for_params(x, drivers, y_obs, N0)
        # if returns penalty vector, return large scalar
        if np.any(np.abs(r) > 1e9):
            return 1e20
        return float(np.sum(r**2))

    print("Starting global optimization (differential_evolution)...")

    de = differential_evolution(loss, bounds=bounds, maxiter=8, popsize=4, workers=1, polish=False, seed=4415216) #DEV
    x0 = de.x
    print("Global opt finished.")

 
    # Now run local refinement using least_squares; must pass bounds in appropriate form
    lb_arr = np.array([b[0] for b in bounds], dtype=float)
    ub_arr = np.array([b[1] for b in bounds], dtype=float)
    print("Running local refinement (least_squares)...")
    res = least_squares(residuals_for_params, x0, args=(drivers, y_obs, N0), bounds=(lb_arr, ub_arr), xtol=1e-4, ftol=1e-4, verbose=2, max_nfev=100) #DEV
    return res.x, res #xtol=1e-8, ftol=1e-8, verbose=2, max_nfev=500



def run_future_scenarios(
    fitted_params,
    hist_df,
    future_df,
    N0,
    scenarios
):
    results = {}

    for name, spec in scenarios.items():
        params = fitted_params.copy()

        # Apply parameter-level shocks
        for k, (mode, val) in spec.get("params", {}).items():
            if mode == "mult":
                params[param_index[k]] *= val
            elif mode == "add":
                params[param_index[k]] += val

        # Build future p,q,K
        p_f, q_f, K_f = build_future_parameters(params, hist_df, future_df)

        # Concatenate with historical
        p_full = np.concatenate([hist_p_t, p_f])
        q_full = np.concatenate([hist_q_t, q_f])
        K_full = np.concatenate([hist_K_t, K_f])

        sim = simulate_hybrid(p_full, q_full, K_full, N0, len(p_full))

        results[name] = sim

        pd.DataFrame({
            "date": pd.concat([hist_df["date"], future_df["date"]]),
            "steam_sim": sim
        }).to_csv(os.path.join(OUTDIR, f"scenario_{name}.csv"), index=False)
    return results

# mapping param names to indices in parameter vector
param_index = {"p0":0, "a1":1, "a2":2, "q0":3, "b1":4, "k1":5, "k2":6}
param_names = ["p0","a1","a2","q0","b1","k1","k2"]

# ---------------- Main script ----------------
if __name__ == "__main__":
    # 1) Load files
    print("Loading CSVs...")
    dfs = {}
    for key, path in FILES.items():
        try:
            dfs[key] = load_monthly_csv(path, key)
            print(f"Loaded {key} from {path}, range {dfs[key]['date'].min().date()} -> {dfs[key]['date'].max().date()}")
        except Exception as e:
            print(f"Error loading {key} ({path}): {e}")
            raise


    # 2) Build master timeline and merge
    master = create_master_timeline(dfs)
    merged = master.copy()
    # fill/merge using sensible strategies
    merged = merge_and_fill(merged, dfs["population"], "population", "hold_before")
    merged = merge_and_fill(merged, dfs["internet"], "internet", "hold_before")
    merged = merge_and_fill(merged, dfs["twitch"], "twitch", "hold_before")
    merged = merge_and_fill(merged, dfs["steam"], "steam", "hold_before")
    merged = merge_and_fill(merged, dfs["cpi"], "cpi", "hold_before")
    merged = merge_and_fill(merged, dfs["ppi"], "ppi", "hold_before")
    #merged = merged.iloc[96:180] #DEV
    print(merged.tail())
    


    # 3) Normalization (for drivers) -- numeric conversion
    for col in ["population", "internet", "twitch", "steam", "cpi", "ppi"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    # Fill remaining NaNs sensibly (interpolate then back/forward-fill)
    merged[["population","internet","twitch","steam","cpi","ppi"]] = (
        merged[["population","internet","twitch","steam","cpi","ppi"]]
        .interpolate(limit_direction='both')
        .fillna(method='bfill')
        .fillna(method='ffill')
    )

    merged["pop_norm"] = merged["population"] #/ merged["population"].max() #removed population normalization
    merged["internet_norm"] = merged["internet"] / (merged["internet"].max() + 1e-12)
    merged["twitch_norm"] = (merged["twitch"] - merged["twitch"].min()) / (merged["twitch"].max() - merged["twitch"].min() + 1e-9)
    #merged["twitch_norm"] = np.log1p(merged["twitch"]) #experimental log twitch stats
    merged["cpi_norm"] = (merged["cpi"] - merged["cpi"].min()) / (merged["cpi"].max() - merged["cpi"].min() + 1e-9)
    merged["ppi_norm"] = (merged["ppi"] - merged["ppi"].min()) / (merged["ppi"].max() - merged["ppi"].min() + 1e-9)


    
    hist_df, future_df = split_historical_future(merged, FIT_END_YEAR)



    # 4) Prepare fitting arrays (use steam observed over merged timeline)
    drivers = {
        "internet_norm": hist_df["internet_norm"].values,
        "pop_norm": hist_df["pop_norm"].values,
        "twitch_norm": hist_df["twitch_norm"].values,
        "cpi_norm": hist_df["cpi_norm"].values,
        "ppi_norm": hist_df["ppi_norm"].values
    }
    y_obs_hist = hist_df["steam"].values.astype(float)  # units: whatever CSV uses
    if np.any(np.isnan(y_obs_hist)):
        # fill steam NaNs sensibly
        y_obs_hist = pd.Series(y_obs_hist).interpolate().fillna(method='bfill').fillna(method='ffill').values
    N0 = float(y_obs_hist[0])
    t_len = len(y_obs_hist)


    # 5) Fit parameters automatically
    print("Fitting parameters automatically (global + local)...")
    fitted_params, fit_result = fit_parameters_auto(drivers, y_obs_hist, N0)
    print("Fitted params:", fitted_params)
    # save params
    params_dict = dict(zip(param_names, fitted_params.tolist()))
    with open(os.path.join(OUTDIR, "fitted_params.json"), "w") as f:
        json.dump(params_dict, f, indent=2)


    # 6) Simulate with fitted params and save
    p0, a1, a2, q0, b1, k1, k2 = fitted_params
    hist_p_t = p0 + a1 * hist_df["cpi_norm"].values + a2 * hist_df["ppi_norm"].values
    hist_q_t = q0 + b1 * hist_df["twitch_norm"].values
    hist_K_t = k1 * hist_df["pop_norm"].values + k2 * hist_df["internet_norm"].values
    sim_base = simulate_hybrid(hist_p_t, hist_q_t, hist_K_t, N0, t_len)
    out_df = pd.DataFrame({"date": hist_df["date"], "steam_obs": y_obs_hist, "steam_sim_fitted": sim_base})
    out_df.to_csv(os.path.join(OUTDIR, "simulation_fitted_timeseries.csv"), index=False)
    print("Saved fitted timeseries to simulation_fitted_timeseries.csv")

    out_df2 = pd.DataFrame({"date": hist_df["date"], "p_t": hist_p_t, "q_t": hist_q_t, "K_t": hist_K_t})
    out_df2.to_csv(os.path.join(OUTDIR, "simulation_fitted_variables.csv"), index=False)
    # print("p_t (first 10):", p_t[:10])
    # print("q_t (first 10):", q_t[:10])
    # print("K_t (first 10):", K_t[:10])
    # print("min/max p_t:", p_t.min(), p_t.max())
    # print("min/max K_t:", K_t.min(), K_t.max())

    # Save variable plots
    plt.plot(hist_df['date'], hist_p_t); plt.title('p(t)'); plt.savefig(os.path.join(OUTDIR, "1p_t.png"));plt.close()
    plt.plot(hist_df['date'], hist_q_t); plt.title('q(t)'); plt.savefig(os.path.join(OUTDIR, "1q_t.png"));plt.close()
    plt.plot(hist_df['date'], hist_K_t); plt.title('K(t)'); plt.savefig(os.path.join(OUTDIR, "1K_t.png"));plt.close()

    # Generate main plot (obs vs fit)
    plt.figure(figsize=(10,5))
    plt.plot(hist_df["date"], y_obs_hist, label="Observed Steam", linewidth=2)
    plt.plot(hist_df["date"], sim_base, label="Fitted Hybrid Model", linewidth=2)
    plt.xlabel("Date")
    plt.ylabel("Users (units from CSV)")
    plt.title("Observed vs Fitted - Hybrid Logistic-Bass Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "obs_vs_fit.png"))
    plt.close()


    # 7) Extrapolate Variables
    print("Extrapolating drivers beyond", FIT_END_YEAR)

    PARAM_EXTRAP_METHOD = {
    "population": "linear",
    "internet": "logistic",
    "twitch": "exp",
    "cpi": "linear",
    "ppi": "linear"
    }

    last_obs_date = merged[merged["date"].dt.year <= FIT_END_YEAR]["date"].max()
    months_out = (
        (EXTEND_TO_YEAR - last_obs_date.year) * 12
        + (12 - last_obs_date.month)
    )

    for col, method in PARAM_EXTRAP_METHOD.items():
        series = merged.loc[merged["date"] <= last_obs_date, col]
        series.index = merged.loc[merged["date"] <= last_obs_date, "date"]

        ext = extrapolate_driver(series, months_out, method)

        merged.loc[merged["date"] > last_obs_date, col] = (
            ext.loc[ext.index > last_obs_date].values
        )


    scenarios = {
        "baseline": {},
        "high_social": {"params": {"b1": ("mult", 1.5)}},
        "low_macro": {"params": {"a1": ("mult", 0.7), "a2": ("mult", 0.7)}},
        "better_tech": {"params": {"a1": ("mult", 2), "a2": ("mult", 2)}},
    }

    scenario_results = run_future_scenarios(
        fitted_params=fitted_params,
        hist_df=hist_df,
        future_df=future_df,
        N0=N0,
        scenarios=scenarios
    )

    plt.figure(figsize=(10,5))
    plt.plot(hist_df["date"], y_obs_hist[:len(hist_df)], label="Observed", lw=2)

    for name, sim in scenario_results.items():
        plt.plot(
            pd.concat([hist_df["date"], future_df["date"]]),
            sim,
            label=name
        )

    plt.axvline(hist_df["date"].iloc[-1], ls="--", color="k", alpha=0.5)
    plt.legend()
    plt.title("Future Adoption Scenarios")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "future_scenarios.png"))
    plt.close()



    # Done
    print("All outputs written to", OUTDIR)
    
    #print("Reference example report path (for your writeup):", REPORT_PDF_PATH)
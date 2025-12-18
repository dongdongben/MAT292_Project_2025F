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

# CSV names (must exist in same folder or change these paths)
FILES = {
    "population": os.path.join(BASE_DIR, "world_population_04_01.csv"),
    "internet": os.path.join(BASE_DIR, "internet_penetration_04_01.csv"),
    "twitch": os.path.join(BASE_DIR, "twitch_views_04_01.csv"),
    "steam": os.path.join(BASE_DIR, "steam_users_04_01.csv"),
    "cpi": os.path.join(BASE_DIR, "tech_cpi_04_01.csv"),
    "ppi": os.path.join(BASE_DIR, "tech_ppi_04_01.csv")
}

# Path to the example fire report you uploaded (for your report reference)
REPORT_PDF_PATH = "/mnt/data/Example Project Report - A Mathematical Fire Spread Model.pdf"

# Extend horizon (year) for scenario projections
EXTEND_TO_YEAR = 2030

# How many months of history to use when fitting local trend for extrapolation
FIT_WINDOW_MONTHS = 60  # last 3 years trend for extrapolation

# Initial guess / bounds for parameters for fitting (p0,a1,a2,q0,b1,k1,k2)
PARAM_BOUNDS = {
    "lower": [1e-4, 0.0, 0.0, 1e-4, 0.0, 0.0, 0.0],
    "upper": [1.0, 20.0, 20.0, 2.0, 10.0, 50.0, 50.0]
}
# -----------------------------------------------------



def load_monthly_csv(filepath, col_name_hint):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CSV not found: {filepath}")

    df = pd.read_csv(filepath)

    # find date column
    date_col_candidates = [
        c for c in df.columns
        if "date" in c.lower() or "time" in c.lower() or "month" in c.lower()
    ]
    if not date_col_candidates:
        raise ValueError(f"No date-like column in {filepath}")

    date_col = date_col_candidates[0]

    # value column
    val_cols = [c for c in df.columns if c != date_col]
    if not val_cols:
        raise ValueError(f"No value column found in {filepath}")

    val_col = val_cols[0]

    df = df[[date_col, val_col]].rename(columns={date_col: "date", val_col: col_name_hint})

    # --- FIX: CLEAN DATE STRINGS ---
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
    # but we'll extend to EXTEND_TO_YEAR
    end_extended = datetime(EXTEND_TO_YEAR, 12, 1) #DEV, end vs end_extended
    dates = pd.date_range(start=start, end=end_extended, freq="MS")  # month starts
    return pd.DataFrame({"date": dates})



def merge_and_fill(master, df, col, fill_strategy="zero_before"):
    merged = master.merge(df, on="date", how="left")
    # linear interpolate internal missing values
    merged[col] = merged[col].interpolate(method='linear', limit_direction='backward')
    # handle months before earliest available depending on strategy
    if df["date"].min() > merged["date"].min():
        if fill_strategy == "zero_before":
            merged.loc[merged["date"] < df["date"].min(), col] = 0.0
        elif fill_strategy == "hold_before":
            first_val = merged.loc[merged["date"] >= df["date"].min(), col].iloc[0]
            merged.loc[merged["date"] < df["date"].min(), col] = first_val
    return merged



def extrapolate_future(series, months_out, method="linear", fit_window=FIT_WINDOW_MONTHS):
    """
    series: pandas Series indexed by datetime (monthly) with no NaNs
    months_out: number of months to extend beyond last index
    """
    # Ensure index is datetime and monotonic
    series = series.copy()
    series.index = pd.to_datetime(series.index)
    series = series.sort_index()
    n = len(series)
    if n == 0:
        # nothing to extrapolate - return zeros for months_out
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
        # else fallback to linear
    # linear fallback
    lr = LinearRegression()
    lr.fit(t_fit, y_fit)
    future_t = np.arange(n, n+months_out).reshape(-1,1)
    y_future = lr.predict(future_t).ravel()
    future_index = pd.date_range(start=series.index[-1] + relativedelta(months=1), periods=months_out, freq='MS')
    ext = pd.Series(y_future, index=future_index)
    return pd.concat([series, ext])



# ODE simulation (N in same units as y_obs)
def simulate_hybrid(p_t, q_t, K_t, N0, t_len):
    """
    Simulate the hybrid ODE over t_len monthly steps.
    Always returns a numpy array of shape (t_len,) even if the solver fails
    (on failure returns a large penalty array).
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
            method="RK45"
        )
    except Exception as e:
        # If solver crashes, return large penalty array
        warnings.warn(f"ODE solver exception: {e}; returning penalty array.")
        return np.ones(t_len) * 1e12

    # Validate solution shape and status
    if sol.status < 0 or sol.y.shape[1] != t_len:
        # solver failed or returned fewer points; return large penalty to reject these params
        warnings.warn("ODE solver did not return expected number of points; returning penalty array.")
        return np.ones(t_len) * 1e12

    # success
    return sol.y[0]



# Objective residual function for least-squares
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

    p_t = np.maximum(p_t, 1e-9) #added
    q_t = np.maximum(q_t, 0.0)
    K_t = np.maximum(K_t, np.max(y_obs)*1.05)  # ensure K at least slightly above observed max

    # safeguard K_t
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
    # Build bounds list of (low, high) pairs required by differential_evolution
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
    try: #40, 8 
        de = differential_evolution(loss, bounds=bounds, maxiter=8, popsize=4, workers=1, polish=False) #DEV
        x0 = de.x
        print("Global opt finished.")
    except Exception as e:
        warnings.warn(f"differential_evolution failed: {e}; falling back to random initial guess for local search.")
        # fallback: use midpoint of bounds as starting guess
        x0 = np.array([(b[0] + b[1]) / 2.0 for b in bounds], dtype=float)

    # Now run local refinement using least_squares; must pass bounds in appropriate form
    lb_arr = np.array([b[0] for b in bounds], dtype=float)
    ub_arr = np.array([b[1] for b in bounds], dtype=float)
    print("Running local refinement (least_squares)...")
    res = least_squares(residuals_for_params, x0, args=(drivers, y_obs, N0), bounds=(lb_arr, ub_arr), xtol=1e-4, ftol=1e-4, verbose=2, max_nfev=100) #DEV
    return res.x, res #xtol=1e-8, ftol=1e-8, verbose=2, max_nfev=500



# Scenario runner (unchanged)
def run_scenarios(base_params, drivers_df, N0, t_len, scenarios):
    results = {}
    for name, modifier in scenarios.items():
        x = base_params.copy()
        for k, v in modifier.items():
            if isinstance(v, tuple) and len(v) == 2:
                typ, val = v
                if typ == "mult":
                    x[param_index[k]] *= val
                elif typ == "add":
                    x[param_index[k]] += val
            else:
                x[param_index[k]] *= v
        p0, a1, a2, q0, b1, k1, k2 = x
        p_t = p0 + a1 * drivers_df["cpi_norm"].values + a2 * drivers_df["ppi_norm"].values
        q_t = q0 + b1 * drivers_df["twitch_norm"].values
        K_t = k1 * drivers_df["pop_norm"].values + k2 * drivers_df["internet_norm"].values
        sim = simulate_hybrid(p_t, q_t, K_t, N0, t_len)
        results[name] = {"params": x.copy(), "sim": sim}
        out_df = pd.DataFrame({"date": drivers_df["date"], "steam_sim_millions": sim})
        out_df.to_csv(os.path.join(OUTDIR, f"scenario_{name}.csv"), index=False)
    return results



# Sensitivity analysis (unchanged)
def sensitivity_one_at_a_time(base_params, drivers_df, N0, t_len, param_names, factors=[0.5, 0.75, 1.0, 1.25, 1.5]):
    sens_results = {}
    for i, pname in enumerate(param_names):
        sens_results[pname] = {}
        for factor in factors:
            x = base_params.copy()
            x[i] *= factor
            p0, a1, a2, q0, b1, k1, k2 = x
            p_t = p0 + a1 * drivers_df["cpi_norm"].values + a2 * drivers_df["ppi_norm"].values
            q_t = q0 + b1 * drivers_df["twitch_norm"].values
            K_t = k1 * drivers_df["pop_norm"].values + k2 * drivers_df["internet_norm"].values
            sim = simulate_hybrid(p_t, q_t, K_t, N0, t_len)
            sens_results[pname][factor] = sim
    # Save CSV summaries: end-of-horizon values and timeseries
    for pname, table in sens_results.items():
        rows = []
        for factor, sim in table.items():
            rows.append({"param": pname, "factor": factor, "final_gamers_millions": float(sim[-1])})
        pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, f"sensitivity_{pname}_summary.csv"), index=False)
        ts_df = pd.DataFrame({str(f): table[f] for f in table})
        ts_df.insert(0, "date", drivers_df["date"].values)
        ts_df.to_csv(os.path.join(OUTDIR, f"sensitivity_{pname}_timeseries.csv"), index=False)
    return sens_results



# Heatmap (unchanged)
def heatmap_2d(base_params, drivers_df, N0, t_len, param_x, param_y, x_range, y_range):
    grid = np.zeros((len(y_range), len(x_range)))
    for i, yv in enumerate(y_range):
        for j, xv in enumerate(x_range):
            x = base_params.copy()
            x[param_x] = xv
            x[param_y] = yv
            p0, a1, a2, q0, b1, k1, k2 = x
            p_t = p0 + a1 * drivers_df["cpi_norm"].values + a2 * drivers_df["ppi_norm"].values
            q_t = q0 + b1 * drivers_df["twitch_norm"].values
            K_t = k1 * drivers_df["pop_norm"].values + k2 * drivers_df["internet_norm"].values
            sim = simulate_hybrid(p_t, q_t, K_t, N0, t_len)
            grid[i,j] = float(sim[-1])
    heat_df = pd.DataFrame(grid, index=y_range, columns=x_range)
    heat_df.to_csv(os.path.join(OUTDIR, f"heatmap_{param_x}_{param_y}.csv"))
    plt.figure(figsize=(8,6))
    plt.imshow(grid, origin='lower', aspect='auto', interpolation='nearest')
    plt.colorbar(label='Final gamers (millions)')
    plt.xlabel(f'param_{param_x}')
    plt.ylabel(f'param_{param_y}')
    plt.title(f'Final gamers heatmap param_{param_y} vs param_{param_x}')
    plt.savefig(os.path.join(OUTDIR, f"heatmap_{param_x}_{param_y}.png"))
    plt.close()
    return grid



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

    # 3) Extrapolate into future where necessary (use FIT_WINDOW_MONTHS)
    #last_date = merged["date"].max()
    last_observed_date = max(dfs[key]["date"].max() for key in dfs)
    #months_out = (EXTEND_TO_YEAR - last_date.year) * 12 + (12 - last_date.month)
    months_out = ((EXTEND_TO_YEAR - last_observed_date.year) * 12 + (12 - last_observed_date.month))
    if months_out > 0:
        print(f"Extending each driver by {months_out} months to {EXTEND_TO_YEAR}-12...")
        for col in ["population", "internet", "twitch", "steam", "cpi", "ppi"]:
            ser = pd.Series(merged[col].values, index=merged["date"])
            # KEEP ONLY OBSERVED DATA
            observed = ser.dropna()

            if len(observed) < 3:
                raise ValueError(f"Not enough data to extrapolate {col}")

            months_out = len(ser) - len(observed)

            ext = extrapolate_future(
                observed,
                months_out,
                method="linear",
                fit_window=FIT_WINDOW_MONTHS
            )
            #ser = pd.Series(merged[col].values, index=merged["date"])
            #ext = extrapolate_future(ser, months_out, method="linear", fit_window=FIT_WINDOW_MONTHS)
            # replace merged column with extended; ensure length matches master timeline
            merged[col] = ext.reindex(merged["date"]).values

    # 4) Normalization (for drivers) -- robust numeric conversion
    for col in ["population", "internet", "twitch", "steam", "cpi", "ppi"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    # Fill remaining NaNs sensibly (interpolate then back/forward-fill)
    merged[["population","internet","twitch","steam","cpi","ppi"]] = (
        merged[["population","internet","twitch","steam","cpi","ppi"]]
        .interpolate(limit_direction='both')
        .fillna(method='bfill')
        .fillna(method='ffill')
    )

    merged["pop_norm"] = merged["population"] #/ merged["population"].max()
    merged["internet_norm"] = merged["internet"] / (merged["internet"].max() + 1e-12)
    merged["twitch_norm"] = (merged["twitch"] - merged["twitch"].min()) / (merged["twitch"].max() - merged["twitch"].min() + 1e-9)
    #merged["twitch_norm"] = np.log1p(merged["twitch"]) #experimental log twitch stats
    merged["cpi_norm"] = (merged["cpi"] - merged["cpi"].min()) / (merged["cpi"].max() - merged["cpi"].min() + 1e-9)
    merged["ppi_norm"] = (merged["ppi"] - merged["ppi"].min()) / (merged["ppi"].max() - merged["ppi"].min() + 1e-9)



    # 5) Prepare fitting arrays (use steam observed over merged timeline)
    drivers = {
        "internet_norm": merged["internet_norm"].values,
        "pop_norm": merged["pop_norm"].values,
        "twitch_norm": merged["twitch_norm"].values,
        "cpi_norm": merged["cpi_norm"].values,
        "ppi_norm": merged["ppi_norm"].values
    }
    y_obs = merged["steam"].values.astype(float)  # units: whatever CSV uses
    if np.any(np.isnan(y_obs)):
        # fill steam NaNs sensibly
        y_obs = pd.Series(y_obs).interpolate().fillna(method='bfill').fillna(method='ffill').values
    N0 = float(y_obs[0])
    t_len = len(y_obs)

    # 6) Fit parameters automatically
    print("Fitting parameters automatically (global + local)...")
    fitted_params, fit_result = fit_parameters_auto(drivers, y_obs, N0)
    print("Fitted params:", fitted_params)
    # save params
    params_dict = dict(zip(param_names, fitted_params.tolist()))
    with open(os.path.join(OUTDIR, "fitted_params.json"), "w") as f:
        json.dump(params_dict, f, indent=2)

    # 7) Simulate with fitted params and save
    p0, a1, a2, q0, b1, k1, k2 = fitted_params
    p_t = p0 + a1 * merged["cpi_norm"].values + a2 * merged["ppi_norm"].values
    q_t = q0 + b1 * merged["twitch_norm"].values
    K_t = k1 * merged["pop_norm"].values + k2 * merged["internet_norm"].values
    sim_base = simulate_hybrid(p_t, q_t, K_t, N0, t_len)
    out_df = pd.DataFrame({"date": merged["date"], "steam_obs": y_obs, "steam_sim_fitted": sim_base})
    out_df.to_csv(os.path.join(OUTDIR, "simulation_fitted_timeseries.csv"), index=False)
    print("Saved fitted timeseries to simulation_fitted_timeseries.csv")

    # after fit, inspect time-varying coefficients
    p_t = p0 + a1 * merged["cpi_norm"].values + a2 * merged["ppi_norm"].values
    q_t = q0 + b1 * merged["twitch_norm"].values
    K_t = k1 * merged["pop_norm"].values + k2 * merged["internet_norm"].values

    print("p_t (first 10):", p_t[:10])
    print("q_t (first 10):", q_t[:10])
    print("K_t (first 10):", K_t[:10])
    print("min/max p_t:", p_t.min(), p_t.max())
    print("min/max K_t:", K_t.min(), K_t.max())

    # Plot them
    plt.plot(merged['date'], p_t); plt.title('p(t)'); plt.savefig(os.path.join(OUTDIR, "1p_t.png"));plt.close()
    plt.plot(merged['date'], q_t); plt.title('q(t)'); plt.savefig(os.path.join(OUTDIR, "1q_t.png"));plt.close()
    plt.plot(merged['date'], K_t); plt.title('K(t)'); plt.savefig(os.path.join(OUTDIR, "1K_t.png"));plt.close()

    # 8) Generate main plot (obs vs fit)
    plt.figure(figsize=(10,5))
    plt.plot(merged["date"], y_obs, label="Observed Steam", linewidth=2)
    plt.plot(merged["date"], sim_base, label="Fitted Hybrid Model", linewidth=2)
    plt.xlabel("Date")
    plt.ylabel("Users (units from CSV)")
    plt.title("Observed vs Fitted - Hybrid Logistic-Bass Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "obs_vs_fit.png"))
    plt.close()

    # # 9) Scenarios (predefined)
    # base_params = fitted_params.copy()
    # scenarios = {
    #     "baseline": {},
    #     "tech_acceleration": {"a1": ("mult", 1.5), "a2": ("mult", 1.5), "k2": ("mult", 1.2)},
    #     "economic_slowdown": {"a1": ("mult", 0.6), "a2": ("mult", 0.6), "k2": ("mult", 0.9)},
    #     "disruptive_cloud": {"b1": ("mult", 1.5), "k1": ("mult", 1.1), "k2": ("mult", 1.2)},
    #     "social_boom": {"b1": ("mult", 2.0)}
    # }
    # base_list = base_params.tolist()
    # scenario_results = run_scenarios(base_list, merged, N0, t_len, scenarios)
    # print("Scenarios complete. CSVs saved per scenario.")

    # # 10) Sensitivity analysis (one-at-a-time)
    # sens = sensitivity_one_at_a_time(base_list, merged, N0, t_len, param_names)
    # print("One-at-a-time sensitivity complete.")

    # # 11) Heatmap for b1 vs k2
    # b1_val = fitted_params[param_index["b1"]]
    # k2_val = fitted_params[param_index["k2"]]
    # b1_range = np.linspace(max(0, b1_val*0.5), b1_val*1.5, 10)
    # k2_range = np.linspace(max(0, k2_val*0.5), k2_val*1.5, 10)
    # grid = heatmap_2d(base_list, merged, N0, t_len, param_index["b1"], param_index["k2"], b1_range, k2_range)
    # print("Heatmap complete.")

    # # 12) Save key parameters and scenario summaries for report (CSV)
    # summary_rows = []
    # for sname, sres in scenario_results.items():
    #     final_val = float(sres["sim"][-1])
    #     summary_rows.append({"scenario": sname, "final_gamers": final_val, "params": json.dumps(sres["params"])})
    # pd.DataFrame(summary_rows).to_csv(os.path.join(OUTDIR, "scenario_summary.csv"), index=False)

    # # Save fitted params human-readable
    # pd.DataFrame([params_dict]).to_csv(os.path.join(OUTDIR, "fitted_params_table.csv"), index=False)

    # 13) Done
    print("All outputs written to", OUTDIR)
    print("Reference example report path (for your writeup):", REPORT_PDF_PATH)
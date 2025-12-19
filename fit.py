import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import os

# ---------- CONFIG ----------
# Filenames: replace these with your own CSV paths
TWITCH_FILE = "twitch_views_12_09.csv"
POP_FILE = "world_population_04_01.csv"
INET_FILE = "internet_penetration_04_01.csv"
STEAM_FILE = "steam_users_04_01.csv"

OUTDIR = "output_fitted"
os.makedirs(OUTDIR, exist_ok=True)

# ---------- Helper: load and standardize CSV ----------
def load_monthly_csv(filename, col_name):
    df = pd.read_csv(filename)
    #Try to detect date column automatically
    date_col = [c for c in df.columns if "date" in c.lower()][0]
    val_col = [c for c in df.columns if c != date_col][0]
    df = df.rename(columns={val_col: col_name})
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    return df[[date_col, col_name]]

# def load_monthly_csv(filename, col_name):
#     df = pd.read_csv(filename)
#     # Try to detect date column automatically
#     date_col = [c for c in df.columns if "date" in c.lower()][0]
#     val_col = [c for c in df.columns if c != date_col][0]
#     df = df.rename(columns={val_col: col_name})
#     df[date_col] = pd.to_datetime(df[date_col])
#     df = df.sort_values(date_col).reset_index(drop=True)
#     return df[[date_col, col_name]]

# ---------- Load all four datasets ----------
twitch_df = load_monthly_csv(TWITCH_FILE, "twitch")
pop_df = load_monthly_csv(POP_FILE, "population")
inet_df = load_monthly_csv(INET_FILE, "internet")
steam_df = load_monthly_csv(STEAM_FILE, "steam")

# ---------- Create global monthly timeline ----------
start_date = min(df["date"].min() for df in [pop_df, inet_df, twitch_df, steam_df])
end_date   = max(df["date"].max() for df in [pop_df, inet_df, twitch_df, steam_df])
dates = pd.date_range(start=start_date, end=end_date, freq="MS")  # month start
full_df = pd.DataFrame({"date": dates})

# ---------- Merge all with interpolation ----------
def merge_with_fill(base_df, new_df, col, fill_strategy="zero_before"):
    merged = base_df.merge(new_df, on="date", how="left")
    # Interpolate missing mid-range values
    merged[col] = merged[col].interpolate(limit_direction="both")
    # Fill before dataset start
    if fill_strategy == "zero_before":
        first_valid = new_df["date"].min()
        merged.loc[merged["date"] < first_valid, col] = 0
    elif fill_strategy == "hold_before":
        first_val = new_df[col].iloc[0]
        merged[col].fillna(first_val, inplace=True)
    return merged

df = full_df.copy()
df = merge_with_fill(df, pop_df, "population", fill_strategy="hold_before")
df = merge_with_fill(df, inet_df, "internet", fill_strategy="hold_before")
df = merge_with_fill(df, twitch_df, "twitch", fill_strategy="zero_before")
df = merge_with_fill(df, steam_df, "steam", fill_strategy="hold_before")

# ---------- Normalize and prepare model variables ----------
df["pop_millions"] = df["population"] / 1e6
df["internet_norm"] = (df["internet"] - df["internet"].min()) / (df["internet"].max() - df["internet"].min())
df["twitch_norm"] = (df["twitch"] - df["twitch"].min()) / (df["twitch"].max() - df["twitch"].min())

y_obs = df["steam"].values
internet_drive = df["internet_norm"].values
twitch_drive = df["twitch_norm"].values
pop_m = df["pop_millions"].values

# ---------- Hybrid ODE ----------
def simulate_hybrid_series(p_t_series, q_t_series, K_t_series, N0):
    def rhs(t, N):
        idx = int(np.clip(np.floor(t), 0, len(p_t_series)-1))
        p = p_t_series[idx]
        q = q_t_series[idx]
        K = K_t_series[idx]
        return (p + q*(N/K))*N*(1 - N/K)
    sol = solve_ivp(rhs, (0, len(p_t_series)-1), [N0], t_eval=np.arange(len(p_t_series)), max_step=1.0)
    return sol.y[0]

# ---------- Residuals for fitting ----------
N0 = y_obs[0]
def residuals(x):
    p0, a, q0, b, k_frac = x
    if k_frac <= 0: return 1e6 * np.ones_like(y_obs)
    p_t = p0 + a * internet_drive
    q_t = q0 + b * twitch_drive
    K_t = pop_m * k_frac
    N_sim = simulate_hybrid_series(p_t, q_t, K_t, N0)
    return N_sim - y_obs

# ---------- Fit parameters ----------
x0 = np.array([0.001, 0.005, 0.01, 0.02, 0.02])
bounds = ([0,0,0,0,1e-4], [0.01,0.05,0.2,0.2,0.1])
res = least_squares(residuals, x0, bounds=bounds, xtol=1e-8, ftol=1e-8, verbose=2)

params = res.x
print("Fitted parameters:", params)

# ---------- Simulate fitted curve ----------
p_t_fit = params[0] + params[1]*internet_drive
q_t_fit = params[2] + params[3]*twitch_drive
K_t_fit = pop_m * params[4]
N_fit = simulate_hybrid_series(p_t_fit, q_t_fit, K_t_fit, N0)

# ---------- Save outputs ----------
out_csv = os.path.join(OUTDIR, "aligned_and_fitted.csv")
pd.DataFrame({"date": df["date"], "steam_obs": y_obs, "steam_fit": N_fit}).to_csv(out_csv, index=False)

plt.figure(figsize=(10,5))
plt.plot(df["date"], y_obs, label="Observed Steam Users")
plt.plot(df["date"], N_fit, label="Fitted Model")
plt.title("Logistic–Bass Hybrid Fit (Automatic Alignment)")
plt.xlabel("Date")
plt.ylabel("Users (millions)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "fit_plot.png"))
plt.close()

print(f"✅ Fit complete. Results saved to {OUTDIR}")

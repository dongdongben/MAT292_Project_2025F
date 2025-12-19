import numpy as np
import pandas as pd
import os
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# -------------------------------
# 1. Helper: auto-load CSV
# -------------------------------
def load_monthly_csv(filename, col_name):
    df = pd.read_csv(filename)

    date_col = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()][0]
    val_col = [c for c in df.columns if c != date_col][0]

    df = df.rename(columns={val_col: col_name})
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    return df[[date_col, col_name]]


# -------------------------------
# 2. File paths
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

POP_FILE    = os.path.join(BASE_DIR, "world_population.csv")
INET_FILE   = os.path.join(BASE_DIR, "internet_penetration.csv")
TWITCH_FILE = os.path.join(BASE_DIR, "twitch_views.csv")
STEAM_FILE  = os.path.join(BASE_DIR, "steam_users.csv")
CPI_FILE    = os.path.join(BASE_DIR, "tech_cpi_04_01.csv")
PPI_FILE    = os.path.join(BASE_DIR, "tech_ppi_04_01.csv")

OUTDIR = os.path.join(BASE_DIR, "output_fitted")
os.makedirs(OUTDIR, exist_ok=True)


# -------------------------------
# 3. Load CSVs
# -------------------------------
pop_df    = load_monthly_csv(POP_FILE,  "population")
inet_df   = load_monthly_csv(INET_FILE, "internet")
twitch_df = load_monthly_csv(TWITCH_FILE, "twitch")
steam_df  = load_monthly_csv(STEAM_FILE, "steam")
cpi_df    = load_monthly_csv(CPI_FILE, "cpi")
ppi_df    = load_monthly_csv(PPI_FILE, "ppi")


# -------------------------------
# 4. Create unified monthly timeline
# -------------------------------
start_date = min(df["date"].min() for df in 
                 [pop_df, inet_df, twitch_df, steam_df, cpi_df, ppi_df])
end_date = max(df["date"].max() for df in 
               [pop_df, inet_df, twitch_df, steam_df, cpi_df, ppi_df])

dates = pd.date_range(start=start_date, end=end_date, freq="MS")
full_df = pd.DataFrame({"date": dates})


# -------------------------------
# 5. Merge with interpolation + synthetic fill
# -------------------------------
def merge_with_fill(base_df, new_df, col, fill_strategy="zero_before"):
    merged = base_df.merge(new_df, on="date", how="left")
    merged[col] = merged[col].interpolate(limit_direction="both")

    # Fill periods before dataset begins
    start = new_df["date"].min()
    if fill_strategy == "zero_before":
        merged.loc[merged["date"] < start, col] = 0
    elif fill_strategy == "hold_before":
        first_val = new_df[col].iloc[0]
        merged.loc[merged["date"] < start, col] = first_val

    return merged

df = full_df.copy()
df = merge_with_fill(df, pop_df, "population", "hold_before")
df = merge_with_fill(df, inet_df, "internet", "hold_before")
df = merge_with_fill(df, twitch_df, "twitch", "zero_before")
df = merge_with_fill(df, steam_df, "steam", "hold_before")
df = merge_with_fill(df, cpi_df, "cpi", "hold_before")
df = merge_with_fill(df, ppi_df, "ppi", "hold_before")


# -------------------------------
# 6. Normalize variables where needed
# -------------------------------
df["pop_norm"] = df["population"] / df["population"].max()
df["inet_norm"] = df["internet"] / df["internet"].max()
df["twitch_norm"] = (df["twitch"] - df["twitch"].min()) / (df["twitch"].max() - df["twitch"].min())
df["cpi_norm"] = (df["cpi"] - df["cpi"].min()) / (df["cpi"].max() - df["cpi"].min())
df["ppi_norm"] = (df["ppi"] - df["ppi"].min()) / (df["ppi"].max() - df["ppi"].min())


# -------------------------------
# 7. Prepare model inputs
# -------------------------------
y_obs = df["steam"].values

internet_drive = df["inet_norm"].values
population_drive = df["pop_norm"].values
twitch_drive = df["twitch_norm"].values
cpi_drive = df["cpi_norm"].values
ppi_drive = df["ppi_norm"].values


# -------------------------------
# 8. Hybrid model simulation
# -------------------------------
def simulate_series(p_t, q_t, K_t, N0):
    def rhs(t, N):
        idx = int(np.clip(int(t), 0, len(p_t)-1))
        return (p_t[idx] + q_t[idx] * (N/K_t[idx])) * N * (1 - N/K_t[idx])

    sol = solve_ivp(rhs, (0, len(p_t)-1), [N0], t_eval=np.arange(len(p_t)), max_step=1.0)
    return sol.y[0]


# -------------------------------
# 9. Residuals with NEW PARAMETERS
# -------------------------------
N0 = y_obs[0]

def residuals(x):
    p0, a1, a2, q0, b1, k1, k2 = x

    p_t = p0 + a1*cpi_drive + a2*ppi_drive
    q_t = q0 + b1*twitch_drive
    K_t = k1*population_drive + k2*internet_drive

    if np.any(K_t <= 0):
        return 1e6 * np.ones_like(y_obs)

    N_sim = simulate_series(p_t, q_t, K_t, N0)
    return N_sim - y_obs


# -------------------------------
# 10. Fit parameters
# -------------------------------
x0 = np.array([0.002, 0.01, 0.01, 0.02, 0.05, 0.5, 0.5])
bounds = (
    [0, 0, 0, 0, 0, 0, 0],
    [0.02, 0.5, 0.5, 0.3, 0.3, 5.0, 5.0]
)

res = least_squares(residuals, x0, bounds=bounds, verbose=2)
params = res.x
print("Fitted parameters:\n", params)


# -------------------------------
# 11. Simulate best fit
# -------------------------------
p_t = params[0] + params[1]*cpi_drive + params[2]*ppi_drive
q_t = params[3] + params[4]*twitch_drive
K_t = params[5]*population_drive + params[6]*internet_drive

N_fit = simulate_series(p_t, q_t, K_t, N0)

out = pd.DataFrame({"date": df["date"], "steam_obs": y_obs, "steam_fit": N_fit})
out.to_csv(os.path.join(OUTDIR, "fit_results.csv"), index=False)

plt.figure(figsize=(10,5))
plt.plot(df["date"], y_obs, label="Observed", linewidth=2)
plt.plot(df["date"], N_fit, label="Fitted", linewidth=2)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "fit_plot.png"))

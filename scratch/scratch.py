# Single run for q=0.5 — save artifacts and print paths only
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(3030)

def simulate_1d_process(n: int, burn: int, q: float, sigma: float, y0: float = 0.0):
    T = n + burn
    Y = np.empty(T + 1, dtype=float)
    Y[0] = y0
    for t in range(T):
        mu = np.sign(Y[t]) * (np.abs(Y[t]) ** q)
        Y[t + 1] = mu + rng.normal(0.0, sigma)
    return Y[burn:]

def local_wls_predict_vectorized(y_tr, y1_tr, yq, h: float, ridge: float = 0.0):
    diff2 = (y_tr[None, :] - yq[:, None]) ** 2
    W = np.exp(- diff2 / (2.0 * (h**2) + 1e-18))
    S_w   = np.sum(W, axis=1)
    S_wy  = np.sum(W * y_tr[None, :], axis=1)
    S_wy2 = np.sum(W * (y_tr[None, :] ** 2), axis=1)
    S_wy1 = np.sum(W * y1_tr[None, :], axis=1)
    S_wyy1= np.sum(W * (y_tr[None, :] * y1_tr[None, :]), axis=1)
    A11 = S_w + ridge
    A12 = S_wy
    A22 = S_wy2 + ridge
    det = A11 * A22 - A12 * A12 + 1e-18
    inv11 =  A22 / det
    inv12 = -A12 / det
    inv22 =  A11 / det
    a = inv11 * S_wy1 + inv12 * S_wyy1
    b = inv12 * S_wy1 + inv22 * S_wyy1
    preds = a + b * yq
    return preds

def excess_action_mse(y, h, ridge, sigma2, n_queries=160):
    y0, y1 = y[:-1], y[1:]
    y_tr, y_te, y1_tr, y1_te = train_test_split(y0, y1, test_size=0.3, random_state=123)
    qs = np.linspace(1, 99, n_queries)
    yq = np.percentile(y_tr, qs)
    yq_pred = local_wls_predict_vectorized(y_tr, y1_tr, yq, h=h, ridge=ridge)
    idx = np.searchsorted(yq, y_te)
    idx = np.clip(idx, 0, len(yq) - 1)
    y1_pred = yq_pred[idx]
    mse = np.mean((y1_te - y1_pred) ** 2)
    excess = max(mse - sigma2, 1e-16)
    return excess

def robust_loglog_slope(x, y):
    X = np.log(x).reshape(-1, 1)
    Y = np.log(y)
    model = HuberRegressor(epsilon=1.35, alpha=0.0).fit(X, Y)
    return float(model.coef_[0])

def run_and_save(q, sigma=1e-3, n_list=(5000, 10000, 20000), burn=1000, h_grid=None, tag=""):
    if h_grid is None:
        h_grid = np.geomspace(0.07, 1.5, 11)
    y_full = simulate_1d_process(n=max(n_list), burn=burn, q=q, sigma=sigma)
    sigma2 = sigma**2

    loc = []
    for h in h_grid:
        loc.append((h, excess_action_mse(y_full, h, 0.0, sigma2, n_queries=150)))
    loc = np.array(loc)
    k = len(h_grid)
    mid = slice(int(0.25*k), int(0.85*k))
    beta_hat = robust_loglog_slope(loc[mid,0], loc[mid,1])

    plt.figure(figsize=(6,4))
    plt.loglog(loc[:,0], loc[:,1], marker="o")
    plt.xlabel("Bandwidth h")
    plt.ylabel("Excess action A(h) - A0")
    plt.title(f"Locality curve (q={q}, beta≈{beta_hat:.3f})")
    plt.tight_layout()
    loc_png = f"./data/locality_curve_q{q}_{tag}.png"
    plt.savefig(loc_png, dpi=160)
    plt.show()

    learn = []
    for n in n_list:
        y_n = y_full[:n+1]
        vals = []
        for h in h_grid:
            vals.append(excess_action_mse(y_n, h, 0.0, sigma2, n_queries=130))
        vals = np.array(vals)
        h_star = h_grid[np.argmin(vals)]
        learn.append((n, np.min(vals), h_star))
    learn = np.array(learn, dtype=float)
    alpha_hat = -robust_loglog_slope(learn[:,0], learn[:,1])

    plt.figure(figsize=(6,4))
    plt.loglog(learn[:,0], learn[:,1], marker="s")
    plt.xlabel("n")
    plt.ylabel("min_h Excess action")
    plt.title(f"Learning curve (q={q}, alpha≈{alpha_hat:.3f})")
    plt.tight_layout()
    learn_png = f"./data/learning_curve_q{q}_{tag}.png"
    plt.savefig(learn_png, dpi=160)
    plt.show()

    s_hat = beta_hat/2
    d_hat = (beta_hat*(1-alpha_hat))/alpha_hat if alpha_hat>1e-6 else np.nan
    summary = pd.DataFrame([{
        "q_true(s)": q, "beta_hat": beta_hat, "s_hat": s_hat,
        "alpha_hat": alpha_hat, "d_hat": d_hat
    }])
    summary_csv = f"./data/tier1_summary_q{q}_{tag}.csv"
    summary.to_csv(summary_csv, index=False)

    loc_df = pd.DataFrame({"h": loc[:,0], "excess": loc[:,1]})
    learn_df = pd.DataFrame({"n": learn[:,0], "min_excess": learn[:,1], "h_star": learn[:,2]})
    loc_csv = f"./data/tier1_locality_q{q}_{tag}.csv"
    learn_csv = f"./data/tier1_learning_q{q}_{tag}.csv"
    loc_df.to_csv(loc_csv, index=False)
    learn_df.to_csv(learn_csv, index=False)

    print("SUMMARY_CSV:", summary_csv)
    print("LOCALITY_PNG:", loc_png)
    print("LEARNING_PNG:", learn_png)
    print("LOCALITY_CSV:", loc_csv)
    print("LEARNING_CSV:", learn_csv)
    print("ESTIMATES:", {"beta_hat": beta_hat, "alpha_hat": alpha_hat, "s_hat": s_hat, "d_hat": d_hat})

run_and_save(q=0.5, sigma=1e-3, tag="Q05")

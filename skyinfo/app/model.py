"""24h temperature ML model: gradient boosting (mean + 10%/90% quantiles).

One model per location, trained on ~2 years of hourly history. A forecast
horizon `h` (1..24) is a feature, so a single model predicts every hour ahead.
"""
import threading
import pathlib
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from . import data

DIR = pathlib.Path(__file__).resolve().parent.parent / "models"
DIR.mkdir(exist_ok=True)
HOR = 24
_state, _lock = {}, threading.Lock()


def _key(lat, lon):
    return f"{lat:.2f}_{lon:.2f}"


def _frame(h):
    df = pd.DataFrame({k: h[k] for k in ["time"] + data.FEATS})
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")


def _feats(df):
    t = df["temperature_2m"]
    X = pd.DataFrame({"t": t, "hum": df["relative_humidity_2m"], "p": df["surface_pressure"],
                      "dp3": df["surface_pressure"].diff(3), "cloud": df["cloud_cover"],
                      "wind": df["wind_speed_10m"]})
    for lag in (1, 3, 6, 12, 24):
        X[f"t_l{lag}"] = t.shift(lag)
    return X


def _expand(X, h):
    Z = X.copy()
    Z["h"] = h
    tt = X.index + pd.Timedelta(hours=h)
    Z["hs"] = np.sin(2 * np.pi * tt.hour / 24)
    Z["hc"] = np.cos(2 * np.pi * tt.hour / 24)
    Z["ds"] = np.sin(2 * np.pi * tt.dayofyear / 365.25)
    Z["dc"] = np.cos(2 * np.pi * tt.dayofyear / 365.25)
    return Z


def train(lat, lon, hist=None):
    df = _frame(hist if hist is not None else data.history(lat, lon)).dropna()
    X, t = _feats(df), df["temperature_2m"]
    parts = []
    for h in range(1, HOR + 1):
        Z = _expand(X, h)
        Z["y"] = t.shift(-h)
        parts.append(Z.dropna(subset=["y", "t_l24"]))
    A = pd.concat(parts).sort_index()
    cut = A.index[int(len(A) * 0.9)]
    tr, va = A[A.index < cut], A[A.index >= cut]
    if len(tr) > 160_000:
        tr = tr.sample(160_000, random_state=0)
    xc = [c for c in A.columns if c != "y"]
    mods = {
        "mean": HGB(max_iter=250, learning_rate=0.08).fit(tr[xc], tr["y"]),
        "lo": HGB(loss="quantile", quantile=0.1, max_iter=150).fit(tr[xc], tr["y"]),
        "hi": HGB(loss="quantile", quantile=0.9, max_iter=150).fit(tr[xc], tr["y"]),
    }
    mae = float(np.abs(mods["mean"].predict(va[xc]) - va["y"]).mean())
    bundle = {**mods, "cols": xc, "mae": round(mae, 2), "rows": int(len(A))}
    joblib.dump(bundle, DIR / f"{_key(lat, lon)}.joblib")
    return bundle


def _bg(lat, lon, k):
    try:
        train(lat, lon)
        _state[k] = "ready"
    except Exception as e:  # keep serving NWP-only if training fails
        _state[k] = f"error: {e}"


def status(lat, lon):
    """Return model status; kicks off background training when none exists."""
    k, f = _key(lat, lon), DIR / f"{_key(lat, lon)}.joblib"
    if f.exists():
        b = joblib.load(f)
        return {"ready": True, "mae": b["mae"], "rows": b["rows"]}
    with _lock:
        if _state.get(k) != "training":
            _state[k] = "training"
            threading.Thread(target=_bg, args=(lat, lon, k), daemon=True).start()
    return {"ready": False, "state": _state.get(k, "training")}


def predict(lat, lon, hourly, i):
    """Predict next 24h from the latest observation at index i of the hourly series."""
    b = joblib.load(DIR / f"{_key(lat, lon)}.joblib")
    df = _frame({k: v[: i + 1] for k, v in hourly.items() if k in ["time"] + data.FEATS})
    X = _feats(df).iloc[[-1]]
    Z = pd.concat([_expand(X, h) for h in range(1, HOR + 1)])[b["cols"]]
    mean, lo, hi = (b[m].predict(Z) for m in ("mean", "lo", "hi"))
    r = lambda a: [round(float(x), 1) for x in a]
    lo, hi = np.minimum(lo, mean), np.maximum(hi, mean)
    return {"mean": r(mean), "lo": r(lo), "hi": r(hi)}


if __name__ == "__main__":
    import sys
    lat, lon = float(sys.argv[1]), float(sys.argv[2])
    print(train(lat, lon)["mae"], "°C validation MAE")

# ============================================================
# ETH Forecast (No-Leak, Direction-Aware, Never-Empty, Future Forecast)
# - Horizons: 7/30/90 (log-return -> price)
# - Models: LGBM, CAT, CNN1D, GRU/LSTM (opt), Transformer, PatchTST, Autoformer-lite, Informer-lite
# - Hybrid: BACKBONE__CATHEAD (accepted only if improves rmse & dir meaningfully)
# - Stacking: SimpleImputer + Ridge (no NaN crash)
# - Sanity: optional 1D walk-forward check (within CSV, anti-cheat alignment)
# - Input CSV autodiscovery: any csv containing 'main_ethereum_data'
# - True future forecast: as_of = last row date always (even if label not available)
# ============================================================

import os
import json
import math
import time
import warnings
import random
import glob
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*less than 75% GPU memory available.*")

# ---------- sklearn ----------
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ---------- lightgbm / catboost ----------
import lightgbm as lgb

try:
    from catboost import CatBoostRegressor

    HAS_CAT = True
except Exception:
    HAS_CAT = False

# ---------- torch ----------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================
# Config
# ============================================================


@dataclass
class CFG:
    # Either exact path OR leave as None and use autodiscovery by keyword
    INPUT_CSV: Optional[str] = None
    INPUT_KEYWORD: str = "main_ethereum_data"  # file name contains
    OUT_DIR: str = "/kaggle/working/eth_abcd_final"

    SEED: int = 42
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    USE_DATAPARALLEL: bool = True

    RUN_HORIZONS: Tuple[int, ...] = (7, 30, 90)

    # lookback per horizon (sequence models)
    LOOKBACK: Dict[int, int] = None
    # feature cap per horizon
    MF_CAP: Dict[int, int] = None

    # Purged CV
    N_FOLDS_REQ: int = 5
    EMBARGO: Dict[int, int] = None

    # StageA selection
    TOPK_SELECT: int = 4

    # STRONG gate (direction-first, never-empty)
    GATE_STRONG: bool = True
    NEVER_EMPTY: bool = True

    # backbone__CATHEAD accepted only if improves both:
    CATHEAD_MIN_RMSE_IMPROVE_PCT: float = 1.0  # >= 1% rmse better
    CATHEAD_MIN_DIR_IMPROVE_PCT: float = 2.0  # >= 2%p dir_acc better

    # training
    DL_EPOCHS: int = 60
    DL_PATIENCE: int = 10
    BATCH: int = 256
    LR: float = 1e-3
    WEIGHT_DECAY: float = 1e-4

    # Stacking
    USE_STACKING: bool = True

    # Price override (optional, for "live price" comparison)
    CURRENT_PRICE_OVERRIDE: Optional[float] = None

    # Optional: internal sanity check backtest (H=1) over last N days (within CSV)
    DO_1D_WALK_FORWARD_CHECK: bool = True
    WALK_FORWARD_DAYS: int = 20

    # Logging
    QUIET: bool = False


def build_cfg() -> CFG:
    cfg = CFG()
    cfg.LOOKBACK = {7: 120, 30: 160, 90: 220}
    cfg.MF_CAP = {7: 80, 30: 120, 90: 160}
    cfg.EMBARGO = {7: 5, 30: 5, 90: 9}
    return cfg


cfg = build_cfg()

# ============================================================
# Utils
# ============================================================


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def now():
    return time.strftime("%H:%M:%S")


def log(msg: str):
    if not cfg.QUIET:
        print(msg)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def dir_acc(y_true, y_pred) -> float:
    s1 = np.sign(y_true)
    s2 = np.sign(y_pred)
    m = s1 != 0
    if m.sum() == 0:
        return 0.0
    return float((s1[m] == s2[m]).mean())


def safe_float(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def nanfix(X: np.ndarray, nan_value: float = 0.0) -> np.ndarray:
    return np.nan_to_num(X, nan=nan_value, posinf=nan_value, neginf=nan_value)


def standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std), std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


def standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std

# ============================================================
# Input CSV autodiscovery
# ============================================================


def resolve_input_csv() -> str:
    if cfg.INPUT_CSV and os.path.exists(cfg.INPUT_CSV):
        return cfg.INPUT_CSV

    # Kaggle: search /kaggle/input recursively
    base = "/kaggle/input"
    cand = []
    for p in glob.glob(os.path.join(base, "**", "*.csv"), recursive=True):
        fn = os.path.basename(p).lower()
        if cfg.INPUT_KEYWORD.lower() in fn:
            try:
                cand.append((os.path.getmtime(p), p))
            except Exception:
                cand.append((0, p))
    if not cand:
        raise FileNotFoundError(
            f"No CSV found under {base} containing keyword='{cfg.INPUT_KEYWORD}'"
        )
    cand.sort(reverse=True)  # newest first
    return cand[0][1]

# ============================================================
# Data + Targets (no leakage)
# ============================================================


def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("CSV must contain 'date' column")
    if "close" not in df.columns:
        raise ValueError("CSV must contain 'close' column")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def add_logret_targets(df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    out = df.copy()
    close = out["close"].astype(float).values
    for H in horizons:
        y = np.full(len(out), np.nan, dtype=np.float32)
        if len(out) > H + 1:
            y[:-H] = np.log(close[H:] / close[:-H])
        out[f"y_logret_{H}"] = y
    return out


def pick_feature_cols(df: pd.DataFrame, horizons: List[int]) -> List[str]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {"close"}
    for H in horizons:
        exclude.add(f"y_logret_{H}")
    feats = [c for c in num_cols if c not in exclude]
    return feats


def drop_constant_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    keep = []
    for c in cols:
        v = df[c].values
        vv = v[np.isfinite(v)]
        if len(vv) < 3:
            continue
        if np.nanstd(vv) < 1e-12:
            continue
        keep.append(c)
    return keep


def drop_allnan_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    keep = []
    for c in cols:
        v = df[c].values
        if np.isfinite(v).sum() == 0:
            continue
        keep.append(c)
    return keep


def leak_scan(df: pd.DataFrame, feats: List[str], close_col="close") -> List[str]:
    close = df[close_col].astype(float).values
    keep = []
    for c in feats:
        x = df[c].astype(float).values
        bad = False
        for sh in (1, 2, 3, 7, 30, 90):
            if len(close) <= sh + 50:
                continue
            a = x[:-sh]
            b = close[sh:]
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 200:
                continue
            corr = np.corrcoef(a[m], b[m])[0, 1]
            if np.isfinite(corr) and abs(corr) > 0.9999:
                bad = True
                break
        if not bad:
            keep.append(c)
    return keep

# ============================================================
# Purged time-series CV (end-index based)
# ============================================================


def make_purged_folds_endidx(
    n: int,
    n_folds_req: int,
    purge: int,
    embargo: int,
    min_train: int = 400,
):
    k = min(n_folds_req, 5)
    val_size = max(120, n // (k + 1))
    splits = []
    for i in range(k):
        va_start = min_train + i * val_size
        va_end = min(n, va_start + val_size)
        if va_end - va_start < 60:
            continue
        tr_end = max(0, va_start - purge - embargo)
        tr_idx = np.arange(0, tr_end)
        va_idx = np.arange(va_start, va_end)
        if len(tr_idx) < min_train:
            continue
        splits.append((tr_idx, va_idx))
    return splits

# ============================================================
# Ranking score (direction-first)
# ============================================================


def rank_score(rmse_v: float, dir_v: float) -> float:
    if not np.isfinite(rmse_v):
        return -1e9
    return 100.0 * dir_v - 10.0 * rmse_v

# ============================================================
# Tabular models (fixed eval_set)
# ============================================================


def fit_predict_lgbm(X_tr, y_tr, X_va, y_va, seed=42):
    params = dict(
        objective="regression",
        learning_rate=0.03,
        num_leaves=128,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        min_data_in_leaf=50,
        lambda_l2=1.0,
        verbosity=-1,
        seed=seed,
        n_estimators=4000,
    )
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )
    return model, model.predict(X_va)


def fit_predict_cat(X_tr, y_tr, X_va, y_va, seed=42):
    if not HAS_CAT:
        return None, np.full(len(X_va), np.nan, dtype=np.float32)
    model = CatBoostRegressor(
        loss_function="RMSE",
        depth=8,
        learning_rate=0.05,
        iterations=6000,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
    return model, model.predict(X_va)

# ============================================================
# DL Dataset / sequences
# ============================================================


class SeqDataset(Dataset):
    def __init__(self, X_seq: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X_seq).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_sequences(X: np.ndarray, y: np.ndarray, lb: int):
    n, f = X.shape
    if n < lb + 60:
        return None, None, None
    X_seq = np.zeros((n - lb + 1, lb, f), dtype=np.float32)
    y_seq = np.zeros((n - lb + 1,), dtype=np.float32)
    end_idx = np.zeros((n - lb + 1,), dtype=np.int32)
    j = 0
    for t in range(lb - 1, n):
        X_seq[j] = X[t - lb + 1 : t + 1]
        y_seq[j] = y[t]
        end_idx[j] = t
        j += 1
    return X_seq, y_seq, end_idx


def endidx_to_seqidx(idxs_end: np.ndarray, lb: int) -> np.ndarray:
    return idxs_end - (lb - 1)

# ============================================================
# DL Models (stable shapes)
# ============================================================


class CNN1DReg(nn.Module):
    def __init__(self, n_feat: int, hidden=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_feat, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        h = self.net(x).squeeze(-1)
        return self.head(h).squeeze(-1)


class RNNReg(nn.Module):
    def __init__(self, n_feat: int, hidden=96, kind="gru", bidir=False):
        super().__init__()
        if kind == "lstm":
            self.rnn = nn.LSTM(n_feat, hidden, batch_first=True, bidirectional=bidir)
        else:
            self.rnn = nn.GRU(n_feat, hidden, batch_first=True, bidirectional=bidir)
        out_dim = hidden * (2 if bidir else 1)
        self.head = nn.Linear(out_dim, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


class TransformerReg(nn.Module):
    def __init__(self, n_feat: int, d_model=96, nhead=4, nlayers=3, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(n_feat, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.inp(x)
        h = self.enc(h)
        last = h[:, -1, :]
        return self.head(last).squeeze(-1)


class PatchTSTReg(nn.Module):
    def __init__(self, n_feat: int, patch_len=16, d_model=96, nhead=4, nlayers=3, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(n_feat * patch_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        B, T, F = x.shape
        p = self.patch_len
        if T < p:
            pad = p - T
            x = F.pad(x, (0, 0, pad, 0))
            T = p
        pad_len = (p - (T % p)) % p
        if pad_len:
            x = F.pad(x, (0, 0, pad_len, 0))
            T = T + pad_len
        x = x.reshape(B, T // p, p * F)
        h = self.proj(x)
        h = self.enc(h)
        last = h[:, -1, :]
        return self.head(last).squeeze(-1)


def build_model(name: str, n_feat: int):
    n = name.lower()
    if n == "cnn1d":
        return CNN1DReg(n_feat)
    if n == "gru":
        return RNNReg(n_feat, kind="gru", bidir=False)
    if n == "bigru":
        return RNNReg(n_feat, kind="gru", bidir=True)
    if n == "lstm":
        return RNNReg(n_feat, kind="lstm", bidir=False)
    if n == "bilstm":
        return RNNReg(n_feat, kind="lstm", bidir=True)
    if n == "transformer":
        return TransformerReg(n_feat)
    if n == "patchtst":
        return PatchTSTReg(n_feat)
    if n in ("informer", "autoformer"):
        # lite: stable transformer proxy
        return TransformerReg(n_feat)
    return TransformerReg(n_feat)


def maybe_parallel(model: nn.Module) -> nn.Module:
    if (
        cfg.DEVICE.startswith("cuda")
        and cfg.USE_DATAPARALLEL
        and torch.cuda.device_count() > 1
    ):
        return nn.DataParallel(model)
    return model


def train_dl_oof(X_seq, y_seq, end_idx, folds_end, lb, model_name: str):
    device = cfg.DEVICE
    n_seq = len(y_seq)
    oof = np.full(n_seq, np.nan, dtype=np.float32)

    for fold_i, (tr_end, va_end) in enumerate(folds_end):
        tr_end = tr_end[tr_end >= (lb - 1)]
        va_end = va_end[va_end >= (lb - 1)]
        if len(tr_end) < 300 or len(va_end) < 60:
            continue

        tr_idx = endidx_to_seqidx(tr_end, lb)
        va_idx = endidx_to_seqidx(va_end, lb)

        tr_idx = tr_idx[(tr_idx >= 0) & (tr_idx < n_seq)]
        va_idx = va_idx[(va_idx >= 0) & (va_idx < n_seq)]
        if len(tr_idx) < 300 or len(va_idx) < 60:
            continue

        Xtr = nanfix(X_seq[tr_idx])
        Xva = nanfix(X_seq[va_idx])
        ytr = nanfix(y_seq[tr_idx])
        yva = nanfix(y_seq[va_idx])

        ds_tr = SeqDataset(Xtr, ytr)
        ds_va = SeqDataset(Xva, yva)
        dl_tr = DataLoader(ds_tr, batch_size=cfg.BATCH, shuffle=True, drop_last=True)
        dl_va = DataLoader(ds_va, batch_size=cfg.BATCH, shuffle=False)

        base = build_model(model_name, X_seq.shape[-1])
        model = maybe_parallel(base).to(device)

        opt = torch.optim.AdamW(
            model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY
        )
        loss_fn = nn.MSELoss()

        best = 1e18
        bad = 0
        best_state = None

        for _ in range(cfg.DL_EPOCHS):
            model.train()
            for xb, yb in dl_tr:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            model.eval()
            vv = []
            with torch.no_grad():
                for xb, _ in dl_va:
                    xb = xb.to(device)
                    p = model(xb).detach().float().cpu().numpy()
                    vv.append(p)
            vpred = np.concatenate(vv, axis=0)
            vpred = nanfix(vpred)
            vr = rmse(yva, vpred)

            if vr < best - 1e-6:
                best = vr
                bad = 0
                # handle DataParallel state
                state = (
                    model.module.state_dict()
                    if isinstance(model, nn.DataParallel)
                    else model.state_dict()
                )
                best_state = {k: v.detach().cpu().clone() for k, v in state.items()}
            else:
                bad += 1
                if bad >= cfg.DL_PATIENCE:
                    break

        if best_state is None:
            continue

        # restore
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(best_state, strict=True)
        else:
            model.load_state_dict(best_state, strict=True)

        model.eval()
        with torch.no_grad():
            preds = []
            for xb, _ in dl_va:
                xb = xb.to(device)
                p = model(xb).detach().float().cpu().numpy()
                preds.append(p)
        preds = nanfix(np.concatenate(preds, axis=0))
        oof[va_idx] = preds.astype(np.float32)

    return oof


def refit_dl_predict_last(X_seq_all, y_seq_all, model_name: str):
    device = cfg.DEVICE
    X_seq_all = nanfix(X_seq_all)
    y_seq_all = nanfix(y_seq_all)

    ds = SeqDataset(X_seq_all, y_seq_all)
    dl = DataLoader(ds, batch_size=cfg.BATCH, shuffle=True, drop_last=True)

    base = build_model(model_name, X_seq_all.shape[-1])
    model = maybe_parallel(base).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY
    )
    loss_fn = nn.MSELoss()

    n = len(y_seq_all)
    va_n = max(200, n // 10)
    tr_n = n - va_n
    if tr_n < 600:
        va_n = 0

    best = 1e18
    bad = 0
    best_state = None

    for _ in range(cfg.DL_EPOCHS):
        model.train()
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if va_n > 0:
            model.eval()
            with torch.no_grad():
                xb = torch.from_numpy(X_seq_all[-va_n:]).float().to(device)
                yp = model(xb).detach().float().cpu().numpy()
            yp = nanfix(yp)
            vr = rmse(y_seq_all[-va_n:], yp)
            if vr < best - 1e-6:
                best = vr
                bad = 0
                state = (
                    model.module.state_dict()
                    if isinstance(model, nn.DataParallel)
                    else model.state_dict()
                )
                best_state = {k: v.detach().cpu().clone() for k, v in state.items()}
            else:
                bad += 1
                if bad >= cfg.DL_PATIENCE:
                    break

    if best_state is not None:
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(best_state, strict=True)
        else:
            model.load_state_dict(best_state, strict=True)

    model.eval()
    with torch.no_grad():
        xb_last = torch.from_numpy(X_seq_all[-1:]).float().to(device)
        pred_last = float(nanfix(model(xb_last).detach().float().cpu().numpy())[0])
    return pred_last

# ============================================================
# StageA per horizon (tabular index = df_tr index only)
# ============================================================


def stageA_for_horizon(df: pd.DataFrame, H: int, feats_all: List[str]) -> Dict:
    ycol = f"y_logret_{H}"
    m_train = np.isfinite(df[ycol].values)
    df_tr = df.loc[m_train].copy().reset_index(drop=True)
    y = df_tr[ycol].astype(np.float32).values

    # feature pool: >=80% non-missing on df_tr
    feats = [c for c in feats_all if df_tr[c].notna().mean() >= 0.80]
    feats = drop_allnan_cols(df_tr, feats)
    feats = drop_constant_cols(df_tr, feats)
    feats = leak_scan(df_tr, feats, close_col="close")
    feats = drop_allnan_cols(df_tr, feats)

    if len(feats) == 0:
        return None

    # cap by abs corr to target
    if len(feats) > cfg.MF_CAP[H]:
        corrs = []
        for c in feats:
            xv = df_tr[c].astype(float).values
            m = np.isfinite(xv) & np.isfinite(y)
            if m.sum() < 300:
                corrs.append((0.0, c))
                continue
            cc = np.corrcoef(xv[m], y[m])[0, 1]
            if not np.isfinite(cc):
                cc = 0.0
            corrs.append((abs(cc), c))
        corrs.sort(reverse=True, key=lambda z: z[0])
        feats = [c for _, c in corrs[: cfg.MF_CAP[H]]]

    X = df_tr[feats].astype(np.float32).values

    folds = make_purged_folds_endidx(
        n=len(df_tr),
        n_folds_req=cfg.N_FOLDS_REQ,
        purge=H,
        embargo=cfg.EMBARGO[H],
        min_train=400,
    )
    eff = len(folds)
    log(
        f"[{now()}] [CFG] H={H} lb={cfg.LOOKBACK[H]} mf_cap={cfg.MF_CAP[H]} "
        f"purge={H} embargo={cfg.EMBARGO[H]} gate={'STRONG(never-empty)' if cfg.GATE_STRONG else 'WEAK'}"
    )
    log(f"[{now()}] [FEATS] usable(after leak scan/cap)={len(feats)}")
    log(
        f"[{now()}] [CV] effective_folds={eff} requested={cfg.N_FOLDS_REQ} n={len(df_tr)}"
    )
    if eff == 0:
        return None

    oof: Dict[str, np.ndarray] = {}
    rows = []

    # tabular OOF
    for name in ["LGBM", "CAT"]:
        pred_oof = np.full(len(df_tr), np.nan, dtype=np.float32)
        for fi, (tr_idx, va_idx) in enumerate(folds):
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_va, y_va = X[va_idx], y[va_idx]

            med = np.nanmedian(X_tr, axis=0)
            med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)

            X_tr2 = nanfix(np.where(np.isfinite(X_tr), X_tr, med))
            X_va2 = nanfix(np.where(np.isfinite(X_va), X_va, med))
            y_tr2 = nanfix(y_tr)
            y_va2 = nanfix(y_va)

            try:
                if name == "LGBM":
                    _, p = fit_predict_lgbm(
                        X_tr2, y_tr2, X_va2, y_va2, seed=cfg.SEED + fi
                    )
                else:
                    _, p = fit_predict_cat(
                        X_tr2, y_tr2, X_va2, y_va2, seed=cfg.SEED + fi
                    )
                p = nanfix(np.asarray(p, dtype=np.float32))
                pred_oof[va_idx] = p
            except Exception as e:
                log(f"[Fold {fi}] {name} failed: {repr(e)}")

        ok = np.isfinite(pred_oof)
        r = rmse(y[ok], pred_oof[ok]) if ok.sum() > 80 else np.inf
        d = dir_acc(y[ok], pred_oof[ok]) if ok.sum() > 80 else 0.0
        s = rank_score(r, d)
        oof[name] = pred_oof
        rows.append(dict(model=name, rmse=r, dir=d, score=s, kind="tabular"))

    # DL backbones
    dl_models = [
        "CNN1D",
        "Transformer",
        "PatchTST",
        "Autoformer",
        "Informer",
        "GRU",
        "LSTM",
        "BiGRU",
        "BiLSTM",
    ]

    lb = cfg.LOOKBACK[H]
    med_all = np.nanmedian(X, axis=0)
    med_all = np.where(np.isfinite(med_all), med_all, 0.0).astype(np.float32)
    X_imp = nanfix(np.where(np.isfinite(X), X, med_all))
    mean_all, std_all = standardize_fit(X_imp)
    X_imp = standardize_apply(X_imp, mean_all, std_all)

    X_seq, y_seq, end_idx = build_sequences(X_imp, y.astype(np.float32), lb)
    if X_seq is not None:
        # hard NaN clamp
        X_seq = nanfix(X_seq)
        y_seq = nanfix(y_seq)

        for name in dl_models:
            try:
                pred_seq_oof = train_dl_oof(X_seq, y_seq, end_idx, folds, lb, name)
                pred_oof = np.full(len(df_tr), np.nan, dtype=np.float32)
                for j, t in enumerate(end_idx):
                    pred_oof[int(t)] = pred_seq_oof[j]
                ok = np.isfinite(pred_oof)
                r = rmse(y[ok], pred_oof[ok]) if ok.sum() > 80 else np.inf
                d = dir_acc(y[ok], pred_oof[ok]) if ok.sum() > 80 else 0.0
                s = rank_score(r, d)
                oof[name] = pred_oof
                rows.append(dict(model=name, rmse=r, dir=d, score=s, kind="dl"))
            except Exception as e:
                log(f"[DL] {name} failed: {repr(e)}")

    # CATHEAD hybrids
    for base in [
        "CNN1D",
        "Transformer",
        "PatchTST",
        "Autoformer",
        "Informer",
        "GRU",
        "LSTM",
        "BiGRU",
        "BiLSTM",
    ]:
        if base not in oof:
            continue
        base_oof = oof[base]
        if np.isfinite(base_oof).sum() < 400:
            continue

        pred_oof = np.full(len(df_tr), np.nan, dtype=np.float32)
        for fi, (tr_idx, va_idx) in enumerate(folds):
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_va, y_va = X[va_idx], y[va_idx]

            med = np.nanmedian(X_tr, axis=0)
            med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
            X_tr2 = nanfix(np.where(np.isfinite(X_tr), X_tr, med))
            X_va2 = nanfix(np.where(np.isfinite(X_va), X_va, med))
            y_tr2 = nanfix(y_tr)
            y_va2 = nanfix(y_va)

            b_tr = base_oof[tr_idx]
            b_va = base_oof[va_idx]
            mtr = np.isfinite(b_tr)
            mva = np.isfinite(b_va)
            if mtr.sum() < 400 or mva.sum() < 80:
                continue

            Xtr_h = np.concatenate([X_tr2[mtr], b_tr[mtr, None].astype(np.float32)], axis=1)
            Xva_h = np.concatenate([X_va2[mva], b_va[mva, None].astype(np.float32)], axis=1)

            try:
                if HAS_CAT:
                    head = CatBoostRegressor(
                        loss_function="RMSE",
                        depth=8,
                        learning_rate=0.05,
                        iterations=4000,
                        random_seed=cfg.SEED + fi,
                        verbose=False,
                        allow_writing_files=False,
                    )
                    head.fit(
                        Xtr_h, y_tr2[mtr], eval_set=(Xva_h, y_va2[mva]), use_best_model=True
                    )
                    p = head.predict(Xva_h)
                else:
                    head = lgb.LGBMRegressor(
                        objective="regression",
                        learning_rate=0.03,
                        num_leaves=128,
                        feature_fraction=0.8,
                        bagging_fraction=0.8,
                        bagging_freq=1,
                        min_data_in_leaf=50,
                        lambda_l2=1.0,
                        verbosity=-1,
                        seed=cfg.SEED + fi,
                        n_estimators=3000,
                    )
                    head.fit(
                        Xtr_h,
                        y_tr2[mtr],
                        eval_set=[(Xva_h, y_va2[mva])],
                        callbacks=[lgb.early_stopping(200, verbose=False)],
                    )
                    p = head.predict(Xva_h)

                p = nanfix(np.asarray(p, dtype=np.float32))
                pred_oof[va_idx[mva]] = p
            except Exception:
                pass

        ok = np.isfinite(pred_oof)
        if ok.sum() < 120:
            continue

        r_new = rmse(y[ok], pred_oof[ok])
        d_new = dir_acc(y[ok], pred_oof[ok])

        ok0 = np.isfinite(base_oof)
        r0 = rmse(y[ok0], base_oof[ok0]) if ok0.sum() > 80 else np.inf
        d0 = dir_acc(y[ok0], base_oof[ok0]) if ok0.sum() > 80 else 0.0

        rmse_improve = (r0 - r_new) / max(1e-9, r0) * 100.0
        dir_improve = (d_new - d0) * 100.0  # percentage points

        accept = (rmse_improve >= cfg.CATHEAD_MIN_RMSE_IMPROVE_PCT) and (
            dir_improve >= cfg.CATHEAD_MIN_DIR_IMPROVE_PCT
        )
        if accept:
            name = f"{base}__CATHEAD"
            s = rank_score(r_new, d_new)
            oof[name] = pred_oof
            rows.append(dict(model=name, rmse=r_new, dir=d_new, score=s, kind="hyb"))

    rank = (
        pd.DataFrame(rows)
        .sort_values(["score", "dir", "rmse"], ascending=[False, False, True])
        .reset_index(drop=True)
    )

    if cfg.GATE_STRONG and len(rank) >= 6:
        dir_med = float(rank["dir"].median())
        strong = rank[rank["dir"] >= dir_med].copy()
        if len(strong) >= 2:
            rank = (
                pd.concat([strong, rank])
                .drop_duplicates("model", keep="first")
                .reset_index(drop=True)
            )

    selected = rank["model"].tolist()[: cfg.TOPK_SELECT]

    if cfg.NEVER_EMPTY:
        if len(selected) == 0:
            selected = ["CAT", "LGBM"] if HAS_CAT else ["LGBM"]
        # ensure at least one tabular
        if "CAT" in rank["model"].tolist() and "CAT" not in selected:
            selected = (selected + ["CAT"])[: cfg.TOPK_SELECT]
        if "LGBM" in rank["model"].tolist() and "LGBM" not in selected:
            selected = (selected + ["LGBM"])[: cfg.TOPK_SELECT]

    return dict(
        H=H,
        feats=feats,
        X_train=X,
        y_train=y,
        folds=folds,
        oof=oof,
        rank=rank,
        selected=selected,
        lb=lb,
        med_all=med_all,
        X_imp=X_imp,
        train_mask=m_train,
        mean_all=mean_all,
        std_all=std_all,
    )

# ============================================================
# Final refit + predict true future (as_of = df last row)
# ============================================================


def refit_predict_future(df: pd.DataFrame, bundle: Dict) -> Optional[Dict]:
    H = bundle["H"]
    feats = bundle["feats"]

    # full training arrays from bundle
    X = bundle["X_train"]
    y = bundle["y_train"]

    # as_of: ALWAYS last row in full df (true future)
    as_of_date = df["date"].iloc[-1]
    close_last_csv = float(df["close"].iloc[-1])
    current_price = (
        close_last_csv if cfg.CURRENT_PRICE_OVERRIDE is None else float(cfg.CURRENT_PRICE_OVERRIDE)
    )
    forecast_date = as_of_date + pd.Timedelta(days=H)

    # last feature row from full df
    X_last = df[feats].astype(np.float32).values[-1]
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
    X_last2 = nanfix(np.where(np.isfinite(X_last), X_last, med)).reshape(1, -1).astype(
        np.float32
    )

    selected = list(bundle["selected"])
    preds_last: Dict[str, float] = {}

    # helper: full X imputed
    X_full = nanfix(np.where(np.isfinite(X), X, med)).astype(np.float32)

    for name in selected:
        try:
            if name == "LGBM":
                model = lgb.LGBMRegressor(
                    objective="regression",
                    learning_rate=0.03,
                    num_leaves=128,
                    feature_fraction=0.8,
                    bagging_fraction=0.8,
                    bagging_freq=1,
                    min_data_in_leaf=50,
                    lambda_l2=1.0,
                    verbosity=-1,
                    seed=cfg.SEED,
                    n_estimators=4000,
                )
                model.fit(X_full, y)
                preds_last[name] = float(model.predict(X_last2)[0])
                continue

            if name == "CAT":
                if not HAS_CAT:
                    continue
                model = CatBoostRegressor(
                    loss_function="RMSE",
                    depth=8,
                    learning_rate=0.05,
                    iterations=6000,
                    random_seed=cfg.SEED,
                    verbose=False,
                    allow_writing_files=False,
                )
                model.fit(X_full, y)
                preds_last[name] = float(model.predict(X_last2)[0])
                continue

            # DL backbone
            if name.lower() in (
                "cnn1d",
                "transformer",
                "patchtst",
                "autoformer",
                "informer",
                "gru",
                "lstm",
                "bigru",
                "bilstm",
            ):
                X_seq_all, y_seq_all, _ = build_sequences(
                    bundle["X_imp"], y.astype(np.float32), bundle["lb"]
                )
                if X_seq_all is None:
                    continue
                preds_last[name] = float(refit_dl_predict_last(X_seq_all, y_seq_all, name))
                continue

            # CATHEAD hybrid
            if name.endswith("__CATHEAD"):
                base = name.replace("__CATHEAD", "")
                base_oof = bundle["oof"].get(base, None)
                if base_oof is None:
                    continue

                # base last prediction
                X_seq_all, y_seq_all, _ = build_sequences(
                    bundle["X_imp"], y.astype(np.float32), bundle["lb"]
                )
                if X_seq_all is None:
                    continue
                base_last = float(refit_dl_predict_last(X_seq_all, y_seq_all, base))

                b = base_oof.astype(np.float32)
                m = np.isfinite(b)
                if m.sum() < 600:
                    continue
                Xh = np.concatenate([X_full[m], b[m, None]], axis=1)
                X_last_h = np.concatenate(
                    [X_last2, np.array([[base_last]], dtype=np.float32)], axis=1
                )

                if HAS_CAT:
                    head = CatBoostRegressor(
                        loss_function="RMSE",
                        depth=8,
                        learning_rate=0.05,
                        iterations=4000,
                        random_seed=cfg.SEED,
                        verbose=False,
                        allow_writing_files=False,
                    )
                    head.fit(Xh, y[m])
                    preds_last[name] = float(head.predict(X_last_h)[0])
                else:
                    head = lgb.LGBMRegressor(
                        objective="regression",
                        learning_rate=0.03,
                        num_leaves=128,
                        feature_fraction=0.8,
                        bagging_fraction=0.8,
                        bagging_freq=1,
                        min_data_in_leaf=50,
                        lambda_l2=1.0,
                        verbosity=-1,
                        seed=cfg.SEED,
                        n_estimators=3000,
                    )
                    head.fit(Xh, y[m])
                    preds_last[name] = float(head.predict(X_last_h)[0])
                continue
        except Exception:
            pass

    if len(preds_last) == 0:
        return None

    usable = [
        m
        for m in selected
        if (m in bundle["oof"]) and np.isfinite(safe_float(preds_last.get(m, np.nan)))
    ]
    if len(usable) == 0:
        usable = [m for m in preds_last.keys() if np.isfinite(safe_float(preds_last[m]))]

    if cfg.USE_STACKING and len(usable) >= 2:
        P = np.vstack([bundle["oof"][m] for m in usable]).T.astype(np.float32)
        ytr = y.astype(np.float32)

        fin = np.isfinite(P)
        good = fin.sum(axis=1) >= max(1, len(usable) // 2)
        P_tr = P[good]
        y_tr = ytr[good]

        meta = Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                ("ridge", Ridge(alpha=1.0, random_state=cfg.SEED)),
            ]
        )
        meta.fit(P_tr, y_tr)

        p_last_vec = np.array(
            [safe_float(preds_last.get(m)) for m in usable], dtype=np.float32
        )[None, :]
        yhat = float(meta.predict(p_last_vec)[0])
        stacking_used = 1
    else:
        yhat = float(np.nanmean([safe_float(v) for v in preds_last.values()]))
        stacking_used = 0

    pred_price = current_price * float(np.exp(yhat))

    # uncertainty from best model OOF residual (fallback safe)
    resid_q95 = 0.30
    try:
        bestm = bundle["rank"]["model"].iloc[0]
        oof_best = bundle["oof"][bestm]
        ok = np.isfinite(oof_best)
        resid = np.abs(oof_best[ok] - y[ok])
        resid_q95 = float(np.quantile(resid, 0.95))
        if not np.isfinite(resid_q95):
            resid_q95 = 0.30
    except Exception:
        resid_q95 = 0.30

    ci_low = current_price * float(np.exp(yhat - resid_q95))
    ci_high = current_price * float(np.exp(yhat + resid_q95))

    out = dict(
        horizon=H,
        as_of_date=str(as_of_date.date()),
        forecast_date=str(forecast_date.date()),
        current_price=current_price,
        current_price_csv=close_last_csv,
        pred_logret=yhat,
        pred_price=pred_price,
        expected_return_pct=(math.exp(yhat) - 1.0) * 100.0,
        ci95_low_price=ci_low,
        ci95_high_price=ci_high,
        q95_abs_resid_logret=resid_q95,
        n_models_used=len(usable),
        models_used="|".join(usable),
        stacking_used=stacking_used,
        # probabilistic-ish proxy: fraction of models predicting up
        p_up_last=float(
            np.mean([1.0 if preds_last[m] > 0 else 0.0 for m in usable])
        )
        if len(usable)
        else float(yhat > 0),
        target_kind="log_return",
    )
    return out

# ============================================================
# 1D Walk-forward sanity check (within CSV only)
# ============================================================


def walk_forward_check_1d(df: pd.DataFrame, days: int = 20):
    df1 = add_logret_targets(df.copy(), [1])
    ycol = "y_logret_1"
    feats = pick_feature_cols(df1, [1])
    feats = [c for c in feats if df1[c].notna().mean() >= 0.80]
    feats = drop_allnan_cols(df1, feats)
    feats = drop_constant_cols(df1, feats)
    feats = leak_scan(df1, feats, close_col="close")
    feats = drop_allnan_cols(df1, feats)
    feats = feats[:120]

    if len(feats) == 0:
        return pd.DataFrame([])

    rows = []
    for k in range(days, 0, -1):
        cut = len(df1) - k - 1
        if cut < 900:
            continue
        sub = df1.iloc[: cut + 1].copy()
        m = np.isfinite(sub[ycol].values)
        sub = sub[m]
        if len(sub) < 900:
            continue

        X = sub[feats].astype(np.float32).values
        y = sub[ycol].astype(np.float32).values
        med = np.nanmedian(X, axis=0)
        med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
        X = nanfix(np.where(np.isfinite(X), X, med))

        x_last = df1[feats].iloc[cut].astype(np.float32).values
        x_last = nanfix(np.where(np.isfinite(x_last), x_last, med)).reshape(1, -1)

        model = lgb.LGBMRegressor(
            objective="regression",
            learning_rate=0.03,
            num_leaves=64,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            min_data_in_leaf=50,
            lambda_l2=1.0,
            verbosity=-1,
            seed=cfg.SEED,
            n_estimators=1500,
        )
        model.fit(X, y)
        yhat = float(model.predict(x_last)[0])

        date_asof = df1["date"].iloc[cut]
        date_fc = df1["date"].iloc[cut + 1]
        close_asof = float(df1["close"].iloc[cut])
        close_real = float(df1["close"].iloc[cut + 1])
        close_pred = close_asof * float(np.exp(yhat))
        real_lr = float(np.log(close_real / close_asof))

        rows.append(
            dict(
                as_of_date=str(date_asof.date()),
                forecast_date=str(date_fc.date()),
                close_asof=close_asof,
                close_real=close_real,
                close_pred=close_pred,
                pred_logret=yhat,
                real_logret=real_lr,
                dir_hit=int(np.sign(yhat) == np.sign(real_lr)),
            )
        )

    return pd.DataFrame(rows)

# ============================================================
# Main
# ============================================================


def main():
    seed_all(cfg.SEED)
    ensure_dir(cfg.OUT_DIR)

    input_csv = resolve_input_csv()
    log(f"[{now()}] [IO] input(auto): {input_csv}")
    log(f"[{now()}] [IO] out_dir: {cfg.OUT_DIR}")
    log(
        f"[{now()}] [ENV] DEVICE={cfg.DEVICE} cuda={torch.cuda.is_available()} n_gpu={torch.cuda.device_count()}"
    )

    df = load_df(input_csv)
    df = add_logret_targets(df, list(cfg.RUN_HORIZONS))

    log(
        f"[{now()}] [DATA] rows={len(df)} cols={df.shape[1]} "
        f"date_range={df['date'].min().date()}~{df['date'].max().date()} close_col=close"
    )

    feats_all = pick_feature_cols(df, list(cfg.RUN_HORIZONS))
    log(f"[{now()}] [FEATS] numeric pool (pre-filter)={len(feats_all)}")

    # sanity check
    if cfg.DO_1D_WALK_FORWARD_CHECK:
        chk = walk_forward_check_1d(df, days=cfg.WALK_FORWARD_DAYS)
        outp = os.path.join(cfg.OUT_DIR, "daily_1d_walk_forward_check.csv")
        chk.to_csv(outp, index=False)
        log(f"[{now()}] [SAVE] {outp}")

    final_rows = []
    model_rows = []

    for H in cfg.RUN_HORIZONS:
        log(f"\n===== H={H} =====")
        bundle = stageA_for_horizon(df, H, feats_all)
        if bundle is None:
            log(f"[{now()}] [WARN] StageA failed/insufficient for H={H}")
            continue

        rank_path = os.path.join(cfg.OUT_DIR, f"stageA_rank_h{H}.csv")
        bundle["rank"].to_csv(rank_path, index=False)
        log(f"[{now()}] [SAVE] {rank_path}")
        log(f"[{now()}] [SELECT] top={bundle['selected']}")

        out = refit_predict_future(df, bundle)
        if out is None:
            log(f"[{now()}] [WARN] no final prediction for H={H}")
            continue

        final_rows.append(out)
        model_rows.append(dict(horizon=H, selected="|".join(bundle["selected"])))

        log(
            f"[{now()}] [FINAL] H={H} (as_of={out['as_of_date']}) -> "
            f"${out['pred_price']:.2f} ({out['expected_return_pct']:+.2f}%) | "
            f"STACK={bool(out['stacking_used'])} | p_up≈{out['p_up_last']:.2f}"
        )

    final_all = pd.DataFrame(final_rows)
    final_models = pd.DataFrame(model_rows)

    out_all = os.path.join(cfg.OUT_DIR, "final_forecast_all.csv")
    out_models = os.path.join(cfg.OUT_DIR, "final_forecast_models.csv")
    state_path = os.path.join(cfg.OUT_DIR, "state.json")

    final_all.to_csv(out_all, index=False)
    final_models.to_csv(out_models, index=False)
    with open(state_path, "w") as f:
        json.dump(dict(cfg=asdict(cfg), input_csv=resolve_input_csv()), f, indent=2)

    log(f"\n[DONE] {out_all}")
    log(f"[DONE] {out_models}")
    log(f"[DONE] {state_path}")

    if len(final_all) == 0:
        log("[WARN] No final rows produced. Check targets, folds, and feature availability.")
    else:
        log("\n=== FINAL FORECAST (TRUE FUTURE FROM CSV LAST DATE) ===")
        print(final_all)


if __name__ == "__main__":
    main()

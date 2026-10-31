"""Model zoo. Every model maps a (causal) feature frame to a score in about [-1, +1].

API: fit(F, y, w) with F the full feature frame up to the training cutoff and y NaN on
unlabeled rows; predict(F) returns a score for every row of F; explain(F) returns
per-feature contributions for the last row.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_config
from ..features import RULE_COMPONENTS, SELL_MONOTONE

# LightGBM and PyTorch ship separate OpenMP runtimes; on macOS running both multi-threaded
# in one process deadlocks. The data is small, so both run single-threaded.
os.environ.setdefault("OMP_NUM_THREADS", "1")

MODEL_TYPES = ["rules", "linear", "lgbm", "nn", "ens_fast", "ens_all"]


class Model:
    name = "base"

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}
        self.columns: list[str] = []

    def fit(self, F: pd.DataFrame, y: pd.Series, w: pd.Series) -> "Model":
        raise NotImplementedError

    def predict(self, F: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def predict_with_spread(self, F: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """(score, spread across ensemble members). Spread 0 for single models."""
        s = self.predict(F)
        return s, pd.Series(0.0, index=s.index)

    def explain(self, F: pd.DataFrame) -> pd.Series:
        """Occlusion: change in the last row's score when each feature is set to its training mean."""
        base = float(self.predict(F).iloc[-1])
        out = {}
        tail = F.iloc[-400:] if self.name == "nn" else F.iloc[-1:]
        for c in self.columns:
            G = tail.copy()
            G.iloc[-1, G.columns.get_loc(c)] = self._means.get(c, np.nan) if hasattr(self, "_means") else np.nan
            out[c] = base - float(self.predict(G).iloc[-1])
        return pd.Series(out).sort_values(key=np.abs, ascending=False)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: Path) -> "Model":
        with open(path, "rb") as fh:
            return pickle.load(fh)


def _train_rows(F: pd.DataFrame, y: pd.Series) -> pd.Index:
    return y.dropna().index.intersection(F.index)


class Standardizer:
    def fit(self, X: pd.DataFrame) -> "Standardizer":
        self.mean = X.mean()
        self.std = X.std().replace(0, 1).fillna(1)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Z = (X - self.mean) / self.std
        return Z.clip(-5, 5).fillna(0.0)


# --- Rule composite -----------------------------------------------------------------

class RulesModel(Model):
    """No training: average of historical-percentile readings of classic cycle indicators."""
    name = "rules"

    def fit(self, F, y, w):
        self.columns = [c for c in RULE_COMPONENTS if c in F]
        self._means = {c: 0.5 for c in self.columns}
        return self

    def predict(self, F):
        heat = F[self.columns].mean(axis=1, skipna=True)  # 0 = cold, 1 = hot
        return (1 - 2 * heat).rename("score")


# --- Ridge -----------------------------------------------------------------------------

class LinearModel(Model):
    name = "linear"

    def fit(self, F, y, w):
        from sklearn.linear_model import Ridge
        self.columns = list(F.columns)
        rows = _train_rows(F, y)
        self.std = Standardizer().fit(F.loc[rows])
        self._means = self.std.mean.to_dict()
        self.m = Ridge(alpha=self.cfg.get("alpha", 30.0)).fit(self.std.transform(F.loc[rows]), y.loc[rows], sample_weight=w.loc[rows])
        return self

    def predict(self, F):
        return pd.Series(np.clip(self.m.predict(self.std.transform(F[self.columns])), -1.5, 1.5), index=F.index)

    def explain(self, F):
        z = self.std.transform(F[self.columns].iloc[[-1]]).iloc[0]
        return (z * self.m.coef_).sort_values(key=np.abs, ascending=False)


# --- LightGBM ----------------------------------------------------------------------------

class LGBMModel(Model):
    name = "lgbm"

    def fit(self, F, y, w):
        import lightgbm as lgb
        self.columns = list(F.columns)
        rows = _train_rows(F, y)
        p = {**load_config()["models"]["lgbm"], **self.cfg}
        # The SELL_MONOTONE constraints encode cycle-valuation logic; other horizons opt out.
        mono = {"monotone_constraints": [-1 if c in SELL_MONOTONE else 0 for c in self.columns],
                "monotone_constraints_method": "advanced"} if p.get("monotone", True) else {}
        self.m = lgb.LGBMRegressor(
            n_estimators=p["n_estimators"], learning_rate=p["learning_rate"], num_leaves=p["num_leaves"],
            min_child_samples=p["min_child_samples"], subsample=p["subsample"], subsample_freq=1,
            colsample_bytree=p["colsample_bytree"], reg_lambda=p["reg_lambda"], **mono,
            random_state=p.get("seed", 7), n_jobs=1, verbose=-1)
        self.m.fit(F.loc[rows], y.loc[rows], sample_weight=w.loc[rows])
        return self

    def predict(self, F):
        return pd.Series(self.m.predict(F[self.columns]), index=F.index)

    def explain(self, F):
        contrib = self.m.predict(F[self.columns].iloc[[-1]], pred_contrib=True)[0][:-1]
        return pd.Series(contrib, index=self.columns).sort_values(key=np.abs, ascending=False)


# --- Neural net ----------------------------------------------------------------------------

import torch  # noqa: E402  (after OMP_NUM_THREADS is set)
from torch import nn  # noqa: E402


class _Net(nn.Module):
    """Temporal CNN over the weekly sequence + MLP over today's features -> tanh score."""

    def __init__(self, d_static: int, d_seq: int, drop: float):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d_seq, 16, 5, padding=2), nn.GELU(),
            nn.Conv1d(16, 16, 5, padding=4, dilation=2), nn.GELU())
        self.static = nn.Sequential(nn.Linear(d_static, 64), nn.GELU(), nn.Dropout(drop), nn.Linear(64, 32), nn.GELU())
        self.head = nn.Sequential(nn.Linear(48, 32), nn.GELU(), nn.Dropout(drop), nn.Linear(32, 1))

    def forward(self, x, q):
        h = self.conv(q)
        hq = 0.5 * (h.mean(dim=2) + h[:, :, -1])
        return torch.tanh(self.head(torch.cat([self.static(x), hq], dim=1))).squeeze(1) * 1.2


# Series fed to the temporal CNN as a weekly sequence. Entries absent from the model's feature
# set are skipped, so the ETH/BTC channels only appear in feature sets that include Ether.
SEQ_FEATURES = ["mayer", "p_200w", "mvrv", "puell", "rsi_w", "dd_ath", "ret_30", "vol_30", "hash_ribbon", "rsi_m",
                "ethbtc", "ethbtc_ret_90", "ethbtc_dd_ath"]


class NNModel(Model):
    """Small temporal-CNN + MLP ensemble.

    Inputs: today's standardized feature vector, plus a 52-week sequence (one sample per
    week) of key indicators. Heavily regularized; trained over several seeds whose spread
    is reported as uncertainty.
    """
    name = "nn"

    def _tensors(self, F: pd.DataFrame):
        import torch
        torch.set_num_threads(1)
        Z = self.std.transform(F[self.columns])
        seq_cols = [c for c in self.seq_cols]
        S = Z[seq_cols].to_numpy(dtype=np.float32)
        idx = self._seq_index(len(Z))
        mask = idx < 0
        seq = S[np.clip(idx, 0, None)]
        seq[mask] = 0.0
        return torch.tensor(Z.to_numpy(dtype=np.float32)), torch.tensor(seq).permute(0, 2, 1).contiguous()

    def _seq_index(self, n: int) -> np.ndarray:
        """Row positions forming each sample's sequence: `seq_weeks` weekly steps ending today."""
        W = self.p["seq_weeks"]
        return np.arange(n)[:, None] - 7 * np.arange(W)[::-1][None, :]

    def _seq_features(self) -> list[str]:
        return SEQ_FEATURES

    def _net(self, d_static: int, d_seq: int):
        return _Net(d_static, d_seq, self.p["dropout"])

    def fit(self, F, y, w):
        import torch
        self.p = {**load_config()["models"]["nn"], **self.cfg}
        dev = torch.device("mps" if torch.backends.mps.is_available() and self.p.get("device", "auto") != "cpu" else "cpu")
        self.columns = list(F.columns)
        self.seq_cols = [c for c in self._seq_features() if c in self.columns]
        rows = _train_rows(F, y)
        self.std = Standardizer().fit(F.loc[rows])
        self._means = self.std.mean.to_dict()
        X, Q = self._tensors(F)
        pos = torch.tensor(F.index.get_indexer(rows))
        X, Q = X[pos].to(dev), Q[pos].to(dev)
        Y = torch.tensor(y.loc[rows].to_numpy(dtype=np.float32), device=dev)
        Wt = torch.tensor((w.loc[rows] / w.loc[rows].mean()).to_numpy(dtype=np.float32), device=dev)
        n = len(Y)
        per_epoch = max(256, int(n * self.p.get("epoch_fraction", 0.35)))
        bs = self.p.get("batch_size", 512)
        self.nets = []
        for seed in range(self.p.get("seed_offset", 0), self.p.get("seed_offset", 0) + self.p["seeds"]):
            torch.manual_seed(seed)
            g = torch.Generator().manual_seed(seed)
            net = self._net(X.shape[1], Q.shape[1]).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=self.p["lr"], weight_decay=self.p["weight_decay"])
            for _ in range(self.p["epochs"]):
                net.train()
                perm = torch.randperm(n, generator=g)[:per_epoch].to(dev)
                for b in range(0, per_epoch, bs):
                    i = perm[b:b + bs]
                    xb = X[i] + self.p["input_noise"] * torch.randn_like(X[i])
                    qb = Q[i] + self.p["input_noise"] * torch.randn_like(Q[i])
                    loss = (Wt[i] * (net(xb, qb) - Y[i]) ** 2).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            net.eval()
            self.nets.append(net.to("cpu"))
        return self

    def _all(self, F) -> np.ndarray:
        import torch
        X, Q = self._tensors(F)
        with torch.no_grad():
            return np.stack([net(X, Q).numpy() for net in self.nets])

    def predict(self, F):
        return pd.Series(self._all(F).mean(axis=0), index=F.index)

    def predict_with_spread(self, F):
        a = self._all(F)
        return pd.Series(a.mean(axis=0), index=F.index), pd.Series(a.std(axis=0), index=F.index)


# --- Ensembles ------------------------------------------------------------------------------

class EnsembleModel(Model):
    def __init__(self, members: list[str], cfg: dict | None = None):
        super().__init__(cfg)
        self.members_types = members
        self.name = "ens_" + "+".join(members)

    def fit(self, F, y, w):
        self.columns = list(F.columns)
        self.members = [make_model(t, self.cfg).fit(F, y, w) for t in self.members_types]
        self._means = {c: F[c].mean() for c in self.columns}
        return self

    def predict(self, F):
        return pd.concat([m.predict(F) for m in self.members], axis=1).mean(axis=1)

    def predict_with_spread(self, F):
        P = pd.concat([m.predict(F) for m in self.members], axis=1)
        return P.mean(axis=1), P.std(axis=1)

    def explain(self, F):
        return pd.concat([m.explain(F) for m in self.members], axis=1).fillna(0).mean(axis=1).sort_values(key=np.abs, ascending=False)


def make_model(model_type: str, cfg: dict | None = None) -> Model:
    cfg = cfg or {}
    if model_type == "rules":
        return RulesModel(cfg)
    if model_type == "linear":
        return LinearModel(cfg)
    if model_type == "lgbm":
        return LGBMModel(cfg)
    if model_type == "nn":
        return NNModel(cfg)
    if model_type == "ens_fast":
        return EnsembleModel(["rules", "linear", "lgbm"], cfg)
    if model_type == "ens_all":
        return EnsembleModel(["rules", "linear", "lgbm", "nn"], cfg)
    raise ValueError(model_type)

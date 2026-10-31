"""Models for the trading signals: one per side, each predicting the trade's R multiple.

Reuses the cycle model zoo's `Model` API. Learned models (`calibrated = True`) output an estimate
of expected R, so a trade is also required to clear its own round-trip cost in R units. The rules
baselines output a ranking only, and are gated by their trailing-quantile threshold alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config
from ..models import EnsembleModel, LGBMModel, LinearModel, Model, NNModel

TRADE_MODEL_TYPES = ["srules_momentum", "srules_pullback", "linear", "lgbm", "ens", "nn"]


class SwingRules(Model):
    """Published-style baselines, no training.

    - momentum: time-series momentum (Moskowitz, Ooi & Pedersen 2012) - long when the 20-day
      return is positive, short when negative, ranked by its vol-scaled size.
    - pullback: buy dips in an uptrend / sell rallies in a downtrend - long when price is above its
      50-day average and RSI(14) is low, short when below it and RSI is high.
    """
    calibrated = False

    def __init__(self, cfg: dict | None = None):
        super().__init__(cfg)
        self.rule = self.cfg["rule"]
        self.side = self.cfg["side"]
        self.name = f"srules_{self.rule}"

    def fit(self, F, y, w):
        self.columns = ["ret_20d_z"] if self.rule == "momentum" else ["ma50_dist_z", "rsi_14"]
        self._means = {c: float(F[c].mean()) for c in self.columns}
        return self

    def predict(self, F):
        sign = 1.0 if self.side == "long" else -1.0
        if self.rule == "momentum":
            s = sign * F["ret_20d_z"]
        else:
            trend = np.sign(F["ma50_dist_z"])
            s = (trend == sign) * sign * (50 - F["rsi_14"]) / 50
        return pd.Series(s, index=F.index, dtype=float)


class SwingNN(NNModel):
    """The cycle NN with a daily sequence of fast features instead of a weekly cycle sequence."""
    SEQ = ["ret_1d_z", "ret_5d_z", "rv_10", "rsi_14", "range_pos_20", "volume_z_20", "ma20_dist_z"]

    def _seq_index(self, n: int) -> np.ndarray:
        W = self.p.get("seq_days", 32)
        return np.arange(n)[:, None] - np.arange(W)[::-1][None, :]

    def _seq_features(self) -> list[str]:
        return self.SEQ


def _with_flag(m: Model, calibrated: bool) -> Model:
    m.calibrated = calibrated
    return m


def make_trade_model(model_type: str, side: str, cfg: dict | None = None) -> Model:
    cfg = dict(cfg or {})
    if model_type.startswith("srules_"):
        return SwingRules({"rule": model_type.split("_", 1)[1], "side": side})
    if model_type == "linear":
        return _with_flag(LinearModel({"alpha": cfg.get("alpha", 100.0)}), True)
    lgbm_cfg = {"monotone": False, "seed": cfg.get("seed", 7), **{k: v for k, v in cfg.items() if k in (
        "n_estimators", "learning_rate", "num_leaves", "min_child_samples", "reg_lambda")}}
    if model_type == "lgbm":
        return _with_flag(LGBMModel(lgbm_cfg), True)
    if model_type == "ens":
        m = EnsembleModel(["linear", "lgbm"], {**lgbm_cfg, "alpha": cfg.get("alpha", 100.0)})
        return _with_flag(m, True)
    if model_type == "nn":
        nn_cfg = {**load_config()["models"]["nn"], "seeds": cfg.get("nn_seeds", 4), "epochs": cfg.get("epochs", 60),
                  "seq_days": 32, "seed_offset": cfg.get("seed", 7) * 100}
        return _with_flag(SwingNN(nn_cfg), True)
    raise ValueError(model_type)

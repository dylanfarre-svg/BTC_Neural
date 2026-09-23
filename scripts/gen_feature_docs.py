"""Regenerate the README's feature reference from FEATURE_DOCS (the same text the GUI shows)."""
import warnings; warnings.filterwarnings("ignore")
import re
from collections import defaultdict
from btc_neural.features import FEATURE_DOCS, FEATURE_GROUPS, FEATURE_SETS, columns_for, load_all

daily, feats = load_all()
by_group = defaultdict(list)
for c in feats.columns:
    by_group[FEATURE_GROUPS[c]].append(c)
titles = {
    "price": "Price / technical", "ma": "Moving averages", "onchain": "On-chain valuation & network",
    "clock": "Cycle clock", "exhaust": "Parabola exhaustion", "flow": "Exchange flows, fees, addresses",
    "eth": "Ether (ETH/BTC, ETH/USD)", "macro": "Macro", "intraday": "Intraday (4-hour bars)",
    "calendar": "Calendar (ablation only)", "sentiment": "Sentiment (display only)",
}
sets_with = {g: sorted(k for k, spec in FEATURE_SETS.items()
                       if any(c in columns_for(k, list(feats.columns)) for c in by_group[g])) for g in by_group}
lines = [f"## Feature reference ({len(feats.columns)} features)", "",
         "Names are exactly as they appear in the app's **Features** page, which shows the same description.",
         "Everything is causal (only data available that day) and scale-free. Log values: 0 = equal, +0.69 = double,",
         "−0.69 = half. `_pct4y` = percentile within the last 4 years (0–1); `_pctx` = percentile of all history so far.", ""]
order = ["price", "ma", "onchain", "clock", "exhaust", "flow", "eth", "macro", "intraday", "calendar", "sentiment"]
for g in order:
    cols = by_group.get(g, [])
    if not cols:
        continue
    first = min(str(feats[c].first_valid_index().date()) for c in cols if feats[c].notna().any())
    lines += [f"### {titles[g]} — group `{g}`, {len(cols)} features, from {first}", "",
              "| Feature | Meaning |", "|---|---|"]
    for c in cols:
        lines.append(f"| `{c}` | {FEATURE_DOCS[c]} |")
    lines.append("")
block = "\n".join(lines)

p = "README.md"
s = open(p).read()
start = s.find("## Feature reference")
if start >= 0:
    end = s.find("\n## ", start + 5)
    s = s[:start] + block + (s[end + 1:] if end > 0 else "")
else:
    anchor = "## Experiment log"
    s = s.replace(anchor, block + anchor, 1)
open(p, "w").write(s)
print("wrote feature reference:", len(feats.columns), "features,", len(lines), "lines")

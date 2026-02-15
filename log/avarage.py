import re
import numpy as np
from pathlib import Path

LOG_PATH = r"./finetune20/NDL-day-OS_BCGNET_RFPA_GridSearch0_Huber_BCG1_PI1_FF1_seed0_N10_zscore1_e20_all_fluc-s3-1-darma.log"   # 改成你的路径

# 只匹配你关心的两种“结果块”：
# [Fold:39]
# MAE: (..,..)
# ME : (..,..)
# STD: (..,..)
# PCC: (..,..)
#
# [X1047]
# MAE: ...
# ...
BLOCK_RE = re.compile(
    r"""
    ^\[(?P<tag>Fold:\s*\d+|X\d+)\]\s*$\s*
    ^MAE:\s*\(\s*(?P<mae1>[-+0-9.eE]+)\s*,\s*(?P<mae2>[-+0-9.eE]+)\s*\)\s*$\s*
    ^ME\s*:\s*\(\s*(?P<me1>[-+0-9.eE]+)\s*,\s*(?P<me2>[-+0-9.eE]+)\s*\)\s*$\s*
    ^STD:\s*\(\s*(?P<std1>[-+0-9.eE]+)\s*,\s*(?P<std2>[-+0-9.eE]+)\s*\)\s*$\s*
    ^PCC:\s*\(\s*(?P<pcc1>[-+0-9.eE]+)\s*,\s*(?P<pcc2>[-+0-9.eE]+)\s*\)\s*$
    """,
    re.M | re.S | re.X
)

def collect_pairs(matches, key1, key2):
    a = np.array([float(m.group(key1)) for m in matches], dtype=float)
    b = np.array([float(m.group(key2)) for m in matches], dtype=float)
    return a, b

def summarize(matches, name):
    if not matches:
        print(f"{name}: 没匹配到任何结果块，请检查log格式/缩进/空格。")
        return

    mae1, mae2 = collect_pairs(matches, "mae1", "mae2")
    me1,  me2  = collect_pairs(matches, "me1",  "me2")
    std1, std2 = collect_pairs(matches, "std1", "std2")
    pcc1, pcc2 = collect_pairs(matches, "pcc1", "pcc2")

    def line(metric, x, y):
        # 这里给你三种平均：分别均值 + overall((x+y)/2)
        overall = ((x + y) / 2).mean()
        return (f"{metric}: n={len(x)}  "
                f"mean1={x.mean():.4f}  mean2={y.mean():.4f}  overall={overall:.4f}")

    print(f"\n===== {name} =====")
    print(line("MAE", mae1, mae2))
    print(line("ME ", me1,  me2))
    print(line("STD", std1, std2))
    print(line("PCC", pcc1, pcc2))

def main():
    text = Path(LOG_PATH).read_text(encoding="utf-8", errors="ignore")

    all_matches = list(BLOCK_RE.finditer(text))

    baseline = [m for m in all_matches if m.group("tag").replace(" ", "").startswith("Fold:")]
    personal = [m for m in all_matches if m.group("tag").strip().startswith("X")]

    summarize(baseline, "Baseline (Fold)")
    summarize(personal, "Personal (X)")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
summarize_runs.py -- 把 v2v4real_experiment.py run 的長輸出壓成「每個配對、每種設定一列」，並做跨場景統計。

用法：python summarize_runs.py opv2v_main.log opv2v_bad.log opv2v_test.log [--pairs "302,311 1773,1782"]
標籤：每個區塊以 '=== ' 開頭的一行辨識，需含 pair=A,B。
  - 優先讀 set=clean|bad|nf；舊格式則含 'noise-free' -> nf，含 'n_bad' -> bad，其餘 -> clean。
  - 'seed=N' 用來分組平均種子。
每個配對視為一個場景（請每個場景只挑一個配對，否則跨場景統計會重複計算同一場景）。

σ 掃描模式（--sweep）：標籤用 set=sig0.05、set=sig0.1 ...；nf 視為 σ = 0，clean 視為 σ = 0.6，bad 不納入。
  σ 同時是平移（m）與旋轉（°）的標準差。輸出每個 σ 的跨場景統計，以及每個場景的損益平衡點
  （修正後誤差第一次不再高於「不修正」誤差的 σ，線性內插）。
"""
import re
import sys
from collections import defaultdict

import numpy as np

NUM = r"(-?[\d.]+|nan)"


def parse(path):
    blocks, cur = [], None
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("=== "):
            cur = dict(label=line[4:], rows={})
            blocks.append(cur)
            continue
        if cur is None:
            continue
        if "±" not in line:                                   # 只取第一張表（無 ± 的那張）
            m = re.match(r"^([A-E])\.\s.*?\s+" + r"\s+".join([NUM] * 6) + r"\s*$", line)
            if m and m.group(1) not in cur["rows"]:
                cur["rows"][m.group(1)] = [float(x) for x in m.groups()[1:]]   # pose, rot, box, prec, npairs, rate
        m = re.search(r"正確配對：.*?被丟棄\(q<q_min\)比例 = " + NUM, line)
        if m:
            cur["drop_ok"] = float(m.group(1))
        m = re.search(r"錯誤配對：.*?被丟棄\(q<q_min\)比例 = " + NUM + r"（平均每樣本 " + NUM, line)
        if m:
            cur["drop_bad"], cur["wrong"] = float(m.group(1)), float(m.group(2))
        m = re.search(r"另有 (\d+) 個幀因兩車距離", line)
        if m:
            cur["far"] = int(m.group(1))
        m = re.search(r"可用 (\d+)", line)
        if m and "合計" in line:
            cur["used"] = int(m.group(1))
    return blocks


SPARSE = {"207,216", "3174,3183", "5933,5942", "623,632"}      # 事先標為稀疏（共視 4～6 個）的四個場景


def sigma_of(st):
    if st == "nf":
        return 0.0
    if st == "clean":
        return 0.6
    m = re.fullmatch(r"sig([\d.]+)", st)
    return float(m.group(1)) if m else None


def crossing(xs, ys):
    """ys = 不修正誤差 − 修正後誤差。回傳第一個 ys >= 0 的 x（線性內插）；一開始就 >= 0 回 0；沒有交叉回 None。"""
    for i, (x, y) in enumerate(zip(xs, ys)):
        if y >= 0:
            if i == 0:
                return x
            x0, y0 = xs[i - 1], ys[i - 1]
            return x0 + (0 - y0) * (x - x0) / (y - y0)
    return None


def sweep_report(groups):
    series = defaultdict(dict)                                  # pair -> {σ: metrics}
    for (pair, st), bl in groups.items():
        sg = sigma_of(st)
        if sg is None:
            continue
        m = lambda k, i: float(np.mean([b["rows"][k][i] for b in bl]))
        series[pair][sg] = dict(At=m("A", 0), Ar=m("A", 1), Ab=m("A", 2), Bt=m("B", 0), Br=m("B", 1), Bb=m("B", 2),
                                Cb=m("C", 2), Dt=m("D", 0), n=len(bl))
    sigmas = sorted({sg for d in series.values() for sg in d})
    print("每個 σ 的跨場景統計（σ 同時是平移 m 與旋轉 °；B 優於 A 的場景數以 pose 誤差比較）")
    print(f"{'σ':>5}{'場景':>5} | {'A 平移':>7}{'B 平移':>7}{'B<A':>6} | {'A 旋轉':>7}{'B 旋轉':>7}{'B<A':>6} | {'A box':>6}{'B box':>6}{'C box':>6}{'C<A':>6}{'B<A':>6}")
    for sg in sigmas:
        rs = [d[sg] for d in series.values() if sg in d]
        n = len(rs)
        f = lambda k: np.mean([r[k] for r in rs])
        c = lambda a, b: sum(r[a] < r[b] for r in rs)
        print(f"{sg:>5.2f}{n:>5} | {f('At'):>7.3f}{f('Bt'):>7.3f}{c('Bt','At'):>4}/{n:<2}| {f('Ar'):>7.3f}{f('Br'):>7.3f}{c('Br','Ar'):>4}/{n:<2}| "
              f"{f('Ab'):>6.3f}{f('Bb'):>6.3f}{f('Cb'):>6.3f}{c('Cb','Ab'):>4}/{n:<2}{c('Bb','Ab'):>4}/{n:<2}")

    print("\n每個場景的損益平衡點（σ，單位 m 與 °；修正後誤差第一次不高於不修正誤差的位置，線性內插）")
    print(f"{'配對':<11}{'組別':<5}{'點數':>4} | {'平移':>8}{'旋轉':>8}{'box(C對A)':>11}{'box(B對A)':>11} | {'B平移@各σ':<}")
    rows = []
    for pair, d in sorted(series.items()):
        xs = sorted(d)
        if len(xs) < 3:
            continue
        ys = lambda a, b: [d[x][a] - d[x][b] for x in xs]
        be = dict(t=crossing(xs, ys("At", "Bt")), r=crossing(xs, ys("Ar", "Br")),
                  bc=crossing(xs, ys("Ab", "Cb")), bb=crossing(xs, ys("Ab", "Bb")))
        grp = "稀疏" if pair in SPARSE else "其餘"
        rows.append((pair, grp, be))
        fmt = lambda v: f"{v:>8.3f}" if v is not None else f"{'>' + str(max(xs)):>8}"
        fmt11 = lambda v: f"{v:>11.3f}" if v is not None else f"{'>' + str(max(xs)):>11}"
        print(f"{pair:<11}{grp:<5}{len(xs):>4} | {fmt(be['t'])}{fmt(be['r'])}{fmt11(be['bc'])}{fmt11(be['bb'])} | "
              + " ".join(f"{d[x]['Bt']:.3f}" for x in xs))
    print("（'>x' 代表掃描範圍內沒有交叉；0.000 代表 σ = 0 時修正已不比不修正差）")
    for grp in ("其餘", "稀疏"):
        sub = [be for _, g, be in rows if g == grp]
        if not sub:
            continue
        out = []
        for key, name in (("t", "平移"), ("r", "旋轉"), ("bc", "box(C對A)"), ("bb", "box(B對A)")):
            v = [b[key] for b in sub if b[key] is not None]
            none = len(sub) - len(v)
            out.append(f"{name} 中位 {np.median(v):.3f}（{min(v):.3f}～{max(v):.3f}）" + (f"，{none} 個無交叉" if none else "") if v else f"{name} 全部無交叉")
        print(f"  {grp}（{len(sub)} 個場景）：" + "；".join(out))


def setting(label):
    m = re.search(r"set=([\w.]+)", label)                       # 新格式：set=clean / bad / nf
    if m:
        return m.group(1)
    if "noise-free" in label:                                 # 舊格式
        return "nf"
    return "bad" if "n_bad" in label else "clean"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--sweep", action="store_true", help="σ 掃描模式：輸出每個 σ 的統計與每個場景的損益平衡點")
    ap.add_argument("--pairs", default=None, help='只保留這些配對，空白分隔並加引號，例如 "302,311 1773,1782"')
    a = ap.parse_args()
    keep = set(a.pairs.split()) if a.pairs else None
    groups = defaultdict(list)
    for p in a.logs:
        for b in parse(p):
            m = re.search(r"pair=([\w,]+)", b["label"])
            if not m or len(b["rows"]) < 5 or (keep and m.group(1) not in keep):
                continue
            groups[(m.group(1), setting(b["label"]))].append(b)

    if a.sweep:
        sweep_report(groups)
        return

    def avg(bl, f):
        v = [f(b) for b in bl if f(b) is not None]
        return float(np.mean(v)) if v else float("nan")

    R = lambda k, i: (lambda b: b["rows"][k][i])
    hdr = (f"{'配對':<11}{'設定':<6}{'種':>2}{'樣本':>5}{'匹配':>6} | {'A平移':>6}{'B平移':>6}{'D平移':>6}{'E平移':>6} | "
           f"{'B旋轉':>6}{'D旋轉':>6} | {'A box':>6}{'B box':>6}{'C box':>6}{'D box':>6} | {'錯/樣本':>6}{'丟對':>6}{'丟錯':>6}{'跳過':>5}")
    print(hdr)
    table = {}
    for (pair, st), bl in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        row = dict(
            nseed=len(bl), used=avg(bl, lambda b: b.get("used")), prec=avg(bl, R("B", 3)),
            At=avg(bl, R("A", 0)), Bt=avg(bl, R("B", 0)), Ct=avg(bl, R("C", 0)), Dt=avg(bl, R("D", 0)), Et=avg(bl, R("E", 0)),
            Br=avg(bl, R("B", 1)), Cr=avg(bl, R("C", 1)), Dr=avg(bl, R("D", 1)),
            Ab=avg(bl, R("A", 2)), Bb=avg(bl, R("B", 2)), Cb=avg(bl, R("C", 2)), Db=avg(bl, R("D", 2)),
            wrong=avg(bl, lambda b: b.get("wrong")), dok=avg(bl, lambda b: b.get("drop_ok")),
            dbad=avg(bl, lambda b: b.get("drop_bad")), far=avg(bl, lambda b: b.get("far", 0)))
        table[(pair, st)] = row
        print(f"{pair:<11}{st:<6}{row['nseed']:>2}{row['used']:>5.0f}{row['prec']:>6.2f} | "
              f"{row['At']:>6.3f}{row['Bt']:>6.3f}{row['Dt']:>6.3f}{row['Et']:>6.3f} | {row['Br']:>6.3f}{row['Dr']:>6.3f} | "
              f"{row['Ab']:>6.3f}{row['Bb']:>6.3f}{row['Cb']:>6.3f}{row['Db']:>6.3f} | "
              f"{row['wrong']:>6.1f}{row['dok']:>6.3f}{row['dbad']:>6.3f}{row['far']:>5.0f}")

    print("\n跨場景統計（每個配對 = 一個場景；D - C 為負代表 feedback 較好）")
    for st in ("clean", "bad", "nf"):
        rs = [r for (p, s), r in table.items() if s == st]
        if not rs:
            continue
        d_t = np.array([r["Dt"] - r["Ct"] for r in rs])
        d_r = np.array([r["Dr"] - r["Cr"] for r in rs])
        d_b = np.array([r["Db"] - r["Cb"] for r in rs])
        c_a = np.array([r["Cb"] - r["Ab"] for r in rs])
        b_a = np.array([r["Bb"] - r["Ab"] for r in rs])
        print(f"  [{st}] 場景數 {len(rs)}；B 的 pose 平移均值 {np.mean([r['Bt'] for r in rs]):.3f} m（範圍 {min(r['Bt'] for r in rs):.3f}～{max(r['Bt'] for r in rs):.3f}），"
              f"A 的 {np.mean([r['At'] for r in rs]):.3f} m")
        print(f"        D-C：平移均值 {d_t.mean():+.4f} m（D 較好的場景 {int((d_t < 0).sum())}/{len(rs)}），"
              f"旋轉 {d_r.mean():+.4f}°（{int((d_r < 0).sum())}/{len(rs)}），box {d_b.mean():+.4f} m（{int((d_b < 0).sum())}/{len(rs)}）")
        print(f"        box：B-A {b_a.mean():+.3f} m（B 較好 {int((b_a < 0).sum())}/{len(rs)}），C-A {c_a.mean():+.3f} m（C 較好 {int((c_a < 0).sum())}/{len(rs)}）")


if __name__ == "__main__":
    main()
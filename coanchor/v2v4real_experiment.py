#!/usr/bin/env python3
"""
v2v4real_experiment.py -- 在 V2V4Real 真實軌跡上驗證 CoAnchor object-level 模組（不訓練任何網路）

做法：把資料集的 GT 標註當成「偵測結果」，加上偵測噪聲與漏檢，再人為注入 pose noise 與通訊延遲，
      比較「無修正」與 CoAnchor 各變體。真實的部分是：車輛軌跡、兩車的相對運動、真實 pose。

用法：
  1) 先檢查座標約定（標註在 lidar 座標系還是 world pose 是 lidar->world 還是反向？）
     python v2v4real_experiment.py check --scenario <場景資料夾>
  2) 再跑實驗（把 check 建議的旗標帶進來）
     python v2v4real_experiment.py run --scenario <場景資料夾> --delay_ms 200 --sig_t 0.6 --sig_r 0.6
"""
from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
from scipy.spatial.distance import cdist

from coanchor_lite import VARIANTS, Params, coanchor, pose_err, se2_compose, se2_inv, warp_tracks
from coanchor_tracker import Tracker
from v2v4real_io import Scenario, mat_to_se2

warnings.filterwarnings("ignore", category=RuntimeWarning)


# --------------------------------------------------------------------------- #
# 座標約定
# --------------------------------------------------------------------------- #
def agent_pose(frame, pose_dir):
    """該幀 lidar 的 SE(2) pose（lidar -> world）。pose_dir='w2l' 表示 yaml 內矩陣是反向的。"""
    M = frame.pose if pose_dir == "l2w" else np.linalg.inv(frame.pose)
    return mat_to_se2(M)


def boxes_in_lidar(frame, label_frame, pose_dir):
    if label_frame == "lidar":
        return frame.boxes
    return warp_tracks(se2_inv(agent_pose(frame, pose_dir)), frame.boxes)


# --------------------------------------------------------------------------- #
# check：判斷哪一種約定讓兩車的標註彼此對得上
# --------------------------------------------------------------------------- #
def cmd_check(a):
    sc = Scenario(a.scenario)
    print("agents:", sc.agents, " frames:", sc.n_frames)
    if len(sc.agents) < 2:
        print("找不到兩個 agent 資料夾，無法檢查。"); return
    ae, an = sc.agents[:2]
    n = min(sc.n_frames[ae], sc.n_frames[an])
    idxs = np.linspace(0, n - 1, min(a.n_check, n)).astype(int)

    # 與約定無關的檢查：兩車 lidar_pose 的平移距離（假設 pose 為 lidar->world）
    dist = np.array([np.linalg.norm(sc.frame(ae, i).pose[:2, 3] - sc.frame(an, i).pose[:2, 3]) for i in idxs])
    print(f"兩車 lidar_pose 平移距離（假設 l2w）：中位數 {np.median(dist):.1f} m，最小 {dist.min():.1f}，最大 {dist.max():.1f}")
    print("（兩車若相距上百公尺，本來就不可能共視；請與其他場景比較）\n")

    print(f"{'label_frame':<12}{'pose_dir':<10}{'重疊率(<1.5m)':>16}{'最近距離中位數(m)':>20}{'夥伴車命中率(<3m)':>20}")
    best = None
    for lf in ("lidar", "world"):
        for pd in ("l2w", "w2l"):
            hit, tot, dists, ph, pt = 0, 0, [], 0, 0
            for i in idxs:
                fe, fn = sc.frame(ae, i), sc.frame(an, i)
                be, bn = boxes_in_lidar(fe, lf, pd), boxes_in_lidar(fn, lf, pd)
                T = se2_compose(se2_inv(agent_pose(fe, pd)), agent_pose(fn, pd))   # nb lidar -> ego lidar
                if len(be):                                                         # 夥伴車原點是否落在 ego 的某個標註附近
                    pt += 1
                    ph += np.linalg.norm(be[:, :2] - T[:2], axis=1).min() < 3.0
                if len(be) == 0 or len(bn) == 0:
                    continue
                bn_e = warp_tracks(T, bn)
                m = np.linalg.norm(bn_e[:, :2], axis=1) < 50          # 只看 ego 附近的物件
                if not m.any():
                    continue
                d = cdist(bn_e[m, :2], be[:, :2]).min(1)
                hit += (d < 1.5).sum(); tot += m.sum(); dists.append(d)
            rate = hit / max(tot, 1)
            med = np.median(np.concatenate(dists)) if dists else np.nan
            print(f"{lf:<12}{pd:<10}{rate:>16.2f}{med:>20.2f}{ph / max(pt, 1):>20.2f}")
            if best is None or rate > best[0]:
                best = (rate, lf, pd)
    print(f"\n建議旗標：--label_frame {best[1]} --pose_dir {best[2]}  （重疊率 {best[0]:.2f}）")
    print("判讀：正確的約定重疊率應明顯高於其他三種。『夥伴車命中率』只有在資料集有標註夥伴車時才有意義，"
          "\n請先在已知可行的場景（重疊率高者）上看它，再拿去比較其他場景。")


# --------------------------------------------------------------------------- #
# run：實驗
# --------------------------------------------------------------------------- #
def noisy_dets(boxes, ids, rng, a):
    keep = (np.linalg.norm(boxes[:, :2], axis=1) < a.range) & (rng.random(len(boxes)) > a.miss)
    d = boxes[keep].copy()
    d[:, :2] += rng.normal(0, a.det_sigma, (len(d), 2))
    d[:, 2] += rng.normal(0, np.deg2rad(a.yaw_sigma), len(d))
    return d, list(ids[keep])


def run_scenario(sc, a, rng, rows, st, diag, P, keys, qdiag):
    """處理單一場景，累積到 rows / st / diag；回傳該場景的摘要。"""
    ae, an = sc.agents[:2][::-1] if a.swap else sc.agents[:2]
    nfr = min(sc.n_frames[ae], sc.n_frames[an])
    delay, H, warm = a.delay_ms // 100, 4, a.warm
    pd, lf = a.pose_dir, a.label_frame
    att0, used0, d0 = st["attempt"], len(diag), len(diag)

    def lidar_boxes(agent, i):
        f = sc.frame(agent, i)
        return boxes_in_lidar(f, lf, pd), f.ids

    for t in range(warm + delay, nfr, a.stride):
        u = t - delay
        for _ in range(a.reps):
            st["attempt"] += 1
            hist, meta = {}, {}
            for ag in (ae, an):
                trk = Tracker(H=H, gate=a.trk_gate)
                for s_ in range(u - warm, u + 1):
                    b, ids = lidar_boxes(ag, s_)
                    d, mids = noisy_dets(b, ids, rng, a)
                    T = None
                    if s_ > u - warm:
                        T = se2_compose(se2_inv(agent_pose(sc.frame(ag, s_), pd)),
                                        agent_pose(sc.frame(ag, s_ - 1), pd))
                    trk.update(d, T, mids)
                hist[ag], meta[ag] = trk.export()
            eh, nh = hist[ae], hist[an]
            if len(eh) < 3 or len(nh) < 3:
                st["few_tracks"] += 1
                continue

            Pe_u, Pn_u = agent_pose(sc.frame(ae, u), pd), agent_pose(sc.frame(an, u), pd)
            Pe_t, Pn_t = agent_pose(sc.frame(ae, t), pd), agent_pose(sc.frame(an, t), pd)
            T_true = se2_compose(se2_inv(Pe_u), Pn_u)
            eps = np.array([rng.normal(0, a.sig_t), rng.normal(0, a.sig_t), np.deg2rad(rng.normal(0, a.sig_r))])
            T0 = se2_compose(T_true, eps)
            T_e = se2_compose(se2_inv(Pe_t), Pe_u)
            bt, it = lidar_boxes(ae, t)
            ego_obs_t, _ = noisy_dets(bt, it, rng, a)

            bn_t, in_t = lidar_boxes(an, t)
            T_n2e_t = se2_compose(se2_inv(Pe_t), Pn_t)
            gt = np.full((len(nh), 2), np.nan)
            for k, mid in enumerate(meta[an]):
                j = np.flatnonzero(in_t == mid)
                if len(j):
                    gt[k] = warp_tracks(T_n2e_t, bn_t[j[:1]])[0, :2]
            ok = ~np.isnan(gt[:, 0])
            if not ok.any():
                st["no_gt"] += 1
                continue
            potential = int((cdist(warp_tracks(T_true, nh)[:, 0, :2], eh[:, 0, :2]).min(1) < 1.5).sum())
            diag.append((len(eh), len(nh), potential))
            keys.append((sc.dir, t))

            for name in VARIANTS:
                res = coanchor(eh, nh, T0, delay, T_e, ego_obs_t, p=P, **VARIANTS[name])
                et, er = pose_err(res.T_final, T_true)
                box = np.linalg.norm(res.states[ok, :2] - gt[ok], axis=1).mean()
                prec = np.nan
                if len(res.pairs):
                    nb_e = warp_tracks(T_true, nh[res.pairs[:, 1]])[:, 0, :2]
                    okp = np.linalg.norm(nb_e - eh[res.pairs[:, 0], 0, :2], axis=1) < 1.5
                    prec = okp.mean()
                    if name.startswith("D.") and len(res.q) == len(res.pairs):
                        supp = res.q / np.maximum(res.q0, 1e-12)          # 相對初始權重的剩餘比例
                        gm = lambda m: supp[m].mean() if m.any() else np.nan
                        gd = lambda m: (res.q[m] < P.q_min).mean() if m.any() else np.nan
                        qdiag.append((gm(okp), gm(~okp), (~okp).sum(), gd(okp), gd(~okp)))
                rows[name].append((et, er, box, prec, len(res.pairs), float(len(res.pairs) >= P.n_min_trk)))

    dd = np.array(diag[d0:], float)
    return dict(name=os.path.basename(os.path.normpath(sc.dir)), frames=nfr, att=st["attempt"] - att0,
                used=len(diag) - used0, pot=dd[:, 2].mean() if len(dd) else np.nan,
                ge3=(dd[:, 2] >= 3).mean() if len(dd) else np.nan)


def list_scenarios(a):
    if a.scenario:
        dirs = [a.scenario]
    elif a.root:
        dirs = [os.path.join(a.root, d) for d in sorted(os.listdir(a.root))
                if os.path.isdir(os.path.join(a.root, d)) and not d.startswith(".")]
    else:
        raise SystemExit("請給 --scenario 或 --root")
    out = []
    for d in dirs:
        try:
            sc = Scenario(d)
        except OSError:
            continue
        if len(sc.agents) >= 2 and min(sc.n_frames[x] for x in sc.agents[:2]) > 0:
            out.append(sc)
    return out


def cmd_run(a):
    rng = np.random.default_rng(a.seed)
    names = list(VARIANTS)
    rows = {k: [] for k in names}
    P = Params(n_min_trk=a.n_min_trk)
    st, diag, summ = dict(attempt=0, few_tracks=0, no_gt=0), [], []
    keys, qdiag = [], []
    for sc in list_scenarios(a):
        summ.append(run_scenario(sc, a, rng, rows, st, diag, P, keys, qdiag))

    n = len(rows[names[0]])
    print(f"設定：delay={a.delay_ms}ms pose noise=({a.sig_t}m,{a.sig_r}°) label_frame={a.label_frame} "
          f"pose_dir={a.pose_dir} n_min_trk={P.n_min_trk} range={a.range} miss={a.miss}\n")
    print(f"{'場景':<46}{'幀數':>6}{'嘗試':>7}{'可用':>6}{'可配對上限均值':>16}{'上限≥3 的比例':>15}")
    for r in summ:
        print(f"{r['name']:<46}{r['frames']:>6}{r['att']:>7}{r['used']:>6}{r['pot']:>16.2f}{r['ge3']:>15.2f}")
    if n == 0:
        print("\n沒有可用樣本（軌跡不足 3 條）。請換場景，或放寬 --range / --miss。"); return
    d = np.array(diag, float)
    print(f"\n合計：嘗試 {st['attempt']}；可用 {n}；因軌跡不足 3 條捨棄 {st['few_tracks']}；因無 GT 捨棄 {st['no_gt']}")
    print(f"可用樣本平均：ego 軌跡 {d[:,0].mean():.1f}、neighbor 軌跡 {d[:,1].mean():.1f}、"
          f"真值下可配對上限 {d[:,2].mean():.1f}（配對需 ≥ {P.n_min_trk} 才會做 pose 修正）\n")
    print(f"{'變體':<34}{'pose平移(m)':>12}{'旋轉(°)':>9}{'box誤差(m)':>12}{'匹配正確率':>12}{'配對數':>8}{'修正執行率':>11}")
    for name in names:
        v = np.array(rows[name], float)
        print(f"{name:<32}{v[:,0].mean():>12.3f}{v[:,1].mean():>9.3f}{v[:,2].mean():>12.3f}"
              f"{np.nanmean(v[:,3]):>12.2f}{v[:,4].mean():>8.1f}{v[:,5].mean():>11.2f}")

    # 以「幀」為單位的平均 ± 標準誤（相鄰幀仍相關，實際不確定性更大）
    uniq = {k: i for i, k in enumerate(dict.fromkeys(keys))}
    kid = np.array([uniq[k] for k in keys])
    nk = len(uniq)

    def mean_se(x):
        per = np.array([x[kid == i].mean() for i in range(nk)])
        return per.mean(), (per.std(ddof=1) / np.sqrt(nk) if nk > 1 else np.nan)

    print(f"\n平均 ± 標準誤（以 {nk} 個幀為單位；相鄰幀相關，實際不確定性比這個大）")
    print(f"{'變體':<34}{'pose平移(m)':>18}{'旋轉(°)':>16}{'box誤差(m)':>18}")
    for name in names:
        v = np.array(rows[name], float)
        cells = [mean_se(v[:, j]) for j in (0, 1, 2)]
        print(f"{name:<32}" + "".join(f"{m:>12.3f} ±{se:.3f}" for m, se in cells))

    if qdiag:
        q = np.array(qdiag, float)
        print("\nclosed-loop 權重診斷（D 變體；q/q⁰ 是 feedback 後剩餘的相對權重，越小代表被壓得越低）")
        print(f"  正確配對：平均 q/q⁰ = {np.nanmean(q[:,0]):.3f}，被丟棄(q<q_min)比例 = {np.nanmean(q[:,3]):.3f}")
        print(f"  錯誤配對：平均 q/q⁰ = {np.nanmean(q[:,1]):.3f}，被丟棄(q<q_min)比例 = {np.nanmean(q[:,4]):.3f}"
              f"（平均每樣本 {q[:,2].mean():.1f} 個錯誤配對）")
        print("  判讀：若錯誤配對的 q/q⁰ 並不明顯低於正確配對，feedback 沒有在區分好壞配對。")


def cmd_diag(a):
    """不加噪聲、不用追蹤器：用 GT id 數每個場景到底有多少可用物體。"""
    H = 4
    print(f"{'場景':<46}{'幀數':>6}{'GT/幀(ego)':>11}{'GT/幀(nb)':>10}{'持續4幀(ego)':>13}{'持續4幀(nb)':>12}{'共視持續均值':>13}{'共視≥3比例':>11}")
    for sc in list_scenarios(a):
        ae, an = sc.agents[:2][::-1] if a.swap else sc.agents[:2]
        nfr = min(sc.n_frames[ae], sc.n_frames[an])
        pd, lf = a.pose_dir, a.label_frame

        def inrange(ag, s_):
            f = sc.frame(ag, s_)
            b = boxes_in_lidar(f, lf, pd)
            m = np.linalg.norm(b[:, :2], axis=1) < a.range
            return b[m], f.ids[m]

        g_e, g_n, p_e, p_n, cov = [], [], [], [], []
        for u in range(H - 1, nfr, a.stride):
            per = {}
            for ag in (ae, an):
                sets = [set(inrange(ag, u - k)[1]) for k in range(H)]
                keep = set.intersection(*sets)
                b, ids = inrange(ag, u)
                per[ag] = (b[np.isin(ids, list(keep))], len(ids), len(keep))
            g_e.append(per[ae][1]); g_n.append(per[an][1]); p_e.append(per[ae][2]); p_n.append(per[an][2])
            T_true = se2_compose(se2_inv(agent_pose(sc.frame(ae, u), pd)), agent_pose(sc.frame(an, u), pd))
            be, bn = per[ae][0], per[an][0]
            if len(be) and len(bn):
                cov.append(int((cdist(warp_tracks(T_true, bn)[:, :2], be[:, :2]).min(1) < 1.5).sum()))
            else:
                cov.append(0)
        cov = np.array(cov)
        print(f"{os.path.basename(os.path.normpath(sc.dir)):<46}{nfr:>6}{np.mean(g_e):>11.1f}{np.mean(g_n):>10.1f}"
              f"{np.mean(p_e):>13.1f}{np.mean(p_n):>12.1f}{cov.mean():>13.2f}{(cov >= 3).mean():>11.2f}")
    print("\n判讀：『共視持續均值』是不經追蹤器、不加噪聲時，兩車共同看到且連續 4 幀存在的物體數。"
          "\n若它本身就 < 3，是資料稀疏；若它夠大但 run 的『可配對上限』很低，問題在追蹤器。")


def _tracks_at(sc, ag, u, n_hist, pd, lf, rng_max):
    """用 GT id 串出連續 n_hist 幀都存在的物體，並用 pose 補償到 u 時刻的 lidar 座標系。"""
    frames = [sc.frame(ag, u - k) for k in range(n_hist)]
    bl = [boxes_in_lidar(f, lf, pd) for f in frames]
    sets = [set(f.ids[np.linalg.norm(b[:, :2], axis=1) < rng_max]) for f, b in zip(frames, bl)]
    keep = sorted(set.intersection(*sets))
    Pu = agent_pose(frames[0], pd)
    tr = np.zeros((len(keep), n_hist, 3))
    for k, (f, b) in enumerate(zip(frames, bl)):
        Tk = se2_compose(se2_inv(Pu), agent_pose(f, pd))
        if keep:
            idx = [int(np.flatnonzero(f.ids == i)[0]) for i in keep]
            tr[:, k] = warp_tracks(Tk, b[idx])
    return np.array(keep), tr


def cmd_floor(a):
    """誤差下限診斷：不加噪聲、不用追蹤器，用 GT id 與『真值 pose 下的最近鄰』當配對，
    直接問：標註幾何推出的相對 pose，與資料集 lidar_pose 推出的差多少？"""
    from coanchor_lite import refine_pose
    P = Params()
    print(f"{'場景':<40}{'n_hist':>7}{'nb幀偏移':>9}{'樣本':>6}{'配對數':>8}{'|Δ平移|均值(m)':>16}{'中位數':>8}{'|Δ旋轉|均值(°)':>16}{'中位數':>8}")
    for sc in list_scenarios(a):
        ae, an = sc.agents[:2][::-1] if a.swap else sc.agents[:2]
        nfr = min(sc.n_frames[ae], sc.n_frames[an])
        for n_hist, shift in [(1, 0), (4, 0), (1, -1), (1, 1)]:
            rec = []
            for u in range(max(n_hist - 1, n_hist - 1 - shift), min(nfr, nfr - shift), a.stride):
                un = u + shift
                ie, te = _tracks_at(sc, ae, u, n_hist, a.pose_dir, a.label_frame, a.range)
                inn, tn = _tracks_at(sc, an, un, n_hist, a.pose_dir, a.label_frame, a.range)
                if len(ie) < 3 or len(inn) < 3:
                    continue
                T_true = se2_compose(se2_inv(agent_pose(sc.frame(ae, u), a.pose_dir)),
                                     agent_pose(sc.frame(an, un), a.pose_dir))
                nbw = warp_tracks(T_true, tn)
                D = cdist(te[:, 0, :2], nbw[:, 0, :2])
                ai, bi = D.argmin(1), D.argmin(0)
                pairs = np.array([(i, ai[i]) for i in range(len(ie)) if bi[ai[i]] == i and D[i, ai[i]] < a.thr])
                if len(pairs) < 3:
                    continue
                dT = refine_pose(te, nbw, pairs, np.ones(len(pairs)), P)    # 殘差修正量 = 下限
                rec.append((np.hypot(dT[0], dT[1]), abs(np.degrees(dT[2])), len(pairs)))
            name = os.path.basename(os.path.normpath(sc.dir))
            if not rec:
                print(f"{name:<40}{n_hist:>7}{shift:>9}{0:>6}"); continue
            r = np.array(rec)
            print(f"{name:<40}{n_hist:>7}{shift:>9}{len(r):>6}{r[:,2].mean():>8.1f}{r[:,0].mean():>16.3f}{np.median(r[:,0]):>8.3f}"
                  f"{r[:,1].mean():>16.3f}{np.median(r[:,1]):>8.3f}")
    print("\n判讀：這裡沒有任何注入噪聲。若 n_hist=1、偏移 0 的下限已接近 0.2 m / 0.7°，"
          "\n代表資料集 pose 與標註幾何本身就有這麼大的不一致（或兩車並非同時刻），"
          "\n此時我們用 lidar_pose 當真值的評測，其絕對誤差下限就是這個量級。"
          "\n若 n_hist=4 明顯高於 n_hist=1，下限有一部分來自逐幀 pose 的自身運動補償。"
          "\n注意：配對由真值 pose 下的最近鄰（門檻 --thr）決定，會偏向挑出與真值一致的配對，因此這個下限可能被低估。")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check"); c.add_argument("--scenario", required=True); c.add_argument("--n_check", type=int, default=20)
    dg = sub.add_parser("diag")
    dg.add_argument("--scenario", default=None); dg.add_argument("--root", default=None)
    dg.add_argument("--label_frame", choices=["lidar", "world"], default="lidar"); dg.add_argument("--pose_dir", choices=["l2w", "w2l"], default="l2w")
    dg.add_argument("--range", type=float, default=70.0); dg.add_argument("--stride", type=int, default=1); dg.add_argument("--swap", action="store_true")
    fl = sub.add_parser("floor")
    fl.add_argument("--scenario", default=None); fl.add_argument("--root", default=None)
    fl.add_argument("--label_frame", choices=["lidar", "world"], default="lidar"); fl.add_argument("--pose_dir", choices=["l2w", "w2l"], default="l2w")
    fl.add_argument("--range", type=float, default=70.0); fl.add_argument("--stride", type=int, default=1)
    fl.add_argument("--thr", type=float, default=2.0); fl.add_argument("--swap", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--scenario", default=None, help="單一場景資料夾")
    r.add_argument("--root", default=None, help="含多個場景資料夾的上層目錄")
    r.add_argument("--delay_ms", type=int, default=200)
    r.add_argument("--sig_t", type=float, default=0.6)
    r.add_argument("--sig_r", type=float, default=0.6)
    r.add_argument("--label_frame", choices=["lidar", "world"], default="lidar")
    r.add_argument("--pose_dir", choices=["l2w", "w2l"], default="l2w")
    r.add_argument("--det_sigma", type=float, default=0.2)
    r.add_argument("--yaw_sigma", type=float, default=3.0)
    r.add_argument("--miss", type=float, default=0.1)
    r.add_argument("--range", type=float, default=70.0)
    r.add_argument("--warm", type=int, default=8)
    r.add_argument("--stride", type=int, default=3)
    r.add_argument("--reps", type=int, default=2)
    r.add_argument("--n_min_trk", type=int, default=3, help="論文預設 3；改小是偏離論文的探索性設定")
    r.add_argument("--trk_gate", type=float, default=3.0, help="追蹤器關聯距離門檻 (m)；高速場景可試 6")
    r.add_argument("--swap", action="store_true", help="ego/neighbor 對調")
    r.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"check": cmd_check, "run": cmd_run, "diag": cmd_diag, "floor": cmd_floor}[a.cmd](a)


if __name__ == "__main__":
    main()
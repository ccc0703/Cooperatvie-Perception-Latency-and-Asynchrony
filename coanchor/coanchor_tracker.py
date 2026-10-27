#!/usr/bin/env python3
"""
coanchor_tracker.py -- Phase 1：每個 agent 獨立的輕量追蹤器 + 偵測串流模擬

追蹤器輸出 coanchor_lite.coanchor() 需要的 (N, H, 3) 歷史 box（k=0 為最新、已補償自身運動）。
執行 `python coanchor_tracker.py` 會比較「oracle 歷史（用 GT id 串起來）」與「追蹤器歷史」。
"""
from __future__ import annotations

import argparse
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from coanchor_lite import (VARIANTS, coanchor, pose_err, rot, se2_compose, se2_inv,
                           warp_tracks, wrap_pi)


# --------------------------------------------------------------------------- #
# SORT 風格追蹤器：CV Kalman (x,y,vx,vy) + Hungarian + 自身運動補償
# --------------------------------------------------------------------------- #
class _Track:
    __slots__ = ("x", "P", "yaw", "hist", "age", "meta")

    def __init__(self, det, sig, meta):
        self.x = np.array([det[0], det[1], 0.0, 0.0])
        self.P = np.diag([sig ** 2, sig ** 2, 1.0, 1.0])
        self.yaw = det[2]
        self.hist = det[None].copy()          # (k,3)，newest first
        self.age = 0
        self.meta = meta


class Tracker:
    def __init__(self, H=4, gate=3.0, max_age=2, sig_det=0.3, q_pos=0.05, q_vel=0.05):
        self.H, self.gate, self.max_age = H, gate, max_age
        self.R = np.eye(2) * sig_det ** 2
        self.Q = np.diag([q_pos ** 2, q_pos ** 2, q_vel ** 2, q_vel ** 2])
        self.sig = sig_det
        self.F = np.eye(4); self.F[0, 2] = self.F[1, 3] = 1.0
        self.tracks: list[_Track] = []

    def update(self, dets, T_prev2cur=None, meta=None):
        """dets (N,3) 在「目前這幀的 ego 座標系」；T_prev2cur 把上一幀座標系的點轉到這一幀。"""
        dets = np.asarray(dets, float).reshape(-1, 3)
        meta = list(meta) if meta is not None else [-1] * len(dets)

        if T_prev2cur is not None:            # 自身運動補償
            R = rot(T_prev2cur[2])
            for tr in self.tracks:
                tr.x[:2] = R @ tr.x[:2] + T_prev2cur[:2]
                tr.x[2:] = R @ tr.x[2:]
                tr.yaw = wrap_pi(tr.yaw + T_prev2cur[2])
                tr.hist = warp_tracks(T_prev2cur, tr.hist)

        for tr in self.tracks:                # predict
            tr.x = self.F @ tr.x
            tr.P = self.F @ tr.P @ self.F.T + self.Q

        matched_t, matched_d = set(), set()
        if self.tracks and len(dets):
            pred = np.array([t.x[:2] for t in self.tracks])
            D = cdist(pred, dets[:, :2])
            r, c = linear_sum_assignment(np.where(D > self.gate, 1e6, D))
            for i, j in zip(r, c):
                if D[i, j] > self.gate:
                    continue
                tr = self.tracks[i]
                Hm = np.zeros((2, 4)); Hm[0, 0] = Hm[1, 1] = 1.0
                S = Hm @ tr.P @ Hm.T + self.R
                K = tr.P @ Hm.T @ np.linalg.inv(S)
                tr.x = tr.x + K @ (dets[j, :2] - tr.x[:2])
                tr.P = (np.eye(4) - K @ Hm) @ tr.P
                tr.yaw = dets[j, 2]
                tr.age = 0
                tr.meta = meta[j]
                tr.hist = np.vstack([[*tr.x[:2], tr.yaw], tr.hist])[: self.H]
                matched_t.add(i); matched_d.add(j)

        keep = []
        for i, tr in enumerate(self.tracks):  # 未匹配：滑行（用預測位置補歷史），過久則刪除
            if i not in matched_t:
                tr.age += 1
                if tr.age > self.max_age:
                    continue
                tr.hist = np.vstack([[*tr.x[:2], tr.yaw], tr.hist])[: self.H]
            keep.append(tr)
        self.tracks = keep
        for j in range(len(dets)):            # 新軌跡
            if j not in matched_d:
                self.tracks.append(_Track(dets[j], self.sig, meta[j]))

    def export(self):
        """回傳成熟且本幀有偵測到的軌跡歷史 (N,H,3) 與其 meta。"""
        ok = [t for t in self.tracks if len(t.hist) >= self.H and t.age == 0]
        if not ok:
            return np.zeros((0, self.H, 3)), []
        return np.stack([t.hist for t in ok]), [t.meta for t in ok]


# --------------------------------------------------------------------------- #
# 偵測串流模擬
# --------------------------------------------------------------------------- #
def simulate_streams(rng, delay, sig_t, sig_r_deg, n_obj=16, n_bad=2, det_sigma=0.2,
                     yaw_sigma_deg=3.0, miss=0.1, n_fp=1.0, u=8, H=4, rng_comm=40.0):
    t = u + delay
    while True:
        pos = np.zeros((t + 1, n_obj, 2)); yaw = np.zeros((t + 1, n_obj))
        pos[0] = rng.uniform(-50, 50, (n_obj, 2)); yaw[0] = rng.uniform(-np.pi, np.pi, n_obj)
        spd = rng.uniform(0, 1.5, n_obj)
        for s in range(1, t + 1):
            yaw[s] = yaw[s - 1] + rng.normal(0, 0.01, n_obj)
            pos[s] = pos[s - 1] + spd[:, None] * np.stack([np.cos(yaw[s]), np.sin(yaw[s])], 1)
        th_e, th_n = rng.uniform(-np.pi, np.pi, 2)
        ang, dist = rng.uniform(-np.pi, np.pi), rng.uniform(15, 35)
        E0, N0 = np.zeros(2), dist * np.array([np.cos(ang), np.sin(ang)])
        E = lambda s: np.array([*(E0 + (s - u) * np.array([np.cos(th_e), np.sin(th_e)])), th_e])
        N = lambda s: np.array([*(N0 + (s - u) * np.array([np.cos(th_n), np.sin(th_n)])), th_n])
        ve = np.linalg.norm(pos[u] - E(u)[:2], axis=1) < rng_comm
        vn = np.linalg.norm(pos[u] - N(u)[:2], axis=1) < rng_comm
        if (ve & vn).sum() >= 5:
            break

    bad = rng.choice(np.flatnonzero(ve & vn), min(n_bad, (ve & vn).sum()), replace=False)
    bad_off = {int(i): rng.uniform(2.0, 3.5) * np.array([np.cos(a), np.sin(a)])
               for i, a in zip(bad, rng.uniform(-np.pi, np.pi, len(bad)))}

    def detect(F, s, is_nb):
        idx = np.flatnonzero((np.linalg.norm(pos[s] - F[:2], axis=1) < rng_comm) & (rng.random(n_obj) > miss))
        loc = np.column_stack([(pos[s][idx] - F[:2]) @ rot(F[2]), wrap_pi(yaw[s][idx] - F[2])])
        loc[:, :2] += rng.normal(0, det_sigma, (len(idx), 2))
        loc[:, 2] += rng.normal(0, np.deg2rad(yaw_sigma_deg), len(idx))
        if is_nb:
            for k, i in enumerate(idx):
                if int(i) in bad_off:
                    loc[k, :2] += bad_off[int(i)]
        ids = list(idx)
        for _ in range(rng.poisson(n_fp)):    # 假陽性
            loc = np.vstack([loc, [*rng.uniform(-rng_comm, rng_comm, 2), rng.uniform(-np.pi, np.pi)]])
            ids.append(-1)
        return loc, ids

    out = {}
    for name, P, is_nb in (("ego", E, False), ("nb", N, True)):
        trk = Tracker(H=H)
        frames = []
        for s in range(u + 1):
            d, ids = detect(P(s), s, is_nb)
            frames.append((d, ids))
            T = se2_compose(se2_inv(P(s)), P(s - 1)) if s > 0 else None
            trk.update(d, T, ids)
        hist, meta = trk.export()
        # oracle：用 GT id 串歷史，並補償到 u 時刻座標系
        common = set(frames[u][1]) - {-1}
        for s in range(u - H + 1, u + 1):
            common &= set(frames[s][1])
        oid = sorted(common)
        oh = np.zeros((len(oid), H, 3))
        for k in range(H):
            d, ids = frames[u - k]
            Tk = se2_compose(se2_inv(P(u)), P(u - k))
            for n_, i in enumerate(oid):
                oh[n_, k] = warp_tracks(Tk, d[ids.index(i)][None])[0]
        out[name] = dict(trk_hist=hist, trk_meta=meta, ora_hist=oh, ora_meta=oid)

    ego_obs_t, _ = detect(E(t), t, False)
    T_true = se2_compose(se2_inv(E(u)), N(u))
    eps = np.array([rng.normal(0, sig_t), rng.normal(0, sig_t), np.deg2rad(rng.normal(0, sig_r_deg))])

    def gt_at_t(meta):
        return np.array([(pos[t][i] - E(t)[:2]) @ rot(E(t)[2]) if i >= 0 else [np.nan, np.nan] for i in meta]).reshape(-1, 2)

    return dict(out=out, ego_obs_t=ego_obs_t, T_true=T_true, T0=se2_compose(T_true, eps),
                T_e=se2_compose(se2_inv(E(t)), E(u)), delay=delay, gt_at_t=gt_at_t)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--delay_ms", type=int, default=200)
    ap.add_argument("--sig_t", type=float, default=0.6)
    ap.add_argument("--sig_r", type=float, default=0.6)
    ap.add_argument("--n_bad", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    rows = {}
    for _ in range(a.trials):
        sc = simulate_streams(rng, a.delay_ms // 100, a.sig_t, a.sig_r, n_bad=a.n_bad)
        for src in ("ora", "trk"):
            e, n = sc["out"]["ego"], sc["out"]["nb"]
            eh, nh = e[f"{src}_hist"], n[f"{src}_hist"]
            if len(eh) < 3 or len(nh) < 3:
                continue
            gt = sc["gt_at_t"](n[f"{src}_meta"])
            for vname in ("A. 無修正 (T0 + 等速傳播)", "D. + closed-loop feedback (1輪)"):
                res = coanchor(eh, nh, sc["T0"], sc["delay"], sc["T_e"], sc["ego_obs_t"], **VARIANTS[vname])
                et, er = pose_err(res.T_final, sc["T_true"])
                ok = ~np.isnan(gt[:, 0])
                box = np.linalg.norm(res.states[ok, :2] - gt[ok], axis=1).mean()
                if len(res.pairs):
                    em, nm = e[f"{src}_meta"], n[f"{src}_meta"]
                    prec = np.mean([em[a_] == nm[b_] and em[a_] >= 0 for a_, b_ in res.pairs])
                else:
                    prec = np.nan
                rows.setdefault((src, vname), []).append((et, er, box, prec, len(eh), len(nh)))

    print(f"delay={a.delay_ms}ms noise=({a.sig_t}m,{a.sig_r}°) bad={a.n_bad} trials={a.trials}\n")
    print(f"{'歷史來源':<8}{'變體':<30}{'pose平移(m)':>12}{'旋轉(°)':>9}{'box誤差(m)':>12}{'匹配正確率':>12}{'ego/nb軌跡數':>14}")
    for (src, vname), v in rows.items():
        v = np.array(v, float)
        print(f"{'oracle' if src=='ora' else 'tracker':<8}{vname:<28}{v[:,0].mean():>12.3f}{v[:,1].mean():>9.3f}"
              f"{v[:,2].mean():>12.3f}{np.nanmean(v[:,3]):>12.2f}{v[:,4].mean():>8.1f}/{v[:,5].mean():.1f}")


if __name__ == "__main__":
    main()
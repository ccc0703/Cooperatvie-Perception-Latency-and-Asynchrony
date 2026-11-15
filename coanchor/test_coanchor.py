#!/usr/bin/env python3
"""
test_coanchor.py -- 驗證 CoAnchor object-level 實作（coanchor_lite.py）與實驗管線。

原則：不用「同一份程式再跑一次」來驗證自己。每個測試都對照一個獨立的來源：
  - 獨立的數值求解器（scipy BFGS 解 Huber 目標）
  - 論文公式的獨立重寫（Eq. 5/17/19/21–24/26 的 oracle）
  - 閉式解 / 理論值（Cramér-Rao 下界、chi-square 分布、Rayleigh 分布的期望）
  - 不變性 / 等變性（改變座標系、改變順序，結果必須相應改變或完全不變）
  - 有已知答案的合成資料（含負對照：該失敗的地方必須失敗）

用法： python test_coanchor.py            # 全部測試
       python test_coanchor.py -k nis     # 只跑名稱含 nis 的測試
       python test_coanchor.py --slow     # 增加蒙地卡羅次數
需要同目錄的 coanchor_lite.py、coanchor_tracker.py；T21、T22 另需 v2v4real_experiment.py、opv2v_io.py、v2v4real_io.py、pyyaml。
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np
from scipy.optimize import minimize

import coanchor_lite as cl
from coanchor_lite import (Params, _weighted_rigid, coanchor, make_scene, match_objects, pose_err, propagate,
                           refine_pose, rot, se2_compose, se2_inv, short_history_residual, warp_tracks,
                           wrap_half, wrap_pi)

SLOW = 1


def rand_se2(rng, tmax=30.0, thmax=np.pi):
    return np.array([*rng.uniform(-tmax, tmax, 2), rng.uniform(-thmax, thmax)])


def ang_diff(a, b):
    return wrap_pi(a - b)


def make_problem(rng, N=15, H=4, T_true=None, sigma=0.0, extent=30.0):
    """回傳 ego、nb（nb 的座標使得 T_true 把它對到 ego）。sigma = 每一邊每個點的位置噪聲。"""
    base = rng.uniform(-extent, extent, (N, 1, 2))
    pos = base + np.cumsum(rng.normal(0, 0.4, (N, H, 2)), 1)
    yaw = rng.uniform(-np.pi, np.pi, (N, H))
    ego = np.concatenate([pos, yaw[..., None]], -1)
    nb = warp_tracks(se2_inv(T_true), ego)
    ego = ego.copy(); ego[..., :2] += rng.normal(0, sigma, ego[..., :2].shape)
    nb[..., :2] += rng.normal(0, sigma, nb[..., :2].shape)
    return ego, nb


def huber_obj(x, P, Q, w, d):
    r = np.linalg.norm(P @ rot(x[2]).T + x[:2] - Q, axis=1)
    return float((w * np.where(r <= d, 0.5 * r * r, d * (r - 0.5 * d))).sum())


def flat_points(ego, nb, pairs, q, L):
    a, b = pairs[:, 0], pairs[:, 1]
    return nb[b, :L, :2].reshape(-1, 2), ego[a, :L, :2].reshape(-1, 2), np.repeat(q, L)


# =========================================================================== #
# A. SE(2) 代數
# =========================================================================== #
def test_T01_se2_algebra():
    rng = np.random.default_rng(1)
    for _ in range(300):
        T, S = rand_se2(rng, 50), rand_se2(rng, 50)
        assert np.allclose(se2_compose(T, se2_inv(T)), 0, atol=1e-9)
        assert np.allclose(se2_compose(se2_inv(T), T), 0, atol=1e-9)
        X = rng.normal(size=(6, 4, 3)) * 10
        l, r = warp_tracks(se2_compose(T, S), X), warp_tracks(T, warp_tracks(S, X))
        assert np.allclose(l[..., :2], r[..., :2], atol=1e-8) and np.allclose(ang_diff(l[..., 2], r[..., 2]), 0, atol=1e-9)
    return "逆元、結合律、warp 的合成性：300 組隨機 SE(2) 通過"


def test_T02_wrap_half():
    rng = np.random.default_rng(2)
    a = rng.uniform(-30, 30, 20000)
    w = wrap_half(a)
    assert (w >= -np.pi / 2 - 1e-12).all() and (w < np.pi / 2 + 1e-12).all()
    assert np.allclose(wrap_half(a + np.pi), w, atol=1e-9)                  # π 週期（前後方向模糊）
    assert np.allclose(np.sin(2 * w), np.sin(2 * a), atol=1e-9)
    return "wrap_half 值域 [-π/2, π/2)、π 週期、與 2a 的三角函數一致"


# =========================================================================== #
# B. IRLS 位姿求解
# =========================================================================== #
def test_T03_irls_exact_recovery():
    rng = np.random.default_rng(3)
    p, worst = Params(), 0.0
    for _ in range(200):
        T = rand_se2(rng, 20, 0.6)
        ego, nb = make_problem(rng, 12, 4, T, 0.0)
        pairs = np.stack([np.arange(12), np.arange(12)], 1)
        Th = refine_pose(ego, nb, pairs, rng.uniform(0.2, 1, 12), p)
        worst = max(worst, np.abs(Th[:2] - T[:2]).max(), abs(ang_diff(Th[2], T[2])))
    assert worst < 1e-8, worst
    return f"無噪聲、任意權重下精確還原已知位姿，最大誤差 {worst:.1e}"


def test_T04_irls_vs_independent_solver():
    rng = np.random.default_rng(4)
    p, L = Params(), 4
    dt, dth, dobj = [], [], []
    for _ in range(150):
        T = rand_se2(rng, 1.0, 0.05)
        ego, nb = make_problem(rng, 15, 4, T, 0.2)
        q = rng.uniform(0.3, 1.0, 15)
        pairs = np.stack([np.arange(15), np.arange(15)], 1)
        Th = refine_pose(ego, nb, pairs, q, p)
        P, Q, w = flat_points(ego, nb, pairs, q, L)
        best = min((minimize(huber_obj, x0, args=(P, Q, w, p.huber), method="BFGS", options=dict(gtol=1e-10))
                    for x0 in (np.zeros(3), T, Th)), key=lambda r: r.fun)
        dt.append(np.linalg.norm(Th[:2] - best.x[:2])); dth.append(abs(ang_diff(Th[2], best.x[2])))
        dobj.append(huber_obj(Th, P, Q, w, p.huber) - best.fun)
    dt, dth, dobj = map(np.array, (dt, dth, dobj))
    assert np.percentile(dt, 95) < 5e-3 and np.percentile(dth, 95) < 5e-4, (np.percentile(dt, 95), np.percentile(dth, 95))
    assert (dobj < 1e-4).mean() > 0.95
    return (f"與 scipy BFGS 解同一個 Huber 目標比較：平移差 中位 {np.median(dt):.1e} / 95% {np.percentile(dt, 95):.1e} m，"
            f"旋轉差 95% {np.percentile(dth, 95):.1e} rad，目標值差 95% {np.percentile(dobj, 95):.1e}")


def test_T05_irls_outlier_robustness():
    rng = np.random.default_rng(5)
    p, N = Params(), 16
    e_ls, e_h, e_w = [], [], []
    for _ in range(300):
        T = rand_se2(rng, 1.0, 0.05)
        ego, nb = make_problem(rng, N, 4, T, 0.1)
        bad = rng.choice(N, 4, replace=False)
        for i in bad:
            a = rng.uniform(-np.pi, np.pi)
            nb[i, :, :2] += 3.0 * np.array([np.cos(a), np.sin(a)])
        pairs = np.stack([np.arange(N), np.arange(N)], 1)
        P, Q, w1 = flat_points(ego, nb, pairs, np.ones(N), 4)
        err = lambda Th: np.linalg.norm(Th[:2] - T[:2])
        e_ls.append(err(_weighted_rigid(P, Q, w1)))
        e_h.append(err(refine_pose(ego, nb, pairs, np.ones(N), p)))
        q = np.ones(N); q[bad] = 1e-3
        e_w.append(err(refine_pose(ego, nb, pairs, q, p)))
    e_ls, e_h, e_w = map(np.mean, (e_ls, e_h, e_w))
    assert e_h < 0.6 * e_ls and e_w < 0.5 * e_h, (e_ls, e_h, e_w)
    return f"25% 離群點（3 m）：最小平方 {e_ls:.3f} m → Huber IRLS {e_h:.3f} m → 再加權重壓低離群點 {e_w:.3f} m"


def test_T06_irls_invariances():
    rng = np.random.default_rng(6)
    p = Params()
    for _ in range(100):
        T = rand_se2(rng, 1.0, 0.05)
        ego, nb = make_problem(rng, 12, 4, T, 0.2)
        q = rng.uniform(0.2, 1, 12)
        pairs = np.stack([np.arange(12), np.arange(12)], 1)
        a = refine_pose(ego, nb, pairs, q, p)
        b = refine_pose(ego, nb, pairs, 7.3 * q, p)                  # 權重整體縮放不應改變解
        perm = rng.permutation(12)
        c = refine_pose(ego, nb, pairs[perm], q[perm], p)            # 配對順序不應改變解
        assert np.allclose(a, b, atol=1e-9) and np.allclose(a, c, atol=1e-9)
    return "權重整體縮放、配對順序置換：解不變（誤差 < 1e-9）"


def test_T07_irls_converged_within_10_iterations():
    rng = np.random.default_rng(7)
    dev = []
    for _ in range(200):
        T = rand_se2(rng, 1.0, 0.05)
        ego, nb = make_problem(rng, 15, 4, T, 0.2)
        pairs = np.stack([np.arange(15), np.arange(15)], 1)
        q = np.ones(15)
        a = refine_pose(ego, nb, pairs, q, Params(n_irls=10))
        b = refine_pose(ego, nb, pairs, q, Params(n_irls=200, irls_tol=1e-12))
        dev.append(np.linalg.norm(a[:2] - b[:2]))
    dev = np.array(dev)
    assert np.percentile(dev, 99) < 1e-3
    return f"論文設定 10 次迭代 vs 收斂解：平移差 中位 {np.median(dev):.1e}，99% {np.percentile(dev, 99):.1e} m（遠小於 0.05 m 的殘差）"


def test_T08_irls_statistical_efficiency():
    """已知對應下，IRLS 的誤差應接近 Cramér-Rao 下界（等權重、各點獨立同向性噪聲）。"""
    rng = np.random.default_rng(8)
    p, N, H, sig = Params(), 20, 4, 0.2
    T = np.array([0.3, -0.2, 0.01])
    et, er, crb_t, crb_r = [], [], [], []
    for _ in range(int(1500 * SLOW)):
        ego, nb = make_problem(rng, N, H, T, sig, extent=35.0)
        pairs = np.stack([np.arange(N), np.arange(N)], 1)
        Th = refine_pose(ego, nb, pairs, np.ones(N), p)
        et.append(np.linalg.norm(Th[:2] - T[:2])); er.append(abs(np.rad2deg(ang_diff(Th[2], T[2]))))
        X = (ego[..., :2].reshape(-1, 2))                              # 估計後座標 ≈ ego 座標
        F = np.zeros((3, 3))
        for x, y in X:
            F += np.array([[1, 0, -y], [0, 1, x], [-y, x, x * x + y * y]])
        C = np.linalg.inv(F / (2 * sig ** 2))                          # 每個點的差噪聲方差 = 2σ²
        crb_t.append(np.sqrt(np.pi / 2) * np.sqrt((C[0, 0] + C[1, 1]) / 2)); crb_r.append(np.rad2deg(np.sqrt(2 / np.pi) * np.sqrt(C[2, 2])))
    et, er, ct, cr = np.mean(et), np.mean(er), np.mean(crb_t), np.mean(crb_r)
    assert 0.95 * ct < et < 1.15 * ct and 0.95 * cr < er < 1.15 * cr, (et, ct, er, cr)
    return (f"平移 實測 {et:.4f} m / CRB {ct:.4f} m（{et / ct:.2f}×）；旋轉 實測 {er:.4f}° / CRB {cr:.4f}°（{er / cr:.2f}×）"
            f"——求解器在其操作範圍內接近統計最佳")


# =========================================================================== #
# C. Kalman / NIS（Eq. 4-6, 16-23）
# =========================================================================== #
def oracle_matrices(p, n, q0):
    d = 1.0 / (q0 + p.eps_q)
    Sig0 = np.diag([p.alpha_p * d] * 2 + [p.alpha_v * d] * 2 + [np.deg2rad(p.alpha_psi_deg) ** 2 * d])   # Eq.17
    A = np.eye(5); A[0, 2] = A[1, 3] = n                                                              # Eq.18
    sa2, sq = p.sigma_a ** 2, np.deg2rad(p.sigma_psi_q_deg) ** 2
    Q = np.diag([sa2 * max(n * n, 1)] * 2 + [sa2 * max(n, 1)] * 2 + [sq * max(n, 1)])               # Eq.19
    H = np.zeros((3, 5)); H[0, 0] = H[1, 1] = H[2, 4] = 1.0                                           # Eq.21
    R = np.diag([p.sigma_zp ** 2] * 2 + [np.deg2rad(p.sigma_zpsi_deg) ** 2])
    return Sig0, A, Q, H, R


def one_anchor(x0):
    """由 x0=[x,y,vx,vy,yaw] 造一條 H=4 的歷史，使 propagate 內部算出的狀態正好是 x0。"""
    k = np.arange(4)[:, None]
    pos = x0[:2] - k * x0[2:4]
    return np.concatenate([pos, np.full((4, 1), x0[4])], 1)[None]


def test_T09_nis_calibration():
    """在模型假設成立的合成資料上，NIS 必須服從 chi-square(3)：均值 3、通過 7.815 的比例 0.95。"""
    from scipy.stats import chi2
    rng = np.random.default_rng(9)
    p = Params(assoc_gate=1e9)
    out = []
    for n, q0 in ((0, 0.8), (2, 0.5), (2, 0.2), (4, 0.9)):
        Sig0, A, Q, H, R = oracle_matrices(p, n, q0)
        M = int(8000 * SLOW)
        x0 = np.array([5.0, -3.0, 0.7, 0.2, 0.4])
        e = rng.multivariate_normal(np.zeros(5), Sig0, M)
        w = rng.multivariate_normal(np.zeros(5), Q, M)
        v = rng.multivariate_normal(np.zeros(3), R, M)
        xt = (x0 + e) @ A.T + w
        z = xt @ H.T + v
        hist = one_anchor(x0)
        nis = np.array([propagate(hist, np.zeros(3), n, np.array([[0, 0]]), np.array([q0]), z[i:i + 1], p, False)[1][0] for i in range(M)])
        pass_rate = (nis <= p.chi2_gate).mean()
        ks = __import__("scipy.stats", fromlist=["kstest"]).kstest(nis, "chi2", args=(3,)).statistic
        assert abs(nis.mean() - 3) < 0.12 and abs(pass_rate - 0.95) < 0.01 and ks < 0.02, (n, q0, nis.mean(), pass_rate, ks)
        out.append(f"n={n},q0={q0}: 均值 {nis.mean():.2f}、通過率 {pass_rate:.3f}、KS {ks:.3f}")
    assert abs(chi2.ppf(0.95, 3) - Params().chi2_gate) < 1e-3
    return "NIS ~ χ²(3)（理論 均值 3、通過率 0.95）：" + "；".join(out)


def test_T10_kalman_update_vs_oracle():
    rng = np.random.default_rng(10)
    p = Params(assoc_gate=1e9)
    n_acc = n_rej = 0
    for _ in range(600):
        n, q0 = int(rng.integers(0, 5)), float(rng.uniform(0.1, 1))
        Sig0, A, Q, H, R = oracle_matrices(p, n, q0)
        x0 = np.array([*rng.uniform(-20, 20, 2), *rng.uniform(-1, 1, 2), rng.uniform(-3, 3)])
        xp = A @ x0
        z = np.array([xp[0] + rng.normal(0, 1.2), xp[1] + rng.normal(0, 1.2), xp[4] + rng.normal(0, 0.3)])
        st, nis, acc = propagate(one_anchor(x0), np.zeros(3), n, np.array([[0, 0]]), np.array([q0]), z[None], p, True)
        Pp = A @ Sig0 @ A.T + Q
        nu = np.array([z[0] - xp[0], z[1] - xp[1], wrap_half(z[2] - xp[4])])
        S = H @ Pp @ H.T + R
        nis_o = nu @ np.linalg.inv(S) @ nu
        assert abs(nis[0] - nis_o) < 1e-9
        xn = xp + Pp @ H.T @ np.linalg.inv(S) @ nu if nis_o <= p.chi2_gate else xp
        xn = xn.copy(); xn[4] = wrap_pi(xn[4])
        assert np.allclose(st[0], xn, atol=1e-9) and bool(acc[0]) == (nis_o <= p.chi2_gate)
        n_acc += nis_o <= p.chi2_gate; n_rej += nis_o > p.chi2_gate
    assert n_acc > 50 and n_rej > 50                                     # 兩個分支都有被測到
    return f"Kalman 更新與 NIS 門檻對獨立 oracle 一致（接受 {n_acc}、拒絕 {n_rej} 次，誤差 < 1e-9）"


# =========================================================================== #
# D. 匹配
# =========================================================================== #
def make_tracks(rng, N, H=4, extent=30.0):
    base = rng.uniform(-extent, extent, (N, 1, 2))
    v = rng.normal(0, 1.0, (N, 1, 2))
    pos = base - np.arange(H)[None, :, None] * v + rng.normal(0, 0.0, (N, H, 2))
    return np.concatenate([pos, rng.uniform(-np.pi, np.pi, (N, 1, 1)) * np.ones((N, H, 1))], -1)


def test_T11_matching_recovers_permutation():
    rng = np.random.default_rng(11)
    p = Params()
    acc0, acc1, n1 = [], [], 0
    for trial in range(200):
        N = 15
        ego = make_tracks(rng, N)
        perm = rng.permutation(N)
        nb0 = ego[perm].copy()                                            # 無噪聲
        pairs, _ = match_objects(ego, nb0, p)
        acc0.append(np.mean([perm[b] == a for a, b in pairs]) if len(pairs) else 0)
        nb1 = ego[perm].copy(); nb1[..., :2] += rng.normal(0, 0.15, nb1[..., :2].shape)
        drop_e = rng.random(N) < 0.15
        keep_e = np.flatnonzero(~drop_e)
        pairs, _ = match_objects(ego[keep_e], nb1, p)
        ok = [perm[b] == keep_e[a] for a, b in pairs]
        acc1.append(np.mean(ok) if ok else 0); n1 += len(ok)
    assert np.mean(acc0) == 1.0 and np.mean(acc1) > 0.97, (np.mean(acc0), np.mean(acc1))
    return f"置換還原：無噪聲 {np.mean(acc0):.3f}；0.15 m 噪聲且 15% 漏檢 {np.mean(acc1):.3f}（{n1} 個配對）"


def test_T12_matching_gate_and_one_to_one():
    rng = np.random.default_rng(12)
    p = Params()
    for _ in range(200):
        ego, nb = make_tracks(rng, 14), make_tracks(rng, 11)
        pairs, cost = match_objects(ego, nb, p)
        assert len(set(pairs[:, 0])) == len(pairs) and len(set(pairs[:, 1])) == len(pairs)
        d = np.linalg.norm(ego[pairs[:, 0], 0, :2] - nb[pairs[:, 1], 0, :2], axis=1)
        assert (d <= p.d_gate + 1e-9).all() and (cost >= 0).all()
    return "一對一指派；所有配對的距離 ≤ d_gate = 4 m；成本非負（200 組隨機場景）"


# =========================================================================== #
# E. closed-loop feedback（Eq. 7, 8, 24-26）
# =========================================================================== #
def oracle_short_history(ego, nb, pairs, p):
    out = []
    for a, b in pairs:
        s = 0.0
        for k in range(min(p.L_p + 1, ego.shape[1])):
            dpos = nb[b, k, :2] - ego[a, k, :2]
            dyaw = nb[b, k, 2] - ego[a, k, 2]
            dyaw = 0.5 * np.arctan2(np.sin(2 * dyaw), np.cos(2 * dyaw))              # 獨立的半週期 wrap
            s += dpos @ dpos / p.sigma_zp ** 2 + dyaw ** 2 / np.deg2rad(p.sigma_zpsi_deg) ** 2
        out.append(s / min(p.L_p + 1, ego.shape[1]))
    return np.array(out)


def test_T13_feedback_weights_vs_oracle():
    rng = np.random.default_rng(13)
    p = Params()
    n_checked = 0
    for _ in range(150):
        sc = make_scene(rng, 2, 0.6, 0.6, n_bad=4)
        res = coanchor(sc["ego_hist"], sc["nb_hist"], sc["T0"], 2, sc["T_e"], sc["ego_obs_t"], p, True, True, 1)
        nb = warp_tracks(sc["T0"], sc["nb_hist"])
        pairs, cost = match_objects(sc["ego_hist"], nb, p)
        q0 = np.exp(-cost / p.T_q)
        dT0 = refine_pose(sc["ego_hist"], nb, pairs, q0, p)
        nbc = warp_tracks(dT0, nb)
        nis = propagate(nbc, sc["T_e"], 2, pairs, q0, sc["ego_obs_t"], p, False)[1]
        r_fb = np.nan_to_num(nis, nan=0.0) + oracle_short_history(sc["ego_hist"], nbc, pairs, p)
        q_exp = q0 * np.exp(-p.lam_fb * r_fb)                                            # Eq.(7)(8)
        assert np.allclose(res.q, q_exp, rtol=1e-9, atol=1e-12)
        assert np.allclose(short_history_residual(sc["ego_hist"], nbc, pairs, p), oracle_short_history(sc["ego_hist"], nbc, pairs, p))
        # NIS 未通過的配對（有對應但 d² 超過門檻）必定被丟棄：q <= q0·exp(-λ·7.815) < q_min（q0 <= 1）
        bad = nis > p.chi2_gate
        assert (res.q[bad] < p.q_min).all()
        n_checked += int(bad.sum())
    assert np.exp(-p.lam_fb * p.chi2_gate) < p.q_min
    assert n_checked >= 5, n_checked                       # NIS 未通過本來就少見（見 T26 對丟棄機制的分解）
    return (f"q = q0·exp(−1.5·(d²+r_hist)) 對獨立 oracle 一致（150 個場景，每場景 4 個壞偵測）；NIS 未通過的 {n_checked} 個配對全部 q < q_min；"
            f"上界 exp(−1.5×7.815) = {np.exp(-p.lam_fb * p.chi2_gate):.1e} < 1e-4")


# =========================================================================== #
# F. 整條管線的不變性 / 等變性
# =========================================================================== #
def run(sc, p=None, refine=True, kalman=True, n_loop=1):
    return coanchor(sc["ego_hist"], sc["nb_hist"], sc["T0"], sc["delay"], sc["T_e"], sc["ego_obs_t"], p or Params(), refine, kalman, n_loop)


def test_T14_equivariance_neighbor_frame():
    rng = np.random.default_rng(14)
    worst = 0.0
    for _ in range(80):
        sc = make_scene(rng, 2, 0.6, 0.6)
        r0 = run(sc)
        G = rand_se2(rng, 40)
        sc2 = dict(sc, nb_hist=warp_tracks(se2_inv(G), sc["nb_hist"]), T0=se2_compose(sc["T0"], G))
        r1 = run(sc2)
        assert np.array_equal(r0.pairs, r1.pairs)
        d = max(np.abs(r0.dT - r1.dT).max(), np.abs(r0.states - r1.states).max(),
                np.abs(se2_compose(r0.T_final, G)[:2] - r1.T_final[:2]).max())
        worst = max(worst, d)
        assert d < 1e-7, d
    return f"把 neighbor 自己的座標系任意重新定義（T0→T0∘G）：配對、ΔT、傳播狀態不變，T_final→T_final∘G（最大差 {worst:.1e}）"


def test_T15_equivariance_ego_frame():
    """IRLS 的停止條件 max|T_new−T| < 1e-4 與座標系有關，所以分兩段：(1) 關掉提早停止，必須精確等變；(2) 預設設定下量偏差上限。"""
    rng = np.random.default_rng(15)
    worst = {True: 0.0, False: 0.0}
    for _ in range(80):
        sc = make_scene(rng, 2, 0.6, 0.6)
        Hu, Ht = rand_se2(rng, 40), rand_se2(rng, 40)
        sc2 = dict(sc, ego_hist=warp_tracks(Hu, sc["ego_hist"]), T0=se2_compose(Hu, sc["T0"]),
                   T_e=se2_compose(Ht, se2_compose(sc["T_e"], se2_inv(Hu))),
                   ego_obs_t=warp_tracks(Ht, sc["ego_obs_t"][:, None, :])[:, 0])
        for strict in (True, False):
            p = Params(irls_tol=0.0) if strict else Params()
            r0, r1 = run(sc, p), run(sc2, p)
            assert np.array_equal(r0.pairs, r1.pairs)
            exp = r0.states.copy()
            exp[:, :2] = r0.states[:, :2] @ rot(Ht[2]).T + Ht[:2]
            exp[:, 2:4] = r0.states[:, 2:4] @ rot(Ht[2]).T
            exp[:, 4] = wrap_pi(r0.states[:, 4] + Ht[2])
            d = max(np.abs(exp[:, :4] - r1.states[:, :4]).max(), np.abs(ang_diff(exp[:, 4], r1.states[:, 4])).max(),
                    np.abs(se2_compose(Hu, r0.T_final)[:2] - r1.T_final[:2]).max())
            if strict:
                nn = ~np.isnan(r0.nis)
                d = max(d, np.abs(r0.nis[nn] - r1.nis[nn]).max(initial=0))
            worst[strict] = max(worst[strict], d)
    assert worst[True] < 1e-9, worst
    assert worst[False] < 1e-3, worst
    return (f"ego 在 u 與 t 兩時刻的座標系任意旋轉平移：傳播狀態依剛體變換等變、NIS 不變。關掉提早停止時最大差 {worst[True]:.1e}（精確）；"
            f"預設停止條件下最大差 {worst[False]:.1e} m（停止條件與座標系有關，影響 < 1 mm）")


def test_T16_permutation_and_far_objects_invariance():
    rng = np.random.default_rng(16)
    for _ in range(80):
        sc = make_scene(rng, 2, 0.6, 0.6)
        r0 = run(sc)
        pe, pn, po = (rng.permutation(len(sc["ego_hist"])), rng.permutation(len(sc["nb_hist"])), rng.permutation(len(sc["ego_obs_t"])))
        sc2 = dict(sc, ego_hist=sc["ego_hist"][pe], nb_hist=sc["nb_hist"][pn], ego_obs_t=sc["ego_obs_t"][po])
        r1 = run(sc2)
        inv_n = np.argsort(pn)
        assert np.allclose(r0.T_final, r1.T_final, atol=1e-9) and np.allclose(r0.states, r1.states[inv_n], atol=1e-9)
        far_e = np.concatenate([sc["ego_hist"], make_tracks(rng, 3) + np.array([900.0, 900.0, 0])])
        far_n = np.concatenate([sc["nb_hist"], make_tracks(rng, 3) + np.array([-900.0, 400.0, 0])])
        r2 = run(dict(sc, ego_hist=far_e, nb_hist=far_n))
        assert np.allclose(r0.T_final, r2.T_final, atol=1e-9) and np.allclose(r0.states, r2.states[:len(r0.states)], atol=1e-9)
    return "物件順序任意置換：T_final 不變、狀態對應置換；加入遠離所有配對的物件：T_final 與原物件的狀態不變"


def test_T17_determinism():
    rng = np.random.default_rng(17)
    sc = make_scene(rng, 2, 0.6, 0.6)
    a, b = run(sc), run(sc)
    assert np.array_equal(a.T_final, b.T_final) and np.array_equal(a.states, b.states, equal_nan=True)
    return "相同輸入兩次輸出逐位元相同"


# =========================================================================== #
# G. 追蹤器
# =========================================================================== #
def test_T18_tracker_ego_motion_compensation():
    from coanchor_tracker import Tracker
    rng = np.random.default_rng(18)
    n_obj, S, H = 6, 14, 4
    p0 = rng.uniform(-30, 30, (n_obj, 2)); vel = rng.uniform(-1.2, 1.2, (n_obj, 2)); yaw0 = rng.uniform(-3, 3, n_obj)
    E = [np.array([0.0, 0.0, 0.3])]
    for _ in range(S):
        E.append(se2_compose(E[-1], np.array([1.0 + rng.normal(0, 0.1), rng.normal(0, 0.05), rng.normal(0, 0.05)])))
    def det(s):
        pos = p0 + vel * s
        return np.column_stack([(pos - E[s][:2]) @ rot(E[s][2]), wrap_pi(yaw0 - E[s][2])])
    def truth(s, k):
        pos = p0 + vel * (s - k)
        return np.column_stack([(pos - E[s][:2]) @ rot(E[s][2]), wrap_pi(yaw0 - E[s][2])])
    errs = {}
    for comp in (True, False):
        trk = Tracker(H=H)
        for s in range(S):
            T = se2_compose(se2_inv(E[s]), E[s - 1]) if (s > 0 and comp) else None
            trk.update(det(s), T, list(range(n_obj)))
        hist, meta = trk.export()
        assert len(meta) == n_obj
        order = np.argsort(meta)
        err = max(np.linalg.norm(hist[order][:, k, :2] - truth(S - 1, k)[:, :2], axis=1).max() for k in range(H))
        errs[comp] = err
    assert errs[True] < 0.1 and errs[False] > 1.0, errs
    return f"追蹤器歷史（已補償自身運動）與真值最大差 {errs[True]:.3f} m；負對照（不補償）{errs[False]:.2f} m，測試對此錯誤是敏感的"


# =========================================================================== #
# H. 實驗管線（OPV2V 轉接器與噪聲注入）
# =========================================================================== #
def make_synth_opv2v(root, n_frames=45):
    import yaml
    rng = np.random.default_rng(1)
    agents = {"101": (-10., 0., 20.), "102": (20., 6., 35.), "103": (300., 0., 0.)}
    scene = os.path.join(root, "sceneA")
    for ag in agents:
        os.makedirs(os.path.join(scene, ag), exist_ok=True)
    ids = list(range(500, 520))
    p0 = rng.uniform(-35, 35, (len(ids), 2)); yaw_o = rng.uniform(-180, 180, len(ids)); sp = rng.uniform(0, 10, len(ids))
    for n in range(n_frames):
        t = n * 0.1
        obj = {k: (p0[j] + sp[j] * t * np.array([np.cos(np.deg2rad(yaw_o[j])), np.sin(np.deg2rad(yaw_o[j]))]), yaw_o[j], sp[j] * 3.6)
               for j, k in enumerate(ids)}
        egos = {}
        for ag, (x, y, yw) in agents.items():
            yaw = yw + 3 * t
            egos[ag] = (np.array([x, y]) + 7.0 * t * np.array([np.cos(np.deg2rad(yw)), np.sin(np.deg2rad(yw))]), yaw, 25.2)
        for ag, (pos, yaw, v) in egos.items():
            lid = pos + 0.5 * np.array([np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw))])
            veh = {}
            for k, (p, yw, s) in obj.items():
                if np.linalg.norm(p - lid) < 70:
                    veh[k] = {"angle": [0.0, float(yw), 0.0], "center": [0.1, 0.05, 0.7], "extent": [2.4, 1.0, 0.7],
                              "location": [float(p[0]), float(p[1]), 0.0], "speed": float(s)}
            for o, (p2, y2, s2) in egos.items():
                if o != ag and np.linalg.norm(p2 - lid) < 70:
                    veh[int(o)] = {"angle": [0.0, float(y2), 0.0], "center": [0.1, 0.05, 0.7], "extent": [2.4, 1.0, 0.7],
                                   "location": [float(p2[0]), float(p2[1]), 0.0], "speed": 25.2}
            d = {"lidar_pose": [float(lid[0]), float(lid[1]), 1.9, 0.0, float(yaw), 0.0],
                 "true_ego_pos": [float(pos[0]), float(pos[1]), 0.0, 0.0, float(yaw), 0.0],
                 "predicted_ego_pos": [float(pos[0]) + 0.03, float(pos[1]), 0.0, 0.0, float(yaw), 0.0], "vehicles": veh}
            with open(os.path.join(scene, ag, f"{2000 + 2 * n:06d}.yaml"), "w") as f:
                yaml.safe_dump(d, f)
    return root


def test_T19_opv2v_adapter_geometry():
    import yaml
    import opv2v_io
    from v2v4real_experiment import agent_pose, boxes_in_lidar
    with tempfile.TemporaryDirectory() as tmp:
        # (1) center 偏移與 yaw：box 中心 = location + R(yaw)·center
        d = {"lidar_pose": [1, 2, 3, 0, 10, 0], "vehicles": {7: {"location": [10.0, 20.0, 0.0], "angle": [0, 40.0, 0], "center": [0.3, 0.1, 0.7],
                                                                  "extent": [2, 1, 1], "speed": 3.0}, 99: {"location": [0, 0, 0], "angle": [0, 0, 0], "center": [0, 0, 0], "extent": [2, 1, 1]}}}
        f = os.path.join(tmp, "x.yaml"); yaml.safe_dump(d, open(f, "w"))
        fr = opv2v_io.load_frame(f, drop_id=99)
        th = np.deg2rad(40.0)
        exp = [10 + np.cos(th) * 0.3 - np.sin(th) * 0.1, 20 + np.sin(th) * 0.3 + np.cos(th) * 0.1, th]
        assert fr.ids.tolist() == [7] and np.allclose(fr.boxes[0], exp, atol=1e-12)
        # (2) 兩車的 lidar 座標標註，經各自 pose 轉回世界座標後必須是同一點（floor = 0 的來源）
        make_synth_opv2v(tmp)
        sc = opv2v_io.pair_scenarios(os.path.join(tmp, "sceneA"), "101,102")[0]
        worst, n = 0.0, 0
        for i in range(0, len(sc.files), 4):
            fa, fb = sc.frame("101", i), sc.frame("102", i)
            la, lb = boxes_in_lidar(fa, "world", "l2w"), boxes_in_lidar(fb, "world", "l2w")
            wa, wb = warp_tracks(agent_pose(fa, "l2w"), la), warp_tracks(agent_pose(fb, "l2w"), lb)
            for ia, id_ in enumerate(fa.ids):
                j = np.flatnonzero(fb.ids == id_)
                if len(j):
                    worst = max(worst, np.abs(wa[ia, :2] - wb[j[0], :2]).max()); n += 1
        assert n > 50 and worst < 1e-9
        # (3) 真值相對 pose 與兩車 lidar 座標標註的關係：T_true ∘ box_nb = box_ego
        Te, Tn = agent_pose(fa, "l2w"), agent_pose(fb, "l2w")
        Tt = se2_compose(se2_inv(Te), Tn)
        k = [(ia, int(np.flatnonzero(fb.ids == id_)[0])) for ia, id_ in enumerate(fa.ids) if id_ in fb.ids]
        assert np.abs(warp_tracks(Tt, lb[[b for _, b in k]])[:, :2] - la[[a for a, _ in k]][:, :2]).max() < 1e-9
    return f"center 偏移公式正確；{n} 個共視物體經各自 pose 轉回世界座標誤差 {worst:.0e}；T_true∘box_nb = box_ego"


def run_cli(args, cwd):
    out = subprocess.run([sys.executable, os.path.join(cwd, "v2v4real_experiment.py")] + args, capture_output=True, text=True, cwd=cwd)
    assert out.returncode == 0, out.stderr[-500:]
    return out.stdout


def parse_rows(txt):
    rows = {}
    for line in txt.splitlines():
        if "±" in line:
            continue
        m = re.match(r"^([A-E])\.\s.*?\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$", line)
        if m and m.group(1) not in rows:
            rows[m.group(1)] = [float(x) if x != "nan" else np.nan for x in m.groups()[1:]]
    return rows


def test_T20_noise_injection_closed_form():
    """A 的 pose 誤差只由注入噪聲決定：E‖t‖ = σ√(π/2)，E|θ| = σ√(2/π)。你的 18 個場景實測 A = 0.752 m／0.478° 正是這個值。"""
    here = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory() as tmp:
        make_synth_opv2v(tmp)
        base = ["run", "--root", tmp, "--dataset", "opv2v", "--pair", "101,102", "--max_dist", "70", "--delay_ms", "200", "--stride", "2"]
        txt = run_cli(base + ["--sig_t", "0.6", "--sig_r", "0.6", "--reps", "20", "--seed", "5"], here)
        a = parse_rows(txt)["A"]
        et, er = 0.6 * np.sqrt(np.pi / 2), 0.6 * np.sqrt(2 / np.pi)
        assert abs(a[0] / et - 1) < 0.08 and abs(a[1] / er - 1) < 0.08, (a, et, er)
        txt0 = run_cli(base + ["--sig_t", "0", "--sig_r", "0", "--reps", "2"], here)
        assert parse_rows(txt0)["A"][:2] == [0.0, 0.0]
    return (f"σ=0.6：A 實測 {a[0]:.3f} m／{a[1]:.3f}°，理論 {et:.3f}／{er:.3f}；σ=0 時 A 誤差為 0。"
            f"（你的 18 個場景在 σ=0.05～0.6 的 A 也逐點吻合：0.063/0.126/0.252/0.504/0.752 m）")


# =========================================================================== #
# I. 機制測試與負對照（有已知答案）
# =========================================================================== #
def paired_stats(x):
    x = np.asarray(x)
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x))


def test_T21_mechanism_feedback_targets_bad_detections():
    """feedback 應該只在存在壞地標時有幫助：壞偵測越多，D−C 越負；沒有壞偵測時約為 0 或略差。並與「用真值挑掉壞配對」的 oracle 比較。"""
    rng = np.random.default_rng(21)
    p = Params()
    out, eff = {}, None
    for n_bad in (0, 2, 4):
        dc, rec = [], []
        for _ in range(int(250 * SLOW)):
            sc = make_scene(rng, 2, 0.6, 0.6, n_bad=n_bad)
            rB = coanchor(sc["ego_hist"], sc["nb_hist"], sc["T0"], 2, sc["T_e"], sc["ego_obs_t"], p, True, False, 0)
            rD = coanchor(sc["ego_hist"], sc["nb_hist"], sc["T0"], 2, sc["T_e"], sc["ego_obs_t"], p, True, True, 1)
            eB, eD = pose_err(rB.T_final, sc["T_true"])[0], pose_err(rD.T_final, sc["T_true"])[0]
            # oracle：用真值 pose 判斷每個配對是否為壞地標（殘差 > 1 m），只用好的配對求解
            nb = warp_tracks(sc["T0"], sc["nb_hist"])
            res_true = np.linalg.norm(warp_tracks(sc["T_true"], sc["nb_hist"])[rB.pairs[:, 1], :, :2] - sc["ego_hist"][rB.pairs[:, 0], :, :2], axis=-1).mean(1)
            good = res_true < 1.0
            if good.sum() >= 3:
                dT = refine_pose(sc["ego_hist"], nb, rB.pairs[good], rB.q0[good], p)
                eO = pose_err(se2_compose(dT, sc["T0"]), sc["T_true"])[0]
            else:
                eO = eB
            dc.append(eD - eB); rec.append((eB, eD, eO))
        rec = np.array(rec); m, se = paired_stats(dc)
        out[n_bad] = (m, se, rec.mean(0))
    m0, m2, m4 = out[0][0], out[2][0], out[4][0]
    assert m0 > -0.004, m0                                  # 沒有壞偵測：feedback 不該有明顯好處
    assert m2 < -0.01 and abs(m2) > 5 * out[2][1], out[2]   # 有壞偵測：顯著改善
    assert m4 < m2 < m0                                     # 壞偵測越多，改善越大（劑量反應）
    eB, eD, eO = out[2][2]
    frac = (eB - eD) / (eB - eO)
    assert eO < eD < eB and frac > 0.5, (eB, eD, eO)
    lines = "；".join(f"壞偵測 {k} 個：D−C = {v[0]:+.4f}±{v[1]:.4f} m" for k, v in out.items())
    return f"{lines}。壞偵測 2 個時 B/D/oracle = {eB:.3f}/{eD:.3f}/{eO:.3f} m，feedback 拿回 oracle 增益的 {frac:.0%}"


def test_T22_negative_controls():
    """該失敗的地方必須失敗：證明方法沒有偷看真值。"""
    rng = np.random.default_rng(22)
    p = Params()
    ratio, gain, acc_unrel, acc_rel = [], 0, [], []
    n = int(150 * SLOW)
    for _ in range(n):
        sc = make_scene(rng, 2, 10.0, 15.0)                  # 初始 pose 誤差遠超過 4 m 的匹配門檻
        rA = coanchor(sc["ego_hist"], sc["nb_hist"], sc["T0"], 2, sc["T_e"], sc["ego_obs_t"], p, False, False, 0)
        rB = coanchor(sc["ego_hist"], sc["nb_hist"], sc["T0"], 2, sc["T_e"], sc["ego_obs_t"], p, True, False, 0)
        eA, eB = pose_err(rA.T_final, sc["T_true"])[0], pose_err(rB.T_final, sc["T_true"])[0]
        ratio.append(eB / eA); gain += eB < 0.3 * eA
    for _ in range(n):
        sc = make_scene(rng, 2, 0.3, 0.3)
        other = make_scene(rng, 2, 0.3, 0.3)                 # 完全無關的 neighbor 場景
        r_ok = coanchor(sc["ego_hist"], sc["nb_hist"], sc["T0"], 2, sc["T_e"], sc["ego_obs_t"], p, True, False, 0)
        r_un = coanchor(sc["ego_hist"], other["nb_hist"], sc["T0"], 2, sc["T_e"], sc["ego_obs_t"], p, True, False, 0)
        acc_rel.append(np.mean(r_ok.nis[~np.isnan(r_ok.nis)] <= p.chi2_gate) if (~np.isnan(r_ok.nis)).any() else np.nan)
        acc_un = r_un.nis[~np.isnan(r_un.nis)]
        acc_unrel.append(np.mean(acc_un <= p.chi2_gate) if len(acc_un) else np.nan)
    ratio = np.array(ratio)
    ok, un = np.nanmean(acc_rel), np.nanmean(acc_unrel)
    assert np.median(ratio) > 0.9 and gain / n < 0.15, (np.median(ratio), gain / n)
    assert ok > 0.85 and un < 0.5 * ok, (ok, un)
    return (f"初始誤差 10 m／15°：B 誤差/A 誤差 中位 {np.median(ratio):.2f}，只有 {gain / n:.0%} 的試驗修到 A 的 30% 以下（超出門檻就救不回來）；"
            f"NIS 門檻通過率：對的場景 {ok:.2f}、無關的場景 {un:.2f}（門檻有區分力）")


def test_T08b_irls_efficiency_vs_number_of_pairs():
    """對應稀疏場景的問題：共視物體少時，IRLS 的誤差是否仍接近 CRB？若遠高於 CRB，代表問題不在噪聲而在別處（例如錯誤配對）。"""
    rng = np.random.default_rng(81)
    p, sig, T = Params(), 0.2, np.array([0.3, -0.2, 0.01])
    rows, msg = [], []
    for N in (4, 5, 8, 12, 20, 40):
        et, ct = [], []
        for _ in range(int(1200 * SLOW)):
            ego, nb = make_problem(rng, N, 4, T, sig, extent=35.0)
            pairs = np.stack([np.arange(N), np.arange(N)], 1)
            Th = refine_pose(ego, nb, pairs, np.ones(N), p)
            et.append(np.linalg.norm(Th[:2] - T[:2]))
            F = np.zeros((3, 3))
            for x, y in ego[..., :2].reshape(-1, 2):
                F += np.array([[1, 0, -y], [0, 1, x], [-y, x, x * x + y * y]])
            C = np.linalg.inv(F / (2 * sig ** 2))
            ct.append(np.sqrt(np.pi / 2) * np.sqrt((C[0, 0] + C[1, 1]) / 2))
        # 用中位數比較，避免幾何接近退化時 CRB 發散的少數樣本主導
        r = np.median(et) / np.median(ct)
        rows.append((N, np.median(et), np.median(ct), r))
        msg.append(f"N={N}: 實測 {np.median(et):.3f} / CRB {np.median(ct):.3f} m（{r:.2f}×）")
        assert 0.85 < r < 1.25, (N, r)
    return "；".join(msg) + "。結論：只要對應正確，物體少只會讓誤差依 1/√N 變大，求解器沒有額外損失"


def test_T23_matching_cost_vs_oracle_and_bruteforce():
    """匹配成本（Eq.10–12）用獨立的逐項迴圈重寫，並與暴力列舉的全域最佳指派比較。"""
    import itertools
    rng = np.random.default_rng(23)
    p = Params()

    def sigs(c):
        edge, pair = [], []
        for i in range(len(c)):
            others = [j for j in range(len(c)) if j != i]
            d = np.array([np.linalg.norm(c[i] - c[j]) for j in others])
            nbrs = [others[o] for o in np.argsort(d)[:p.K]]
            edge.append(np.sort(d[np.argsort(d)[:p.K]]))
            pair.append(np.sort([np.linalg.norm(c[a] - c[b]) for ia, a in enumerate(nbrs) for b in nbrs[ia + 1:]]))
        return edge, pair

    def dsort(a, b):
        m = min(len(a), len(b))
        return 0.0 if m == 0 else float(np.mean(np.abs(np.sort(a)[:m] - np.sort(b)[:m])))

    def oracle(ego, nb):
        Ne, Nn, H = len(ego), len(nb), ego.shape[1]
        ee, ep = sigs(ego[:, 0, :2]); ne, np_ = sigs(nb[:, 0, :2])
        C = np.full((Ne, Nn), np.inf)
        for a in range(Ne):
            for b in range(Nn):
                if np.linalg.norm(ego[a, 0, :2] - nb[b, 0, :2]) > p.d_gate:
                    continue
                dv = np.mean([np.linalg.norm((ego[a, k, :2] - ego[a, k + 1, :2]) - (nb[b, k, :2] - nb[b, k + 1, :2])) for k in range(H - 1)])
                dg = (p.w_edge * dsort(ee[a], ne[b]) + p.w_pair * dsort(ep[a], np_[b])) / (p.w_edge + p.w_pair)
                C[a, b] = p.lam_vel * dv / p.T_vel + p.lam_graph * dg / p.T_graph
        return C

    n_cmp = 0
    for _ in range(300):
        Ne, Nn = int(rng.integers(3, 7)), int(rng.integers(3, 7))
        ego = make_tracks(rng, Ne, extent=8.0)
        nb = make_tracks(rng, Nn, extent=8.0)
        k = min(Ne, Nn)
        nb[:k] = ego[:k] + rng.normal(0, 0.3, ego[:k].shape)            # 讓部分物體有真實對應
        C = oracle(ego, nb)
        pairs, cost = match_objects(ego, nb, p)
        # 成本矩陣：回傳的成本必須等於 oracle
        for (a, b), c in zip(pairs, cost):
            assert abs(C[a, b] - c) < 1e-9
        # 暴力列舉：先最大化「未被門檻擋掉」的配對數，再最小化總成本
        best, best_key = None, None
        if Ne <= Nn:
            cands = [list(zip(range(Ne), perm)) for perm in itertools.permutations(range(Nn), Ne)]
        else:
            cands = [[(a, b) for b, a in zip(range(Nn), perm)] for perm in itertools.permutations(range(Ne), Nn)]
        for cand in cands:
            ok = [(a, b) for a, b in cand if np.isfinite(C[a, b])]
            key = (-len(ok), sum(C[a, b] for a, b in ok))
            if best_key is None or key < best_key:
                best_key, best = key, sorted(ok)
        assert sorted(map(tuple, pairs.tolist())) == best, (pairs.tolist(), best)
        n_cmp += len(best)
    return f"匹配成本對獨立 oracle 一致；300 個小場景（3–6 個物體）的指派與暴力列舉的全域最佳完全相同（共 {n_cmp} 個配對）"


def test_T24_no_label_leakage():
    import inspect
    from coanchor_tracker import Tracker
    rng = np.random.default_rng(24)
    assert not any("id" in k.lower() or "meta" in k.lower() or "gt" in k.lower() for k in inspect.signature(coanchor).parameters)
    out = []
    for labels in (list(range(6)), list(rng.permutation(100)[:6]), [-1] * 6):
        trk = Tracker(H=4)
        r = np.random.default_rng(0)
        base = r.uniform(-20, 20, (6, 2)); vel = r.uniform(-1, 1, (6, 2))
        for s in range(8):
            d = np.column_stack([base + vel * s + r.normal(0, 0.2, (6, 2)), r.uniform(-1, 1, 6)])
            trk.update(d, None, labels)
        out.append(trk.export()[0])
    assert all(np.array_equal(out[0], o) for o in out[1:])
    return "coanchor() 的參數不含任何 id 或真值；追蹤器的輸出與傳入的標籤完全無關（3 種標籤方式結果逐位元相同）。id 只用於實驗的評分"


def test_T25_edge_cases():
    rng = np.random.default_rng(25)
    sc = make_scene(rng, 2, 0.6, 0.6)
    p = Params()
    ok = []
    def go(name, ego, nb, obs, delay=2, expect_identity=False):
        r = coanchor(ego, nb, sc["T0"], delay, sc["T_e"], obs, p, True, True, 1)
        assert np.isfinite(r.T_final).all() and r.states.shape == (len(nb), 5) and np.isfinite(r.states).all(), name
        if expect_identity:
            assert np.allclose(r.T_final, sc["T0"], atol=1e-12), name
        ok.append(name)
    z = np.zeros((0, 4, 3))
    go("ego 沒有物體", z, sc["nb_hist"], sc["ego_obs_t"], expect_identity=True)
    go("neighbor 沒有物體", sc["ego_hist"], z, sc["ego_obs_t"], expect_identity=True)
    go("兩邊都沒有物體", z, z, np.zeros((0, 3)), expect_identity=True)
    go("只有 2 個物體（< 3 條配對，不做修正）", sc["ego_hist"][:2], sc["nb_hist"][:2], sc["ego_obs_t"], expect_identity=False)
    go("ego 當前時刻沒有偵測", sc["ego_hist"], sc["nb_hist"], np.zeros((0, 3)))
    go("延遲 0 frame", sc["ego_hist"], sc["nb_hist"], sc["ego_obs_t"], delay=0)
    go("延遲 10 frame", sc["ego_hist"], sc["nb_hist"], sc["ego_obs_t"], delay=10)
    line = np.zeros((8, 4, 3)); line[:, :, 0] = np.linspace(-20, 20, 8)[:, None] - np.arange(4)[None] * 0.5
    go("所有物體共線（幾何退化）", line, line.copy() + rng.normal(0, 0.05, line.shape), line[:, 0])
    same = np.zeros((6, 4, 3)) + np.array([3.0, 3.0, 0.0])
    go("所有物體位置完全重合", same, same.copy(), same[:, 0])
    # 只有兩條配對時必須完全不修正
    r = coanchor(sc["ego_hist"][:2], sc["nb_hist"][:2], sc["T0"], 2, sc["T_e"], sc["ego_obs_t"], p, True, False, 1)
    assert len(r.pairs) < 3 and np.allclose(r.dT, 0, atol=1e-12)
    return "、".join(ok) + "：全部不報錯、輸出有限；少於 3 條配對時確實不做 pose 修正（ΔT = 0）"


def test_T26_sensitivity_to_self_chosen_settings():
    """論文沒寫清楚、由我們自訂的選擇（沒有 ego 對應的配對如何處理、關聯距離、歷史點數）是否會改變結論？
    這裡直接用 coanchor() 的公開參數比較，不重寫流程。"""
    rng = np.random.default_rng(26)
    out = {}
    for n_bad in (0, 2):
        scs = [make_scene(rng, 2, 0.6, 0.6, n_bad=n_bad) for _ in range(int(250 * SLOW))]
        def err(p, n_loop):
            return np.array([pose_err(coanchor(sc["ego_hist"], sc["nb_hist"], sc["T0"], 2, sc["T_e"], sc["ego_obs_t"], p, True, True, n_loop).T_final,
                                       sc["T_true"])[0] for sc in scs])
        e = {"B": err(Params(), 0), "ours": err(Params(), 1), "exclude": err(Params(nan_mode="exclude"), 1),
             "keep_q0": err(Params(nan_mode="keep_q0"), 1), "gate1.5": err(Params(assoc_gate=1.5), 1),
             "gate4.0": err(Params(assoc_gate=4.0), 1), "Lp2": err(Params(L_p=2), 1)}
        out[n_bad] = e
    e0, e2 = out[0], out[2]
    d = lambda e, a, b: (e[a] - e[b]).mean()
    se = lambda e, a, b: (e[a] - e[b]).std(ddof=1) / np.sqrt(len(e[a]))
    # (1) 壞偵測下，兩種「合理」的處理都比不用 feedback 好
    assert d(e2, "ours", "B") < -3 * se(e2, "ours", "B") and d(e2, "exclude", "B") < -3 * se(e2, "exclude", "B")
    # (2) 我們的選擇不劣於論文字面解讀（排除）
    assert d(e2, "ours", "exclude") <= 0.005
    # (3) 關聯距離不敏感
    assert abs(d(e2, "gate1.5", "ours")) < 0.01 and abs(d(e2, "gate4.0", "ours")) < 0.01
    # (4) 「不懲罰、維持原權重」的讀法會讓 feedback 失效甚至更糟：這是需要被記錄的風險，不是我們採用的做法
    assert d(e2, "keep_q0", "B") > 0
    f = lambda a, b, e: f"{d(e, a, b):+.4f}±{se(e, a, b):.4f}"
    return (f"壞偵測 2 個：B {e2['B'].mean():.3f}、D(預設) {e2['ours'].mean():.3f}、D(排除無對應配對) {e2['exclude'].mean():.3f}、"
            f"D(不懲罰) {e2['keep_q0'].mean():.3f} m；關聯距離 1.5／4.0 m 相對 2.5 m 的差 {f('gate1.5', 'ours', e2)}／{f('gate4.0', 'ours', e2)}；"
            f"歷史點 3 個 vs 4 個 {f('Lp2', 'ours', e2)}。乾淨偵測：D(預設)−B = {f('ours', 'B', e0)}，D(排除)−D(預設) = {f('exclude', 'ours', e0)}")


def test_T27_why_pairs_are_dropped():
    """feedback 丟棄配對的機制：NIS 項還是短歷史殘差項？（影響我們怎麼描述這個模組）"""
    rng = np.random.default_rng(27)
    p = Params()
    need = lambda q0: np.log(q0 / p.q_min) / p.lam_fb
    cnt = {}
    for n_bad in (0, 2):
        c = dict(n=0, dropped=0, nan=0, nis_fail=0, by_nis=0, by_hist=0)
        for _ in range(250):
            sc = make_scene(rng, 2, 0.6, 0.6, n_bad=n_bad)
            nb = warp_tracks(sc["T0"], sc["nb_hist"])
            pairs, cost = match_objects(sc["ego_hist"], nb, p)
            q0 = np.exp(-cost / p.T_q)
            dT = refine_pose(sc["ego_hist"], nb, pairs, q0, p)
            nbc = warp_tracks(dT, nb)
            nis = propagate(nbc, sc["T_e"], 2, pairs, q0, sc["ego_obs_t"], p, False)[1]
            h = oracle_short_history(sc["ego_hist"], nbc, pairs, p)
            n0 = np.nan_to_num(nis, nan=0.0)
            drop = q0 * np.exp(-p.lam_fb * (n0 + h)) < p.q_min
            c["n"] += len(pairs); c["dropped"] += drop.sum(); c["nan"] += np.isnan(nis).sum(); c["nis_fail"] += (nis > p.chi2_gate).sum()
            c["by_nis"] += (drop & (n0 > need(q0)) & ~(h > need(q0))).sum(); c["by_hist"] += (drop & (h > need(q0)) & ~(n0 > need(q0))).sum()
        cnt[n_bad] = c
    c = cnt[2]
    assert c["by_hist"] > 0.5 * c["dropped"] and c["by_nis"] < 0.1 * c["dropped"]
    return (f"壞偵測 2 個：{c['dropped'] / c['n']:.1%} 的配對被丟棄，其中 {c['by_hist'] / c['dropped']:.0%} 只靠短歷史殘差項、{c['by_nis'] / c['dropped']:.0%} 只靠 NIS 項；"
            f"NIS 超過門檻的配對只有 {c['nis_fail'] / c['n']:.1%}，沒有 ego 對應（NIS 為 NaN）的有 {c['nan'] / c['n']:.1%}。"
            f"乾淨偵測：{cnt[0]['dropped'] / cnt[0]['n']:.1%} 的配對被丟棄")


# =========================================================================== #
TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_T")]


def main():
    global SLOW
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", default="", help="只跑名稱含此字串的測試")
    ap.add_argument("--slow", action="store_true", help="增加蒙地卡羅次數")
    a = ap.parse_args()
    SLOW = 3 if a.slow else 1
    n_pass = n_fail = 0
    t_all = time.time()
    for fn in TESTS:
        if a.k and a.k not in fn.__name__:
            continue
        t0 = time.time()
        try:
            msg = fn()
            n_pass += 1
            print(f"[PASS] {fn.__name__[5:]:<54} {time.time() - t0:5.1f}s\n       {msg}")
        except Exception as e:                                                     # noqa: BLE001
            n_fail += 1
            print(f"[FAIL] {fn.__name__[5:]:<54} {time.time() - t0:5.1f}s\n       {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)
    print(f"\n{n_pass} 通過、{n_fail} 失敗（{time.time() - t_all:.0f}s）")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
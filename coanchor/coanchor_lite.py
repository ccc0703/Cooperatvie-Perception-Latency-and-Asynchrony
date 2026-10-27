#!/usr/bin/env python3
"""
coanchor_lite.py -- CoAnchor 的 object-level 模組（rule-based，只需 numpy + scipy）

對應論文：Sec. 4.3-4.5 與附錄 B.2。不含 BEV 特徵融合（Sec. 4.6）。
執行 `python coanchor_lite.py` 會在合成場景上比較四種變體。

座標與資料約定（這是我自己定的，論文沒有明說）
- track 陣列形狀 (N, H, 3)，最後一維是 [x, y, yaw]；axis=1 的 k=0 是延遲時刻 u_j，
  k 越大越舊。H = k_hist + 1 = 4。
- ego 與 neighbor 的歷史 box 都已補償各自的自身運動，統一表示在「u_j 時刻自己的座標系」。
- 速度單位 m/frame；延遲以 frame 數表示（10 Hz，200 ms = 2 frames）。
- SE(2) 以 np.array([x, y, theta]) 表示。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist

# --------------------------------------------------------------------------- #
# 參數：Table 5 / Table 6。標 [自訂] 的是論文沒給的。
# --------------------------------------------------------------------------- #
@dataclass
class Params:
    d_gate: float = 4.0
    lam_vel: float = 1.0
    lam_graph: float = 0.5
    T_vel: float = 2.0
    T_graph: float = 3.0
    K: int = 3
    w_edge: float = 1.0
    w_pair: float = 0.5
    n_min_trk: int = 3
    n_irls: int = 10
    huber: float = 0.5
    irls_tol: float = 1e-4
    chi2_gate: float = 7.815      # chi2_0.95(3)
    lam_fb: float = 1.5
    q_min: float = 1e-4
    eps_q: float = 1e-6
    T_q: float = 1.5
    alpha_p: float = 0.1
    alpha_v: float = 0.03
    alpha_psi_deg: float = 3.0
    sigma_a: float = 0.03
    sigma_psi_q_deg: float = 4.0
    sigma_zp: float = 0.7
    sigma_zpsi_deg: float = 8.0
    L_p: int = 3                  # 用 k = 0..L_p 的歷史點做 IRLS
    assoc_gate: float = 2.5       # [自訂] 傳播後 anchor 與 ego 當前偵測的最近鄰關聯距離


# --------------------------------------------------------------------------- #
# SE(2) 工具
# --------------------------------------------------------------------------- #
def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def wrap_half(a):
    """對應附錄 Eq.(22) 的 wrap_{1/2}：映射到 [-pi/2, pi/2)。"""
    return (a + np.pi / 2) % np.pi - np.pi / 2


def rot(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


IDENTITY = np.zeros(3)


def se2_compose(A, B):
    """A∘B：先套用 B 再套用 A。"""
    return np.array([*(rot(A[2]) @ B[:2] + A[:2]), wrap_pi(A[2] + B[2])])


def se2_inv(T):
    R = rot(T[2])
    return np.array([*(-R.T @ T[:2]), wrap_pi(-T[2])])


def warp_tracks(T, tracks):
    out = tracks.copy()
    out[..., :2] = tracks[..., :2] @ rot(T[2]).T + T[:2]
    out[..., 2] = wrap_pi(tracks[..., 2] + T[2])
    return out


def velocities(tracks):
    """(N,H,3) -> (N,H-1,2)，單位 m/frame，v[k] = p[k] - p[k+1]。"""
    return tracks[:, :-1, :2] - tracks[:, 1:, :2]


# --------------------------------------------------------------------------- #
# 4.3 匹配（Eq.10-12）
# --------------------------------------------------------------------------- #
def _graph_signatures(centers, K):
    D = cdist(centers, centers)
    edge, pair = [], []
    for i in range(len(centers)):
        order = np.argsort(D[i])
        order = order[order != i][:K]
        edge.append(np.sort(D[i, order]))
        pair.append(np.sort(pdist(centers[order])) if len(order) >= 2 else np.empty(0))
    return edge, pair


def _d_sort(a, b):
    m = min(len(a), len(b))
    return 0.0 if m == 0 else float(np.mean(np.abs(np.sort(a)[:m] - np.sort(b)[:m])))


def match_objects(ego, nb, p: Params):
    """ego (Ne,H,3)、nb (Nn,H,3)（nb 已用原始 pose 轉到 ego 座標系）。
    回傳 pairs (M,2) = [ego_idx, nb_idx] 與每對的匹配成本 (M,)。"""
    if len(ego) == 0 or len(nb) == 0:
        return np.zeros((0, 2), int), np.zeros(0)
    ve, vn = velocities(ego), velocities(nb)
    d_vel = np.linalg.norm(ve[:, None] - vn[None], axis=-1).mean(-1)

    ee, ep = _graph_signatures(ego[:, 0, :2], p.K)
    ne, np_ = _graph_signatures(nb[:, 0, :2], p.K)
    d_graph = np.zeros_like(d_vel)
    for a in range(len(ego)):
        for b in range(len(nb)):
            d_graph[a, b] = (p.w_edge * _d_sort(ee[a], ne[b]) +
                             p.w_pair * _d_sort(ep[a], np_[b])) / (p.w_edge + p.w_pair)

    cost = p.lam_vel * d_vel / p.T_vel + p.lam_graph * d_graph / p.T_graph
    BIG = 1e6
    cost = np.where(cdist(ego[:, 0, :2], nb[:, 0, :2]) > p.d_gate, BIG, cost)
    r, c = linear_sum_assignment(cost)
    ok = cost[r, c] < BIG / 2
    return np.stack([r[ok], c[ok]], 1), cost[r[ok], c[ok]]


# --------------------------------------------------------------------------- #
# IRLS 位姿（Eq.3 / 14 / 15）
# --------------------------------------------------------------------------- #
def _weighted_rigid(P, Q, w):
    W = w.sum()
    if W < 1e-12:
        return IDENTITY.copy()
    mp, mq = (w[:, None] * P).sum(0) / W, (w[:, None] * Q).sum(0) / W
    Pc, Qc = P - mp, Q - mq
    s = (w * (Pc[:, 0] * Qc[:, 1] - Pc[:, 1] * Qc[:, 0])).sum()
    c = (w * (Pc[:, 0] * Qc[:, 0] + Pc[:, 1] * Qc[:, 1])).sum()
    th = np.arctan2(s, c)
    return np.array([*(mq - rot(th) @ mp), th])


def refine_pose(ego, nb, pairs, q, p: Params):
    a, b = pairs[:, 0], pairs[:, 1]
    L = min(p.L_p + 1, ego.shape[1])
    P = nb[b, :L, :2].reshape(-1, 2)
    Q = ego[a, :L, :2].reshape(-1, 2)
    w_pair = np.repeat(q, L)
    T = IDENTITY.copy()
    for _ in range(p.n_irls):
        r = np.linalg.norm(P @ rot(T[2]).T + T[:2] - Q, axis=1)
        hw = np.where(r <= p.huber, 1.0, p.huber / np.maximum(r, 1e-9))  # Huber 的 IRLS 權重
        T_new = _weighted_rigid(P, Q, w_pair * hw)
        done = np.abs(T_new - T).max() < p.irls_tol
        T = T_new
        if done:
            break
    return T


# --------------------------------------------------------------------------- #
# 4.4 時間傳播 + Kalman 修正 + NIS gate（Eq.4-6, 16-23）
# --------------------------------------------------------------------------- #
def propagate(nb_corr, T_e, delay, pairs, q0, ego_obs_t, p: Params, apply_kalman=True):
    """把所有 neighbor 物件推到 ego@t。只有 pairs 內的物件是 anchor，才做 Kalman 修正。
    回傳 states (Nn,5)、nis (M,)（無對應則 NaN）、accepted (M,) bool。"""
    n = float(delay)
    vel = velocities(nb_corr).mean(1)
    x0 = np.column_stack([nb_corr[:, 0, :2], vel, nb_corr[:, 0, 2]])
    # ego 自身運動改寫：u_j 的 ego 座標系 -> t 的 ego 座標系
    R = rot(T_e[2])
    x0[:, :2] = x0[:, :2] @ R.T + T_e[:2]
    x0[:, 2:4] = x0[:, 2:4] @ R.T
    x0[:, 4] = wrap_pi(x0[:, 4] + T_e[2])

    A = np.eye(5)
    A[0, 2] = A[1, 3] = n
    states = x0 @ A.T                                    # Eq.(4)，等速模型

    M = len(pairs)
    nis, accepted = np.full(M, np.nan), np.zeros(M, bool)
    if M == 0 or len(ego_obs_t) == 0:
        return states, nis, accepted

    sa2 = p.sigma_a ** 2
    sq = np.deg2rad(p.sigma_psi_q_deg) ** 2
    Q = np.diag([sa2 * max(n * n, 1)] * 2 + [sa2 * max(n, 1)] * 2 + [sq * max(n, 1)])   # Eq.(19)
    Hm = np.zeros((3, 5)); Hm[0, 0] = Hm[1, 1] = Hm[2, 4] = 1.0                          # Eq.(21)
    Rm = np.diag([p.sigma_zp ** 2, p.sigma_zp ** 2, np.deg2rad(p.sigma_zpsi_deg) ** 2])
    apsi = np.deg2rad(p.alpha_psi_deg)

    for k, (_, b) in enumerate(pairs):
        d = 1.0 / (q0[k] + p.eps_q)
        Sig0 = np.diag([p.alpha_p * d] * 2 + [p.alpha_v * d] * 2 + [apsi ** 2 * d])      # Eq.(17)
        Pp = A @ Sig0 @ A.T + Q                                                          # Eq.(5)
        xp = states[b]
        dist = np.linalg.norm(ego_obs_t[:, :2] - xp[:2], axis=1)
        j = int(np.argmin(dist))
        if dist[j] > p.assoc_gate:
            continue                                                                     # 無可靠 ego 對應
        z = ego_obs_t[j]
        nu = np.array([z[0] - xp[0], z[1] - xp[1], wrap_half(z[2] - xp[4])])             # Eq.(22)
        S = Hm @ Pp @ Hm.T + Rm
        Sinv = np.linalg.inv(S)
        nis[k] = nu @ Sinv @ nu                                                          # Eq.(23)
        if nis[k] <= p.chi2_gate:
            accepted[k] = True
            if apply_kalman:
                xn = xp + Pp @ Hm.T @ Sinv @ nu
                xn[4] = wrap_pi(xn[4])
                states[b] = xn
    return states, nis, accepted


# --------------------------------------------------------------------------- #
# 4.5 closed-loop posterior scoring（Eq.7, 8, 24-26）
# --------------------------------------------------------------------------- #
def short_history_residual(ego, nb_corr, pairs, p: Params):
    a, b = pairs[:, 0], pairs[:, 1]
    L = min(p.L_p + 1, ego.shape[1])
    pos = ((nb_corr[b, :L, :2] - ego[a, :L, :2]) ** 2).sum(-1) / p.sigma_zp ** 2
    yaw = wrap_half(nb_corr[b, :L, 2] - ego[a, :L, 2]) ** 2 / np.deg2rad(p.sigma_zpsi_deg) ** 2
    return (pos + yaw).mean(1)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    T_final: np.ndarray        # ΔT ∘ T0，neighbor 座標系 -> ego@u_j
    dT: np.ndarray
    states: np.ndarray         # (Nn,5) 傳到 ego@t 的 neighbor 物件 [x,y,vx,vy,yaw]
    pairs: np.ndarray
    q0: np.ndarray
    q: np.ndarray
    nis: np.ndarray
    accepted: np.ndarray       # 哪些 pair 在最終傳播後仍是 active anchor


def coanchor(ego_hist, nb_hist, T0, delay, T_e, ego_obs_t, p: Params = Params(),
             refine=True, kalman=True, n_loop=1) -> Result:
    """
    ego_hist   (Ne,H,3)  ego 物件歷史，在 ego@u_j 座標系
    nb_hist    (Nn,H,3)  neighbor 物件歷史，在 neighbor 自己的座標系
    T0         (3,)      有噪聲的 neighbor->ego 相對 pose
    delay      通訊延遲（frame 數）
    T_e        (3,)      ego@u_j -> ego@t 的自身運動
    ego_obs_t  (Nt,3)    ego 在當前時刻 t 的偵測 [x,y,yaw]（ego@t 座標系）
    refine=False 時完全不做物件級修正（只用 T0 + 等速傳播）；n_loop=0 即無 feedback。
    """
    nb = warp_tracks(T0, nb_hist)
    pairs, q0, q, dT = np.zeros((0, 2), int), np.zeros(0), np.zeros(0), IDENTITY.copy()

    if refine:
        pairs, cost = match_objects(ego_hist, nb, p)
        q0 = np.exp(-cost / p.T_q)                                                       # Eq.(13)
        q = q0.copy()
        if len(pairs) >= p.n_min_trk:
            dT = refine_pose(ego_hist, nb, pairs, q0, p)
        for _ in range(n_loop):
            nb_corr = warp_tracks(dT, nb)
            _, nis, _ = propagate(nb_corr, T_e, delay, pairs, q0, ego_obs_t, p, apply_kalman=False)
            r_fb = np.nan_to_num(nis, nan=0.0) + short_history_residual(ego_hist, nb_corr, pairs, p)   # Eq.(7)
            q = q0 * np.exp(-p.lam_fb * r_fb)                                            # Eq.(8)
            keep = q >= p.q_min
            if keep.sum() >= p.n_min_trk:
                dT = refine_pose(ego_hist, nb, pairs[keep], q[keep], p)
            else:
                break                                                                    # 跳過後續 refinement

    nb_corr = warp_tracks(dT, nb)
    states, nis, acc = propagate(nb_corr, T_e, delay, pairs, q0, ego_obs_t, p, apply_kalman=kalman)
    return Result(se2_compose(dT, T0), dT, states, pairs, q0, q, nis, acc)


# --------------------------------------------------------------------------- #
# 合成場景（用來驗證，不依賴任何資料集）
# --------------------------------------------------------------------------- #
def make_scene(rng, delay, sig_t, sig_r_deg, n_obj=16, n_bad=2, det_sigma=0.2,
               yaw_sigma_deg=3.0, miss_prob=0.1, H=4, rng_comm=40.0):
    u, t = H - 1, H - 1 + delay
    S = t + 1
    while True:
        # 世界座標下的物件軌跡（近似等速，帶一點轉向）
        pos = np.zeros((S, n_obj, 2)); yaw = np.zeros((S, n_obj))
        pos[0] = rng.uniform(-50, 50, (n_obj, 2))
        yaw[0] = rng.uniform(-np.pi, np.pi, n_obj)
        spd = rng.uniform(0, 1.5, n_obj)
        for s in range(1, S):
            yaw[s] = yaw[s - 1] + rng.normal(0, 0.01, n_obj)
            pos[s] = pos[s - 1] + spd[:, None] * np.stack([np.cos(yaw[s]), np.sin(yaw[s])], 1)

        def pose_at(p0, th, v, s):
            return np.array([p0[0] + v * (s - u) * np.cos(th), p0[1] + v * (s - u) * np.sin(th), th])

        th_e, th_n = rng.uniform(-np.pi, np.pi, 2)
        ang = rng.uniform(-np.pi, np.pi); dist = rng.uniform(15, 35)
        E = lambda s: pose_at([0, 0], th_e, 1.0, s)
        N = lambda s: pose_at([dist * np.cos(ang), dist * np.sin(ang)], th_n, 1.0, s)

        vis_e = np.linalg.norm(pos[u] - E(u)[:2], axis=1) < rng_comm
        vis_n = np.linalg.norm(pos[u] - N(u)[:2], axis=1) < rng_comm
        if (vis_e & vis_n).sum() >= 5:
            break

    def to_frame(w_xy, w_yaw, F):
        return np.column_stack([(w_xy - F[:2]) @ rot(F[2]), wrap_pi(w_yaw - F[2])])

    def observe(idx, F, steps):
        out = np.zeros((len(idx), len(steps), 3))
        for k, s in enumerate(steps):
            o = to_frame(pos[s][idx], yaw[s][idx], F)
            o[:, :2] += rng.normal(0, det_sigma, (len(idx), 2))
            o[:, 2] += rng.normal(0, np.deg2rad(yaw_sigma_deg), len(idx))
            out[:, k] = o
        return out

    steps_hist = [u - k for k in range(H)]
    ids_e, ids_n = np.flatnonzero(vis_e), rng.permutation(np.flatnonzero(vis_n))
    ego_hist = observe(ids_e, E(u), steps_hist)
    nb_hist = observe(ids_n, N(u), steps_hist)

    # 低品質偵測：某些 neighbor 物件整段歷史帶固定偏移
    for i in rng.choice(len(ids_n), min(n_bad, len(ids_n)), replace=False):
        a = rng.uniform(-np.pi, np.pi)
        nb_hist[i, :, :2] += rng.uniform(2.0, 3.5) * np.array([np.cos(a), np.sin(a)])

    # ego 在 t 的當前偵測（會漏檢）
    vis_t = np.flatnonzero(np.linalg.norm(pos[t] - E(t)[:2], axis=1) < rng_comm)
    vis_t = vis_t[rng.random(len(vis_t)) > miss_prob]
    ego_obs_t = observe(vis_t, E(t), [t])[:, 0]

    T_true = se2_compose(se2_inv(E(u)), N(u))
    eps = np.array([rng.normal(0, sig_t), rng.normal(0, sig_t), np.deg2rad(rng.normal(0, sig_r_deg))])
    T0 = se2_compose(T_true, eps)
    T_e = se2_compose(se2_inv(E(t)), E(u))
    gt_nb_t = to_frame(pos[t][ids_n], yaw[t][ids_n], E(t))[:, :2]
    return dict(ego_hist=ego_hist, nb_hist=nb_hist, T0=T0, T_true=T_true, T_e=T_e,
                ego_obs_t=ego_obs_t, delay=delay, gt_nb_t=gt_nb_t)


def pose_err(T, T_true):
    return np.linalg.norm(T[:2] - T_true[:2]), abs(np.rad2deg(wrap_pi(T[2] - T_true[2])))


# --------------------------------------------------------------------------- #
# Demo：四種變體（對應論文 ablation 的精神）
# --------------------------------------------------------------------------- #
VARIANTS = {
    "A. 無修正 (T0 + 等速傳播)":      dict(refine=False, kalman=False, n_loop=0),
    "B. + 軌跡 IRLS 位姿修正":        dict(refine=True, kalman=False, n_loop=0),
    "C. + anchor Kalman/NIS":         dict(refine=True, kalman=True, n_loop=0),
    "D. + closed-loop feedback (1輪)": dict(refine=True, kalman=True, n_loop=1),
    "E. + closed-loop feedback (2輪)": dict(refine=True, kalman=True, n_loop=2),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--delay_ms", type=int, default=200)
    ap.add_argument("--sig_t", type=float, default=0.6)
    ap.add_argument("--sig_r", type=float, default=0.6)
    ap.add_argument("--n_bad", type=int, default=2, help="低品質偵測的 neighbor 物件數")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    delay = a.delay_ms // 100
    acc = {k: [] for k in VARIANTS}
    for _ in range(a.trials):
        sc = make_scene(rng, delay, a.sig_t, a.sig_r, n_bad=a.n_bad)
        for name, kw in VARIANTS.items():
            res = coanchor(sc["ego_hist"], sc["nb_hist"], sc["T0"], sc["delay"], sc["T_e"],
                           sc["ego_obs_t"], **kw)
            et, er = pose_err(res.T_final, sc["T_true"])
            box = np.linalg.norm(res.states[:, :2] - sc["gt_nb_t"], axis=1).mean()
            acc[name].append((et, er, box, res.accepted.sum() if len(res.accepted) else 0))

    print(f"delay={a.delay_ms}ms  pose noise=({a.sig_t}m, {a.sig_r}°)  bad dets={a.n_bad}  trials={a.trials}\n")
    print(f"{'變體':<34}{'pose平移誤差(m)':>16}{'pose旋轉誤差(°)':>16}{'t時刻box誤差(m)':>17}")
    for name, v in acc.items():
        v = np.array(v)
        print(f"{name:<32}{v[:,0].mean():>16.3f}{v[:,1].mean():>16.3f}{v[:,2].mean():>17.3f}")


if __name__ == "__main__":
    main()
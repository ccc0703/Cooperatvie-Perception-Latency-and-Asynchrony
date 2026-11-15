#!/usr/bin/env python3
"""
opv2v_probe.py -- 在寫 OPV2V 轉接器之前，先用數字確認 yaml 的約定。只讀 yaml，不碰 pcd / png。

要回答的問題：
  1. 三個 pose 欄位 (lidar_pose / true_ego_pos / predicted_ego_pos) 彼此差多少？哪個是沒有噪聲的真值？
  2. vehicles 的 location 是世界座標（兩個 agent 對同一輛車給出相同座標）還是各自的 lidar 座標？
  3. 兩車（以各 pose 欄位算）相距多遠、每幀共視幾輛、連續 4 幀都共視的有幾輛？
  4. vehicles 的 speed 單位是 m/s 還是 km/h？（用位移 / 0.1 s 反推）
  5. 幀編號間隔是否固定、是否剛好 100 ms。

用法：python opv2v_probe.py --scene <場景資料夾> [--stride 5] [--range 70]
依賴：numpy、pyyaml，以及同目錄的 v2v4real_io.py（只用它的 load_yaml）。
"""
from __future__ import annotations

import argparse
import itertools
import os

import numpy as np

from v2v4real_io import load_yaml

POSE_KEYS = ["lidar_pose", "true_ego_pos", "predicted_ego_pos"]
STEP_S = 0.1          # 假設：編號間隔 2 × fixed_delta_seconds 0.05；下面會用資料再核對編號間隔


def xy(v):
    return np.asarray(v, float)[:2]


class Loader:
    def __init__(self, scene):
        self.scene, self.cache = scene, {}

    def get(self, agent, idx):
        k = (agent, idx)
        if k not in self.cache:
            p = os.path.join(self.scene, agent, f"{idx:06d}.yaml")
            self.cache[k] = load_yaml(p) if os.path.exists(p) else None
        return self.cache[k]


def veh_locs(d):
    """vehicles -> {id: xy(world?)}"""
    return {int(k): xy(v["location"]) for k, v in (d.get("vehicles") or {}).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--range", type=float, default=70.0)
    a = ap.parse_args()

    agents = sorted(d for d in os.listdir(a.scene) if os.path.isdir(os.path.join(a.scene, d)))
    idxs = {ag: sorted(int(f[:-5]) for f in os.listdir(os.path.join(a.scene, ag)) if f.endswith(".yaml")) for ag in agents}
    common = sorted(set.intersection(*[set(v) for v in idxs.values()]))
    gaps = np.diff(common)
    print(f"場景 {os.path.basename(os.path.normpath(a.scene))}：agents={agents}，共同幀 {len(common)}")
    print("編號間隔分布：", {int(k): int(v) for k, v in zip(*np.unique(gaps, return_counts=True))}, "（預期只有 2）")
    L = Loader(a.scene)

    # ---- 1. pose 欄位彼此的差異（每個 agent）----
    print("\n[1] 各 agent 的 pose 欄位差異（xy 距離，公尺；yaw 差，度）")
    for ag in agents:
        dl, dt, dp = [], [], []
        for i in common[::a.stride]:
            d = L.get(ag, i)
            P = {k: np.asarray(d[k], float) for k in POSE_KEYS}
            dl.append(np.linalg.norm(P["lidar_pose"][:2] - P["true_ego_pos"][:2]))
            dt.append(abs((P["lidar_pose"][4] - P["true_ego_pos"][4] + 180) % 360 - 180))
            dp.append(np.linalg.norm(P["lidar_pose"][:2] - P["predicted_ego_pos"][:2]))
        print(f"  agent {ag}: |lidar - true| xy 均值 {np.mean(dl):.3f} m, yaw {np.mean(dt):.3f}°；"
              f"|lidar - predicted| xy 均值 {np.mean(dp):.3f} m")

    # ---- 2~4. 逐對 agent ----
    for A, B in itertools.combinations(agents, 2):
        print(f"\n===== agent {A} 與 {B} =====")
        dist = {k: [] for k in POSE_KEYS}
        shared_diff, n_shared, cov_now, cov_persist = [], [], [], []
        partner = {k: [] for k in POSE_KEYS}
        for i in common[::a.stride]:
            dA, dB = L.get(A, i), L.get(B, i)
            for k in POSE_KEYS:
                dist[k].append(np.linalg.norm(xy(dA[k]) - xy(dB[k])))
            lA, lB = veh_locs(dA), veh_locs(dB)
            sh = set(lA) & set(lB)
            n_shared.append(len(sh))
            shared_diff += [np.linalg.norm(lA[v] - lB[v]) for v in sh]
            # 夥伴車是否出現在對方標註，以及它的 location 與夥伴自己的 pose 欄位差多少
            for k in POSE_KEYS:
                if int(B) in lA:
                    partner[k].append(np.linalg.norm(lA[int(B)] - xy(dB[k])))
            # 共視：兩邊都標註，且（以 lidar_pose 為原點）在範圍內
            pA, pB = xy(dA["lidar_pose"]), xy(dB["lidar_pose"])
            near = {v for v in sh if np.linalg.norm(lA[v] - pA) < a.range and np.linalg.norm(lB[v] - pB) < a.range}
            cov_now.append(len(near))
            ids_hist = [near]
            ok = True
            for s in (1, 2, 3):
                j = i - 2 * s
                if j not in idxs[A] or j not in idxs[B]:
                    ok = False
                    break
                eA, eB = L.get(A, j), L.get(B, j)
                lA2, lB2 = veh_locs(eA), veh_locs(eB)
                pA2, pB2 = xy(eA["lidar_pose"]), xy(eB["lidar_pose"])
                ids_hist.append({v for v in set(lA2) & set(lB2)
                                 if np.linalg.norm(lA2[v] - pA2) < a.range and np.linalg.norm(lB2[v] - pB2) < a.range})
            if ok:
                cov_persist.append(len(set.intersection(*ids_hist)))

        for k in POSE_KEYS:
            d = np.array(dist[k])
            print(f"  [3] 兩車距離（{k}）：中位 {np.median(d):.1f} m，最小 {d.min():.1f}，最大 {d.max():.1f}")
        if shared_diff:
            sd = np.array(shared_diff)
            print(f"  [2] 兩邊都標註的同 id 車輛，location 差：中位 {np.median(sd):.4f} m，最大 {sd.max():.3f}"
                  f"（≈0 表示標註在共同的世界座標系；很大表示各自座標系）")
        else:
            print("  [2] 兩邊沒有共同標註的車輛，無法判斷標註座標系")
        for k in POSE_KEYS:
            if partner[k]:
                print(f"  [2b] 夥伴車在對方標註的 location，與夥伴 {k} 的 xy 距離：中位 {np.median(partner[k]):.3f} m")
        cp = np.array(cov_persist) if cov_persist else np.array([np.nan])
        print(f"  [3] 共同標註數均值 {np.mean(n_shared):.1f}；範圍 {a.range:.0f} m 內共視 {np.mean(cov_now):.1f}；"
              f"連續 4 幀共視 {np.nanmean(cp):.2f}，≥3 的比例 {np.nanmean(cp >= 3):.2f}")

    # ---- 4. speed 單位 ----
    print("\n[4] vehicles 的 speed 單位（用相鄰編號的世界座標位移 / 0.1 s 反推）")
    ratios = []
    for ag in agents:
        for i in common[::a.stride]:
            if i + 2 not in idxs[ag]:
                continue
            d0, d1 = L.get(ag, i), L.get(ag, i + 2)
            v0, v1 = d0.get("vehicles") or {}, d1.get("vehicles") or {}
            for k in set(v0) & set(v1):
                disp = np.linalg.norm(xy(v1[k]["location"]) - xy(v0[k]["location"]))
                est = disp / STEP_S                      # m/s
                sp = float(v0[k].get("speed", np.nan))
                if est > 1.0 and sp > 1.0:               # 太慢的不算，避免比值被噪聲主導
                    ratios.append(sp / est)
    if ratios:
        r = np.array(ratios)
        print(f"  speed / (位移/0.1s) 的中位數 {np.median(r):.2f}，四分位 {np.percentile(r, 25):.2f}～{np.percentile(r, 75):.2f}，樣本 {len(r)}"
              f"（≈3.6 表示 km/h；≈1 表示 m/s）")
    else:
        print("  沒有足夠的移動樣本。")


if __name__ == "__main__":
    main()
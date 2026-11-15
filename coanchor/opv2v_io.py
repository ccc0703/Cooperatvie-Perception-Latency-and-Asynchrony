#!/usr/bin/env python3
"""
opv2v_io.py -- OPV2V（含 test_culvercity）的 yaml 轉接器，介面與 v2v4real_io.Scenario 相同。

約定（已用 opv2v_probe.py 在 4 個場景上驗證，不是猜的）
- agent 資料夾以數字 id 命名；檔名 6 位數，編號間隔固定為 2；fixed_delta_seconds = 0.05，所以相鄰檔案 = 100 ms。
- vehicles 的 location / angle 在共同的世界座標系（同 id 車輛在兩個 agent 的檔案中座標完全相同）。
  angle = [roll, yaw(度), pitch]；center 是 box 中心相對 location 的偏移（車輛座標系）。
- 取 lidar_pose = [x, y, z, roll, yaw(度), pitch] 當感測器 pose（= true_ego_pos 加固定 0.5 m 安裝偏移，無噪聲）。
  不使用 predicted_ego_pos（帶定位噪聲）。只取 x、y、yaw，忽略 z / roll / pitch。
- vehicles.speed 單位為 km/h。
- 標註只含「至少有一個點打到該 agent 的 LiDAR」的車輛（OpenCOOD 文件），所以每個 agent 的標註不同。

因為標註是世界座標、再用同一個 lidar_pose 轉成各車 lidar 座標，兩車的標註在建構上彼此一致。
這正是 OPV2V 當乾淨基準的用處，但也代表 floor 診斷在這裡不再是獨立檢查。

用法：python opv2v_io.py --scene <場景資料夾> [--pair A,B]    # 印出物體速度分布
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np

from v2v4real_io import Frame, load_yaml


def pose6_to_mat(p):
    """[x, y, z, roll, yaw(度), pitch] -> 4x4（只保留 x, y, yaw）。"""
    th = np.deg2rad(float(p[4]))
    M = np.eye(4)
    M[:2, :2] = [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]
    M[0, 3], M[1, 3] = float(p[0]), float(p[1])
    return M


def load_frame(path, drop_id=None, pose_key="lidar_pose") -> Frame:
    d = load_yaml(path)
    ids, boxes, ext, speed = [], [], [], []
    for k, v in (d.get("vehicles") or {}).items():
        if drop_id is not None and int(k) == drop_id:      # 防呆：agent 自己若出現在自己的標註中就移除
            continue
        loc, ang, c = v["location"], v["angle"], v.get("center", [0.0, 0.0, 0.0])
        yaw = np.deg2rad(float(ang[1]))
        cx = loc[0] + np.cos(yaw) * c[0] - np.sin(yaw) * c[1]   # box 中心 = location + R(yaw) · center
        cy = loc[1] + np.sin(yaw) * c[0] + np.cos(yaw) * c[1]
        ids.append(int(k)); boxes.append([cx, cy, yaw]); ext.append(v["extent"]); speed.append(float(v.get("speed", np.nan)))
    fr = Frame(pose=pose6_to_mat(d[pose_key]), ids=np.array(ids, int), boxes=np.array(boxes, float).reshape(-1, 3),
               extent=np.array(ext, float).reshape(-1, 3), types=[""] * len(ids))
    fr.speed = np.array(speed, float)                           # km/h
    return fr


class PairScenario:
    """一個場景裡的兩個 agent。frame(agent, i) 的 i 是『共同檔案清單』的第 i 個（相鄰 = 100 ms）。"""

    def __init__(self, scene_dir, ego, nb, pose_key="lidar_pose"):
        self.root, self.pose_key = scene_dir, pose_key
        self.dir = f"{scene_dir.rstrip('/')}__{ego}-{nb}"       # 只用來命名，不是真實路徑
        self.agents = [ego, nb]
        nums = {ag: {int(f[:-5]) for f in os.listdir(os.path.join(scene_dir, ag)) if f.endswith(".yaml")} for ag in self.agents}
        self.files = sorted(nums[ego] & nums[nb])
        gaps = np.unique(np.diff(self.files)) if len(self.files) > 1 else np.array([])
        self.regular = len(gaps) <= 1
        if not self.regular:
            print(f"警告：{self.dir} 的編號間隔不固定 {gaps.tolist()}，相鄰幀不一定是 100 ms。", file=sys.stderr)
        self.n_frames = {ag: len(self.files) for ag in self.agents}
        self._cache = {}

    def frame(self, agent, idx) -> Frame:
        key = (agent, idx)
        if key not in self._cache:
            path = os.path.join(self.root, agent, f"{self.files[idx]:06d}.yaml")
            self._cache[key] = load_frame(path, drop_id=int(agent), pose_key=self.pose_key)
        return self._cache[key]


def agent_dirs(scene_dir):
    return sorted(d for d in os.listdir(scene_dir) if os.path.isdir(os.path.join(scene_dir, d)))


def pair_scenarios(scene_dir, pair=None):
    """pair 為 'A,B' 或 None（None = 這個場景裡所有兩兩組合）。"""
    try:
        ags = agent_dirs(scene_dir)
    except OSError:
        return []
    if pair:
        a, b = [x.strip() for x in pair.split(",")]
        if a not in ags or b not in ags:
            return []                                           # 這個場景沒有指定的 agent，略過
        combos = [(a, b)]
    else:
        combos = list(itertools.combinations(ags, 2))
    return [PairScenario(scene_dir, a, b) for a, b in combos]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--pair", default=None)
    ap.add_argument("--stride", type=int, default=5)
    a = ap.parse_args()
    for sc in pair_scenarios(a.scene, a.pair):
        sp, near = [], []
        for i in range(0, len(sc.files), a.stride):
            fe = sc.frame(sc.agents[0], i)
            m = np.linalg.norm(fe.boxes[:, :2] - fe.pose[:2, 3], axis=1) < 70
            sp.append(fe.speed[m])
        sp = np.concatenate(sp)
        print(f"{os.path.basename(sc.dir)}: 幀 {len(sc.files)}，ego 70 m 內標註物體 speed (km/h) "
              f"P10/P50/P90 = {np.percentile(sp, 10):.1f}/{np.percentile(sp, 50):.1f}/{np.percentile(sp, 90):.1f}，"
              f"速度 < 5 km/h 的比例 {np.mean(sp < 5):.2f}（多半是停放車輛）")
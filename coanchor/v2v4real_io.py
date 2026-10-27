#!/usr/bin/env python3
"""
v2v4real_io.py -- 讀取 V2V4Real（OPV2V 格式）的 yaml 標註

為什麼不用 yaml.load：這些 yaml 內含 !!python/object 標籤，yaml.unsafe_load 能讀，
但會允許 yaml 檔執行任意程式碼。這裡自訂 Loader，只放行 numpy 純量 / dtype / ndarray 這幾種標籤。

資料夾結構（你的 unzip 輸出）：
  <root>/testoutput_CAV_data_2022-03-15-09-54-40_0/<agent_id>/000000.yaml, 000000.pcd ...
  agent_id 通常是 0 與 1（兩台車）。
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import yaml


# --------------------------------------------------------------------------- #
# 安全 Loader
# --------------------------------------------------------------------------- #
class _NumpySafeLoader(yaml.SafeLoader):
    pass


_ALLOWED_APPLY = {
    "numpy.core.multiarray.scalar", "numpy._core.multiarray.scalar",
    "numpy.dtype",
    "numpy.core.multiarray._reconstruct", "numpy._core.multiarray._reconstruct",
}


def _args_state(loader, node):
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True), None
    m = loader.construct_mapping(node, deep=True)
    return m.get("args", []), m.get("state")


def _construct_apply(loader, suffix, node):
    if suffix not in _ALLOWED_APPLY:
        raise yaml.constructor.ConstructorError(None, None, f"不允許的標籤: {suffix}", node.start_mark)
    args, state = _args_state(loader, node)
    if suffix == "numpy.dtype":
        return np.dtype(args[0])
    if suffix.endswith("scalar"):
        dtype, raw = args
        return np.frombuffer(raw, dtype=dtype)[0]
    # _reconstruct：state = (version, shape, dtype, is_fortran, raw_bytes)
    _, shape, dtype, fortran, raw = state
    return np.frombuffer(raw, dtype=dtype).reshape(shape, order="F" if fortran else "C").copy()


def _construct_name(loader, suffix, node):
    if suffix != "numpy.ndarray":
        raise yaml.constructor.ConstructorError(None, None, f"不允許的標籤: {suffix}", node.start_mark)
    return np.ndarray


_NumpySafeLoader.add_multi_constructor("tag:yaml.org,2002:python/object/apply:", _construct_apply)
_NumpySafeLoader.add_multi_constructor("tag:yaml.org,2002:python/name:", _construct_name)
_NumpySafeLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", lambda l, n: tuple(l.construct_sequence(n, deep=True)))


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.load(f, Loader=_NumpySafeLoader)


# --------------------------------------------------------------------------- #
# SE(2) 與一幀資料
# --------------------------------------------------------------------------- #
def mat_to_se2(M):
    """4x4 齊次矩陣 -> [x, y, yaw]（忽略 roll/pitch/z）。"""
    return np.array([M[0, 3], M[1, 3], np.arctan2(M[1, 0], M[0, 0])])


@dataclass
class Frame:
    pose: np.ndarray        # 4x4，lidar_pose（假設為 lidar -> world；用 check_frames 驗證）
    ids: np.ndarray         # (N,) 物件 id（同一台車的標註 id，跨幀一致）
    boxes: np.ndarray       # (N,3) [x, y, yaw(rad)]，在 label_frame 指定的座標系
    extent: np.ndarray      # (N,3)
    types: list


def load_frame(path) -> Frame:
    d = load_yaml(path)
    veh = d.get("vehicles", {}) or {}
    ids, boxes, ext, types = [], [], [], []
    for k, v in veh.items():
        loc, ang = v["location"], v["angle"]
        ids.append(int(k))
        boxes.append([loc[0], loc[1], np.deg2rad(ang[1])])   # angle = [roll, yaw, pitch]（度）
        ext.append(v["extent"])
        types.append(v.get("obj_type", ""))
    return Frame(pose=np.asarray(d["lidar_pose"], float), ids=np.array(ids, int),
                 boxes=np.array(boxes, float).reshape(-1, 3), extent=np.array(ext, float).reshape(-1, 3),
                 types=types)


class Scenario:
    """一個場景資料夾（含 agent 0、1 的逐幀 yaml）。frame 會 lazy 載入並快取。"""

    def __init__(self, scenario_dir):
        self.dir = scenario_dir
        self.agents = sorted(a for a in os.listdir(scenario_dir) if os.path.isdir(os.path.join(scenario_dir, a)))
        self._cache = {}
        self.n_frames = {a: len(glob.glob(os.path.join(scenario_dir, a, "*.yaml"))) for a in self.agents}

    def frame(self, agent, idx) -> Frame:
        key = (agent, idx)
        if key not in self._cache:
            self._cache[key] = load_frame(os.path.join(self.dir, agent, f"{idx:06d}.yaml"))
        return self._cache[key]


if __name__ == "__main__":
    import sys
    f = load_frame(sys.argv[1])
    np.set_printoptions(precision=3, suppress=True)
    print("lidar_pose:\n", f.pose)
    print("ids:", f.ids, "types:", f.types)
    print("boxes [x,y,yaw(rad)]:\n", f.boxes)
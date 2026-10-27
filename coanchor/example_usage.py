import numpy as np
from coanchor_lite import coanchor, Params
from coanchor_tracker import Tracker, simulate_streams

# 1) 用模擬資料取得一組輸入（換成你自己的資料時，只要準備好同樣形狀的陣列）
rng = np.random.default_rng(1)
sc = simulate_streams(rng, delay=2, sig_t=0.6, sig_r_deg=0.6)
ego_hist = sc["out"]["ego"]["trk_hist"]   # (Ne, 4, 3)  [x, y, yaw]，k=0 最新，在 ego@u 座標系
nb_hist  = sc["out"]["nb"]["trk_hist"]    # (Nn, 4, 3)  在 neighbor 自己的座標系

# 2) 呼叫主函式
res = coanchor(
    ego_hist, nb_hist,
    T0=sc["T0"],                # [x, y, theta] 有噪聲的 neighbor->ego 相對 pose
    delay=2,                    # 延遲 frame 數（200 ms = 2）
    T_e=sc["T_e"],              # [x, y, theta] ego 從 u 到 t 的自身運動
    ego_obs_t=sc["ego_obs_t"],  # (Nt, 3) ego 當前時刻的偵測
    p=Params(),                 # 論文參數，可改
    refine=True, kalman=True, n_loop=1,
)

# 3) 讀結果
print("修正後的相對 pose  :", res.T_final)
print("真實相對 pose      :", sc["T_true"])
print("傳播到 t 的 neighbor 物件 (x,y,vx,vy,yaw):\n", res.states.round(2))
print("匹配對 [ego_idx, nb_idx]:\n", res.pairs.T)
print("最終 pair 權重 q:", res.q.round(3))
print("仍是 active anchor 的 pair:", res.accepted)
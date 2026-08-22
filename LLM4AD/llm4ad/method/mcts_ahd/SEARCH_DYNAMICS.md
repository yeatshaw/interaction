# 无反思搜索动力学记录

`MCTS_AHD(..., dynamics_log_dir=...)` 会在指定目录创建：

- `expansion_events.jsonl`：每次 LLM 生成尝试，以及候选是否被接入树；
- `search_states.jsonl`：每个 MCTS search round 结束后的树状态。

当前实现不改变 `e1/e2/m1/m2/s1` 的 Prompt、算子权重或 UCT 选择逻辑。日志默认关闭。

扩展事件包含算子、选中路径、父节点 ID/深度/访问次数/分数、`lambda_t`、算法描述、函数代码、子代分数、父子增益和有效性。

每轮核心状态为：

\[
S_t=[\Delta B_t,Y_t,H_t,G_t,L_t]
\]

- `best_improvement`：全局最优分数的本轮变化；
- `improvement_rate`：被接入树的子代中优于父代的比例；
- `root_visit_entropy`：根分支访问分布熵；
- `tree_growth`：本轮新增树节点数；
- `stagnation_length`：连续未刷新最佳分数的轮数。

同时记录分支集中度、树大小、最大深度、重复率及各算子的有效率和收益。

## 运行无反思基线

```powershell
$env:LLM4AD_API_HOST = "your-api-host"
$env:LLM4AD_API_KEY = "your-api-key"
$env:LLM4AD_MODEL = "your-model"
$env:LLM4AD_MAX_SAMPLES = "100"
python example/tsp_construct/run_mcts_ahd_dynamics.py
python llm4ad/method/mcts_ahd/analyze_dynamics.py logs/mcts_ahd_dynamics/dynamics
```

该入口不启用反思或 PID，用于建立后续实验的基线轨迹。

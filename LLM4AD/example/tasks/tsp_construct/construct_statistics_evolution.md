# TSP Construct 统计量驱动进化流程

## 1. 目标与边界

研究对象是 TSP 的逐步构造函数：

```python
select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix)
```

EoH 只进化这个函数。每个候选函数在构造路径的每一步选择一个未访问节点，最终得到一条完整 tour 和最终 `loss`。

统计量的作用是解释候选代码的决策机制，并在后续 EoH 生成代码时提供方向性反馈。真实 TSP `loss` 始终是候选代码的唯一主 fitness；统计量不替代真实评估，也不直接跳过候选评估。

当前评估入口为 `llm4ad/task/optimization/tsp_construct/evaluation.py`。它为每个实例从节点 `0` 开始，重复调用 `select_next_node`，再根据完整 tour 计算路径长度。

## 2. 一次实验的初始化

### 2.1 固定实验配置

开始一个 EoH 实验前，创建一个不可变的实验配置：

```text
run_id: YYYYMMDD_HHMMSS_select_next_node
random_seed: EoH 随机种子
n_instance: TSP 实例数
problem_size: 城市数
max_sample_nums: EoH 最大候选数
max_generations: EoH 最大代数
analysis_block_size: 每多少个有效候选触发一次分析，例如 30
stage_count: 10
```

同一 run 内的 TSP 实例集必须固定。否则候选间的 loss 差异可能来自不同实例，而不是代码机制。

结果目录建议为：

```text
operator_statistics_results/
└── YYYYMMDD_HHMMSS_select_next_node/
    ├── raw/                 # 每次构造决策的阶段累计数据
    ├── scores/              # 每个候选的最终平均 loss
    ├── candidate_metrics.csv
    ├── analysis_blocks/
    ├── feature_registry.json
    ├── mechanism_feedback.json
    └── figures/
```

### 2.2 初始化基础统计量库

第一版不要求候选代码返回额外字段。插件仅使用当前函数的输入和输出：

```text
输入：current_node, destination_node, unvisited_nodes, distance_matrix
输出：next_node
```

基础统计量如下：

| 名称 | 计算 | 机制含义 |
|---|---|---|
| `normalized_chosen_rank` | `next_node` 在 `unvisited_nodes` 中的位置除以候选数 | 选择偏近邻贪心还是偏远距探索 |
| `edge_cost_ratio` | `d(current,next) / min(d(current,u))` | 为探索付出的即时边代价 |
| `edge_cost_percentile` | 被选边长度在候选边长度中的分位数 | 所选边在局部候选中的相对长短 |
| `return_home_rank` | `d(next,start)` 在候选节点到起点距离中的分位数 | 对最终闭环的前瞻程度 |
| `candidate_spread` | 当前节点到候选节点距离的变异系数 | 当前决策环境是否存在明显优劣候选 |

每个统计量还必须注册语义，不只注册计算公式：

```json
{
  "name": "return_home_rank",
  "meaning": "所选节点到起点距离在当前候选中的排名",
  "higher_means": "更少考虑最终闭环",
  "lower_means": "更重视最终闭环",
  "availability": "pre_evaluation",
  "status": "active"
}
```

这些语义字段用于之后自动生成机制摘要。

### 2.3 初始化 EoH 种群

EoH 根据任务描述和函数接口生成初始 `pop_size` 个 `select_next_node` 候选代码。初始 prompt 只包含：

```text
TSP 任务描述
函数签名与返回值约束
必须返回未访问节点
当前可用基础机制提示（初始通常为空）
```

初始候选也必须运行真实评估并记录统计量；它们是后续发现“机制与性能关系”的第一批样本。

## 3. 一个候选的评估与记录

### 3.1 包装候选函数

评估器不修改候选函数内部。它以插件 wrapper 包装 `select_next_node`：

```text
调用原始候选函数得到 next_node
→ 校验 next_node 是否属于未访问节点
→ 插件用输入和 next_node 计算基础统计量
→ 将 next_node 原样返回给原始 TSP 构造流程
```

若代码返回重复节点、越界节点、非整数节点或异常，则该候选评估失败，记录失败原因，不将其作为有效统计样本。

### 3.2 按构造进度增量累计

构造第 `t` 步的进度定义为：

\[
p = \frac{t}{n - 1}
\]

其中 `n` 是城市数。根据 `p` 分到十个阶段：

```text
0-10%, 10-20%, ..., 90-100%
```

插件对每个 `实例 × 阶段 × 统计量` 仅保存：

```text
sum
sum_of_squares（需要方差时）
count
```

因此无需保存每次决策的完整矩阵，也不会显著增加评估开销。

### 3.3 完整 tour 完成后的结构统计量

当一个实例构造完成后，评估器已经拥有完整路径。可额外计算：

| 名称 | 含义 |
|---|---|
| `edge_cost_cv` | tour 内边长的变异系数，衡量是否出现极长跳跃边 |
| `closing_edge_ratio` | 回到起点的最后一条边相对平均边长的比例 |
| `two_opt_improvability` | 2-opt 邻域中仍可改进的程度，衡量路径结构缺陷 |

这些属于 `post_evaluation` 统计量：可用于解释为什么路径好或差，但不应在同一次构造决策中反馈给候选函数。

### 3.4 候选级汇总

一个候选在所有固定 TSP 实例上运行完后，形成：

\[
(P_i, L_i, z_{i,0}, z_{i,1}, \ldots, z_{i,9})
\]

其中：

```text
P_i：候选代码及其 candidate_id
L_i：所有实例的平均最终 tour loss
z_i,s：第 s 个阶段的统计量向量
```

多次重复运行时，先按 `candidate_id × instance × stage` 聚合，再按实例数加权汇总；不能将同一候选的多次决策当作独立的“代码性能样本”。

## 4. EoH 的常规选择

候选的主排序保持原始 EoH 逻辑：

```text
较低 final loss → 更可能被保留、重组或变异
```

统计量在初期只进入分析和 prompt，不参与主排序。这样新颖但暂未被现有统计量解释的代码仍有机会被真实 loss 保留。

## 5. 自动统计机制分析

每累计 `analysis_block_size` 个有效候选，例如 30 个，自动创建一个分析 block。

### 5.1 构造分析表

每一行对应：

```text
candidate_id × stage
```

字段包含：

```text
final_loss
log_loss = log10(max(final_loss, epsilon))
所有 active 统计量
```

loss 使用连续值，不用“前 20% 是好、后 20% 是坏”的硬切分。用于展示时，可使用：

```text
<=1e-8
(1e-8,1e-4]
(1e-4,1]
(1,10]
(10,100]
(100,1000]
>1000
```

### 5.2 单调关系检测

对每个 `统计量 × 阶段` 计算：

\[
\rho_{f,s}=\operatorname{Spearman}(z_{i,f,s}, \log_{10}(\max(L_i,\epsilon)))
\]

解释规则：

```text
rho > 0：统计量越大，最终 loss 越高
rho < 0：统计量越大，最终 loss 越低
rho 接近 0：未发现稳定单调关系
```

### 5.3 非单调有效区间检测

不少 construct 机制存在合适区间。例如 `normalized_chosen_rank` 过小可能过度贪心，过大可能产生长边。

自动处理方式：

```text
将统计量按候选分位数划为五组
→ 分别计算各组的 loss 加权中位数
→ 找到 loss 最低的连续一组或多组
→ 输出有效区间
```

例如：

```text
normalized_chosen_rank @ 0-20%
有效区间：[0.15, 0.32]
```

### 5.4 稳定性、去重与状态更新

每个候选代码是独立样本。bootstrap 时应重采样候选代码，而不是重采样同一个候选内部的构造步骤。

一个机制进入 `validated` 的条件可以设为：

```text
有效候选数 >= 20
bootstrap 置信区间的关系方向稳定
连续两个分析 block 的方向一致
与已有 active 统计量的相关性未超过阈值，例如 0.9
```

特征状态为：

```text
proposed → collecting → validated
                    └→ inactive
```

只有 `validated` 统计量可进入进化反馈。连续多个 block 没有贡献的统计量标记为 `inactive`，但历史记录不删除。

## 6. 自动生成机制摘要

分析器输出结构化证据：

```json
{
  "feature": "return_home_rank",
  "stage": "70-100%",
  "relation": "lower_is_better",
  "rho": 0.41,
  "best_range": [0.00, 0.25],
  "sample_n": 64,
  "stable_blocks": 2
}
```

根据 Feature Registry 中的语义，用固定模板生成摘要：

```text
已验证机制：
在构造后期（70-100%），较低的 return_home_rank
与较低的最终 tour loss 稳定相关。

机制解释：
性能较好的构造策略会在后期更重视候选节点到起点的距离，
以降低最终闭环边过长的风险。

进化建议：
当未访问节点较少时，适度考虑候选节点到起点的距离；
仍需同时考虑当前边代价，避免退化为单一最近起点规则。
```

每次最多保留证据强度最高的 2-3 条，避免 prompt 过长。

## 7. 反馈到下一批 EoH 代码生成

下一批 EoH prompt 在原始任务说明后追加：

```text
当前已验证的机制反馈：
1. ...
2. ...

请在不改变函数签名、必须返回未访问节点的前提下，
探索能体现上述机制的不同节点选择策略。
不要把统计量区间硬编码为常数；应从未访问节点比例、距离矩阵
和当前节点状态中自适应地实现。
```

EoH 继续通过真实 loss 进行选择。机制摘要只改变 LLM 的搜索先验。

## 8. 动态发现新机制与新统计量

基础统计量不可能覆盖所有后期出现的代码机制。每隔较长周期，例如 3 个分析 block，启动一次机制发现流程。

### 8.1 输入代码集合

选择：

```text
top-k：低 loss 代码
diverse-k：行为或代码结构不同的中等代码
bottom-k：失败代码
当前 Feature Registry
当前未解释的性能差异
```

### 8.2 机制发现 Agent 的职责

Agent 不直接决定代码好坏，而是识别：

```text
优秀代码使用了什么新的节点选择机制？
这个机制是否未被现有统计量表示？
需要候选函数额外暴露哪些诊断信息才能测量？
```

输出必须是结构化提案：

```json
{
  "mechanism": "adaptive_lookahead",
  "hypothesis": "根据剩余节点数量动态调整当前边与回程代价的权重",
  "diagnostics": ["current_weight", "return_weight"],
  "feature": "return_weight_progress_correlation",
  "formula": "corr(progress, return_weight)",
  "availability": "pre_evaluation"
}
```

### 8.3 新诊断返回协议

若新统计量无法从现有输入/输出得到，下一版候选函数允许逻辑上返回：

```python
return next_node, diagnostics
```

其中：

```python
diagnostics = {
    "current_weight": current_weight,
    "return_weight": return_weight,
}
```

wrapper 做两件事：

```text
读取 diagnostics 并计算新统计量
只把 next_node 返回给原始 evaluation.py
```

因此原 TSP 构造流程不需要知道 diagnostics。

### 8.4 前瞻性验证

旧运行没有记录的新诊断量不能被严格回溯计算。正确时间线是：

```text
Block t：从旧代码提出机制假设
Block t+1：生成携带 diagnostics 的新代码
Block t+2：积累新统计量和真实 loss
Block t+3：判断该统计量能否稳定解释 loss
```

新统计量在通过稳定性检验前只能是 `collecting`，不能进入 EoH 反馈。

## 9. 最终实验验证

至少比较：

```text
A. 原始 EoH
B. EoH + 无关或随机提示
C. EoH + 自动统计机制反馈
```

报告：

```text
最终最优 loss
达到相同 loss 所需候选评估次数
多随机种子的中位数与置信区间
被反馈统计量是否向有效方向或有效区间移动
新机制从 proposed 到 validated 的数量与稳定性
```

若 C 在多种子下获得更低 loss 或更快收敛，且相应统计量按摘要所述发生变化，则可以支持结论：统计量不仅解释 construct 代码性能，也能有效指导 LLM 自动算法进化。

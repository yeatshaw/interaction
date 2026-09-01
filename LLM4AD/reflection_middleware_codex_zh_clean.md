# Reflection Middleware
## 一个可跨 AAD/AHD 框架迁移的即插即用 Reflection 插件

> **暂定英文标题（推荐）**  
> **Reflection as Middleware: A Plug-and-Play, State-Conditioned Reflection Layer for LLM-Based Automatic Heuristic Design**
>
> **备选标题**  
> - *When Does Reflection Help? A Universal Reflection Interface for Automatic Heuristic Design*  
> - *AdaReflect: A Plug-and-Play Reflection Policy for LLM-Based Automatic Heuristic Design*  
> - *Less Is More: State-Conditioned Reflection Middleware for Automatic Heuristic Design*
>
> **核心定位**  
> 本工作不是提出一种新的 AHD 搜索框架，也不是让 agent 主动获取新的环境证据；而是研究并构建一个可以插入不同 AHD 方法中的 **统一 Reflection Middleware**：
>
> \[
> \boxed{
> \text{Native AHD State}
> \rightarrow
> \text{Universal Reflection Middleware}
> \rightarrow
> \text{Reflection Guidance}
> \rightarrow
> \text{Native Generator}
> }
> \]
>
> 在尽量不修改原方法的 selection、population、tree search、evaluator、operator 和 generation mechanism 的前提下，用同一个 reflection layer 替换原方法各自不统一的 reflection，并希望在多个 AHD backbone 上稳定提高搜索质量或效率。

---

# 1. 为什么要把 Version D 改成“插件/中间件”方向

之前 Version D 的核心问题是：

> reflection 中到底哪些 evidence 有用？它们是否依赖 search state？

这个问题依然成立。

但如果论文最后只得到：

```text
early 阶段应该看 A
stagnation 阶段应该看 B
local exploitation 应该看 C
```

再做一个只适用于单个框架的 router，仍然容易被 reviewer 认为：

```text
针对某个 AHD framework 做 prompt / policy tuning
```

更强的研究目标应该是：

> **不同 AHD 方法虽然 reflection prompt 和 search framework 完全不同，但 reflection 所承担的本质功能是否可以抽象成一个共享、可迁移的信息接口？**

如果答案是肯定的，我们就不只是：

```text
研究 ReEvo 的 reflection
```

而是在研究：

\[
\boxed{
\text{AHD 中 Reflection 的通用计算接口}
}
\]

最终形成一个：

# Plug-and-Play Reflection Middleware

这会让论文同时具有：

```text
mechanism study
+
unified formulation
+
portable method
+
cross-framework empirical validation
```

而不是单纯做消融。

---

# 2. 研究边界

本工作只研究 AAD/AHD 中 **已有搜索信息如何被 reflection 选择、压缩、组织和注入下一步生成**。

Reflection Middleware 只能使用原生 AHD framework 在当前搜索步骤已经自然产生的信息，例如：

```text
current heuristic
parent / lineage
absolute score
relative score
population
recent search history
search progress
diversity
execution error（仅当原框架本身已经提供）
```

为了保证研究对象纯粹，第一版方法明确不做：

```text
主动运行额外 evaluator
主动构造 diagnostic instances
额外请求 execution trace
主动获取图片或多模态证据
主动调用外部环境工具
额外进行 counterfactual rollout
改变 solver 或 evaluation environment
```

因此方法严格作用于：

\[
\boxed{
\text{Existing Search State}
\rightarrow
\text{Reflection Middleware}
\rightarrow
\text{Native Generator}
}
\]

它研究的是 **Reflection Information Policy**，而不是重新设计整个 AHD search framework。

---

# 4. 为什么“插件”这个方向是合适的

我认为合适，但论文里最好少用纯工程意义上的 “plugin”。

更建议使用：

```text
Reflection Middleware
Universal Reflection Layer
Reflection Interface
Plug-and-Play Reflection Policy
```

原因是：

“插件”只是 deployment form。

真正的 scientific claim 应该是：

> **Reflection 的有效信息可以被抽象成一个 backbone-agnostic search-state representation，并由一个共享的 state-conditioned reflection policy 跨不同 AHD framework 使用。**

如果只写：

> 我们写了一个插件，放进 ReEvo、MCTS-AHD、XXX 都变好了。

容易被认为是工程整合。

真正应该证明的是：

\[
\boxed{
\text{Reflection policy has cross-framework invariance.}
}
\]

也就是说：

> 虽然不同 AHD 框架的 population、tree、operators 和 prompts 不一样，但决定 “当前应该看什么 reflection evidence” 的规律具有可迁移性。

这才是论文核心。

---

# 5. 最强的一句话 Research Thesis

建议将论文核心 hypothesis 改成：

> **The utility of reflection is determined more by the underlying search state than by the specific AHD framework in which reflection is implemented.**

中文：

> **Reflection 的效用主要由当前搜索状态决定，而不是由某一个特定 AHD 框架或某一种 reflection prompt 决定。**

因此我们可以学习：

\[
\pi_{\text{reflect}}(z_t)
\]

其中 \(z_t\) 是 backbone-agnostic search state，

然后将同一个：

\[
\pi_{\text{reflect}}
\]

插入不同 AHD framework。

这是比：

> “early 用 parent，stagnation 用 history”

更高级的一层。

---

# 6. 论文希望证明的三个核心 Hypothesis

## H1：No Universal Fixed Reflection

不存在一个固定 reflection recipe 在所有 search state 上都最优。

即：

\[
\exists s_i,s_j:
\quad
\arg\max_r U(r|s_i)
\neq
\arg\max_r U(r|s_j).
\]

---

## H2：Cross-Framework Reflection Invariance

对于不同 backbone \(b\)：

\[
U(r|s,b)
\]

虽然绝对值可能不同，但 recipe ranking 主要由 search state 决定。

理想结果：

\[
Rank_r U(r|s,b_1)
\approx
Rank_r U(r|s,b_2).
\]

例如：

```text
在 stagnation state：

ReEvo：
history + population 最好

另一个 AHD：
history + population 也最好
```

这说明 reflection policy 可以跨框架迁移。

---

## H3：Universal Middleware Improves Native Reflection

学习一个：

\[
\pi(r|z_t)
\]

在多个 backbone 中替换 native reflection 后：

```text
>= native reflection quality
< native reflection context/token cost
```

或者：

```text
> native reflection quality
≈ same compute
```

并且最好：

> 在没有针对新 backbone 再训练的情况下依然有效。

---

# 7. 一个特别强的目标：Train Once, Plug Everywhere

如果能实现，论文的核心实验应该是：

```text
Train Reflection Router:
    Backbone A
    Backbone B

Freeze Router

Plug into:
    unseen Backbone C

No finetuning
No prompt retuning
```

然后仍然超过 C 自己的 native reflection。

这会非常有说服力。

因为它证明：

\[
\boxed{
\text{reflection knowledge is transferable across AHD frameworks}
}
\]

而不是：

```text
我们给每个 framework 单独调了最优 prompt
```

---

# 8. 什么叫 Reflection Middleware

整个插件位于：

```text
Evaluator / Search State
        ↓
[ Reflection Middleware ]
        ↓
Native Generator
```

它不替换整个 AHD。

它只替换：

```text
evaluation/search information
        ↓
reflection
        ↓
generation guidance
```

这一层。

---

# 9. Reflection Middleware 的三个模块

建议架构非常简单：

\[
\boxed{
Adapter
\rightarrow
Reflection Core
\rightarrow
Injector
}
\]

---

# 10. Module 1：Backbone Adapter

不同 AHD 方法的内部状态格式不同。

例如：

```text
ReEvo:
parent pair + fitness + reflection history

Population AHD:
population + elite + mutation history

Tree-based AHD:
current node + ancestors + visits/reward

Metacognitive AHD:
current strategy prompt + generated code + feedback
```

我们为每个 backbone 写一个很薄的 adapter：

\[
A_b:
X_t^b
\rightarrow
X_t^{canonical}.
\]

---

# 11. Canonical Search State

所有 framework 都映射到一个统一 schema。

例如：

```json
{
  "current_candidate": {
    "code": "...",
    "thought": "...",
    "fitness": 8.21,
    "rank": 3
  },

  "lineage": {
    "parent_fitness": 8.34,
    "relative_improvement": 0.015,
    "edit_distance": 0.23
  },

  "search_progress": {
    "normalized_step": 0.42,
    "recent_improvement_slope": 0.006,
    "consecutive_failures": 2,
    "offspring_success_rate": 0.31
  },

  "population": {
    "fitness_variance": 0.12,
    "diversity": 0.58
  },

  "available_evidence": {
    "absolute_score": true,
    "relative_score": true,
    "lineage": true,
    "history": true,
    "population": true
  }
}
```

注意：

> Canonical state 只包含框架已经拥有的信息。

不能额外运行 evaluator 或 tool 来补齐字段。

字段不存在：

```text
标记 unavailable
```

即可。

---

# 12. 为什么 Canonical Search State 是一个真正的贡献

现有 reflection 方法通常直接和 framework prompt 耦合。

例如：

```text
某个方法：
把 parent A/B + score 塞进 reflection prompt

另一个方法：
把 top candidates + history 塞给 critic

另一个方法：
把 trajectory 放入 memory
```

它们无法直接比较。

Canonical State 将 reflection 从：

```text
framework-specific prompt artifact
```

抽象成：

```text
framework-independent search information
```

这使：

```text
reflection study
cross-framework transfer
universal router
```

第一次变得可实现。

---

# 13. Module 2：Reflection Core

Reflection Core 再分两部分：

\[
\boxed{
State\ Encoder
+
Evidence\ Router
+
Canonical\ Reflector
}
\]

---

# 14. State Encoder

只读取通用的 search dynamics：

```text
normalized search progress
recent improvement
offspring success rate
consecutive failures
parent-child delta
population fitness variance
population diversity
```

得到：

\[
z_t.
\]

必须尽量做到：

```text
problem agnostic
framework agnostic
```

不要用：

```text
TSP-specific feature
某个 EoH operator id
某个特定 framework 内部 token
```

否则无法迁移。

---

# 15. Evidence Router

Router 决定：

> 当前 reflection 需要读取 canonical state 中哪些 evidence。

例如 evidence bank：

```text
E0 = current candidate
E1 = absolute outcome
E2 = relative outcome
E3 = lineage
E4 = short history
E5 = population context
```

Router 输出：

\[
m_t\in\{0,1\}^{|E|}.
\]

例如：

```json
{
  "absolute_score": false,
  "relative_score": true,
  "lineage": true,
  "history": false,
  "population": false
}
```

---

# 16. 为什么不能只做 Reflection Trigger

已经有工作会在 stagnation 达到 patience threshold 时才触发 reflection。

因此我们不能把 innovation 写成：

> “我们发现 stagnation 时需要 reflection，所以动态决定是否 reflect。”

我们的 action space 必须更加丰富：

\[
\boxed{
\text{what to reflect on}
}
\]

而不仅仅是：

\[
\boxed{
\text{whether to reflect}
}
\]

即：

```text
不只是：
reflect / no-reflect

而是：
minimal
relative
lineage
history
population
relative+lineage
history+population
...
```

trigger 只是 router 的一个特殊情况。

---

# 17. Canonical Reflector

选出 evidence 后，通过固定统一的 reflector：

\[
R_t=F(E_{m_t})
\]

生成结构化 reflection。

建议 schema：

```json
{
  "diagnosis": "...",
  "retain": [
    "..."
  ],
  "change": [
    "..."
  ],
  "search_intent": "explore|exploit|recover",
  "uncertainty": [
    "..."
  ],
  "recommended_edit_scope": "local|moderate|structural"
}
```

所有 backbone 都使用同一个 reflector prompt。

这样 reflection 的 semantic transformation 是统一的。

---

# 18. 为什么 Reflector 本身要尽量固定

因为论文主要研究：

\[
\boxed{
State\rightarrow Evidence\ Selection
}
\]

如果每个 backbone 又用不同 reflector prompt：

```text
router + prompt + generator
```

全部一起变，很难解释。

所以第一版：

```text
固定 Canonical Reflector
主要学习 Evidence Router
```

后续才研究：

```text
不同 reflection functions
```

---

# 19. Module 3：Backbone Injector

最后需要将 canonical reflection 放回原 AHD framework。

定义：

\[
I_b:
R_t
\rightarrow
\text{native generation context}.
\]

这个 injector 应该非常薄。

例如：

```text
ReEvo:
replace original short/long reflection text

Framework B:
prepend canonical reflection to mutation prompt

Framework C:
replace critic guidance field
```

Injector 不能：

```text
改变 parent selection
改变 population update
改变 tree policy
改变 evaluator
改变 operator schedule
```

否则不再是 reflection plugin。

---

# 20. 整个 Plugin 的完整形式

对于任意 backbone \(b\)：

\[
X_t^b
\xrightarrow{A_b}
X_t^{canonical}
\xrightarrow{\phi}
z_t
\xrightarrow{\pi}
m_t
\xrightarrow{F}
R_t
\xrightarrow{I_b}
C_t^b
\xrightarrow{G_b}
h_{t+1}.
\]

其中：

- \(A_b\)：backbone adapter；
- \(\phi\)：通用 state encoder；
- \(\pi\)：reflection evidence router；
- \(F\)：canonical reflector；
- \(I_b\)：backbone injector；
- \(G_b\)：原 backbone generator。

真正共享的是：

\[
\boxed{
\phi,\pi,F
}
\]

只有：

\[
A_b,I_b
\]

是 framework-specific glue code。

---

# 21. 这为什么不太“工程”

关键在于论文不应该贡献：

```text
Adapter 写得多漂亮
接口多统一
```

而应该贡献一个 scientific result：

\[
\boxed{
\text{A shared reflection policy transfers across heterogeneous AHD search processes.}
}
\]

插件只是验证这一 hypothesis 的实验载体。

如果换一个 backbone 后：

```text
只写 50 行 adapter
冻结 Reflection Core
直接提升
```

这个结果就很强。

---

# 22. Reflection Episode Benchmark 仍然保留，而且更加重要

插件训练之前先从多个 AHD backbone 收集 frozen episodes。

定义：

\[
\mathcal E_t^b
=
(X_t^{canonical},b).
\]

然后同一个 episode 运行不同 reflection recipes。

例如：

```text
Minimal
Absolute
Relative
Lineage
Relative+Lineage
History
Population
All
```

每个 recipe 生成固定 \(k\) 个 children。

测：

```text
one-step improvement
best-of-k gain
catastrophic rate
diversity
token cost
```

---

# 23. Episode Benchmark 现在新增一个最关键的分析

不仅研究：

\[
Recipe\times State
\]

还要研究：

\[
\boxed{
Recipe\times State\times Backbone
}
\]

我们希望发现：

\[
Recipe\times State
\]

interaction 强，

而：

\[
Recipe\times State\times Backbone
\]

interaction 相对较弱。

直观来说：

> 某种 search state 需要什么 reflection，主要由 state 决定，而不是由 framework 决定。

这就是 universal plugin 存在的统计依据。

---

# 24. 一个很漂亮的 Mixed-Effects 模型

可以拟合：

\[
Gain
=
\beta_0
+
\beta_R Recipe
+
\beta_S State
+
\beta_B Backbone
+
\beta_{R\times S}
+
\beta_{R\times B}
+
\beta_{R\times S\times B}
+
u_{\text{task}}
+
u_{\text{LLM}}
+
\epsilon.
\]

理想结果：

```text
Recipe main effect:
一般

Recipe × State:
很强

Recipe × Backbone:
较弱

Recipe × State × Backbone:
较弱
```

那就非常支持：

> Reflection policy 是 cross-framework transferable 的。

---

# 25. 最重要的科学问题已经变了

旧问题：

> 什么 reflection 最好？

新问题：

> **Reflection utility 的决定因素到底是 framework identity，还是 underlying search dynamics？**

如果答案更偏后者：

> 就应该建立 universal reflection middleware。

这是一个清楚而且可证伪的问题。

---

# 26. 第一版 Router 怎么做

依然不建议 RL。

## Oracle Router

根据离线 episode 数据：

\[
r_s^*
=
\arg\max_r
U(r|s).
\]

先验证 adaptive upper bound。

---

## Simple Learned Router

输入：

```text
search state features
```

输出：

```text
recipe
```

可用：

```text
decision tree
XGBoost
small MLP
contextual bandit
```

论文不要强调模型 architecture。

甚至 decision tree 如果跨 framework transfer 很好，会更漂亮。

---

# 27. 训练方式：Leave-One-Backbone-Out

这是插件论文最核心的设置。

假设三个 backbone：

```text
A
B
C
```

做：

### Fold 1

```text
Train router:
A + B

Test:
C
```

### Fold 2

```text
Train:
A + C

Test:
B
```

### Fold 3

```text
Train:
B + C

Test:
A
```

全程：

```text
不对 held-out backbone 微调 router
```

可以允许写 adapter，因为 adapter 只是字段映射，不含 learned parameters。

这就是：

# Cross-Backbone Zero-Shot Reflection Transfer

---

# 28. 哪些 backbone 更适合作为实验对象

第一版不要追求数量。

至少选择 reflection 形态明显不同的 2～3 种。

建议优先：

## Backbone A：ReEvo

理由：

```text
reflection 是核心组件
容易定义 native reflection baseline
```

比较：

```text
Native ReEvo Reflection
No Reflection
Reflection Middleware
```

---

## Backbone B：一种 trajectory / history-oriented 方法

例如 PathWise 类 framework，或者你们能稳定复现、具有 critic/history reflection 的方法。

重点是：

```text
reflection granularity 与 ReEvo 不同
```

---

## Backbone C：另一种 population/metacognitive reflection framework

选择代码能稳定复现的工作即可。

不要为了覆盖论文名单牺牲可复现性。

---

# 29. EoH 应该怎么处理

EoH 原生并不等价于 ReEvo 式 reflection。

因此不要写：

> “替换 EoH 的 reflection”。

更严谨的是：

```text
对于已有 reflection 的方法：
replace native reflection

对于没有显式 reflection 的方法：
insert middleware as optional reflective guidance
```

论文核心 comparison 应以前一类为主。

后一类只是证明插件还能：

```text
augment a non-reflective backbone
```

---

# 30. 最核心的 End-to-End Baselines

对每个有 native reflection 的 backbone：

```text
B0: No Reflection
B1: Native Reflection
B2: All-Context Canonical Reflection
B3: Best Fixed Canonical Reflection
B4: Random Reflection Router
B5: Backbone-Specific Trained Router
B6: Universal Reflection Middleware
```

其中最关键的是：

\[
B6 \text{ vs } B1
\]

以及：

\[
B6 \text{ vs } B5.
\]

---

# 31. 如果 Universal Plugin 比 Backbone-Specific Router 还强

这会是一个很有意思的结果。

原因可能是：

```text
multi-backbone training 提供更丰富的 search-state coverage
减少对某个 framework artifact 的过拟合
```

可以进一步讲：

> universal reflection acts as a regularizer over framework-specific reflection heuristics.

---

# 32. 如果 Universal Plugin 只接近 Native Reflection，也可能有价值

假设：

```text
Native:
quality = 100
tokens = 100%

Universal:
quality = 99.8
tokens = 55%
```

依然可能是很好的结果。

因为插件的价值不一定只是最高 objective。

它也可以是：

\[
\boxed{
\text{same performance with less reflection context}
}
\]

这就是 evidence bottleneck 的故事。

---

# 33. 插件必须严格控制计算公平性

Native Reflection 可能：

```text
1 reflection call
```

Universal Middleware 也应该尽量：

```text
1 reflection call
```

如果 Router 是小模型/规则：

```text
成本几乎忽略
```

并报告：

```text
generation calls
reflection calls
input tokens
output tokens
evaluator calls
wall-clock
```

---

# 34. 必须有 More-Sampling Baseline

如果：

```text
Reflection = 1 LLM call
```

reviewer 会问：

> 为什么不把这个 call 用来多生成一个 candidate？

比较：

### Native / Plugin Reflection

```text
1 reflect
+
1 generate
```

### Extra Sampling

```text
2 generate
```

在相近 token/evaluator budget 下比较。

---

# 35. 插件的输出不能直接代替 generator

Reflection Middleware 只能输出：

```text
design guidance
```

不能自己生成最终 heuristic。

否则它和 generator/agent 的边界又混了。

即：

\[
Middleware:
X_t\rightarrow R_t
\]

而不是：

\[
Middleware:
X_t\rightarrow h_{t+1}.
\]

这样才能真正叫 reflection layer。

---

# 36. 一个推荐的 Canonical Reflection Schema

```json
{
  "state_assessment": {
    "mode": "explore|exploit|stagnate|recover",
    "confidence": 0.82
  },

  "diagnosis": [
    {
      "claim": "...",
      "support": ["relative_score", "lineage"]
    }
  ],

  "preserve": [
    "..."
  ],

  "change": [
    "..."
  ],

  "avoid": [
    "..."
  ],

  "recommended_search_behavior": {
    "edit_scope": "local|moderate|structural",
    "novelty_level": "low|medium|high"
  },

  "uncertainty": [
    "..."
  ]
}
```

不同 AHD framework 都消费这一套 reflection。

---

# 37. 推荐的 Evidence Recipes

第一篇仍然保持少量：

```text
R0: Minimal
    current candidate only

R1: Outcome
    current + absolute fitness

R2: Comparative
    current + relative parent-child outcome

R3: Lineage
    current + parent + edit information

R4: Local Refinement
    comparative + lineage

R5: History
    recent successes/failures

R6: Population
    elite/diversity/population summary

R7: Recovery
    relative + failures + history

Rall:
    all available evidence
```

插件 Router 选择 recipe，而不是任意组合每个 field。

这样 search space 更可控、更容易解释。

---

# 38. 为什么 “Minimal Sufficient Reflection” 仍然重要

插件不是一定选信息最多的 recipe。

目标依然是：

\[
r^*(s)
=
\arg\max_r
[
U(r,s)-\beta C(r)
].
\]

如果：

```text
local exploitation:
R4 就够

stagnation:
R6 才需要

early:
R0 反而最好
```

那么插件自然实现：

```text
只在必要的时候使用复杂 reflection。
```

---

# 39. 一个更强的 Plugin 目标：Framework-Native Evidence Only

为了保持研究对象纯粹：

Reflection Middleware 的原则应明确写成：

> **The middleware may only consume information already available to the native AHD framework at the current step.**

它不能为了 reflection：

```text
额外跑新的 evaluator instance
请求新的 trace
主动构建 contrastive sample
画图
做 counterfactual experiment
```

这条建议直接写进 Method。

这样可以确保方法始终只研究 reflection information flow。

---

# 40. 当前最新工作带来的额外边界

## AHD Agent

AHD Agent 会根据 state 主动调用工具获得 targeted evidence。

我们的区别：

```text
AHD Agent:
state-conditioned evidence acquisition

Reflection Middleware:
state-conditioned processing of already-available evidence
```

插件没有 environment tool action。

---

## Patience-Triggered Reflection 类方法

已经有框架只在 search stagnation 时触发 reflection。

因此我们的贡献不能只是：

```text
when to reflect
```

而应该是：

```text
what evidence to reflect on
+
how to standardize reflection across frameworks
+
whether the policy transfers across frameworks
```

---

## MeLA / MeEvo

这些方法已经会演化 metacognitive strategy/reflection history。

我们的区别：

```text
MeLA / MeEvo:
design a new AHD architecture around metacognitive evolution

Reflection Middleware:
keep the backbone unchanged and replace only its reflection interface
```

以及：

> 我们的重点是 cross-backbone portability，而不是在单一新 architecture 中演化 reflection strategy。

---

# 41. 第一篇最有力的 Contribution 重新写法

如果实验成功，建议贡献写成：

## Contribution 1：Universal Reflection Interface

提出一个统一的 AHD reflection abstraction，将不同 framework-specific search information 映射为 canonical search state，使 reflection 能够从具体 AHD architecture 中解耦。

---

## Contribution 2：State-Conditioned Reflection Utility

通过跨 framework 的 frozen episode benchmark 系统证明：

> reflection recipe 的 utility 强烈依赖 underlying search state，而对具体 backbone identity 的依赖相对较弱。

这是论文最核心的 scientific finding。

---

## Contribution 3：Reflection Middleware

提出一个 plug-and-play Reflection Middleware，根据 backbone-agnostic search dynamics 选择 minimal sufficient evidence，并通过统一 reflector 生成 guidance。

---

## Contribution 4：Cross-Backbone Transfer

证明同一个 reflection policy 可以：

```text
train on some AHD backbones
zero-shot plug into unseen backbone
```

并超过：

```text
native reflection
all-context reflection
fixed reflection
```

或以明显更低 reflection cost 达到相当性能。

---

# 42. 推荐的论文 Figure 1

不是首先画方法框架。

先画：

# Same Search State, Same Reflection Preference Across Different AHD Frameworks

例如三列：

```text
ReEvo
Framework B
Framework C
```

三行：

```text
Early
Improving
Stagnation
```

显示每个状态中最优 recipe。

如果结构相似：

```text
Early:
Minimal

Improving:
Relative+Lineage

Stagnation:
History+Population
```

那 universal middleware 的 motivation 一眼成立。

---

# 43. Figure 2：Plugin Diagram

```text
          ReEvo State ── Adapter ─┐
                                  │
       PathWise State ─ Adapter ──┼──> Canonical State
                                  │
 Framework-C State ─── Adapter ───┘
                                         │
                                         ▼
                                 Search-State Encoder
                                         │
                                         ▼
                                  Evidence Router
                                         │
                                         ▼
                               Canonical Reflector
                                         │
                                         ▼
                               Structured Reflection
                                  ┌──────┼──────┐
                                  ▼      ▼      ▼
                               ReEvo   B     C
                              Injector
```

真正共享的是中间部分。

---

# 44. Figure 3：Leave-One-Backbone-Out Transfer

横轴：

```text
Native Reflection
Backbone-Specific Router
Universal Router
All Context
```

纵轴：

```text
Normalized search performance
```

每个 held-out backbone 一组。

这是最终最有力的结果图。

---

# 45. Figure 4：Performance–Context Pareto

横轴：

\[
Reflection\ Tokens
\]

纵轴：

\[
Search\ Performance
\]

如果 Universal Middleware：

```text
更左
更上
```

就形成 information bottleneck 的故事。

---

# 46. 第一阶段应该怎么改现有实验

如果你同学现在已经做了很多：

```text
parent / no-parent
score / no-score
history / no-history
```

先不要丢。

第一步把这些实验全部转成：

# Offline Reflection Episode Dataset

每条记录加：

```text
backbone_id
task
search step
recent improvement
stagnation
diversity
parent-child delta
recipe
child gain
```

---

# 47. 先做一个最便宜的分析

拟合：

\[
Gain
\sim
Recipe
+
State
+
Backbone
+
Recipe:State
+
Recipe:Backbone
+
Recipe:State:Backbone.
\]

看：

```text
Recipe × State 是否强
Recipe × Backbone 是否弱
```

如果是：

> 插件方向值得继续。

如果反过来：

```text
每个 framework 都需要完全不同 recipe
```

则 universal plugin 可能不成立。

这一步非常重要，应该在写大量新代码之前完成。

---

# 48. 第一阶段 GO 条件

以下结果越多，越值得做 plugin：

### GO-1

同一 recipe 在不同 state 上 effect 翻转。

### GO-2

相同 state 下，不同 backbone 的最优 recipe 大致一致。

### GO-3

All-context 不是稳定最优。

### GO-4

Native reflection 可以被统一 canonical reflection 匹配或超过。

### GO-5

Oracle universal router 明显优于 best global fixed recipe。

### GO-6

在未见 backbone 上仍保持提升。

其中最关键：

\[
\boxed{
GO2 + GO5 + GO6
}
\]

如果这三个成立，论文故事非常清楚。

---

# 49. NO-GO 条件

如果发现：

```text
ReEvo 的最优 reflection 和其它框架完全不同；
recipe ranking 主要由 backbone 决定而非 state；
统一 canonical reflector 让性能普遍下降；
held-out framework transfer 基本失败；
```

那么不要强行做 universal plugin。

这时可以退回：

```text
framework-specific reflection science
```

但论文 ceiling 会稍低。

---

# 50. 一个现实的两阶段策略

为了降低风险，可以先做：

## Paper Core

```text
Universal Reflection Interface
+
Cross-Framework Episode Study
+
Oracle/Rule-based Reflection Middleware
```

如果已经能跨 framework 提升，足够形成强故事。

---

## Strong Extension

再做：

```text
Learned universal router
+
Leave-one-backbone-out
+
cross-task / cross-LLM
```

不要一开始就必须训练复杂 policy。

---

# 51. 插件最终应该长什么样

理想使用方式类似：

```python
reflection = middleware.reflect(
    backbone_state=current_state,
    backbone="reevo"
)

child = native_generator(
    parent=current_parent,
    reflection=reflection
)
```

对于另一 framework：

```python
reflection = middleware.reflect(
    backbone_state=current_state,
    backbone="framework_b"
)

child = framework_b_generator(
    ...,
    reflection=reflection
)
```

中间：

```text
Reflection Core 完全不改。
```

---

# 52. 为了证明“真正即插即用”，必须限制 per-backbone customization

每个 backbone 允许：

```text
field mapping adapter
output placement injector
```

不允许：

```text
单独调 router
单独改 canonical reflector prompt
单独选择 recipe library
单独调 threshold
```

否则所谓 plug-and-play 很容易被 reviewer 质疑。

---

# 53. 一个很强的 Generalization Setting

除了 leave-one-backbone-out，再做：

```text
Train:
TSP/BPP episodes from Backbone A/B

Test:
CVRP/JSSP on Backbone C
```

同时跨：

```text
framework
+
task
```

如果仍有效：

> reflection policy 学到的更可能是真正 general search dynamics。

---

# 54. LLM 也应该有一个跨模型测试

例如：

```text
Router/recipe discovery:
Model A

Middleware test:
Model B
```

或者：

```text
同一个 router
插到两个不同 generator LLM
```

如果仍然成立：

> 插件更接近 AHD-level primitive，而不是 model-specific prompt trick。

---

# 55. 论文中“插件”应该怎么表述

建议：

工程层面：

```text
plug-and-play
drop-in
middleware
```

科学层面：

```text
backbone-agnostic reflection policy
universal reflection interface
state-conditioned evidence selection
cross-framework reflection transfer
```

不要把 contribution 写成：

> “我们实现了第一个 reflection plugin。”

而是：

> “We identify transferable structure in reflection utility across heterogeneous AHD frameworks and instantiate it as a plug-and-play middleware.”

---

# 56. 最推荐的最终 Research Question

整个 Version D v2 可以压成：

\[
\boxed{
\text{Can reflection be decoupled from individual AHD frameworks and treated as a transferable, state-conditioned information policy?}
}
\]

中文：

> **Reflection 能否从具体 AHD 框架中解耦，成为一个由通用搜索状态驱动、可以跨方法迁移的信息策略？**

这比：

> 哪种 reflection prompt 最好？

强很多。

---

# 57. 最推荐的论文主张

如果数据允许：

> Existing AHD methods tightly couple reflection with framework-specific prompts and search procedures. We show that the utility of reflection is governed largely by shared search dynamics rather than framework identity. Based on this observation, we introduce a universal reflection middleware that maps heterogeneous AHD states into a canonical representation, selects minimal state-relevant evidence, and produces structured guidance that can replace native reflection without modifying the underlying search algorithm.

---

# 58. 最建议现在立刻做的事情

不要先实现完整 plugin。

先用现有实验数据验证插件成立的必要条件：

## Step 1

将已有 reflection runs 切成 frozen episodes。

## Step 2

为每个 episode 计算统一 search-state features。

## Step 3

如果有两个以上 AHD backbone，做：

\[
Recipe\times State\times Backbone.
\]

## Step 4

检验：

\[
\boxed{
Recipe\times State
>
Recipe\times Backbone
}
\]

是否在统计和 effect size 上成立。

## Step 5

如果成立，再实现 Canonical State + Universal Router。

---

# 59. 如果目前只有一个 Backbone 怎么办

也没问题。

先完成：

```text
State-dependent utility study
+
Canonical Reflection Interface
+
Oracle Router
```

同时开始把第二个 backbone 接进来。

第二个 backbone 不需要完整复现实验矩阵。

先抽少量 episode 验证：

> 第一个 backbone 上发现的 recipe-state ranking 是否迁移。

如果完全不迁移，可以很早止损。

---

# 60. Version D v2 的最终边界

这篇工作严格坚持：

```text
不做主动 evidence acquisition
不做 counterfactual intervention
不做 agent tool use
不改 evaluator
不改 solver
不改 population/tree search
不设计新的 heuristic operators
```

只做：

\[
\boxed{
\text{Existing Search State}
\rightarrow
\text{Better Reflection}
\rightarrow
\text{Native Generation}
}
\]

因此该方法可以作为独立的 reflection primitive 被研究和评估。

---

# 61. 最终一句话总结

本工作的目标是：

> **让不同 AAD/AHD 方法共享同一套“如何反思已有搜索信息”的通用 Reflection Middleware。**

如果最终同一个 Reflection Core 能够：

```text
插入不同 AHD backbone
不修改原生搜索机制
不做 per-framework reflection retuning
仍稳定改善搜索质量或降低 reflection 成本
```

那么可以支持一个更一般的结论：

\[
\boxed{
\text{Reflection can be treated as a transferable, state-conditioned information primitive in AHD.}
}
\]

也就是说，reflection 不再只是某个算法框架内部的一段 prompt，而可以成为一个独立、可替换、可迁移的搜索组件。

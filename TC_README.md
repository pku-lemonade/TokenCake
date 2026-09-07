# TokenCake 当前实现与性能说明

本文介绍本仓库中的 TokenCake 优化、已经取得的性能提升，以及相关代码、测试和实验工具的用途。
实现和结果以 2026-09-07 完成的五档 QPS、四种配置测试为准。

TokenCake 面向包含多个模型调用和工具等待的 agent 应用。
它在 vLLM 原生调度器和 KV 缓存管理之上增加三项能力：

1. 根据 agent 和 DAG 中的依赖关系、重要性及等待情况安排请求顺序。
2. 动态调整 KV 容量预留，允许借用闲置额度，并在准入时考虑请求的后续增长。
3. 利用工具等待时间，把后续仍可复用的 KV 保存到 CPU，在需要时恢复到 GPU。

## 阅读入口

本文章节：[主要优化](#主要优化)、[当前实验负载](#当前实验负载)、[性能结果](#性能结果)、[文件和目录说明](#文件和目录说明)、[使用 TokenCake](#使用-tokencake)、[复现实验和读取结果](#复现实验和读取结果)。

| 内容 | 文档或目录 |
| --- | --- |
| 完整性能结果、图表和分析 | [性能评估报告](tools/tokencake_experiments/reports/component-evaluation.md) |
| 实验条件、指标定义和统计方法 | [实验方法](tools/tokencake_experiments/reports/component-methodology.md) |
| 服务配置、请求元数据和工具事件 | [功能与接口说明](docs/features/tokencake.md) |
| 调度和 KV 保存实现 | [vllm/tokencake](vllm/tokencake) |
| 实验准备、运行和结果分析 | [实验工具说明](tools/tokencake_experiments/README.md) |
| 功能、服务和实验工具测试 | [tests/tokencake](tests/tokencake) |

## 主要优化

### 1. 预留主要影响后续准入

动态预留按 agent 类型分配一部分逻辑 KV 容量，其余容量用于共享。
预留比例根据压力在 5%～30% 内调整，闲置额度可以借给其他请求。
等待队列先按正常预留规则检查，随后可以借出仍未使用的额度。
重要请求的分数满足额外条件时，也可以提前借用其他类型尚未使用的预留。

已经接纳的请求继续沿原生路径申请下一步所需 KV。
如果分类预留调整后，其增长超过了当前分类额度，新增占用先记入共享容量账目。
之后的新准入和已完成请求释放的容量会逐步消化这部分占用。

这减少了由预留调整引起的反复抢占。
原生物理分配失败时，调度器仍然可以选择具体牺牲者释放空间；已接纳请求并非无条件免于抢占。

### 2. 准入时计入完整输入和后续生成需求

当前默认启用 `reserve_generation_tokens=true`，并使用 `generation_reserve_mode="all"`。
调度器不仅检查本次分块，还计算请求的完整输入及声明的最大生成预算所需容量。
所有已接纳请求尚未分配的增长需求，都需要在后续准入时扣除。

可以用下面的关系理解这项检查：

```text
可用于新准入的容量 = 原生空闲 block - 已接纳请求尚未分配的增长需求
```

例如，忽略共享 KV 时，100 个空闲 block 中有两个候选请求。
它们当前各需 20 block，但完整执行各需 70 block。
只检查本次分块会同时接纳两者；完整需求检查则会让其中一个先等待，避免后续共同超过容量。

具体需求复用原生多缓存组计算，考虑已有分配、GPU 前缀命中和各缓存组规则。
实际 KV 仍通过原生接口逐步分配，没有另外建立 GPU 内存池。
异步 CPU KV 回载也属于已接纳工作，其已预留容量不会在回载完成前被重新用于其他准入。

这项策略的代价是：声明的生成预算远大于实际输出时，准入可能偏保守，降低并发。
因此当前的 `all` 是已验证的默认配置，不代表它在所有负载上都最优。

### 3. 保留重要性排序，改善分支汇合和 GPU 缓存共享

等待请求继续使用 agent 重要性、关键路径、DAG 深度、等待时间和接近完成程度等信息评分。
客户端提供这些元数据，服务端据此调度；服务端不会根据 prompt 自动推断完整 DAG。

在此基础上，当前实现增加了两项配合规则：

- 同一应用中带有相同 `join_group` 的活跃分支可以继承组内最高分数，帮助尚未完成的依赖分支推进到汇合点。
- 有容量压力时，重要性相近的候选可以优先复用运行中请求持有的 GPU 前缀，降低新增 KV 需求。

共享前缀偏好默认限定在 500 分的范围内，并且共享 block 必须足以覆盖候选的新增容量需求。
只有很短的共同开头，不会因此改变调度顺序。
候选调整顺序后，仍需通过完整容量和预留检查。

### 4. 抢占考虑实际释放空间和重算成本

物理容量不足时，先在重要性相近的低分候选中选择牺牲者，再考虑：

1. 能否释放足够空间，让当前需要容量的请求继续执行。
2. 实际可释放多少 block；仍被其他请求引用的共享 block 不能全部计入。
3. 为恢复该请求预计需要重算多少 token，以及每释放一个 block 对应的重算量。
4. 请求是否已经生成了声明预算的 90%，尽量保护这类接近完成的请求。

启用 CPU KV 保存时，已经就绪、当前可以恢复的前缀会从重算估计中扣除。
尚未完成的传输不会被当作可恢复数据。
重要性用于限制候选范围，重算成本在候选内部比较，不把重要性分数和毫秒直接相加。

被抢占的请求重新准入时，还会检查可用容量是否增加、是否有其他请求完成。
这可以减少容量条件没有改善时立即重演同一失败的情况。

### 5. 使用原生 prefill 分块，并提前检查明显不足的容量

`decode_prefill_token_budget` 当前默认为 `0`，表示沿用原生 prefill 预算。
正式实验的 `max_num_batched_tokens` 为 8,192，这是整批预算，不是每个请求每步都能获得的 token 数。
当前没有启用固定 256-token 上限。
此前测试过的 1,024-token decode 压力上限出现 E2E 回退，未作为最终默认值。

对当前使用的完整注意力模型，在没有前瞻 token 和编码器输入的适用条件下，调度器先根据 GPU 命中和完整需求检查容量，再查询 CPU KV。
已经确定放不下的请求可以提前暂缓，减少重复 CPU 缓存查询及状态处理。
容量允许后，再进行原生命中检查和异步回载。

### 6. 在工具等待期间保存可复用 KV

一次模型调用完成后，应用可能需要等待工具，再携带之前的上下文发起下一次调用。
原生 GPU 前缀缓存已经可以复用相同输入，但这些缓存可能在工具等待期间被其他工作覆盖。
TokenCake 利用这段等待时间，按条件保存可复用 KV 到原生 CPU 缓存。

保存决策需要同时检查：请求是否声明前缀可复用、是否允许保存、当前等待需求、剩余工具时间、CPU 容量和预计传输成本。
开始工具等待的事件得到确认，并不代表一定执行了保存；没有收益或没有合格前缀时可以不复制。

早期每次最多保存 32 block 的默认限制，已经改为根据剩余工具窗口和 CPU 容量选择保存长度。
当前 `offload.max_relief_blocks=0` 表示采用这种计算方式，仍保留传输时间余量及多缓存组对齐检查。
保存按可命中的前缀选择，避免中间缺失导致已保存的后续部分无法命中。

已经完成的完整注意力前缀即使仍被其他请求共享，也可以作为保存来源。
复制过程复用原生传输同步，保护源 block 在传输完成前不被覆盖，并保持已有 GPU 引用关系。
其他缓存类型仍有相应的可用性限制。

后继请求到达后，先使用 GPU 命中的前缀，再通过原生接口回载需要的 CPU KV，计算剩余输入并继续生成。
当前 H2D 由后继请求的实际命中触发，尚未实现预测式提前回载。
保存 CPU 副本不会增加 GPU 的物理容量，回载本身也仍有成本。

## 当前实验负载

正式结果使用 `conversation-tools`，每次运行包含完整 24 个 DAG、648 次模型调用和 155,136 个输出 token。
每个 DAG 有 29 个执行节点和 36 条依赖边。
工具阶段按声明时长进行等待模拟，并不实际执行搜索、文件操作或外部测试。

此前经允许调整负载时，做了两项与性能有关的改动：

1. 使用追加式上下文。后继调用保留此前输入和输出，再追加工具结果与新指令；汇合处去掉重复的共同上下文。
2. 将每个 DAG 中 13 次工具指令调用的输出目标设为 128 token。规划和审查保留 200 token，代码验证和修复保留 400 token。

追加式上下文有助于前缀命中。例如：

```text
调用 A：已有上下文 + A 指令
调用 B：已有上下文 + A 指令 + A 输出 + 工具结果 + B 指令
```

这些输入和输出目标的调整属于实验客户端，由补丁应用到独立的负载副本。
TokenCake 服务端不会自动改写普通用户的 prompt，也不会自动缩短其输出预算。

相较早期负载，每组输出目标从 271,200 降为 155,136 token，减少约 42.8%。
当前四种配置使用相同的调整后负载，因此可以做组间对比；不能用早期负载的原生耗时计算当前相对降幅。
本次完整矩阵期间没有再修改 DAG、输出预算、工具等待或到达序列，也没有使用 speculative decoding。
实验尚未验证缩短工具指令前后的任务质量等价。

## 性能结果

### 实验条件

| 项目 | 设置 |
| --- | --- |
| 模型 | Qwen2.5-14B-Instruct，BF16 |
| 单个服务使用的 GPU | 一张 A800-SXM4-80GB，TP=1、PP=1 |
| 显存比例 | 0.5 |
| 最大上下文 | 32,768 token |
| 整批 token 预算 | 8,192 |
| GPU KV | 3,050 block，每 block 16 token，其中一个为原生 null block |
| offload 配置的 CPU KV 容量 | 100 GiB |
| 应用到达 QPS | 0.05、0.1、0.2、0.5、1.0，等间隔到达 |
| 原生基准版本 | 独立且未修改的原生 vLLM 代码副本，提交 `0b3ba88f165976e77ca5e6a7a3f5bba4562b80af` |

这里的 QPS 是应用 DAG 的到达速率，不是模型调用速率。
原生基准也启用了 prefix caching、分块 prefill 和完整输入长度检查。
三种 TokenCake 配置使用同一个目标运行版本，各次运行重新启动服务。

### 四种配置的总 E2E

总 E2E 从客户端启动计到退出，包含应用到达间隔和全部 DAG 完成前的等待，不包含服务初始化。
下表单位为秒，降幅按 `(base - agent_offload) / base` 计算。

| QPS | base | agent | offload | agent_offload | 组合相对 base 降幅 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 875.74 | 806.39 | 677.62 | 676.86 | 22.71% |
| 0.1 | 940.76 | 825.72 | 672.67 | 671.28 | 28.65% |
| 0.2 | 938.79 | 815.69 | 681.66 | 640.45 | 31.78% |
| 0.5 | 938.99 | 830.47 | 713.16 | 648.05 | 30.98% |
| 1.0 | 960.44 | 837.78 | 691.01 | 631.77 | 34.22% |

报告中的 `base` 对应驱动参数 `native`，`agent_offload` 对应 `offload-agent`。
`agent` 只启用调度，`offload` 只启用工具窗口 KV 保存。
其中 `offload` 的保存策略仍使用重要性和保存资格等元数据，不能将它解释为完全不使用 agent 信息的缓存策略。

20 组初始测试和 8 次复测全部完成，共 28 次运行、672 个 DAG、18,144 次模型调用，无排除运行。
QPS 0.05 和 0.1 的 base、组合配置各运行三次，其余组合各运行一次。
表格按各项指标的中位数汇总，所有合格运行都保留。
单次观测不代表统计稳定性，图中的最小值和最大值范围也不是置信区间。

### 哪些优化贡献最大

QPS 1.0 时，仅开启 offload 的 E2E 为 691.01 秒，相对原生降低 28.05%。
组合配置进一步降到 631.77 秒，相对仅 offload 又降低 8.57%。
仅开启调度相对原生降低 12.77%。这些节省可能重叠，不能直接相加。

| QPS 1.0 配置 | 首次 prefill 计算 token | GPU 前缀命中 token | CPU KV 命中 token | 原生总抢占次数 |
| --- | ---: | ---: | ---: | ---: |
| base | 3,282,597 | 2,591,264 | 0 | 41 |
| agent | 2,705,768 | 3,169,408 | 0 | 0 |
| offload | 1,039,577 | 2,647,472 | 2,187,632 | 58 |
| agent_offload | 952,780 | 3,193,296 | 1,726,736 | 0 |

组合配置的首次 prefill 计算量相对原生减少 70.97%，平均请求排队从 62.97 秒降到 28.78 秒。
这支持长前缀复用和排队改善是主要收益来源，但 token 降幅不等于 E2E 降幅。

最新一次默认策略调整，是从选择性生成预留改为对所有已接纳请求使用 `all`。
QPS 1.0 单次诊断中，物理抢占从 58 次降到 0，实际重算从 38,958 token 降到 0，E2E 从 663.98 秒降到 639.22 秒。
约 3.73% 是不同 A800 上的诊断观测，不能单独解释最终相对原生的 34.22% 收益。
扩大共享前缀偏好范围至 1,000 分的诊断只有约 0.17% 的 E2E 改善，同时执行 token 增加、应用 P50 变差，因此仍保留 500 分。

### 请求完成、尾延迟和传输

启用调度的 14 次正式运行中，物理抢占、预留策略引发的抢占和实际重算 token 均为 0。
物理抢占和恢复路径仍然保留，并通过独立的真实模型服务测试验证。
原生和仅 offload 配置没有对应的实际执行区间观测，其重算量为 N/A，不能记为 0。

准入拒绝与抢占分开统计。
例如 QPS 1.0 组合配置有 1,456 次预留准入拒绝和 511,519 次生成容量准入拒绝，但全部请求完成，没有发生抢占。
这些数字统计调度尝试，同一等待请求可以被多次计数，不等于失败的请求数量。

应用时延从一个 DAG 开始执行计到完成。P95 是这些时延的第 95 百分位，用于观察较慢应用的完成时间。

| QPS | base 应用 P95 | 组合应用 P95 |
| --- | ---: | ---: |
| 0.05 | 455.96 秒 | 252.14 秒 |
| 0.1 | 725.33 秒 | 449.19 秒 |
| 0.2 | 821.64 秒 | 520.62 秒 |
| 0.5 | 883.84 秒 | 598.34 秒 |
| 1.0 | 929.12 秒 | 604.50 秒 |

组合配置在所有已测 QPS 改善了应用 P95，但部分关键节点的调用 P95 慢于仅 offload。
例如 QPS 0.2 的关键节点调用 P95 为 77.26 秒，仅 offload 为 51.85 秒。
启用调度的正式运行没有关键准入等待达到 180 秒的记录，但有限 24-DAG 负载不足以证明持续到达下始终没有饥饿。

QPS 1.0 组合配置的 D2H 累计传输时间为 2.105 秒，H2D 为 13.656 秒。
这些是传输作业的累计时间，作业可能重叠，不能直接作为串行部分从 E2E 中扣除。
所有正式 offload 运行均没有传输失败或 CPU 容量拒绝记录。

QPS 0.05 的总 E2E 只下降 22.71%，未达到 25%。
该负载的到达阶段持续 460 秒，减少请求排队不会同比例缩短整个到达和完成区间。
当前结果不能推断为其他模型、硬件、负载或稳态服务速率下的同等提升。

## 文件和目录说明

### 运行代码：`vllm/tokencake`

这一目录包含服务运行时使用的新增实现。

| 文件 | 用途 |
| --- | --- |
| [config.py](vllm/tokencake/config.py) | 定义服务端配置、默认值、参数范围和前置条件检查 |
| [protocol.py](vllm/tokencake/protocol.py) | 校验请求中的 agent、DAG 和生命周期元数据 |
| [scheduling.py](vllm/tokencake/scheduling.py) | 实现请求评分、预留与借用、增长需求计算、共享前缀偏好和抢占选择 |
| [offload_policy.py](vllm/tokencake/offload_policy.py) | 评估等待需求、保存收益、工具时间和传输成本 |
| [offloading.py](vllm/tokencake/offloading.py) | 选择有效 KV 前缀，注册原生 CPU 保存作业并跟踪传输完成 |
| [lifecycle.py](vllm/tokencake/lifecycle.py) | 管理生成完成、工具开始与结束、超时，以及 CPU 缓存的临时保留 |
| [events.py](vllm/tokencake/events.py) | 定义工具等待事件和 EngineCore 内部消息 |
| [api_router.py](vllm/tokencake/api_router.py) | 提供 `/v1/tokencake/events` HTTP 接口并返回处理结果 |
| [metrics.py](vllm/tokencake/metrics.py) | 定义调度、保存、等待、命中和重算指标 |

### 对原生 vLLM 的接入修改

优化需要接入原生请求和缓存流程，因此还修改了以下已有文件或目录。
这里复用原生执行、分配和传输接口，没有替换模型推理内核。

| 位置 | 修改目的 |
| --- | --- |
| [vllm/config/vllm.py](vllm/config/vllm.py) | 解析 TokenCake 配置并检查可用的缓存后端 |
| [vllm/entrypoints/openai](vllm/entrypoints/openai) 下的 completion、chat_completion、responses | 校验并传递请求元数据，限制一个生命周期对应一次生成 |
| [vllm/entrypoints/serve/__init__.py](vllm/entrypoints/serve/__init__.py) | 注册工具等待事件接口 |
| [vllm/engine/protocol.py](vllm/engine/protocol.py) 与 [vllm/v1/engine](vllm/v1/engine) | 将事件从 API 进程传入 EngineCore，并返回确认结果 |
| [vllm/v1/core/sched/scheduler.py](vllm/v1/core/sched/scheduler.py) | 接入排序、准入、抢占、执行统计和生命周期更新 |
| [vllm/v1/core/block_pool.py](vllm/v1/core/block_pool.py) | 将 GPU block 复用与未完成的原生传输同步 |
| [vllm/distributed/kv_transfer/kv_connector](vllm/distributed/kv_transfer/kv_connector) | 注册 TokenCake connector，接入原生保存、回载和缓存命中计算 |
| [vllm/v1/kv_offload/cpu/manager.py](vllm/v1/kv_offload/cpu/manager.py) | 支持生命周期内对 CPU KV 的临时保留和释放 |
| [vllm/v1/metrics](vllm/v1/metrics) | 将 TokenCake 统计与原生指标一起导出 |

### 实验工具：`tools/tokencake_experiments`

这个目录用于准备和运行可复查的性能实验，以及生成分析报告。
正常启动和使用 TokenCake 服务不需要调用这些脚本；这里的调度是实验任务的安排，不是模型请求的运行时调度。

这些工具解决四类问题：保证不同配置使用可比较的输入；记录实际执行版本和参数；管理服务与客户端进程；从完整运行记录计算性能及正确性结果。

| 文件 | 用途 |
| --- | --- |
| `driver.py` | 实验入口，提供 `plan`、`prepare`、`run-cases` 等命令，管理运行记录和复测 |
| `campaign.py` | 定义 QPS、配置组合、GPU/CPU 绑定、模型位置及不同版本的启动命令 |
| `preflight.py` | 运行前检查环境、模型和负载，保存固定输入与启动配置 |
| `runtime.py` | 启停服务与客户端，记录进程、GPU、指标采样和退出状态 |
| `materialize.py` | 从固定的原始版本创建独立实验客户端副本，应用指定补丁 |
| `launch_client.py` | 在目标环境中运行实验客户端，处理 tokenizer 导入位置的兼容 |
| `source_adapter.py` | 在独立进程中调用原始负载构造和结果分析代码 |
| `provenance.py` | 记录代码版本、依赖、模型、数据集及文件哈希 |
| `verify_environment.py` | 检查实际导入的 vLLM、Torch 和原生扩展是否来自指定环境 |
| `report.py` | 检查运行是否合格，计算验收结果并确定需要复测的比较项 |
| `scheduling_reference.py` | 从固定的原始调度实现导出参考结果，供迁移测试使用 |
| `component_graph.py` | 提取执行节点和依赖，供关键路径分析使用 |
| `component_analysis.py` | 分析每次运行的时延、执行量、命中、传输、采样和关键路径 |
| `component_matrix.py` | 汇总五档 QPS、四种配置，生成 JSON、CSV 和 Markdown 表格 |
| `component_plots.py` | 根据同一分析数据生成时延、吞吐、缓存来源等 PNG/PDF 图表 |
| `reports/` | 保存正式报告、方法说明、优化诊断和可分享的结果附件 |
| `README.md` | 记录本机环境准备、实验命令、历史负载和结果校验方式 |

这些工具目前针对已记录的双 A800 环境、模型路径和原始代码版本编写。
它们不是可以直接在任意机器上运行的通用基准工具。
在其他环境中使用时，需要明确调整硬件和输入检查，并生成新的可比较结果。

同目录中的 `.patch` 文件作用于实验客户端副本：

| 文件 | 用途 |
| --- | --- |
| `launcher.patch` | 适配当前请求元数据和工具等待事件协议 |
| `continuation.patch` | 历史 `continuation` 负载，仅调整部分同角色后继调用的输入拼接 |
| `conversation.patch` | 追加式上下文、汇合时共同前缀去重，以及工具调用输出保留 |
| `tool-budget.patch` | 将指定工具指令调用的输出目标设为 128 token |

当前 `conversation-tools` 使用 `conversation.patch` 和 `tool-budget.patch`，并配合客户端协议适配。
`continuation.patch` 对应另一个历史负载选项，不是当前负载中叠加的步骤。
原始应用和数据来自相邻的 `../vllm_agent` 固定版本；补丁不会直接修改该原始代码目录。

### 测试：`tests/tokencake`

| 文件或分组 | 验证内容 |
| --- | --- |
| `test_config.py`、`test_protocol.py` | 配置默认值、无效参数和请求元数据校验 |
| `test_events.py`、`test_lifecycle.py` | 事件处理、重复通知、超时和缓存保留状态 |
| `test_scheduling.py` | 排序、预留借用、完整增长、共享 KV 和抢占恢复规则 |
| `test_offloading.py` | 保存候选、收益判断、CPU 缓存命中及原生 connector 接入 |
| `test_offloading_cuda.py` | 真实 GPU/CPU KV 传输和数据一致性 |
| `test_serving.py`、`test_event_serving.py` | 模型服务接口、请求元数据和事件 API |
| `test_scheduling_serving.py`、`test_offloading_serving.py` | 真实模型服务中的受压恢复、KV 回载和输出检查 |
| `test_experiment_*.py`、`test_component_analysis.py` | 实验准备、客户端、工作量、验收及结果分析 |
| `experiment_workload_probe.py` | 检查独立客户端实际构造的 DAG、输入和输出目标 |
| `scheduling_reference.json` | 原始调度实现生成的固定参考结果 |

此外，已有的 `tests/v1/core/test_scheduler.py` 调整了相关测试设置。
原生 connector 和 CPU KV 管理器的测试也纳入验证，检查接入后是否保持原生行为。

### 文档、历史记录和本地生成文件

| 位置 | 用途 |
| --- | --- |
| `TC_README.md` | 当前实现、性能结果和文件分布的中文入口 |
| `docs/features/tokencake.md` | 服务使用和接口参考 |
| `docs/features/disagg_prefill.md` | 在原生 KV 传输文档中增加 TokenCake 的入口链接 |
| `tools/tokencake_experiments/reports/` | 当前正式结果和后续优化诊断 |
| `pyproject.toml` | 为固定外部接口的已有拼写和依赖名称增加拼写检查例外，不改变推理行为 |
| `openspec/` | 已有迁移计划、历史决策和阶段性证据；不参与服务运行 |
| `.agents/skills/openspec-*` | 已有文档工作流说明；不属于 TokenCake 算法，也不是服务或测试的运行依赖 |
| `.venv/` | 本机 Python 环境、原生基准代码副本和临时验证记录，不是运行算法源码 |
| `/root/autodl-tmp/tokencake-optimization/` | 本机完整实验的原始日志、配置、指标和逐次结果 |

当前正式原始记录位于 `/root/autodl-tmp/tokencake-optimization/components-final-01`。
仓库内的 `reports/components-final-01/` 保存汇总 JSON/CSV、表格和图表，逐次原始分析位于外部结果目录的 `analysis.json`。
复制仓库不会自动带上本地模型、虚拟环境和全部原始实验记录。

## 使用 TokenCake

### 服务端配置

所有 Python 命令使用仓库的 `.venv/bin/python`，依赖安装通过 `uv`。
模型、CUDA 环境、当前版本的原生扩展和 CPU 内存需要事先准备好。
下例使用本机模型，启用调度及 CPU KV 保存：

```bash
env CUDA_VISIBLE_DEVICES=0 VLLM_USE_SIMPLE_KV_OFFLOAD=0 \
  PATH="$PWD/.venv/bin:$PATH" \
  .venv/bin/python -m vllm.entrypoints.cli.main serve \
  /root/autodl-tmp/model/Qwen/Qwen2.5-14B-Instruct \
  --served-model-name tokencake-model \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.5 \
  --max-model-len 32768 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 1024 \
  --kv-offloading-size 100 \
  --kv-offloading-backend native \
  --additional-config '{"tokencake":{}}'
```

CPU KV 容量单位是 GiB，100 GiB 是正式实验使用的设置，部署时需按实际内存条件选择。
开启 TokenCake 要求 `data_parallel_size=1`。
CPU KV 保存要求原生 CPU 后端及注册的 connector，不能使用 simple KV offload 或任意替代 connector。
真实模型性能目前验证到单 GPU、TP=1、PP=1。

| 模式 | `--additional-config` 的值 | CPU offload 参数 |
| --- | --- | --- |
| 组合模式 | `'{"tokencake":{}}'` | 需要 |
| 仅调度 | `'{"tokencake":{"offload":{"enabled":false}}}'` | 无需提供 |
| 仅 KV 保存 | `'{"tokencake":{"scheduling":{"enabled":false}}}'` | 需要 |
| 关闭 TokenCake | 不提供 `tokencake` 命名空间 | 不由 TokenCake 启用 |

配置表中的关闭方式说明功能开关；正式性能基准仍使用独立的原生 vLLM 代码副本。

主要默认值如下，完整范围见 [config.py](vllm/tokencake/config.py)：

| 配置项 | 默认值 | 作用 |
| --- | --- | --- |
| `scheduling.reserve_generation_tokens` | `true` | 准入时计入声明的生成预算 |
| `scheduling.generation_reserve_mode` | `"all"` | 对所有已接纳请求计算后续生成容量 |
| `scheduling.decode_prefill_token_budget` | `0` | 使用原生 prefill 预算 |
| `scheduling.cache_affinity_score_band` | `500` | 允许共享 KV 偏好调整顺序的分数范围 |
| `scheduling.inherit_join_priority` | `true` | 启用同一应用汇合分支的优先级继承 |
| `scheduling.priority_borrow_score_margin` | `500` | 提前借用其他等待类型的闲置预留所需分数优势 |
| `scheduling.reserve_ratio_min`、`scheduling.reserve_ratio_max` | `0.05 / 0.30` | 动态预留比例上下限 |
| `offload.max_relief_blocks` | `0` | 按工具窗口和 CPU 容量计算保存长度 |

### 客户端请求和工具事件

仅开启服务端配置还不足以获得完整的 agent 调度和工具窗口保存行为。
应用需要在请求的 `vllm_xargs.tokencake` 中提供元数据，并在工具等待开始和结束时发送事件。

发送到 `/v1/completions` 的请求示例：

```json
{
  "model": "tokencake-model",
  "request_id": "tc-4c3d2e1f001142228333abcdef123456",
  "prompt": "Write a Python function that sums a list of integers.",
  "max_tokens": 128,
  "vllm_xargs": {
    "tokencake": {
      "lifecycle_id": "tc-4c3d2e1f001142228333abcdef123456",
      "agent_type": "programmer",
      "importance": 3.0,
      "offload_eligible": true,
      "reusable_prefix": true
    }
  }
}
```

示例 ID 仅用于展示格式。每次生成及重试都要创建新的 `tc-` 加小写 UUID4 hex，并让顶层 `request_id` 与 `lifecycle_id` 相同。
一个生命周期只对应一次生成，不能用同一个 ID 展开多个 prompt、多个输出或内置工具导致的多次模型调用。
后续需要继承实际输入前缀；仅声明 `reusable_prefix=true` 不会自动使不匹配的输入命中。

生成成功后，在实际工具操作开始前，向 `/v1/tokencake/events` 发送：

```json
{
  "event": "stall_started",
  "lifecycle_id": "tc-4c3d2e1f001142228333abcdef123456",
  "kind": "file_write",
  "estimated_duration_s": 5.0
}
```

工具完成后，向同一接口发送：

```json
{
  "event": "stall_finished",
  "lifecycle_id": "tc-4c3d2e1f001142228333abcdef123456"
}
```

客户端应等待开始事件成功确认后执行工具，等待结束事件成功确认后提交后继生成。
事件确认表示服务端已处理状态变更，异步 D2H 仍可能尚未完成。
仅启用调度时不发送工具事件。未携带 TokenCake 元数据的请求保持原生请求排序；启用 connector 后仍可使用原生自动 CPU 保存和回载。

## 复现实验和读取结果

### 创建新的完整组件测试

先按 [实验工具 README](tools/tokencake_experiments/README.md) 准备本机环境、原生基准和原始负载版本。
以下命令创建五档 QPS、四种配置的 20 组初始测试；结果目录必须尚未存在。

```bash
TC_RUN_DIR=/root/autodl-tmp/tokencake-optimization/components-new-01
.venv/bin/python -m tools.tokencake_experiments.driver plan --components
.venv/bin/python -m tools.tokencake_experiments.driver prepare "$TC_RUN_DIR" \
  --components --snapshot-target --workload-profile conversation-tools
.venv/bin/python -m tools.tokencake_experiments.driver run-cases "$TC_RUN_DIR"
```

`prepare` 保存输入、代码和配置的固定记录，并创建独立客户端。
每次正式运行都检查这些记录，完成全部 24 个 DAG，不截断后段工作。
带 CPU offload 的服务之间串行，避免在当前 240 GiB 容器内同时申请两份 100 GiB KV 容量及模型加载内存。
等待实验资源发生在客户端计时前。

`run-cases` 只运行已准备的初始配置，不自动执行边界复测。
本次报告额外的 8 次复测由 `Runner.repeat_affected(include_old=False)` 按原有规则完成，沿用同一份运行记录，并通过文件锁防止多个进程同时修改。
具体执行记录见 [完整报告中的复现说明](tools/tokencake_experiments/reports/component-evaluation.md#复现命令)。
修改负载或运行代码后，应创建新实验记录，并使用同条件的原生基准重新比较。

### 生成分析、表格和图表

所有组合都有合格结果后，在上面的新结果目录中运行：

```bash
.venv/bin/python -m tools.tokencake_experiments.component_graph \
  "$TC_RUN_DIR/launcher" "$TC_RUN_DIR/graph.json"
.venv/bin/python -m tools.tokencake_experiments.component_analysis \
  "$TC_RUN_DIR" --graph "$TC_RUN_DIR/graph.json" \
  --output "$TC_RUN_DIR/analysis.json"
.venv/bin/python -m tools.tokencake_experiments.component_matrix \
  "$TC_RUN_DIR/analysis.json" "$TC_RUN_DIR/tables"
.venv/bin/python -m tools.tokencake_experiments.component_plots \
  "$TC_RUN_DIR/analysis.json" "$TC_RUN_DIR/figures"
```

输出文件和目录使用新路径，避免覆盖已有记录。
分析会检查完整性、工作量、到达序列和文件哈希，再按各指标汇总合格运行。
来源堆叠图与应用时延 CDF 使用 E2E 中位数对应的实际运行，避免把不同运行的分项拼成一个不存在的样本。

阅读指标时注意：

- 首次 prefill 来源包含本地计算、GPU 命中和 CPU 命中，与抢占后的实际重算是不同指标。
- 抢占恢复的 GPU/CPU 命中只统计被抢占请求再次准入；工具后继调用的命中另计。
- 关键准入等待报告所有运行中的实际最大值，不能只看每次最大值的中位数。
- 原生 KV block 占用不等于 CUDA 已分配显存，也不覆盖空闲队列中所有仍可复用的前缀。
- GPU 活跃度不等于有效计算比例；有限负载的完成吞吐也不等于可持续服务 QPS。

### 验证记录

正式测试前，464 项 CPU 测试和 4 项真实 Qwen2.5-14B 服务测试通过；CPU 测试明确筛除了 3 项。
报告最终修改后，17 项分析测试再次通过。
适用的 pre-commit 检查也已通过，具体命令、筛选条件和 JUnit 记录见 [最终版本验证](tools/tokencake_experiments/reports/stage-22-affinity-convergence.md#final-version-validation)。

日常修改可以先运行相关测试，例如：

```bash
env CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 .venv/bin/python -m pytest \
  tests/tokencake/test_scheduling.py \
  tests/tokencake/test_offloading.py \
  tests/tokencake/test_component_analysis.py -q
```

这类测试用于检查功能和分析逻辑，不能替代完整 DAG 性能实验。
当前进一步优化应重点检查低压力下的准入保守程度，以及重要性和共享 KV 偏好对个别关键节点尾延迟的影响。

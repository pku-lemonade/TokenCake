# Agent Benchmark 首轮两组试跑

日期：2026-09-08。两组执行和评分已完成。

其余两组也已完成，完整对照见 [四组试跑报告](agent-benchmark-four-mode-pilot.md)；本文保留首对的详细依据。
用户已要求试跑后停止并取消正式全量，收尾结论见 [试跑最终分析](agent-benchmark-pilot-final-analysis.md)。
三个口径的定义、全程平均的时间权重，以及前 30 分钟之后的任务与排队情况，见该报告第 2–5 部分。

SWE 固定 20 题中，`base` 通过 0 题，`agent_offload` 通过 1 题。
实际缓存复用、生成间隔与抢占指标出现改善，同时排队和超时增加。
按用户指定口径，共同前 30 分钟的输出吞吐量提高 27.6%，各自全程提高 8.6%，客户端完整响应提高 6.8%。
这些数据支持本轮输出吞吐量更高；输出包含失败任务的内容，解题效率和稳定收益仍需独立判断。

首对使用 GPU 0 `base`、GPU 1 `agent_offload`，两张 A800 独立运行相同 Qwen2.5-14B-Instruct BF16 配置。
每组应用并发上限为 16，共享 36 核 CPU 配额和 240 GiB cgroup 内存。
两组实际 GPU KV 容量均为 48,800 token；`agent_offload` 配置 100 GiB CPU KV。
任务采用真实生成历史和真实工具操作，闭环释放；相同并发规则不保证相同实际到达时刻或生成量。

输入与环境依据：

- [已确认清单](/root/autodl-tmp/tokencake-agent-bench/manifests/confirmed-parallel-20260908-01/manifest.json)。
- [20 题原始版本与参考修复验证](/root/autodl-tmp/tokencake-agent-bench/grading/confirmed-environment-validation-summary-02.json)。
- [60 个独立环境身份核对](/root/autodl-tmp/tokencake-agent-bench/grading/confirmed-prepared-environment-audit-01.json)。
- [并行运行器与适配验证索引](/root/autodl-tmp/tokencake-agent-bench/results/parallel-platform-validation-index-20260908.json)。

## BFCL 四题功能验证

两组均完成四题执行和官方评分，解决率均为 0/4。
这些是 Base 与 Long Context 各两题的功能样例，实际应用并发最多为 4。
原生环境准备曾与功能验证同时运行，不能用这一批数据得出独立性能或 16 并发压力结论。

| 指标 | base | agent_offload |
| --- | ---: | ---: |
| 计划 / 完成执行 / 完成评分 | 4 / 4 / 4 | 4 / 4 / 4 |
| 官方任务通过数 | 0 | 0 |
| 应用执行区间，秒 | 24.95 | 108.87 |
| 平均应用 E2E，秒 | 14.75 | 36.69 |
| 模型响应数 | 29 | 28 |
| 实际输入 token | 305,548 | 330,506 |
| 实际输出 token | 1,272 | 5,392 |
| GPU KV 使用率采样峰值 | 62.32% | 62.87% |

`agent_offload` 的实际输出约为 base 的 4.24 倍，轨迹和工作量已经不同，不能将耗时差直接归因于框架。
这四题未观察到非零 KV 传输计数；真实工具调用通常不足 1 毫秒，也没有持续的容量压力。

| 任务 | 官方评分指出的主要失败 |
| --- | --- |
| `multi_turn_base_198` | TwitterAPI 未达到参考认证和发布状态 |
| `multi_turn_base_103` | base 第三轮无有效调用；agent_offload 使用错误订单价格 |
| `multi_turn_long_context_97` | 缺少参考要求的路线查询执行结果 |
| `multi_turn_long_context_96` | 第二轮无有效函数调用 |

首批执行遇到官方工具状态包含 Python `set` 的 JSON 保存问题，修复后使用新服务和新输出目录完整重跑。
修复复用 BFCL 官方序列化函数；包含真实 MessageAPI 状态的适配回归检查已通过。
首批失败仍保留并计入测量预算，不混入修复后的四题成绩。

证据：[评分](/root/autodl-tmp/tokencake-agent-bench/grading/pilot-bfcl-parallel-first-pair-02/scores.json)、[完整指标](/root/autodl-tmp/tokencake-agent-bench/results/pilot-bfcl-parallel-first-pair-02/report.json)、[结果分析](/root/autodl-tmp/tokencake-agent-bench/results/pilot-bfcl-parallel-first-pair-02/assessment.json)。

## SWE 20 题试跑

两组各 20 题，使用已验证的独立任务环境，每组应用并发上限为 16。
测量期间没有环境安装和离线评分；20/20 初始输入及生成参数逐题一致。
评分使用独立本地环境与冻结的官方脚本、解析器和 resolved 规则，属于 `custom_local` 结果。
原始版本和参考修复验证通过，不意味着本地环境与官方 Docker 镜像完全等价。

### 优先比较正确结果

主指标为相同墙钟预算内通过官方评分的题数。
时间从每组开始释放任务计起；使用最终提交产生的时间，之后的离线判分不计入模型执行预算。
失败、空补丁和超时不会增加正确完成曲线，其消耗的时间仍计入预算。

| 相同时间预算 | base 正确完成数 | agent_offload 正确完成数 |
| --- | ---: | ---: |
| 15 分钟 | 0 / 20 | 1 / 20 |
| 30 分钟 | 0 / 20 | 1 / 20 |
| 60 分钟 | 0 / 20 | 1 / 20 |

唯一通过题为 `sympy__sympy-16766`，在 `agent_offload` 测量开始后 306.45 秒提交。
该题本身 E2E 为 203.08 秒，其余时间为闭环释放前的等待；完成曲线使用前者，不能使用任务自身 E2E 冒充全局完成时间。
`base` 的固定 20 题在 36.64 分钟全部结束，此后曲线保持为零，没有向它补充新题。
这是固定批次结果，不能外推持续到达场景的每小时处理能力；也不能把 1 对 0 写成有限倍数加速。

![相同预算内正确完成曲线](/root/autodl-tmp/tokencake-agent-bench/results/pilot-swe-parallel-first-pair-01/correct-completions.png)

双方共同通过的题数为 **0**，因此共同成功题的配对 E2E **不可用**。
单方成功为 `agent_offload` 1 题、`base` 0 题，其余 19 题双方均未解决。
20 题、单次运行、仅 1 个成功，不足以认定质量等价或稳定提升。

### 工作量、失败与计时

| 指标 | base | agent_offload |
| --- | ---: | ---: |
| 计划 / 已记录终态 / 已评分 | 20 / 20 / 20 | 20 / 20 / 20 |
| Submitted / 其中非空补丁 | 14 / 6 | 8 / 7 |
| 空补丁 | 8 | 1 |
| 上下文溢出 | 5 | 1 |
| 重复格式错误终止 | 1 | 0 |
| 请求读取超时 | 0 | 11 |
| 官方测试通过 | 0 | 1 |
| 应用执行区间，分钟 | 36.64 | 60.02 |
| 模型响应 / 尝试数 | 587 / 587 | 834 / 845 |
| 响应报告的输入 token | 7,972,616 | 10,812,859 |
| 响应报告的输出 token | 95,692 | 167,360 |
| 请求重试 | 0 | 0 |

`agent_offload` 的 11 个读取超时中，2 个触及独立的 600 秒模型请求上限；另 9 个发生在任务的一小时时限到达时，请求上限已经按任务剩余时间收紧。
输出 token 只统计成功返回的响应，超时请求可能已执行但未交付的 token 不可从客户端恢复。
两组工具调用次数分别为 581 和 817，中位工具时间约为 14.74 和 15.97 毫秒，没有人工增加等待。

提交后的五个 `test_output_missing` 均由模型补丁引入的 Python 语法或缩进错误导致：
`base` 的 Sympy 15976、Sympy 21379、Scikit-learn 25102，以及 `agent_offload` 的 Django 13933、Sympy 22080。
两组各有一个补丁在完整官方应用策略后仍失败：`base` 的 Django 11820 和 `agent_offload` 的 Django 13794。
这些结果保留在 20 题分母内；没有丢弃失败后只比较成功 HTTP 请求的总耗时。

评分适配现沿用冻结官方的四种补丁应用策略、重试间的部分应用复位及反向检查。
已保存的全部预测统一重新评分，模型执行未重跑；第一版仅尝试一次 `git apply` 的评分目录保留，本文以 `pilot-swe-parallel-first-pair-02` 的评分为准。
包含真实 Git 和 GNU patch 的回归验证了部分应用复位与最终 fuzz 应用，避免把适配器缺失误算为模型失败。

后续修复了部分 Django 评分脚本的 locale 初始化适配，并在新环境中复核受影响的四题。
首对及其余两组共 16 份保存预测的评分状态与通过结果均未改变；见 [Django 复核报告](agent-benchmark-django-environment-validation.md)。

### 系统与机制诊断

输出吞吐量按用户指定的方法重新计算，统一使用 `vllm:generation_tokens_total`。
每个采样仅汇总当前模型各 engine 的累计值，减去 `metrics-start.prom` 的初始值，再除以一次窗口墙钟时间。
两组初始值均为 0；不累计不同时刻的计数器，不计入 prompt token，不除以并发数或请求耗时总和。

共同起点取计时账本的 `1788832262.4999597`，前 30 分钟截止为 `1788834062.4999597`。
各自全程截止使用对应 `tasks/execution.json` 的 `ended_at`，分母严格使用其 `wall_s`。
每个窗口取 `metrics.jsonl` 中 `timestamp <= 截止时间` 的最后一个有效采样，不使用服务关闭前的 `metrics.prom`。

| 口径，单位 token/s/GPU | base | agent_offload | 相对变化 |
| --- | ---: | ---: | ---: |
| 服务端：共同前 30 分钟 | 83,292 / 1800 = 46.27 | 106,239 / 1800 = 59.02 | +27.6% |
| 服务端：各自全程 | 95,659 / 2198.273171 = 43.52 | 170,156 / 3600.932717 = 47.25 | +8.6% |
| 客户端：完整响应 | 95,692 / 2198.273171 = 43.53 | 167,360 / 3600.932717 = 46.48 | +6.8% |

表中墙钟显示到小数点后六位，计算使用原文件中的完整精度。
墙钟包含排队、prefill、decode、工具执行、运行中的等待和失败耗时；不包含加载模型或本组结束后等待另一组的空闲时间。
客户端分子来自报告的 `model_work.reported_output_tokens`，只含完整返回并记录 usage 的响应。
服务端分子还可能包含超时或中断请求已生成但未完整交付的 token，两个口径分别报告。

| 窗口末次采样距截止时间 | base | agent_offload |
| --- | ---: | ---: |
| 前 30 分钟 | 1.730 秒 | 0.177 秒 |
| 各自全程 | 1.703 秒 | 1.957 秒 |

全部 1783 / 1744 条采样均有效，未观察到计数器回退、序列缺失、重复指标序列或时间戳逆序。
运行期间最大采样间隔分别为 2.129 / 2.150 秒，没有明显断档；窗口末尾存在上述采样滞后，因此服务端吞吐是近似值。
程序逐 engine 检查计数器下降，避免某个 engine 的重置被其他 engine 的增长掩盖；发生重置时不给出正常吞吐结论。

各自全程时长和实际轨迹不同，因此同时展示共同前 30 分钟与完整运行结果。
输出 token 包含最终失败任务的内容，即使客户端完整收到响应，也不能据此认定任务解决正确。
这些数字衡量输出吞吐，不能直接解释为解题效率提升。

另一次独立核算直接只读原始指标、任务计时和原 `report.json`，复现全部六个 token 校验值及窗口末次采样。
输入文件核算前后的哈希一致；证据见 [原始文件独立复核](/root/autodl-tmp/tokencake-agent-bench/results/pilot-first-pair-throughput-readonly-audit-20260908-01.json)。

以下其他服务端计数和时间诊断也已改用本组结束前的最后有效采样；旧版报告保留。

| 指标 | base | agent_offload |
| --- | ---: | ---: |
| 实际缓存复用 token / 输入 token | 9.97% | 10.39% |
| 其中 CPU 回载 token / 输入 token | 0% | 1.14% |
| 服务端平均 TTFT | 21.20 秒 | 37.11 秒 |
| 服务端平均排队时间 | 18.06 秒 | 34.65 秒 |
| 平均相邻生成 token 间隔 | 46.80 ms | 34.25 ms |
| 按请求平均 TPOT | 51.73 ms | 36.55 ms |
| GPU KV 使用率采样峰值 | 100% | 91.83% |
| 原生抢占计数 | 11 | 0 |
| GPU → CPU 保存 | 不可用 | 3 次，0.94 GiB |
| CPU → GPU 回载 | 不可用 | 70 次，22.61 GiB |
| 保存 / 回载传输累计时间 | 不可用 | 0.042 / 0.991 秒 |

缓存复用率使用服务端实际输入来源计数，包含 GPU 命中和 CPU 回载；不使用缓存查找命中率代替实际复用比例。
CPU 回载包含同一已保存前缀的重复使用，因此回载累计量可以高于保存量。
`agent_offload` 调度器报告重算为 0 token，原生基线缺少相同计数器，不能据此计算重算减少比例。

按服务端 token 间隔加权，生成间隔缩短约 26.8%；按请求加权 TPOT 缩短约 29.3%。
实际复用率提高约 0.42 个百分点；平均排队时间增加约 91.8%。
这些聚合量的请求长度、执行轨迹和观测总体不同，超时后服务端与客户端统计覆盖范围也会不同。
可表述为“本次运行出现局部机制改善”，不能表述为控制同等工作量后的框架加速。

已生成输入长度与输出长度的联合分组诊断：每组给出请求数、HTTP E2E 分布、失败时间和对全程输出吞吐的贡献。
失败请求的输出长度标为未知；HTTP 返回成功也可能属于最终失败的任务。
当前请求采用非流式响应，没有逐请求首 token 或 decode 时间戳，**按长度分组的 TTFT 和 TPOT 不可用**。
聚合 Prometheus 时间不能反向分配到长度分组，也不能用 HTTP 耗时除输出长度冒充纯生成 TPOT。
后续若要控制长度后比较这些指标，需要两组统一补充可关联请求 ID 的服务端时间戳，再运行同一配置。

主机内存是两组共享的 cgroup 观测，采样峰值为 239.99 GiB，限制为 240 GiB，无 OOM/OOM kill。
它包含共享内存和文件缓存，不能将该峰值当作某一组的私有 RSS，或简单认定工具进程耗尽物理内存。
两组均未发现外部 GPU 计算进程、监测查询失败或 CPU affinity 违规。

原始证据：

- [按用户窗口口径更新的完整报告](/root/autodl-tmp/tokencake-agent-bench/results/pilot-swe-parallel-first-pair-01/report-throughput-02.json)与[分析代码身份](/root/autodl-tmp/tokencake-agent-bench/results/pilot-swe-parallel-first-pair-01/report-implementation-02.json)。
- [吞吐边界与异常回归](/root/autodl-tmp/tokencake-agent-bench/results/window-throughput-tests-20260908.xml)，覆盖多 engine、窗口后输出、计数器重置与缺失采样。
- [统一重新评分结果](/root/autodl-tmp/tokencake-agent-bench/grading/pilot-swe-parallel-first-pair-02/scores.json)与[评分代码身份](/root/autodl-tmp/tokencake-agent-bench/grading/pilot-swe-parallel-first-pair-02/grading-implementation.json)。
- [20 题初始输入对齐](/root/autodl-tmp/tokencake-agent-bench/results/pilot-swe-parallel-first-pair-01/initial-input-pair-audit.json)。
- [指标回归测试](/root/autodl-tmp/tokencake-agent-bench/results/quality-report-tests-20260908.xml)与[官方补丁应用回归](/root/autodl-tmp/tokencake-agent-bench/results/grading-patch-fallback-tests-20260908.xml)。

运行证据：[配置与适配代码哈希](/root/autodl-tmp/tokencake-agent-bench/results/pilot-swe-parallel-first-pair-01/experiment.json)。
预算记录：[共享计时账本](/root/autodl-tmp/tokencake-agent-bench/results/parallel-pilot-budget.jsonl)，并行区间只计一次。

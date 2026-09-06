# TokenCake 性能优化与完整组件评估

状态：20 组初始测试及 8 组必要复测全部完成，28 次运行均通过完整性检查，无排除运行。

## 结论

最终调度版本保留 agent/DAG 重要性排序、动态 KV 预留与借用、工具等待窗口内的可复用 KV 保存。
本轮新增的核心改动是：准入时逻辑计入所有已接纳请求的剩余生成预算，避免已经接纳的工作共同承诺过多后续 KV 容量。
真实 KV 仍按原生路径逐步分配。

QPS 0.1、0.2、0.5、1.0 的总 E2E 相对原生 vLLM 分别降低 28.65%、31.78%、30.98%、34.22%。
QPS 0.05 的三次中位数降幅为 22.71%，未达到 25% 目标。
本次是四档达标、一档未达标，不能概括为所有五档均达到 25%。

单独开启 offload 已提供大部分收益。组合配置在 QPS 0.2、0.5、1.0 进一步降低整体 E2E，在 QPS 0.1、0.05 则与单独 offload 基本持平。
组合配置改善了应用级尾延迟，但并不在每一个关键节点的调用 P95 上优于单独 offload。

## 实验条件

每组执行完整 24-DAG、648 次模型调用，生成 155,136 个输出 token。
使用 Qwen2.5-14B-Instruct、BF16、单张 A800-SXM4-80GB、0.5 显存比例、3,050 个 GPU KV block、16 token/block。
保留原生 prefix caching、异步调度、完整输入长度检查及 8,192-token 批处理预算。
两种 offload 配置使用 100 GiB CPU KV 容量和原生传输路径。

| 标签 | Agent 空间调度 | 工具窗口 KV 保存 |
| --- | --- | --- |
| base | 关闭，独立原生 vLLM checkout | 关闭 |
| agent | 开启 | 关闭 |
| offload | 关闭 | 开启 |
| agent_offload | 开启 | 开启 |

正式矩阵中的三种组件模式使用同一个目标 runtime，base 使用独立原生版本。
同 QPS 使用同一份冻结等间隔到达序列；最低 QPS 的最后一个 DAG 在第 460 秒到达。
总 E2E 包含到达和排空，不含服务初始化。
QPS 0.1 和 0.05 的 base、agent_offload 按既定边界规则各运行三次，其余组合各运行一次。
所有合格运行保留，按每项指标的中位数汇总；单次运行不代表统计稳定性。

负载沿用此前选择的 `conversation-tools`，本轮没有再次改变任务、输出预算或等待时长，也没有使用 speculative decoding。
此前该负载已将工具指令输出目标缩减到 128 token，并采用追加式对话上下文。
当前四种模式的工作量一致，但这不是早期 271,200-output-token 负载下的纯算法加速，也没有证明任务质量等价。

完整定义、冻结版本、到达分布、硬件绑定和正确性边界见 [实验方法](component-methodology.md)。
报告结构参考论文的端到端、组件及传输成本评估；硬件、应用规模、到达分布与回载机制有差异，不复用论文数值作为验收依据。
参见 [TokenCake 论文第 7 节](https://arxiv.org/html/2510.18586v4#S7)。

## 端到端结果

单位为秒；降幅使用 `1 - T_agent_offload / T_base`。

| QPS | base E2E | agent E2E | offload E2E | agent_offload E2E | 相对 base 降幅 | 25% 目标 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0.05 | 875.74 | 806.39 | 677.62 | 676.86 | 22.71% | 未达到 |
| 0.1 | 940.76 | 825.72 | 672.67 | 671.28 | 28.65% | 达到 |
| 0.2 | 938.79 | 815.69 | 681.66 | 640.45 | 31.78% | 达到 |
| 0.5 | 938.99 | 830.47 | 713.16 | 648.05 | 30.98% | 达到 |
| 1.0 | 960.44 | 837.78 | 691.01 | 631.77 | 34.22% | 达到 |

| QPS | base 应用均值 | 组合应用均值 | base 应用 P95 | 组合应用 P95 |
| --- | ---: | ---: | ---: | ---: |
| 0.05 | 326.36 | 204.44 | 455.96 | 252.14 |
| 0.1 | 559.02 | 359.77 | 725.33 | 449.19 |
| 0.2 | 661.92 | 448.81 | 821.64 | 520.62 |
| 0.5 | 776.26 | 529.65 | 883.84 | 598.34 |
| 1.0 | 864.15 | 525.25 | 929.12 | 604.50 |

![总 E2E、应用均值及 P95](/root/TokenCake/vLLM-TokenCake-upstream/tools/tokencake_experiments/reports/components-final-01/figures/latency.png)

最低 QPS 的结果需要同时看总 E2E 和单个应用时延。
该点的应用 P95 降低 44.70%，但总 E2E 只降低 22.71%。
等间隔到达持续 460 秒，后续仍有实际生成与工具依赖，减少排队不会等比例缩短整个有限负载的测量区间。
这一解释不改变 25% 的验收口径，也不能据此宣称该点达标。

## 收益来源

QPS 1.0 的四配置对比给出最直接的分解：

| 配置 | 总 E2E | 首次 prefill 计算 token | GPU 前缀命中 token | CPU KV 命中 token | 原生总抢占 |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 960.44 | 3,282,597 | 2,591,264 | 0 | 41 |
| agent | 837.78 | 2,705,768 | 3,169,408 | 0 | 0 |
| offload | 691.01 | 1,039,577 | 2,647,472 | 2,187,632 | 58 |
| agent_offload | 631.77 | 952,780 | 3,193,296 | 1,726,736 | 0 |

agent 通过重要性、依赖和受限共享前缀偏好改变执行顺序，减少排队和首次输入计算。
工具窗口保存进一步减少必须重新计算的输入前缀。
组合配置的首次 prefill 计算量相对 base 降低 70.97%，平均请求排队从 62.97 秒降至 28.78 秒。
这些记录支持缓存复用和排队改善的解释，但不能把某一项 token 降幅直接视为相同幅度的 E2E 降幅。

![首次 prefill 的计算、GPU 命中与 CPU 命中来源](/root/TokenCake/vLLM-TokenCake-upstream/tools/tokencake_experiments/reports/components-final-01/figures/prefill-sources.png)

来源图采用每组 E2E 中位数对应的实际运行，20 根柱子的三个来源之和均核对为该次运行的完整输入 token 总量。

组合相对单独 offload 的额外 E2E 降幅，在 QPS 0.2、0.5、1.0 分别为 6.05%、9.13%、8.57%。
QPS 0.1 的中位数对比只有约 0.21%，QPS 0.05 约 0.11%，不足以证明稳定的额外收益。
两种单独策略的降幅不能相加。使用 `T_agent + T_offload - T_base - T_combined` 定义的交互项，在全部五档 QPS 为负，与两种策略节省的工作存在重叠的解释一致。
负交互项不等于组合配置相对 base 退化。

## 抢占与容量

agent 的五次运行和 agent_offload 的九次运行均完成，14 次运行的物理抢占、预留抢占、实际重复计算 token 均为 0。
这表明在当前完整负载中，后续增长承诺避免了已接纳请求反复被迫重算。
物理分配失败时的抢占与恢复路径仍保留，并由独立真实服务测试覆盖。

预留拒绝与抢占必须分开。例如 QPS 1.0 组合配置记录了 1,456 次预留准入拒绝和 511,519 次生成容量准入拒绝，但没有任何抢占，全部请求最终完成。
这些数字统计调度尝试，同一等待请求可以多次计数，不是数十万个失败请求。

base 和 offload 没有启用 TokenCake 的实际执行区间观测，不能将缺失的重算 token 记成 0。
保留其原生总抢占和首次 prefill 来源统计。
启用调度的运行没有发生抢占，因此抢占恢复的 GPU/CPU 命中为 0；工具后继调用的 CPU 命中属于首次 prefill 来源，两者不能混用。

## 重要请求

下表的最大准入等待取该模式该 QPS 所有运行中的实际最大值，不取最大值的中位数。

| QPS | agent 最大关键准入等待 | 组合最大关键准入等待 | 组合关键准入等待达到 180 秒 |
| --- | ---: | ---: | ---: |
| 0.05 | 95.506 | 58.142 | 0 |
| 0.1 | 115.692 | 96.312 | 0 |
| 0.2 | 114.911 | 108.454 | 0 |
| 0.5 | 109.208 | 92.971 | 0 |
| 1.0 | 134.409 | 107.021 | 0 |

已测负载中没有遗留未完成请求，组合配置没有关键准入等待达到 180 秒的记录。
这是有限 24-DAG 负载的观测，不能证明任意持续到达下没有饥饿。

固定 DAG 每次包含 408 次标注关键节点的模型调用，其调用 P95 显示了另一项取舍：

| QPS | base 关键节点 P95 | agent 关键节点 P95 | offload 关键节点 P95 | 组合关键节点 P95 |
| --- | ---: | ---: | ---: | ---: |
| 0.05 | 41.46 | 41.70 | 21.88 | 27.06 |
| 0.1 | 66.97 | 105.96 | 41.62 | 62.88 |
| 0.2 | 78.08 | 91.68 | 51.85 | 77.26 |
| 0.5 | 107.86 | 87.68 | 74.81 | 86.55 |
| 1.0 | 165.64 | 96.33 | 106.56 | 86.89 |

组合配置在 0.05、0.1、0.2、0.5 的这一节点指标上慢于单独 offload。
例如 QPS 0.2 的组合关键节点 P95 为 77.26 秒，对方为 51.85 秒，尽管组合的应用级 P95 从 569.00 秒降到 520.62 秒。
节点级检查中，组合配置的 `reviewer_1`、`reviewer_2` 调用 P95 约为 102 秒，属于该点较慢的节点。
调用时延包含排队、推理和调用开销；聚合准入指标不足以把某个节点的完整耗时全部归因于容量控制。

## KV 与传输

QPS 1.0 组合配置累计 D2H 为 52,996,079,616 字节、104 个作业、2.105 秒；H2D 为 339,490,111,488 字节、327 个作业、13.656 秒。
单独 offload 的 H2D 为 433,475,026,944 字节、360 个作业、17.328 秒。
组合配置通过更多 GPU 侧复用减少了 CPU 回载量，CPU KV 仍承担了大量后继输入复用。
传输耗时是作业累计值，可以相互重叠，不能作为串行部分直接从总 E2E 中扣除。

QPS 1.0 的两种 offload 运行均没有 CPU 容量拒绝或传输失败记录。
当前证据不支持将增加 CPU KV 容量作为该点的优先优化项。

同一 QPS 下，base、agent、offload、组合配置的平均 KV block 占用分别为 84.54%、84.40%、86.33%、87.75%，GPU 活跃度分别为 98.75%、99.22%、97.81%、98.81%。
更低 KV 占用或更高 GPU 活跃度都不能单独说明性能更好。
这些 KV 占用指标不包含空闲队列中仍可复用的全部前缀，也不表示预分配 CUDA buffer 的大小改变。

## 关键路径

按每个 DAG 实际最晚完成的依赖回溯，QPS 1.0 的平均关键路径 LLM 时长从 base 的 836.74 秒降到组合的 496.10 秒。
工具部分分别为 27.39 秒与 29.14 秒，编排间隙均约 0.01 秒。
工具配置未变化，但不同运行的实际关键分支可以变化，因此选中路径上的工具时间不要求逐项相等。

分解逐个核对 LLM、工具、剩余处理和编排间隙之和等于应用时延。
并行分支的全部耗时不会直接相加为 E2E。
当前收益主要体现在关键路径上的模型调用与其排队，不能解释成工具操作本身被加速。

## 优化收敛

本轮把生成预算承诺从选择性回收模式调整为覆盖所有已接纳请求。
QPS 1.0 诊断中，物理抢占从 58 降为 0，实际重算从 38,958 token 降为 0，E2E 从 663.98 秒降到 639.22 秒。
这是不同 A800 上的单次诊断，3.73% 的观测降幅不是独立、稳定的因果估计。

进一步将共享前缀偏好的分数范围从 500 扩大到 1000，只观测到 0.17% 的额外 E2E 降幅，同时执行 token 增加 32,556，P50 应用时延变差。
最终保留 500，进入完整矩阵验证。
这表示已测试参数的收益趋于收敛，不表示已经穷尽全部优化，也不把接近 99% 的 GPU 活跃度当作硬件吞吐上限证明。

诊断和验证细节见 [完整生成承诺](stage-21-generation-commitments.md) 与 [共享前缀范围诊断](stage-22-affinity-convergence.md)。
下一轮值得研究的方向是低压力时的准入保守程度，以及重要性和缓存偏好对个别关键节点尾延迟的影响。
当前正式矩阵未在运行中改变这些策略或负载。

## 验证与复现

正式记录位于 `/root/autodl-tmp/tokencake-optimization/components-final-01`。
28 次运行共完成 672 个 DAG、18,144 次模型调用和 4,343,808 个输出 token。
全部通过版本、工作量、到达序列、终止状态、无重试、无输入减半及原始采样文件哈希检查。
正式运行没有排除项；此前诊断中的启动失败仍保留在独立诊断目录，不混入当前矩阵。

KV 采样的最低区间覆盖率为 99.9964%，GPU 采样覆盖率为 100%。
所有 offload 运行的传输失败与 CPU 容量拒绝计数均为 0。
基准服务已退出。统计汇总增加实际最大关键等待、绘图修正分项聚合与移动图例的修改均在测量结束后完成，不影响冻结的运行版本。

运行最终版本前，464 项 CPU 测试通过，3 项明确筛除；另有 4 项真实 Qwen2.5-14B 服务测试通过。
服务测试覆盖受压抢占恢复、不同队列策略、工具前缀保存和原生 CPU KV 回载及输出检查。
相关修改已通过适用的 pre-commit 检查。
完整命令和 JUnit 记录位置见前述诊断报告。
报告分析测试另通过 17 项，记录为 `.venv/component-final-report.xml`。

## 数据与图表

- [完整四配置表格](components-final-01/tables.md)：包含均值、P50、P90、P95、P99、最大应用时延、吞吐、调度及关键路径。
- [汇总 CSV](components-final-01/measurements.csv) 与 [汇总 JSON](components-final-01/summary.json)：含 20 组指标、观测范围、采样覆盖率、传输开销及原始结果路径。
- [逐次分析 JSON](/root/autodl-tmp/tokencake-optimization/components-final-01/analysis.json)：保留全部 28 次运行的身份、逐应用时延、节点时延、关键路径、采样统计和校验记录。
- [最终验收判定](/root/autodl-tmp/tokencake-optimization/components-final-01/reports/0001.json)：保留 QPS 0.05 的未达标状态及所有中位数输入。

| 图 | PNG | PDF |
| --- | --- | --- |
| 总体与应用时延 | [PNG](components-final-01/figures/latency.png) | [PDF](components-final-01/figures/latency.pdf) |
| 吞吐、KV 占用、GPU 活跃度 | [PNG](components-final-01/figures/throughput-memory.png) | [PDF](components-final-01/figures/throughput-memory.pdf) |
| 组件贡献与交互项 | [PNG](components-final-01/figures/component-effects.png) | [PDF](components-final-01/figures/component-effects.pdf) |
| 首次 prefill 来源 | [PNG](components-final-01/figures/prefill-sources.png) | [PDF](components-final-01/figures/prefill-sources.pdf) |
| 应用时延 CDF | [PNG](components-final-01/figures/application-cdf.png) | [PDF](components-final-01/figures/application-cdf.pdf) |

图中的范围线是实际最小值与最大值，不是置信区间。
CDF 与来源堆叠图使用 E2E 中位数对应的实际运行，其他曲线按指标中位数汇总。

| 记录 | SHA-256 |
| --- | --- |
| 冻结清单 `frozen.json` | `a0764f901f1ec8d9516294dcb9b9c948f81f10626a15ba1899b711bd06f16cfc` |
| 冻结 DAG `graph.json` | `0172f4bc1271445879918931f17196535317fd9206eda00f26633b751f4cbbb1` |
| 全部运行分析 `analysis.json` | `963038ad886e05b9449e5622cc202a690e2bc3a14388e487553be2687477ad11` |
| 汇总 `summary.json` | `088fce7f88a021569c433583bec8fafc43d5ec5d1f2b2494387bb8bc1c02701c` |
| 汇总 `measurements.csv` | `c2573c1f03b3564fecbae6d554c5bbca98f512ce7da91a811853d634296db200` |

## 复现命令

初始队列使用本机 `.venv/optimization_run_pair.py`，复测使用 `.venv/optimization_repeat_native.py`。
它们调用已提交的 `Runner.run_queues` 和 `Runner.repeat_affected(include_old=False)`，持有同一 campaign 锁并执行冻结检查。
每次服务和客户端的实际命令、环境、GPU 归属及退出记录均保存在对应 `cases/.../launch-N` 目录。
新的完整矩阵可按工具 README 的 `prepare --components --snapshot-target --workload-profile conversation-tools` 和 `run-cases` 创建，复测必须沿用同一运行账本。

从当前原始记录重新生成分析，输出到新目录：

```bash
TOKENCAKE_RESULT_ROOT=/root/autodl-tmp/tokencake-optimization/components-final-01
TOKENCAKE_REPORT_DIR=$(mktemp -d /root/autodl-tmp/tokencake-report.XXXXXX)
.venv/bin/python -m tools.tokencake_experiments.component_analysis \
  "$TOKENCAKE_RESULT_ROOT" --graph "$TOKENCAKE_RESULT_ROOT/graph.json" \
  --output "$TOKENCAKE_REPORT_DIR/analysis.json"
.venv/bin/python -m tools.tokencake_experiments.component_matrix \
  "$TOKENCAKE_REPORT_DIR/analysis.json" "$TOKENCAKE_REPORT_DIR/tables"
.venv/bin/python -m tools.tokencake_experiments.component_plots \
  "$TOKENCAKE_REPORT_DIR/analysis.json" "$TOKENCAKE_REPORT_DIR/figures"
```

报告分析测试：

```bash
env CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 .venv/bin/python -m pytest \
  tests/tokencake/test_component_analysis.py -q \
  --junitxml=.venv/component-final-report.xml
```

结论限定于固定模型、显存比例、当前对话工具负载及有限到达序列。
不把完成吞吐解释为稳态可持续 QPS，不把 24 个相互竞争的 DAG 当作 24 次独立系统实验，也不从性能测试推断任务质量等价。

# Agent Benchmark 四组试跑对照

更新日期：2026-09-08。BFCL 四组功能检查和 SWE 四组 20 题试跑均已完成执行与独立评分。

用户已要求试跑后停止并取消正式全量，最终范围与结论见 [试跑最终分析](agent-benchmark-pilot-final-analysis.md)。
首次阅读建议先看该报告第 2–5 部分的口径解释与分段分析；本文保留四组完整指标和诊断明细。

SWE 中 `agent_offload` 为 1/20，其余三组均为 0/20；BFCL 四组均为 0/4。
按用户指定口径，`agent_offload` 相对 `base` 的前 30 分钟、各自全程、客户端完整响应输出吞吐分别为 +27.6%、+8.6%、+6.8%。
`agent` 只在前 30 分钟输出吞吐更高，完整运行略低；`offload` 出现长时间排队停滞，三个口径均更低。
这是单次试跑的输出吞吐结果，不能由此认定稳定的正确解题效率提升。

共同配置和环境依据见 [实施状态](agent-benchmark-platform-status.md)，优先两组的详细结果见 [首对报告](agent-benchmark-first-pair-pilot.md)。
执行分成两对：GPU 0 `base` 与 GPU 1 `agent_offload`，随后 GPU 0 `agent` 与 GPU 1 `offload`。
每组独占一张 A800，两组共享 36 核 CPU 配额和 240 GiB cgroup 内存。
各对在不同时间运行，不能将四组称为同时测量；共同前 30 分钟均相对于所在并行批次的计时账本起点。

## BFCL 功能检查

每组运行相同四题，Base 与 Long Context 各两题，均完成执行和官方评分。
没有模型请求重试；工具事件按对应模式启用。

| 指标 | base | agent | offload | agent_offload |
| --- | ---: | ---: | ---: | ---: |
| 官方通过数 | 0 / 4 | 0 / 4 | 0 / 4 | 0 / 4 |
| 已记录终态 | 4 | 4 | 4 | 4 |
| 实际模型响应数 | 29 | 28 | 30 | 28 |
| 客户端输出 token | 1,272 | 1,401 | 5,340 | 5,392 |
| 应用执行区间，秒 | 24.95 | 23.71 | 107.56 | 108.87 |
| 服务端窗口输出 token | 1,235 | 1,391 | 5,340 | 5,333 |
| 服务端全程 token/s/GPU | 49.50 | 58.66 | 49.65 | 48.99 |
| 客户端完整响应 token/s/GPU | 50.98 | 59.08 | 49.65 | 49.53 |
| 末次采样距本组结束，秒 | 1.399 | 0.806 | 0.145 | 1.799 |

服务端全程分子取本组结束前最后一个采样，分母为本组 `wall_s`；客户端使用完整响应的 usage。
四组均未覆盖完整 30 分钟，该窗口速率明确为不可用。
四题最多只有四个应用并行，不能代表 16 并发压测；首对功能检查还与环境准备重叠，因此上述速率仅作记录。
轨迹和输出长度不同，全部题目未通过，不能将输出量或某组较短的终止时间当作解题能力提升。

原始对照见 [四组 BFCL 数据](/root/autodl-tmp/tokencake-agent-bench/results/pilot-bfcl-four-mode-01.json)。

## SWE 20 题

每组使用相同 20 题、相同初始提示、16 并发、固定生成配置及原定工具与任务预算。
80 个独立工作环境与 20 个独立评分环境均已验证或核对身份；SWE 测量期间没有环境安装和离线评分。
20/20 题的四组初始输入 token、提示哈希和生成参数一致，后续真实生成历史与闭环到达时刻不同。
评分统一使用冻结的官方脚本、解析器、补丁应用回退和 resolved 规则，环境类型为 `custom_local`，不宣称与官方 Docker 镜像完全等价。
首对保存预测已使用同一评分实现重新评分；历史结果和原始日志均保留。

后续补齐 Django 评分脚本的任务本地 locale 初始化，重建并验证受影响的四题评分环境。
四组共 16 份保存预测复核后，评分状态与 resolved 均未改变，完整 20 题成绩保持下表结果；见 [Django 环境与评分复核](agent-benchmark-django-environment-validation.md)。

### 相同预算内的正确结果

| 指标 | base | agent | offload | agent_offload |
| --- | ---: | ---: | ---: | ---: |
| 计划 / 已记录终态 / 已评分 | 20 / 20 / 20 | 20 / 20 / 20 | 20 / 20 / 20 | 20 / 20 / 20 |
| 15 分钟内正确完成 | 0 | 0 | 0 | 1 |
| 30 分钟内正确完成 | 0 | 0 | 0 | 1 |
| 60 分钟内正确完成 | 0 | 0 | 0 | 1 |
| 整批最终通过数 | 0 / 20 | 0 / 20 | 0 / 20 | 1 / 20 |

唯一通过题是 `agent_offload` 的 `sympy__sympy-16766`，于该组开始后 306.45 秒提交，离线评分确认通过。
失败和空补丁不增加曲线，失败消耗的时间仍包含在相同预算内。
`base` 提前完成固定批次后正确数保持为零，不假定额外题目或持续到达吞吐。
任意两组都没有共同成功题，配对成功 E2E 不可用；与 `agent_offload` 比较时，单方成功为该题，其他 19 题双方均失败。
不能将 1 对 0 写成有限倍数的解题加速，也不能用 20 题单次结果证明质量等价。

![四组输出和正确完成曲线](/root/autodl-tmp/tokencake-agent-bench/results/pilot-swe-four-mode-curves-01.png)

图中输出曲线只画各组自身测量区间；正确曲线在固定批次结束后保持不变。
上图的累计输出包含最终失败任务，不能代替下图的正确结果。
可导出的矢量版本见 [SVG](/root/autodl-tmp/tokencake-agent-bench/results/pilot-swe-four-mode-curves-01.svg)。

### 用户指定的三个输出吞吐口径

服务端使用精确指标名 `vllm:generation_tokens_total`，每次采样仅合计当前模型各 engine，减去 `metrics-start.prom` 初始值。
四组初始计数均为 0。窗口结束取 `metrics.jsonl` 中 `timestamp <= 截止时间` 的最后有效采样，不使用关闭前的 `metrics.prom`。
共同前 30 分钟分母固定为 1800 秒；各自全程以本组 `ended_at` 截止，分母严格使用对应 `wall_s`。
客户端分子取报告中的 `model_work.reported_output_tokens`，仅包含完整响应的 usage。

| 分子与分母 | base | agent | offload | agent_offload |
| --- | ---: | ---: | ---: | ---: |
| 前 30 分钟服务端输出 token | 83,292 | 98,343 | 75,457 | 106,239 |
| 全程截止采样输出 token | 95,659 | 152,722 | 122,416 | 170,156 |
| 客户端完整响应输出 token | 95,692 | 152,749 | 122,081 | 167,360 |
| 各自 wall_s，秒 | 2198.273171 | 3646.511274 | 3896.437140 | 3600.932717 |

| 吞吐，token/s/GPU | base | agent | offload | agent_offload |
| --- | ---: | ---: | ---: | ---: |
| 服务端：前 30 分钟 | 46.27 | 54.64 | 41.92 | 59.02 |
| 相对 base | — | +18.1% | -9.4% | +27.6% |
| 服务端：各自全程 | 43.52 | 41.88 | 31.42 | 47.25 |
| 相对 base | — | -3.8% | -27.8% | +8.6% |
| 客户端：完整响应 | 43.53 | 41.89 | 31.33 | 46.48 |
| 相对 base | — | -3.8% | -28.0% | +6.8% |

计算使用 JSON 原始精度，表中仅显示有限小数；相对变化为 `(variant / base - 1) * 100%`。
每组一张 GPU，并发请求的 token 合计后除以一次墙钟，不除以并发数，不平均逐请求速率，不用请求耗时总和作分母。
墙钟包含排队、prefill、decode、真实工具、等待、失败和超时；不包括模型启动与本组全部结束后的空闲等待。
服务端可能包含超时或中断请求已生成但未完整交付的 token；末次采样滞后也可能使服务端窗口计数略小于客户端累计数。
完整响应同样可能属于失败任务，因此三个口径均是输出吞吐，不能直接解释为正确解题效率。

两对的共同起点分别为 `1788832262.4999597` 和 `1788840641.3909407`，对应截止为 `1788834062.4999597` 和 `1788842441.3909407`。
每题的 3600 秒时限从该题到达起算，闭环后续题可能较晚释放；因此一批 20 题的总时间可以超过 60 分钟。
`agent` 全部结束后等待 `offload` 的约 250 秒不计入自身分母。

| 采样核验 | base | agent | offload | agent_offload |
| --- | ---: | ---: | ---: | ---: |
| 前 30 分钟末次采样滞后，秒 | 1.730 | 0.689 | 0.587 | 0.177 |
| 全程末次采样滞后，秒 | 1.703 | 1.329 | 1.324 | 1.957 |
| JSONL 有效计数采样总数 | 1783 | 1920 | 1884 | 1744 |
| 运行期间最大采样间隔，秒 | 2.129 | 2.111 | 2.132 | 2.150 |

未观察到计数器重置、指标序列缺失、重复序列、损坏采样或时间戳逆序，也没有超过 4 秒的采样间隔。
检查按 engine 分别进行，避免一个 engine 的重置被其他 engine 增长掩盖。
采样总数可包含模式结束后的观测，但这些采样不进入本组全程分子；两种窗口的末次采样均滞后不足 2 秒，因此服务端结果为近似值。
先前保存的 [30 分钟检查点](/root/autodl-tmp/tokencake-agent-bench/results/pilot-swe-parallel-second-pair-01/checkpoint-30min.json)与最终分析一致。
首对还完成了 [独立原始采样核算](/root/autodl-tmp/tokencake-agent-bench/results/pilot-first-pair-throughput-readonly-audit-20260908-01.json)，全部六个 token 校验值匹配，原始输入文件哈希保持一致。

### 失败与实际工作量

| 指标 | base | agent | offload | agent_offload |
| --- | ---: | ---: | ---: | ---: |
| Submitted / 非空补丁 | 14 / 6 | 6 / 5 | 3 / 1 | 8 / 7 |
| 空补丁 | 8 | 1 | 2 | 1 |
| ContextLimitError | 5 | 1 | 2 | 1 |
| RepeatedFormatError | 1 | 2 | 0 | 0 |
| ReadTimeout | 0 | 11 | 15 | 11 |
| 其中独立 600 秒请求超时 | 0 | 5 | 11 | 2 |
| 其中剩余任务时限耗尽 | 0 | 6 | 4 | 9 |
| 模型响应 / 尝试 | 587 / 587 | 831 / 842 | 832 / 847 | 834 / 845 |
| 完整响应输入 token | 7,972,616 | 11,727,780 | 10,215,156 | 10,812,859 |
| 请求重试 | 0 | 0 | 0 | 0 |

四组均无缺失终态、缺失 journal、损坏 journal 或完整响应 usage 缺失。
第二对的 `agent` 有四个非空补丁进入评分但未解决任务，其中 Sphinx 7462 的补丁引入缩进错误，造成测试收集失败；不能把官方判为失败的 P2P 数量解释成逐项运行后的失败数。
`agent` 的 Django 11820 与 `offload` 的 Django 11211 经完整官方补丁应用回退后仍失败。
第二对没有 scorer 异常；模型生成失败、无效补丁及测试失败均保留在 20 题分母内。
首对模型补丁引入的语法错误与应用失败详见首对报告。

### 机制观测与排队停滞

以下均为本组截止前采样中的聚合诊断，未按相同请求长度或相同任务轨迹配对。

| 指标 | base | agent | offload | agent_offload |
| --- | ---: | ---: | ---: | ---: |
| 实际缓存复用 token / 输入 token | 9.97% | 8.95% | 78.88% | 10.39% |
| 其中 CPU 回载 token / 输入 token | 0% | 0% | 65.30% | 1.14% |
| 平均 TTFT，秒 | 21.20 | 40.13 | 35.23 | 37.11 |
| 平均排队，秒 | 18.06 | 37.28 | 30.14 | 34.65 |
| 相邻生成 token 平均间隔，ms | 46.80 | 35.09 | 31.80 | 34.25 |
| 按请求平均 TPOT，ms | 51.73 | 36.29 | 32.45 | 36.55 |
| GPU KV 使用率采样峰值 | 100% | 85.77% | 99.97% | 91.83% |
| 原生抢占计数 | 11 | 0 | 13 | 0 |
| GPU → CPU 累计传输，GiB | 不可用 | 不可用 | 50.79 | 0.94 |
| CPU → GPU 累计传输，GiB | 不可用 | 不可用 | 1251.25 | 22.61 |
| 保存 / 回载累计时间，秒 | 不可用 | 不可用 | 2.232 / 53.754 | 0.042 / 0.991 |
| 排队但无运行、无输出增长的采样区间总长，秒 | 未观察到 | 未观察到 | 2277.00 | 未观察到 |

`offload` 的缓存复用率更高、相邻生成 token 间隔更短，但整体输出吞吐最低。
它有四段主要停滞：

| 相对本组开始的采样区间，秒 | 持续，秒 | 排队请求数范围 |
| --- | ---: | ---: |
| 620.873–1186.326 | 565.453 | 15–16 |
| 1310.288–1878.022 | 567.734 | 13 |
| 2263.743–2850.079 | 586.336 | 6–8 |
| 3058.732–3601.711 | 542.979 | 1–5 |

另有 2.070 秒和 12.430 秒的短区间，全部六段合计 2277.002 秒，约占 `offload` 全程 58.4%。
判定条件为连续采样同时满足运行数为 0、排队数大于 0、累计生成 token 不变；间隔超过 4 秒则中断区间。
只合计连续样本之间可确认的时间，不推断两端未观测时间；其他组未观察到该条件不等于没有普通排队。
多次恢复发生在客户端超时取消之后，期间没有手动重启或修改参数。现有聚合日志不能进一步确定调度器阻塞点。
传输累计时间不能直接从墙钟扣除，也不能单凭其数值解释全部停滞。

进一步只读核对发现，全部六段内已完成 H2D/D2H 的字节与耗时计数都不增长。
首段 565.453 秒内，GPU KV 占用保持 82.814%，`capacity` 排队为 11–12 个，`deferred` 为 4 个。
后两类标签分别来自普通等待队列与暂缓队列，不能据此识别某个请求的具体阻塞原因；已完成传输计数不增长也不能排除尚未完成的传输。
后续定位需要请求状态、持有 KV、分配需求及传输完成通知的对应记录；现有数据不足以在这些原因之间作出结论。
核对过程和全部六段范围保存于 [停滞补充诊断](/root/autodl-tmp/tokencake-agent-bench/results/offload-waiting-diagnosis-20260908-01.json)，上述时间仍完整保留在输出吞吐量分母中。

将这些区间与保存的资源采样对齐后，前三段长停滞内分别有 261、262、272 个资源样本，offload GPU 利用率均为 0%。
这三段共享 cgroup 平均 CPU 使用量约为 2.12、2.05、2.03 核，CPU 限流计数与内存 full pressure 累计时间均未增加。
第四段长停滞内有 251 个资源样本，其中末尾约第 3600.834 秒有一个 GPU 利用率 100% 的样本，其余为 0%；输出计数不增长不能排除边界附近的 prefill。
12.430 秒短区间可核对到 6 个资源样本；2.070 秒短区间仅有 1 个，无法计算资源计数器区间增量。
CPU 与内存仍是两组共享数据，采样边界也不完全重合；这些观察削弱 CPU 配额耗尽或 cgroup 内存停顿的解释，尚不能确定根因。
完整边界、计数器检查、输入哈希和重算脚本见 [资源对齐诊断](/root/autodl-tmp/tokencake-agent-bench/results/offload-resource-alignment-20260908-01.json)。

新增无模型调度夹具复现了一种队头容量阻塞：两个请求完成 KV 回载后持有 GPU 块，队头还需的空间不足，循环退出使后续本可运行的请求也无法得到调度；取消队头后恢复。
相同夹具启用 agent 调度及完整生成容量预留后，两个请求均可完成；同步与异步调度均验证了这一差异，也保留了两组都能完成的等长控制。
这说明该机制可以发生，并不能代替对原试跑逐请求状态的确认；详见 [调度器最小诊断](agent-benchmark-offload-stall-diagnosis.md)。

四组均已生成输入、输出长度联合分组的 HTTP E2E 和吞吐贡献；超时请求输出长度保持未知。
当前非流式日志没有逐请求首 token 与 decode 时间戳，按长度分组的 TTFT/TPOT 仍为不可用，不能由聚合 Prometheus 计时反推。
另行检查了冻结两套源码的原生追踪，并用无模型夹具验证正常完成请求的 HTTP/protobuf span 导出；取消请求在该路径没有 span。
夹具时间由程序构造，不能补算本轮逐请求延迟；采集建议及缺失处理保留在 [已取消的下一阶段提案](agent-benchmark-next-stage-proposal.md)。
`agent` 与 `agent_offload` 的调度重算计数为 0；`offload` 关闭相应调度模块，其零值不代表真实重算为零，原生基线也没有可直接对应的计数器。

两对共享 cgroup 内存的采样峰值均约 239.99 GiB，限制为 240 GiB，没有 OOM 或 OOM kill。
它包含共享内存及文件缓存，不能视为某一模式私有占用，也不能将同一对的采样相加。
四组资源监测均未发现外部 GPU 计算进程，第二对服务日志未发现 CUDA 错误或引擎崩溃。

### 证据与最终范围

- [四组完整 SWE 数据](/root/autodl-tmp/tokencake-agent-bench/results/pilot-swe-four-mode-01.json)与[汇总分析源码](/root/autodl-tmp/tokencake-agent-bench/results/pilot-swe-four-mode-analysis-01.py)。
- [首对最新报告](/root/autodl-tmp/tokencake-agent-bench/results/pilot-swe-parallel-first-pair-01/report-throughput-02.json)与[第二对报告](/root/autodl-tmp/tokencake-agent-bench/results/pilot-swe-parallel-second-pair-01/report.json)。
- [首对统一重新评分](/root/autodl-tmp/tokencake-agent-bench/grading/pilot-swe-parallel-first-pair-02/scores.json)与[第二对评分](/root/autodl-tmp/tokencake-agent-bench/grading/pilot-swe-parallel-second-pair-01/scores.json)。
- [四组初始输入核对](/root/autodl-tmp/tokencake-agent-bench/results/pilot-swe-four-mode-initial-input-audit-01.json)。
- [第二对运行配置](/root/autodl-tmp/tokencake-agent-bench/results/pilot-swe-parallel-second-pair-01/experiment.json)、[启动时源码副本](/root/autodl-tmp/tokencake-agent-bench/results/pilot-swe-parallel-second-pair-01/adapter-source-identity.json)及[分析代码身份](/root/autodl-tmp/tokencake-agent-bench/results/pilot-swe-parallel-second-pair-01/report-implementation.json)。
- [剩余两组 40 个环境身份核对](/root/autodl-tmp/tokencake-agent-bench/grading/remaining-prepared-environment-audit-01.json)与[全部 20 题参考修复验证](/root/autodl-tmp/tokencake-agent-bench/grading/confirmed-environment-validation-summary-02.json)。

共享测量预算累计使用 7821.614 秒，约 2.17 小时，包含保留的 BFCL 首次失败运行；环境准备、模型加载与离线评分不计入这一预算。
第二对已完整结束，两个服务均已停止，未启动新的模型任务。

本轮完成的是四组试跑：BFCL 每组 4 题，SWE 每组 20 题。
用户已取消正式 BFCL 400 题、SWE 500 题、负载点与三次重复；后续模型复跑、观测接入和环境准备也已停止。
本轮未修改被测模型、任务集合、上下文或超时配置，原始数据及未完成的环境准备记录完整保留。

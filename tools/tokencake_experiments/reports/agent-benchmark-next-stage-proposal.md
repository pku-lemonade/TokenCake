# Agent Benchmark 下一阶段方案

日期：2026-09-08。最终状态：**不执行，用户决定停在试跑**。

用户最新要求为“跑完试跑就停止，开始进行实验结果分析”和“不要跑正式全量实验了”。
下文保留此前建议及已完成夹具的技术记录；接入逐请求采集、复跑 20 题和正式全量均取消，不再等待确认。
已有模型试跑的最终交付见 [试跑最终分析](agent-benchmark-pilot-final-analysis.md)。

四组试跑已完成，SWE 为 `base` 0/20、`agent` 0/20、`offload` 0/20、`agent_offload` 1/20；BFCL 功能样例四组均为 0/4。
原建议为补齐正常请求的逐请求计时、定位 offload 停滞，再复跑原 20 题；该建议没有转为新运行。
本文件保留已有验证和观测缺口，不改动已确认清单或历史结果。

## 1. 当前证据能够说明什么

按用户要求的窗口重新分析，`agent_offload` 相对 `base` 的输出吞吐变化为：

| 口径 | base，token/s/GPU | agent_offload，token/s/GPU | 相对变化 |
| --- | ---: | ---: | ---: |
| 共同前 30 分钟，服务端 | 46.27 | 59.02 | +27.6% |
| 各自任务全程，服务端 | 43.52 | 47.25 | +8.6% |
| 各自任务全程，客户端完整响应 | 43.53 | 46.48 | +6.8% |

服务端分子取对应截止时间前最后一条有效生成计数，客户端分子只取完整响应的 usage；失败、取消和工具等待占用的墙钟均保留。
本轮采样与窗口末尾相差小于 2 秒，结果为近似窗口统计；不同生成轨迹及低解题通过数仍限制因果解释。
详见 [四组试跑报告](agent-benchmark-four-mode-pilot.md)，其中也保留 `agent`、`offload` 的全部结果。

`offload` 的六段有排队但无生成增长区间合计 2277.002 秒，约占本组全程 58.4%。
已完成传输的字节和耗时计数在六段内均不增长；前三段长区间内的 GPU 利用率采样为 0%，共享 CPU 平均约使用 2 核，未观察到 CPU 限流或内存 full pressure 增量。
第四段长区间末尾有一个 GPU 利用率 100% 的样本，短至 2.070 秒的区间资源采样不足。
因此目前不能把停滞归因为 CPU 配额耗尽，也不能排除未完成传输或完成通知未被处理。

只读源码检查确认零生成 token 的执行路径仍调用 connector 的 `get_finished`，调度器随后处理 KV 输出。
这排除了“源码完全没有在空批次轮询传输”的简单解释，但不能证明本轮每次轮询及通知均实际完成。
需要将等待状态、GPU KV 分配需求、已持有块、传输 job 和完成通知与同一请求关联，才能进一步定位。
证据见 [等待与传输诊断](/root/autodl-tmp/tokencake-agent-bench/results/offload-waiting-diagnosis-20260908-01.json)和[资源对齐诊断](/root/autodl-tmp/tokencake-agent-bench/results/offload-resource-alignment-20260908-01.json)。

后续无模型夹具已在同步及异步调度中复现队头容量不足阻挡后续就绪请求的机制，取消队头后恢复；启用 agent 容量预留后同一夹具可完成。
这给出优先检查方向，仍需真实运行的请求持有块、申请需求和完成通知状态来确认历史根因；完整过程及等长控制见 [调度器最小诊断](agent-benchmark-offload-stall-diagnosis.md)。

## 2. 逐请求计时的建议接入

建议使用两套服务已有的原生 `llm_request` 追踪，维持当前非流式客户端。
两套源码的 `do_tracing`、`abort_requests`、完成统计及追踪初始化函数已核对 AST 一致；服务解释器已有所需 OpenTelemetry 依赖。
这项核对只覆盖列出的函数，不表示两套调度实现相同。

| 项目 | 建议配置 |
| --- | --- |
| 服务开关 | 两组均设置 `--otlp-traces-endpoint`，指向各自本机接收器的 `/v1/traces` |
| 协议 | `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf` |
| 采样 | 两组均使用 `OTEL_TRACES_SAMPLER=always_on`，核对实际采样与导出覆盖率 |
| 采集范围 | 原生普通请求 span；`collect_detailed_traces` 保持当前未启用状态 |
| 存储 | 数据盘各次新运行的 `service/traces/`，保存原始 protobuf、接收时间、解析记录和导出错误 |
| 去重与关联 | 按 trace/span ID 去重；以服务返回的请求 ID 关联客户端 attempt、任务终态与官方评分 |
| 关闭流程 | 先结束服务请求并刷新导出，再关闭接收器；保存未导出、无法关联及取消数量 |

当前客户端将 `tc-…` 放在顶层 `request_id`，聊天接口会生成 `chatcmpl-tc-…`。
已完成请求应优先使用日志里实际保存的 `model_response.response.id` 关联 `gen_ai.request.id`，不能把生命周期 UUID 直接当作服务 span ID。
真实服务检查还需验证 HTTP 失败、重试和超时的关联；每次 attempt 分开记录，不能把一次工具调用中的多个 attempt 合并为一个请求。

| 字段或指标 | 来源及算法 | 边界 |
| --- | --- | --- |
| TTFT | `gen_ai.latency.time_to_first_token`，单位秒 | 服务前端接收至前端观察首 token；不等于客户端网络首 token 延迟 |
| 首次排队 | `gen_ai.latency.time_in_queue` | 首次入队至首次调度，不覆盖后续全部等待 |
| decode 时间 | `gen_ai.latency.time_in_model_decode` | 首个至最后一个生成 token 的核心时间戳差，包含其中的抢占等待 |
| TPOT | decode 秒数 / (`gen_ai.usage.completion_tokens` − 1) | 仅输出大于 1 token 且计时有效时计算；单 token 为不可用 |
| 长度分组 | `gen_ai.usage.prompt_tokens` 与 `gen_ai.usage.completion_tokens` | 使用每条 span 自己的 token 数，并对照客户端 usage 检查差异 |
| 客户端完整响应 E2E | 现有客户端 monotonic 起止时间 | 包含网络与服务等待，继续单独统计 |

来源为冻结的 [请求追踪实现](/root/autodl-tmp/code/TokenCake/vLLM-TokenCake-upstream/vllm/v1/engine/output_processor.py:712)和[完成统计实现](/root/autodl-tmp/code/TokenCake/vLLM-TokenCake-upstream/vllm/v1/metrics/stats.py:429)。
前端 arrival 使用墙钟，核心 queued/scheduled/first/last 使用单调时钟；只使用原生已经计算好的延迟或同一时钟内的差，不跨时钟相减。
span 的结束时间也不能替代原生记录的首 token 和 decode 字段。

### 已完成的无模型验证

通过两套实际 `init_tracer` 的 HTTP/protobuf exporter 向临时本机接收器导出，并直接调用实际 `do_tracing` 与 `abort_requests`。
两套均成功导出两条完成请求 span，字段与构造的时间和 token 数一致；一个输出 1 token 的请求按上述规则得到 TPOT 不可用。
每套另测了首 token 前取消和已产生部分 token 后取消，两者均未导出请求 span。
夹具没有加载模型，构造的 1.25 秒 TTFT 和 0.5 秒 TPOT **不是测量成绩**，也不能用于补算旧实验。

原始证据见 [base 夹具](/root/autodl-tmp/tokencake-agent-bench/results/native-tracing-fixture-20260908-01/base/result.json)、[TokenCake 夹具](/root/autodl-tmp/tokencake-agent-bench/results/native-tracing-fixture-20260908-01/tokencake/result.json)和[源码与依赖身份核验](/root/autodl-tmp/tokencake-agent-bench/results/post-pilot-validation-20260908-01.json)。
同目录保留原始 protobuf、解析输出和[夹具源码](/root/autodl-tmp/tokencake-agent-bench/results/native-tracing-fixture-20260908-01/probe.py)。

### 必须保留的覆盖率限制

正常结束的服务端请求不等同于客户端已完整收到响应，更不等同于任务解决。
原生取消路径的缺口意味着它不能提供所有 attempt 的逐请求 TTFT/TPOT。
按模式与输入长度同时报告已记录 attempt、完整响应、已关联 span、超时/取消、缺失或无效计时的数量；缺失值不得置零。
报告中的 TTFT/TPOT 必须注明其观测子集，成功请求与失败任务中的正常响应也要保留对应标签。
服务端生成计数和墙钟吞吐继续统计取消前的输出与全部失败耗时，不改为“只有有 span 的请求”的吞吐。

如果要观测取消前已经产生的首 token，需要增加两套服务对称的取消路径记录；目前没有可从原生 span 直接取得的数据。
若后续定位需要调度状态或传输 job 记录，应另行明确记录位置、字段、触发频率及开销，保存观察补丁身份，不能将修改后成绩与本轮合并。
真正接入前还需验证并发导出、重复/损坏 payload、关闭刷新和缺失报告；无模型夹具尚未验证实际服务开销。

## 3. 已取消的下一阶段建议

1. 完成上述本机接收和分析适配，在两套真实服务上验证正常请求、取消与 ID 关联，记录采集开销与丢失情况。
2. 根据现有停滞证据做有针对性的状态检查；如需新增调度观察补丁，先给出具体补丁与验证结果供核对。
3. 在可解释的采集条件下复跑原固定 20 道 SWE 题，四组各一次：先 GPU 0 `base` / GPU 1 `agent_offload`，再 GPU 0 `agent` / GPU 1 `offload`；各组 16 并发。
4. 使用原模型、任务顺序、提示、环境快照、上下文、步数和超时，按用户的三种输出吞吐口径、正确完成曲线及独立评分重新出报告。

原建议使用现有 12 小时试跑测量预算的余额，已使用 7821.614 秒，未用 35378.386 秒，约 9.83 小时；该余额不再用于新实验。
原设计按两组运行区间并集记账，预算耗尽时保存取消和未完成状态；一次性环境准备、模型加载和离线评分另行记录。
新旧运行应分别报告，不能通过挑选表现更好的一轮作结论。
相同初始任务与 temperature=0 仍不能保证生成轨迹相同，按长度分组也不能完全消除工作量差异。

全量 BFCL 400 题、SWE 500 题的正式 QPS、到达序列、三次重复、总预算和环境存储安排均未冻结，现已取消。
目前 SWE 环境进度为 46 题通过规定的 F2P/P2P 验证、6 题参考验证失败、6 题下载失败、6 题准备取消、436 题未进入准备，详见 [实施状态](agent-benchmark-platform-status.md)。
部分 Django 评分脚本的 locale 初始化已补齐；试跑受影响四题重新验证和四组保存预测复核后，原有通过数保持一致，详见 [Django 复核](agent-benchmark-django-environment-validation.md)。
完整任务集合保持 500 题；环境验证失败不能靠换题或从分母删除来解决。

此前后续确认流程的来源是 [实现计划第 7 节](agent-benchmark-implementation-plan.md#7-运行阶段与任务清单)：
“试跑 20 题全部未解决，或出现频繁环境失败、格式错误、上下文溢出时，先分析原因并确认后续安排。”
用户最新停止要求已经取代该流程。已完成的日志分析和无模型夹具证据保留；环境准备停止，不再启动新增模型运行。

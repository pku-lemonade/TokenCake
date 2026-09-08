# TokenCake 的 Agent Benchmark 调研与选型

调研日期：2026-09-07。

本报告根据当前仓库实现、相关系统论文、benchmark 原论文和官方评测资料进行选型。
重点核对工作负载、实验方法、评分规则和版本变化；尚未安装或运行候选 benchmark。
文中的适配程度和收益潜力是结合代码作出的判断，不是新测得的性能结果。

## 1. 选型结论

建议优先考虑以下三个互补的候选：

| 候选 | 最适合回答的问题 | 选择理由 |
| --- | --- | --- |
| BFCL 多轮子集，先考虑 Base 和 Long Context | 多轮工具调用在并发下能否减少排队和重复 prefill，同时保持正确性？ | 接入较轻、可执行评测、已有 agent serving 论文采用 |
| SWE-bench Verified + mini-SWE-agent | 真实读文件、改代码、运行测试的 agent，能否获得端到端收益？ | 通用认知度高，有真实工具等待、追加式历史和测试判定 |
| HotpotQA + LATS 的并行执行负载 | 并行分支、共享前缀和汇合依赖能否受益于调度？ | 有公开方法和系统论文采用依据，适合覆盖 DAG 相关优化 |

只想先接入一个、尽快建立可重复的性能实验，选 **BFCL 多轮子集**。
只想先接入一个、优先强调真实 agent 工作和任务质量，选 **SWE-bench Verified + mini-SWE-agent**。
要覆盖目前框架的主要优化，建议最终采用以上三项组合；单个常用 benchmark 通常覆盖不全。

这里必须区分三个对象：benchmark 定义任务与评分；agent 决定怎样解决任务；负载驱动决定多个任务怎样到达和竞争资源。
例如，SWE-bench 本身不规定必须有多个 agent，HotpotQA 本身也不规定必须搜索、并行执行或调用工具。

## 2. 当前实现需要怎样的负载

依据 [TC_README.md](../../../TC_README.md)、[实验方法](component-methodology.md)、[接口说明](../../../docs/features/tokencake.md) 和 [协议定义](../../../vllm/tokencake/protocol.py)，需要分别检验以下机制。

| 当前机制 | 能暴露收益的条件 | 不充分的测试 |
| --- | --- | --- |
| 工具窗口内保存 CPU KV，后继命中后回载 | 真实追加的长前缀、工具间隔、其他请求造成的 GPU 缓存竞争 | 单会话无竞争，或只模拟长等待却没有可复用前缀 |
| 完整输入和生成增长的准入检查 | 多任务并发、输入和输出长短不一、实际 KV 压力 | 所有请求都很短，或统一设置远大于实际输出的预算 |
| agent 重要性与等待时间排序 | 不同角色、不同任务长度、持续到达 | 只看一次模型调用的时延 |
| `join_group` 优先级继承 | 同一应用中同时活跃、确实需要汇合的多个模型调用分支 | 单 agent 串行工具循环 |
| 运行中 GPU 共享前缀偏好 | 同时活跃的请求共享较长 token 前缀，且共享量足以影响准入 | 多个请求只有很短的相同系统提示 |
| 抢占牺牲者与恢复策略 | 原生基线出现容量压力，能够观察抢占、实际重算和公平性 | 把多次准入暂缓误算成失败或抢占 |

当前正式负载只有 24 个 DAG，工具按时长模拟，也没有任务质量评分。
因此新增测试最有价值的补充是：公开任务、可验证结果、真实工具执行、更多应用实例以及持续负载。

两个实现边界需要保留：目前 H2D 由后继请求的实际命中触发，没有预测式提前回载；保存 CPU 副本也不增加 GPU 物理容量。
论文中其他版本的功能和百分比不能直接作为这个仓库的能力或预期收益。

## 3. 相关论文是怎样测试的

### 3.1 系统论文

| 论文与所查部分 | 负载及实验方式 | 对本项目的启发 |
| --- | --- | --- |
| [TokenCake，v4，2026，§7](https://arxiv.org/html/2510.18586v4#S7) | Code-Writer、Deep Research 应用；ShareGPT/AgentCode 派生输入；改变应用到达率；调度和 offload 消融 | 保留组件拆分，但这类自建应用不等于独立的通用任务 benchmark |
| [Autellix，2025，§6](https://arxiv.org/html/2502.13965v1#S6)；正式版 [Agentix，NSDI 2026，§6](https://www.usenix.org/system/files/nsdi26-luo.pdf) | ShareGPT 完整会话、BFCL V3、LATS + HotpotQA、等比例混合；按 program 生成 Poisson 到达 | 最直接的选型依据：同时覆盖串行多轮和并行调用，改变的是应用到达率 |
| [InferCept，2024，§2、§5](https://arxiv.org/html/2402.01869v2) | 数学工具、HotpotQA、ALFWorld、聊天、图像生成、TTS；比较保留、丢弃重算和交换等策略 | 工具时间、上下文长度和中断次数必须一起分析；短工具也应纳入 |
| [Parrot，OSDI 2024，§8](https://arxiv.org/html/2405.19888#S8) | 长文档 chain/map-reduce、应用提示、MetaGPT 编程工作流、混合聊天；部分工作量按记录的输出长度构造 | 能借鉴依赖、并发和共享前缀的控制方法，但输出长度匹配不是任务质量验证 |
| [Teola，v3，2025，§7](https://arxiv.org/html/2407.00326#S7) | 搜索增强、基础和高级 RAG、上下文检索；使用 HotpotQA 等数据，比较端到端编排 | 必须分解 LLM、搜索、检索、编排等时间，才能解释推理优化为何没有等比例反映到 E2E |
| [Preble，2024，§4、附录 A](https://arxiv.org/html/2407.00023) | ToolBench、具身交互、编程、视频和长文档 QA；Poisson 与 Azure 到达轨迹，含工具流行度偏斜实验 | 共享前缀和数据局部性要测真实 token 命中；其多副本调度收益不能直接外推到当前单 GPU |
| [AgentRace，公开预印本，实验设置](https://agent-race.github.io/AgentRace.pdf) | 多种 agent 框架；ReAct、RAG、MoA；GAIA、HumanEval、ScienceWorld 等 | 适合参考框架开销和轨迹记录方法；更换编排器会同时改变模型调用量，不能归因给 vLLM 后端 |
| [AgentSysBench，2026-08，§3](https://arxiv.org/html/2608.15127#S3) | 十种应用，包含 SWE-bench、Terminal-Bench、WebArena 等任务来源；统一调用与资源观测 | 近期的系统测量参考；发布较新，本次未确认成熟的独立复用生态及可直接部署的官方代码入口 |

Agentix 延续了 Autellix 的实验路线，但名称已变化，引用正式论文时应使用 Agentix。
其 BFCL 和 LATS 负载的平均调用次数分别约为 10.75 和 159.7，这是该论文构造的轨迹统计，不是 benchmark 的固定任务规格。
它还使用应用总时延除以输出 token 数的归一化指标；本项目应同时保留未归一化的任务时延，避免不同生成量影响解释。

InferCept 的部分归一化延迟会扣除工具中断时间。
可以把类似指标作为诊断，但主结果应包含用户实际经历的工具时间；并行 DAG 的工具累计时间也不能直接从墙钟 E2E 相减。

### 3.2 Agent 方法和任务论文

| 论文 | 如何构造或评价 agent 行为 | 借鉴点 |
| --- | --- | --- |
| [ReAct，ICLR 2023，§3、§4](https://arxiv.org/html/2210.03629) | 在 HotpotQA 中交替推理与检索；在 ALFWorld、WebShop 中执行多步动作并评估结果 | 数据集必须与实际工具循环结合；只发一次问答请求无法代表该负载 |
| [LATS，ICML 2024，§5](https://arxiv.org/html/2310.04406#S5) | 在 HotpotQA、编程、WebShop 等任务中搜索多个轨迹，控制扩展数和搜索预算 | 适合研究分支与共享前缀；原 HotpotQA 设置含答案正确性反馈，必须披露 |
| [BFCL，ICML 2025](https://proceedings.mlr.press/v267/patil25a.html) | 包括函数选择、参数、拒绝调用和有状态多步任务 | 主测多轮子集，保留官方校验；单轮 AST 得分不够 |
| [SWE-bench，§2、附录 A](https://arxiv.org/html/2310.06770) | 修复真实仓库问题，用测试判断补丁是否解决问题及保持原功能 | 补上当前实验缺少的任务质量判据 |
| [Terminal-Bench，2026，§2、§3](https://arxiv.org/html/2601.11868) | 独立终端环境、任务测试和统一执行框架；模型与 agent 组合重复运行 | 固定环境、资源、超时和 agent，再比较服务端 |
| [τ-bench，2024](https://arxiv.org/html/2406.12045)；[τ²-bench，2025，§3、§4](https://arxiv.org/html/2506.07982) | 用户模拟器与工具交互；世界状态检查和重复成功率；τ² 加入双方均可操作环境的 telecom | 用户模拟器也是额外工作，应固定其模型和部署，避免与被测后端混淆 |
| [AppWorld，ACL 2024，§3](https://arxiv.org/html/2407.18901) | 多应用交互代码，通过数据库检查评估任务和场景完成度 | 使用可重置、可程序化评分的工具环境 |
| [WebArena，§3](https://arxiv.org/html/2307.13854#S3) | 在自托管网站执行多步操作，检查最终功能结果 | 浏览器本身提供真实工具阶段，但页面观察和历史保留策略影响前缀复用 |
| [GAIA，§3](https://arxiv.org/html/2311.12983#S3) | 搜索、文件和多模态工具完成问题，按最终明确答案评分 | 可作为 Deep Research 的公开任务来源，环境复现难度较高 |
| [AgentBench，ICLR 2024](https://arxiv.org/html/2308.03688) | 八类交互环境与相应评分，统一客户端和环境服务 | 适合跨领域覆盖，但总分不能代替分环境性能分析 |
| [StableToolBench，ACL Findings 2024，§3](https://arxiv.org/html/2403.07714#S3) | API 响应缓存与模拟器，加上可解任务过滤和模型评判 | 能减少外部 API 漂移；缓存响应及模拟耗时不等于真实 API 延迟 |

上述来源包括通用任务 benchmark、agent 算法和系统测量工作，三者的证据用途不同。
采用程度的判断依据是公开评测生态、正式论文及其他研究中的实际使用，不以仓库星数给出精确排名。

## 4. 各候选的具体比较

以下“潜力”表示有机会覆盖相应优化，最终依赖模型、轨迹、并发和硬件；不是预测加速倍数。
“并行 DAG”特指同一应用内的并行模型调用及其依赖，并非多个独立任务同时运行。

| 候选 | 采用依据 | 长前缀与多轮 | 工具等待 offload 潜力 | 并行 DAG 覆盖 | 接入成本判断 |
| --- | --- | --- | --- | --- | --- |
| BFCL 多轮 | 通用工具调用榜单；Agentix 采用 | 强 | 本地工具版有限 | 通常弱 | 低到中 |
| SWE-bench Verified + mini | 成熟代码任务榜单与公开 harness | 强，取决于上下文管理 | 较强，需实测工具时间 | mini 默认弱 | 中到高 |
| Terminal-Bench 2.0 | 公开终端 agent 榜单与跨模型实验 | 强，可能超出 32K | 较强，但工具可能主导 E2E | 取决于 agent | 高 |
| HotpotQA + 并行 LATS | ReAct/LATS 方法论文；Agentix 采用 | 分支共享强，需核查实际拼接 | 取决于检索实现 | 强，需真实并行执行 | 中到高 |
| τ²-bench 文本域 | 成熟客服 agent 评测体系 | 强 | 取决于用户模型和工具部署 | 双方交互不等于并行 DAG | 中 |
| AppWorld | ACL 资源论文与公开榜单 | 较强 | 本地 API 阶段可能很短 | 默认通常弱 | 中 |
| WebArena | 成熟浏览器 agent 任务与生态 | AgentLab 默认动态重建，长前缀需核查 | 不确定，可能被前缀失效限制 | 默认通常弱 | 高 |
| GAIA + 研究型 agent | 通用助手榜单；研究应用采用 | 较强 | 较强，环境波动也大 | 取决于所选 agent | 高 |
| StableToolBench | 工具学习论文、工具型负载研究采用 | 较强 | 取决于缓存和 API 模拟器 | DFS 不自动等于并行 | 中 |
| ALFWorld + ReAct | ReAct、AgentBench、InferCept 采用 | 多步历史可累积 | 本地文本环境通常有限 | 弱 | 低到中 |
| AgentBench 子环境 | ICLR 综合 agent benchmark | 取决于环境 | 取决于环境 | 默认通常弱 | 中到高 |

### 4.1 BFCL：最适合首先接入

任务是在文件系统、旅行、交易等有状态 API 环境中完成多轮操作。
建议从 `multi_turn_base`、`multi_turn_long_context` 开始，两类在 V3 官方说明中各有 200 个条目。
多轮评分要求每一轮同时通过状态检查与必要执行路径检查，不能仅检查返回 JSON 能否解析。[V3 任务与评分说明](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html)

它适合当前框架，因为多轮历史可产生连续的前缀复用机会，多个会话交错到达也会制造缓存竞争。
Base 和 Long Context 可以自然比较上下文长度的影响，不必人为填充 prompt。
局限是工具多为本地模拟的状态操作，原始执行窗口可能不足以触发有收益的 offload；这本身就是应该保留的对照结果。

官方 runner 支持已有兼容服务端点和 `--skip-server-setup`，因此不用让其另起一套 vLLM。
仍需适配模型请求和工具执行位置，添加 TokenCake 元数据及事件。[官方运行说明](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)

当前官方榜单已是 V4，页面明确给出了复现用版本；应冻结所选数据和代码提交。
V4 加入搜索、记忆等任务，但这些任务测的是 agent 能力，其中的“记忆”不等于 GPU/CPU KV 缓存。[当前榜单](https://gorilla.cs.berkeley.edu/leaderboard)、[V4 说明](https://gorilla.cs.berkeley.edu/blogs/15_bfcl_v4_web_search.html)

Missing Function 类可能改变工具 schema，从而改变提示前部，适合后续检验前缀失效时的行为。
BFCL 中的 parallel function calling 也不代表存在多个并行 LLM 分支。

### 4.2 SWE-bench Verified：最有说服力的真实编程测试

给定真实 GitHub issue 和仓库快照，agent 修改代码，再由测试判断是否解决问题。
Verified 有 500 个实例；Lite 有 300 个实例，是不同的筛选子集，不应假定 Lite 是 Verified 的子集。
质量指标是 `% Resolved`。[官方任务与榜单](https://www.swebench.com/)、[原论文](https://arxiv.org/html/2310.06770)

推荐搭配 mini-SWE-agent：它使用 shell 执行动作，历史按步骤追加，便于接入本地模型、记录工具时间和审计 prompt。
这是公开使用的 agent，而非为 TokenCake 临时设计的工作流。[mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent)

预期可以检验长会话在文件读取、代码执行和测试间隔内的 KV 保存，以及不同长度修复任务之间的准入与排队。
mini 的默认模型调用是串行的，无法单独验证 `join_group`；若采用多 agent 修复器，必须另行固定其实现并给所有后端使用。

主要成本是容器镜像、磁盘、测试 CPU 和较长执行时间。
在当前 Qwen2.5-14B-Instruct、32K 设置上，应先检查成功率和上下文溢出；不能因为快但经常失败就认为性能更好。
初始化和最终隐藏测试评分应单独计时；agent 在任务过程中主动执行的测试则属于真实工具等待。

建议预先确定一个用于接入验证的分层小样本，再运行完整选定子集。
小样本只能叫 pilot，不能宣称获得完整 SWE-bench Verified 分数。

### 4.3 Terminal-Bench 2.0：更广泛的终端工具负载

2.0 包含 89 个真实终端任务，每项带环境、参考解和验证测试，涵盖代码、系统操作等工作。
通过 Harbor 运行；论文对模型与 agent 组合至少重复五次。[论文 §2、§3](https://arxiv.org/html/2601.11868)、[官方任务库](https://github.com/harbor-framework/terminal-bench-2)

编译、测试和命令运行提供真实等待阶段，长轨迹也可能带来较大 KV 工作集，适合观察 offload 与准入策略。
但某些任务的大头是工具执行，推理明显变快仍可能只带来有限 E2E 改善；shell 轮询之间的实际等待也可能很短。

接入时选一个能连接本地模型的公开 agent，固定任务资源、超时和上下文压缩策略。
若先挑选兼容当前硬件的任务，需要公开筛选规则和任务 ID，标为子集；不能把删掉慢任务后的结果当作完整 89 题成绩。
它比 SWE-bench 更跨领域，但作为第一项测试的运行和排错成本更高。

### 4.4 HotpotQA + LATS：最适合检验并行依赖

HotpotQA 是多跳问答数据；LATS 是通过树搜索、行动和反思求解问题的方法，不是另一个独立任务集。
Agentix 已把 LATS + HotpotQA 用作多线程 agent serving 负载，因此这个组合有直接的系统论文依据。[LATS](https://arxiv.org/html/2310.04406#S5)、[Agentix §6](https://www.usenix.org/system/files/nsdi26-luo.pdf)

同一搜索节点展开多个候选时，可共享祖先上下文；评价和后续选择又依赖候选结果。
在确实并行执行的实现中，这能覆盖分支竞争、共享前缀、汇合等待和应用级时延。
进一步核对公开代码发现，原版 `get_values()` 顺序评价候选，候选生成使用单次请求的 `n` 参数；不能把原版直接当作当前接口可用的并行 DAG。
要验证该机制，需要固定搜索语义后适配并发执行，详见第 7 节。
若各分支的原始调度分数相同，优先级继承也可能不改变顺序。
因此应记录实际分数变化、汇合等待和缓存竞争，不能仅凭存在分支便宣称该机制产生收益。

重要限制：LATS 原论文的 HotpotQA 实验使用 100 题子集，并由环境反馈答案是否正确。
采用该设置时必须明确标为含 oracle feedback 的实验；若取消反馈，则是新的求解配置，质量不能直接对照原表。
固定扩展宽度、最大迭代数、反思提示、候选排序和停止条件，防止后端调度改变搜索预算。

当前 TokenCake 不接受一个带元数据的请求使用 `n > 1`。
若原客户端一次生成多个候选，须改成多个独立、`n=1` 的调用，并在所有基线采用同样的客户端。
只标注运行时已知的分支和汇合关系，不把事后完整搜索树当成事前已知 DAG。

检索语料应冻结；在线 Wikipedia 与本地快照的延迟分别报告。
它适合作为 DAG 机制测试，与 SWE-bench 或 BFCL 的真实闭环测试组合使用。

### 4.5 τ-bench / τ²-bench：客服型长对话与可靠性

任务围绕零售、航空等领域，在业务规则约束下与用户模拟器对话并操作工具。
τ² 增加 telecom，用户和 agent 都能改变环境状态。
评分依据任务配置检查数据库、状态等，`pass^k` 衡量重复 k 次都成功的可靠性，与“k 次至少成功一次”的 `pass@k` 不同。[τ-bench](https://arxiv.org/html/2406.12045)、[τ²-bench §3、§4](https://arxiv.org/html/2506.07982)

多轮追加历史有利于观察前缀复用，但用户模拟器的推理时间不是自然的人类等待时间。
建议把模拟器部署在固定的独立服务上，单独记录其耗时和调用量；若与被测模型共享服务，也应作为明确的混合负载实验。
本地数据库工具可能非常快，不能默认它比 BFCL 更能触发 offload。

本次核对发现官方仓库已经升级为 τ³-bench，增加知识检索、语音和任务修订。
复现 τ² 应固定旧版本；采用当前文本域则记录当前版本，不能和旧分数混称。
语音并不符合当前纯文本模型的首批测试范围。[官方版本说明](https://github.com/sierra-research/tau2-bench)

它适合希望证明框架对非编程业务 agent 同样有效的情况。
双参与者交替对话不等于具有并行分支的多 agent DAG。

### 4.6 AppWorld：本地可控、结果可验证的应用操作

AppWorld 提供 9 个应用、457 个 API 和 750 个任务，agent 通过交互式代码完成跨应用工作。
论文划分 Train 105、Dev 60、Test-N 168、Test-C 417，使用任务完成率 TGC 和场景完成率 SGC。[原论文 §3](https://arxiv.org/html/2407.18901)

这是控制环境漂移的有力候选：世界状态可重置，任务有程序化判据，能记录每次模型调用与代码执行。
适合希望兼顾工具真实性、稳定评分和本地复现的人；公开仓库也提供评测与榜单工具。[官方仓库](https://github.com/stonybrooknlp/appworld)

其 API 通常在本地执行，真实工具等待可能较短。
一个模型生成的代码块可以执行很多 API，API 数量不能当作 LLM 轮数；TokenCake 事件宜覆盖整个实际阻塞的代码执行阶段。
长前缀和多任务调度值得测，但默认 agent 一般不足以覆盖并行汇合。

### 4.7 WebArena：浏览器工作流

原版包含 812 个自托管网站任务，按最终功能结果评价网页操作。
官方建议使用 AgentLab/BrowserGym 生态开展实验，支持统一环境和并行评测。[论文](https://arxiv.org/html/2307.13854)、[官方实现说明](https://github.com/web-arena-x/webarena)

网页访问和交互可构成真实工具阶段，DOM 或可访问性树也可能产生较长输入。
进一步核对 AgentLab 的 `GenericAgent` 发现，它逐步重建提示，并把最新观察放在历史之前；页面内容变化可能让后方历史和动作说明无法继续前缀命中。
即使历史文本仍在，也未必获得当前追加式负载那样的前缀复用。
应先统计 token 层面的共同前缀和工具时长，再评估适配程度。

成本在于网站镜像、浏览器、账户和数据库重置。
并发任务需要隔离可写状态，避免两个 agent 修改同一个网站世界而改变任务结果。
当前文本模型优先选文本观察路线，图像版另作模型与系统配置实验。

### 4.8 GAIA：面向研究型 agent 的公开任务

GAIA 要求综合搜索、文件处理、推理和部分多模态工具，最终答案通常明确，按难度分级。
本地质量实验应使用带答案的验证部分；测试部分答案不公开，数据本身也有访问条件。[论文](https://arxiv.org/html/2311.12983)、[官方数据说明](https://huggingface.co/datasets/gaia-benchmark/GAIA)

若搭配包含并行研究分支的公开 agent，它可以成为当前 Deep Research 类应用的任务来源。
长检索上下文与外部工具可能覆盖 offload，实际并行分支也可能覆盖 DAG 调度；这些结构由 agent 决定，不是 GAIA 自带。

难点是在线网页变化、搜索服务配额、附件处理以及多模态模型需求。
筛选纯文本或本地附件任务时必须标注子集，不能报完整 GAIA 得分。
适合后续展示研究应用的泛化，作为第一项稳定性能基准的成本偏高。

### 4.9 StableToolBench：大规模工具型任务的补充

它在 ToolBench 基础上引入 API 响应缓存与模拟器，并使用可解任务上的通过率和相对胜率。
原论文从 1,100 个评测任务中筛出 765 个可解任务，并依赖模型评判。[论文 §3](https://arxiv.org/html/2403.07714)、[官方代码](https://github.com/THUNLP-MT/StableToolBench)

有助于研究大量工具文档、工具集合间共享前缀以及不同工具类别的负载。
不过，API 模拟器和评判模型引入额外依赖；缓存命中快、模拟器生成慢，两者都不能直接解释成真实 API 的工具窗口。
更适合作为工具多样性和控制轨迹的补充，不优先用作证明真实 offload 收益的唯一依据。

DFS 工具规划通常是顺序搜索，不能据此声称已验证并行 DAG。

### 4.10 ALFWorld 与 AgentBench：经典对照和跨域补充

ALFWorld 的文本环境让 agent 通过动作序列完成家庭场景目标。
ReAct 采用了 134 个未见评测实例，InferCept 也将其作为交互负载。[ALFWorld](https://arxiv.org/abs/2010.03768)、[ReAct §4](https://arxiv.org/html/2210.03629#S4)、[代码](https://github.com/alfworld/alfworld)

它的优点是任务和成功判定清晰，适合当前文本模型的多步交互验证。
本地文本动作通常很快，所以更适合用来测低开销和无收益场景的退化情况，而不是专门证明长工具窗口 offload。

AgentBench 覆盖 OS、数据库、知识图谱、游戏、购物和浏览等八类环境。
可以选 OS、DB、ALFWorld 等子环境增加跨度，并逐项报告任务质量和系统指标。[论文](https://arxiv.org/html/2308.03688)、[官方代码](https://github.com/THUDM/AgentBench)

上述八类对应原版；当前主分支已经切换到 AgentBench FC / AgentRL。复现原论文可固定 `v0.2`，不要混用两版任务及分数。

一次部署完整套件成本较高，其聚合能力分数也不能说明哪一类负载改善了缓存或调度。
它是候选的组织方式，不必与已经单独选定的重叠子环境重复计为独立证据。

## 5. 如何使用 benchmark 才能体现当前优化

### 5.1 主实验运行真实闭环

保留 benchmark 官方任务、环境和评分，固定 agent、模型、工具配置与上下文管理。
所有后端使用同样的提示模板、最大步数、停止条件和输出预算；用实际模型输出决定下一步动作。
分别报告成功、任务失败、推理错误、工具错误、上下文溢出和超时，失败任务不得从分母消失。

当前实验的固定输出 token 验收不能直接套到这些任务上。
真实 agent 应允许正常结束，也不能为了性能把所有工具输出统一压成 128 token。
固定的是预算和规则，实际轮数及 token 数可能不同，需同时记录并用于解释质量与时延差异。

### 5.2 控制轨迹仅用于辅助归因

从公开任务的运行中记录调用依赖、完整 token 输入、实际输出、工具结果和时间，再做控制实验。
回放时后继请求仍需等待前驱完成及对应工具间隔，不应把整条轨迹拆成无关的独立请求。

尤其注意：记录的后继 prompt 如果包含旧输出，而当前回放生成了不同输出，实际可复用前缀会改变。
只固定输入输出长度、或宣称“追加式”并不能保证缓存行为一致，应审计实际 token 前缀与 GPU/CPU 命中。
固定轨迹不能作为新一次任务成功率；原生工具时间、记录时间回放和人为放大的延迟须分开标注。

### 5.3 应用级负载与公平基线

| 实验维度 | 建议 |
| --- | --- |
| 到达方式 | 以完整任务或会话为到达单位，使用固定种子的 Poisson 到达；保留等间隔到达作为对照 |
| 压力范围 | 先测轻载，再增加到接近容量上限；不能照搬原 24-DAG 负载的 QPS 数值 |
| 并发验证 | 固定并发用于接入测试；持续到达用于观察排队、尾延迟和可持续吞吐 |
| 原有组件矩阵 | 保留 `base`、`agent`、`offload`、`agent_offload` |
| 额外缓存基线 | 增加同 GPU 和同 CPU KV 容量的原生通用 CPU offload，单独验证工具事件策略的增益 |
| 原生优化 | 保持 prefix caching、chunked prefill、精度、模型和批处理预算公平 |
| 模型与容量 | 先用当前 Qwen2.5-14B、A800 配置做 pilot，再改变 KV 容量；必要时另列更合适的模型实验 |
| 重复与随机化 | 重要比较安排独立重复运行，预先固定重复规则，随机化配置顺序，保留全部合格运行 |
| 冷启动和环境 | 模型初始化、镜像构建、环境启动、任务运行和最终评分分别计时；明确主 E2E 的边界 |

原有 `offload` 配置依然使用重要性和保存资格元数据，并不等于通用、无 agent 信息的 CPU KV 基线。
第三方系统跨版本、跨执行引擎的比较可作补充，不能替代当前原生版本上的受控组件实验。
CPU offload 与测试容器还会竞争主机内存和带宽，必须固定资源；不能只检查显存。

### 5.4 最少需要保留的指标

| 层次 | 指标 | 目的 |
| --- | --- | --- |
| 任务质量 | 官方成功率、对应可靠性指标、失败和超时率 | 确保更快不是因为早退、少执行或做错 |
| 应用体验 | 从任务提交到完成的平均、P50、P95 时延 | 直接反映 agent 完成任务的时间 |
| 服务容量 | 完成任务/s；成功且满足预定 SLO 的任务/s | 衡量有用的吞吐，不能由 token/s 替代 |
| 模型调用 | 首轮及工具后续轮 TTFT、排队、生成时间、TPOT、调用数 | 解释收益发生在哪个阶段 |
| KV 与调度 | GPU/CPU 命中 token、prefill 计算、抢占、重算、准入等待 | 对照实际优化机制 |
| 工具与编排 | 工具实测时长、用户模拟器时间、事件确认开销、编排间隙 | 排除 CPU、网络和客户端瓶颈 |
| 分支与公平性 | 实际关键路径、join 等待、不同任务类别 P95、最长等待 | 验证 DAG 与优先级策略是否帮助整体完成 |
| 资源与传输 | KV 容量、CPU RSS、D2H/H2D 字节和时间、GPU 活跃度 | 说明收益的资源代价 |

SLO 应在看性能结果前固定，并按任务类别制定。可将 goodput 明确定义为：

```text
goodput = 观测窗口内成功且满足预定任务时限的完成数 / 观测窗口长度
```

报告进入观测窗口的任务数、窗口边界和最后排空情况，避免只统计先完成的短任务。
任务时延与完成吞吐应覆盖所有尝试；成功任务时延只能作为附加统计。
当前原生基线缺少的实际重算区间统计仍应记为 N/A，不能用导出的初始零值代替观测。

### 5.5 元数据适配要遵守当前接口

1. 每次模型调用及重试生成新的 `tc-` UUID4 标识，同时作为 `request_id` 与 `lifecycle_id`。
2. 为同一任务记录一致且大于零的 `application_started_at_s`；当前汇合分组使用它与 `join_group` 的组合，不同任务不得意外碰撞。角色、重要性和 DAG 字段由固定的 agent 策略提供。
3. 只有输入会实际复用时才声明 `reusable_prefix`，并按保存资格设置 `offload_eligible`。
4. 工具执行前后发送并等待 `stall_started`、`stall_finished` 确认，把事件请求开销计入真实 E2E。
5. 时间估计使用事前配置或过去观测，不使用尚未发生的实际工具耗时；未知时可省略估计。
6. `join_group` 只描述真实的汇合；未知未来路径、剩余深度和接近完成状态不能从标准答案或未来轨迹取得。
7. 每个带元数据请求只对应一次生成；并行候选使用独立 `n=1` 请求，所有配置使用相同的客户端拆分。

接口约束见 [功能文档](../../../docs/features/tokencake.md) 和 [协议代码](../../../vllm/tokencake/protocol.py)。

### 5.6 保留可能无收益或退化的对照

应包含轻载、短工具等待、短前缀、前缀失效和较宽松 KV 容量等条件。
同时记录 `max_tokens` 相对实际输出的富余程度，检验默认 `generation_reserve_mode=all` 是否因过度预留而限制并发。
任何输出预算变化都要同时检查质量，不能只选有利预算。

对于长工具阶段主导的串行任务，推理时间缩短并不意味着总时延同比缩短。
在并行应用中应从实际关键路径解释这种上限；工具累计时间、传输累计时间和所有分支 LLM 时间不能直接相加为 E2E。
较小的真实端到端收益仍有价值，它说明了框架适用范围。

## 6. 可选的测试组合

| 优先目标 | 建议组合 | 能支持的结论 |
| --- | --- | --- |
| 最快建立公开基准 | BFCL Base + Long Context | 多轮工具正确性、前缀复用和并发准入 |
| 真实代码 agent | SWE-bench Verified + mini-SWE-agent | 编程任务质量、工具等待中的缓存收益、应用级时延 |
| 当前优化覆盖较完整 | BFCL + SWE-bench + 并行 LATS/HotpotQA | 串行多轮、真实工具、并行 DAG 三种证据互补 |
| 跨业务泛化 | 上述组合再加 τ² 文本域或 AppWorld | 非编程领域的工具 agent 也可受益 |
| 研究与浏览器应用 | GAIA + 明确的研究型 agent，或 WebArena + AgentLab | 搜索、浏览与较长外部等待场景 |
| 更重的终端任务 | Terminal-Bench 2.0 | 编程之外的终端工作与更长任务轨迹 |

我的选型建议是先在 BFCL 和 SWE-bench 之间决定第一项，再用 LATS 补齐并行 DAG 覆盖。
AgentSysBench、AgentRace 可以参考其观测方法，但现有证据不足以把它们优先当作“已广泛采用”的主 benchmark。
最终选择前最关键的未知量，是当前模型在候选任务上的成功率、实际 token 前缀复用率和真实工具时间分布；这三项需要 pilot 实测。

## 7. 配套 Agent 与当前优化的逐项适配

本节进一步核对了公开 agent 的模型适配层、主循环和提示构造。
以下都有公开 agent 或官方交互循环可用，但“已有代码”“可连接本地模型”“已支持 TokenCake”是三个不同状态。
本次未发现这些入口已经包含本仓库的 TokenCake 元数据与事件适配，不能仅更换模型地址就视为完成接入。

### 7.1 所有候选共有的条件

串行 agent 也可以受益：多个独立任务共用一套 vLLM，一个任务等待工具时，其他任务会使用 GPU 并可能覆盖它的前缀。
CPU KV 保存的价值是后续少重算；如果原生 GPU 缓存始终命中，新增保存和回载没有相同的收益空间。
当前策略还要求实际存在合格的等待需求，单独逐题运行可能不会选择保存。

从单个后继调用的关键路径看，应检验：

```text
避免的原生 prefill 重算时间
    > 回载时间 + 未被等待窗口遮住的保存时间 + 事件与调度开销
```

这个关系用于解释观测，不是现有策略的精确评分公式；还应检查对其他请求的带宽和排队影响。
较长工具时间能提供保存窗口，但工具占总时间过高，也会限制 E2E 的相对改善。

准入、抢占和排队策略需要并发竞争才能检验；单一角色负载可以测这些机制，却不足以验证跨角色预留与借用。
`join_group` 需要同一应用内同时活跃的多个模型请求，并且原始分数存在差异时继承才可能改变调度。
父 agent 等待一个子 agent，或者一个模型返回多个工具调用，都不能自动满足这个条件。

GPU 运行中前缀偏好又是另一种机制：串行后继复用已完成请求的缓存，与多个运行中请求共享同一前缀并不相同。
只有共享 block 足以覆盖新增需求且分数接近时，当前偏好才可能改动顺序。

### 7.2 可用实现与模型连接方式

| Benchmark | 已核对的公开 Agent / 运行器 | 本地模型连接方式 | 默认调用结构 |
| --- | --- | --- | --- |
| BFCL 多轮 | 官方 `BaseHandler` 多轮循环与 `OSSHandler` | 已有兼容端点；本地模型路径使用 Completions API | 单任务循环调用模型、执行工具、追加结果 |
| SWE-bench | mini-SWE-agent 的 `DefaultAgent` | LiteLLM 的 provider、`api_base` 等配置 | 追加消息，模型与 shell 执行交替 |
| Terminal-Bench | Harbor 的 `Terminus2` | `api_base` 与模型配置，支持额外调用参数 | 终端交互循环，含上下文压缩调用 |
| HotpotQA + LATS | LATS 官方 HotpotQA 实现 | 旧版 SDK 与 `OPENAI_API_BASE`，需兼容适配 | 顺序搜索和评价，单次 `n` 采样 |
| τ² 文本域 | 官方 `LLMAgent` 与用户模拟器 | LiteLLM，agent 和 user 可分别配置 | 用户、agent、工具交替，保留各自历史 |
| AppWorld | 官方 simplified ReAct code、function calling、full code agents | 官方模型配置支持 `base_url` 和自托管 vLLM | 多步代码或函数执行，具体轮数由 agent 决定 |
| WebArena | AgentLab `GenericAgent` + BrowserGym | `VLLMChatModel` / `VLLM_API_URL` | 每步重建页面观察提示，再执行浏览器动作 |
| GAIA | Hugging Face smolagents 的 Open Deep Research 示例 | LiteLLM / `OpenAIServerModel`，还需配置辅助模型 | manager、search 子 agent、文件和搜索工具 |
| StableToolBench | 仓库内 CoT / `single_chain` 与 DFSDT | 配置 API base、模型名，适配现有工具调用协议 | CoT 顺序链或 DFS 搜索；跨题线程不等于题内分支 |
| ALFWorld | ReAct 官方 `alfworld.ipynb` | 原版旧 Completion SDK，需替换模型调用入口 | 动作与观察追加进提示，顺序执行 |
| AgentBench 原版 | `v0.2` 官方 `HTTPAgent` 与任务客户端 | URL、请求体和角色映射配置 | 各环境内多轮交互，外部 assigner 分发任务 |

这里的模型连接方式只说明具备接入基础，不代表当前 Qwen2.5-14B 在对应任务上的质量、上下文长度和解析行为已经验证。

### 7.3 BFCL：循环齐全，主要收益看多会话调度

官方 `BaseHandler` 已实现多轮推理、工具执行和消息追加，具备完整 agent 循环；不必另写规划器。
`OSSHandler` 支持已有服务端点，并由模型 handler 构造完整提示后调用 Completions API。
可在模型请求处加入元数据，在 `execute_multi_turn_func_call()` 周围加入事件。[循环代码](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl_eval/model_handler/base_handler.py)、[本地模型适配](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl_eval/model_handler/local_inference/base_oss_handler.py)

多轮历史有利于前缀复用，压力下可测准入与排队；本地状态工具很快时，工具窗口保存可能被跳过。
单任务没有默认的并行模型分支，角色也较单一，因而 DAG 汇合和跨角色预留不是其强项。
另一个具体风险是当前通用 OSS handler 在上下文允许时可声明 4,096 个生成 token；短工具输出也可能因此触发当前 `all` 模式的保守预留。
应记录声明预算与实际输出，先保持官方配置，再单列有质量验证的预算敏感性实验。

判断：接入优先级高，多轮和负载调度适配较好，原始本地工具版的 offload 收益预期应保守。

### 7.4 SWE-bench：mini 的执行形态最贴近当前缓存机制

mini 的 `DefaultAgent` 直接把模型响应和工具观察加入 `messages`，在 `query()` 后执行 `env.execute()`。
它有公开的本地模型配置文档，可固定文本 shell 或工具调用版本。[主循环](https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/agents/default.py)、[本地模型接入](https://github.com/SWE-agent/mini-swe-agent/blob/main/docs/models/local_models.md)

追加历史、真实测试与命令等待，使它适合检验同一 agent 暂停后恢复的缓存收益。
多个修复任务交错时，较大的上下文工作集也能检验生成预留和抢占；只运行一个任务，原生 GPU 前缀可能始终保留。
mini 的默认循环不产生并行汇合，所有任务都是同一角色时也不能声称验证了跨角色调度。
模型能否产生有效修复、32K 是否够用、测试容器是否争抢 CPU 内存，都是 pilot 必须确认的条件。

判断：若优先寻找真实 agent 上的优化证据，这是当前最值得首先实跑的候选；适配程度较高不等于保证加速。

### 7.5 Terminal-Bench：现成终端 Agent，但压缩会影响命中

Harbor 提供参考 agent Terminus-2，使用 tmux 交互，可以配置自定义 API 地址。
其 `Chat` 在正常轮次追加消息；Terminus-2 默认允许总结上下文，总结完成后会改写消息历史。[参考 Agent 文档](https://www.harborframework.com/docs/agents/terminus-2)、[Agent 源码](https://github.com/harbor-framework/harbor/blob/main/src/harbor/agents/terminus_2/terminus_2.py)、[消息历史](https://github.com/harbor-framework/harbor/blob/main/src/harbor/llms/chat.py)

模型调用与长命令交替时可覆盖 offload，多个长任务也可施加准入压力。
但需要测实际 tmux 等待窗口，不能把后台命令的全部运行时间视为模型调用之间的连续空闲。
上下文总结前后应分开看命中率；总结中的辅助模型调用也要计入成本和服务负载。
当前实现中的这些调用不能自动当作多分支 DAG，长命令占比高也可能稀释 E2E 改善。

判断：适配潜力较高，接入和运行成本高于 mini，适合第二阶段扩展。

### 7.6 LATS：有正式 Agent，原版需要并发与接口改造

已核对原版 `get_values()` 中的顺序循环，以及模型适配层通过 `n=cnt` 生成多个候选的做法。
它不是可直接接入当前 TokenCake 的并行请求实现。[搜索与评价](https://github.com/lapisrocks/LanguageAgentTreeSearch/blob/main/hotpot/lats.py)、[模型调用](https://github.com/lapisrocks/LanguageAgentTreeSearch/blob/main/hotpot/models.py)

拆成独立 `n=1` 调用，并把独立评价分支并发执行后，才有条件同时观察活跃共享前缀和汇合等待。
所有后端必须使用相同改造；候选顺序、缓存、停止条件和搜索预算要保持一致。
原版还使用全局环境与反思状态，不能简单增加线程而忽略分支状态隔离。
生成、评价和反思的提示不同，能共享多长前缀需要按 token 统计，不能按“同一搜索树”推断。

判断：DAG 机制测试的潜力高，但原版现成度低于此前几项；保留原版串行测试，再单列并行适配版本最清楚。

### 7.7 τ²：官方对话 Agent 齐全，用户等待可以成为真实暂停

官方 `LLMAgent` 会追加收到的消息和自身输出，通过 LiteLLM 调用模型；用户模拟器是独立的交互参与者。
应使用常规 `LLMAgent`，源码中的 `LLMGTAgent` 会获得预期解题步骤，不适合作为正常能力测试。[Agent 实现](https://github.com/sierra-research/tau2-bench/blob/main/src/tau2/agent/llm_agent.py)、[模型适配与缓存](https://github.com/sierra-research/tau2-bench/blob/main/src/tau2/utils/llm_utils.py)

客服 agent 等用户模型回复时，它自己的长历史暂时不用，适合声明真实的等待阶段。
并发会话可制造缓存竞争；本地数据库操作本身仍可能很快。
应固定并单独记录用户模型，关闭会绕过被测推理服务的完整响应缓存；若两方共用服务，明确作为混合模型调用负载。
文本轮流对话没有默认的并行汇合；旧 τ² 与当前 τ³ 文本版也必须明确版本。

判断：非编程任务中很值得考虑，尤其适合多会话前缀与等待；offload 效果取决于用户模型等待和实际竞争。

### 7.8 AppWorld：已有多种 Agent，优先交互式 ReAct

官方已提供 simplified ReAct code、function calling 和 full code agents，并有自托管模型配置。
ReAct 实现追加工具结果和模型输出，但也支持缩减较早观察及删除历史块。[Agent 与模型配置](https://github.com/StonyBrookNLP/appworld#-agent-experiments)、[ReAct 主循环和历史裁剪](https://github.com/StonyBrookNLP/appworld/blob/main/experiments/code/simplified/react_code_agent.py)

交互式 ReAct 更容易产生多轮复用；full code 一次执行很多 API 时，可能只有很少模型往返。
本地工具和数据库状态可控，适合性能与质量一起评估；工具窗口偏短则 offload 潜力有限。
历史裁剪会让早期分歧之后的 KV 无法命中，应保留裁剪规则并观测，而不是为制造收益改变主实验策略。
默认 agent 通常串行，公开的多进程任务执行功能并不提供题内 `join_group`。

判断：多轮缓存和跨任务调度的适配较好，环境稳定性有优势，offload 预期低于真实长工具任务。

### 7.9 WebArena：Agent 可用，但默认提示顺序不利于长前缀

AgentLab 的 `GenericAgent` 可运行 BrowserGym 环境，已有 `VLLMChatModel` 连接本地端点。
它每步构建 `MainPrompt`，其顺序为任务说明、当前观察、历史、动作说明等。[主循环](https://github.com/ServiceNow/AgentLab/blob/main/src/agentlab/agents/generic_agent/generic_agent.py)、[提示顺序](https://github.com/ServiceNow/AgentLab/blob/main/src/agentlab/agents/generic_agent/generic_agent_prompt.py)、[vLLM 适配](https://github.com/ServiceNow/AgentLab/blob/main/src/agentlab/llm/chat_api.py)

当前页面较早发生变化时，后面即使有大量相同历史，也无法单独命中前缀缓存。
浏览器等待确实存在，但缺少后继可复用长前缀时，CPU 保存不能凭空产生收益。
长页面、多任务竞争仍适合测准入；默认单浏览器任务没有并行模型汇合。
改为追加式历史属于 agent 配置或实现变化，应单列实验并重新检查质量，不能把改写客户端的收益归给 TokenCake。

判断：可用现成 agent，但针对当前 offload 的优先级需要下调，适合作为前缀易失效的泛化测试。

### 7.10 GAIA：已有研究型 Agent，角色存在不代表分支并行

GAIA 本身提供任务；smolagents 的 Open Deep Research 示例提供 `CodeAgent` manager、`ToolCallingAgent` search 子 agent，以及搜索和文件工具。
示例使用 LiteLLM；库也提供兼容服务端点的模型适配。[公开 GAIA 实现](https://github.com/huggingface/smolagents/tree/main/examples/open_deep_research)、[Agent 构造](https://github.com/huggingface/smolagents/blob/main/examples/open_deep_research/run.py)、[模型适配](https://huggingface.co/docs/smolagents/en/reference/models)

manager 等子 agent 时，其历史有暂存价值；搜索与文件处理也能提供真实等待，角色信息比单一 ReAct 更丰富。
但不同角色的系统提示不同，不能默认跨 agent 共用长前缀；该示例也没有预先建立多个并行研究分支的固定结构。
计划更新、记忆处理、辅助文件理解模型和多模态工具都会影响实际调用图与成本，需要全部观测。
工具服务、外部模型和任务附件配置的复现成本较高，当前文本模型也不能独自完成所有多模态任务。

判断：适合作为后续真实多角色场景；offload 潜力较好，DAG 收益仍需实际并行轨迹支持。

### 7.11 StableToolBench：Agent 已有，DFS 是搜索形态

仓库包含 CoT 单链和 DFS/DFSDT 推理代码，并允许设置模型 API 地址。
DFS 原版递归展开和调用模型；多题线程与工具调用列表都不能自动算作并行 LLM DAG。[运行说明](https://github.com/THUNLP-MT/StableToolBench)、[单链实现](https://github.com/THUNLP-MT/StableToolBench/blob/master/toolbench/inference/Algorithms/single_chain.py)、[DFS 实现](https://github.com/THUNLP-MT/StableToolBench/blob/master/toolbench/inference/Algorithms/DFS.py)

长工具说明与多轮观察可检验缓存，DFS 回溯也可能复用祖先前缀；跨任务能否共享则取决于工具集合及其排列。
缓存 API 响应、模拟器推理和真实请求的等待性质不同，需分开记录；评判模型不属于在线完成阶段。
若大部分响应直接命中工具缓存，offload 收益可能较弱；调用模拟器时则可能更明显，但只能解释为模拟器负载。

判断：适合作为大规模工具负载补充，准入与前缀值得测，原版不适合单独证明汇合调度。

### 7.12 ALFWorld：经典 Agent 可用，适合测短工具窗口

ReAct 官方 notebook 包含完整动作循环：调用模型、`env.step()`、追加动作与观察。
模型入口使用旧 Completion API，可替换为当前服务并固定提示和终止规则。[原版 Agent](https://github.com/ysymyth/ReAct/blob/master/alfworld.ipynb)

历史自然累积，同类型任务还可能共享较长的 few-shot 示例，因此可以检查前缀命中和并发调度。
文本动作本地执行，窗口可能不足以覆盖传输余量；频繁事件自身的开销应计入 E2E。
原版没有并行汇合；环境叙述中的动作不表示真的执行相同持续时间的物理过程。

判断：适合低开销和短工具场景对照，多轮优化有机会，offload 预期较弱。

### 7.13 AgentBench：已有运行器，效果必须按子环境拆开

原版 `v0.2` 提供 `HTTPAgent`，可配置请求地址、模型和消息角色；任务客户端与 assigner 负责交互和分发。
当前主分支的 FC / AgentRL 版本是另一套需要固定版本的路径。[原版模型配置](https://github.com/THUDM/AgentBench/blob/v0.2/configs/agents/openai-chat.yaml)、[原版 HTTPAgent](https://github.com/THUDM/AgentBench/blob/v0.2/src/client/agents/http_agent.py)、[版本说明](https://github.com/THUDM/AgentBench)

OS 命令、DB 查询与 ALFWorld 动作的等待分布不同，不能给整套 benchmark 一个统一的 offload 强弱结论。
较长 OS 工作更值得检查保存窗口，快查询和文本环境则更适合测调度开销；提示保留规则也要逐环境审计。
跨环境混合负载有利于验证类别预留和公平性，但固定的重要性策略必须适用于全部后端，不得根据任务答案选择优先级。
多个环境任务同时执行仍属于应用间竞争，默认不构成同一任务内的分支汇合。

判断：跨类型准入与公平性测试有价值，整体接入较重，建议先选择 OS、DB 两个环境。

### 7.14 第一轮实测需要回答的问题

1. 当前模型能否走完有代表性的有效轨迹，官方成功率和上下文溢出情况如何？
2. 后继实际 token 输入与已保存前缀重合多少，压缩或工具调用序列化在何处产生分歧？
3. 工具或子 agent 等待期间，原生 GPU KV 是否真的被其他任务覆盖，当前策略是否满足保存条件？
4. 避免的 prefill 是否抵消传输、事件和调度开销，任务成功率是否保持？
5. 是否观测到真正的并行模型分支、分数继承和运行中共享前缀排序变化？

若优先验证真实效果，第一轮推荐 mini-SWE-agent + SWE-bench；若优先降低接入成本，推荐 BFCL 多轮。
若优先检验并行 DAG，需明确采用经过审计的并行 LATS 适配版本，而不能把原版直接列为已覆盖。
所有强弱判断都需要上述观测支持，不能把现有模拟负载上的 34.22% 端到端降幅套到这些 agent 上。

# Agent Benchmark 平台实施状态

更新日期：2026-09-08。对应 [实现计划](agent-benchmark-implementation-plan.md)。

最终状态：**试跑完成，实验已停止，正式全量取消**。
用户要求“跑完试跑就停止，开始进行实验结果分析”和“不要跑正式全量实验了”。
本轮交付范围为已完成的四组 BFCL 4 题功能试跑、四组 SWE 20 题试跑及 [最终分析](agent-benchmark-pilot-final-analysis.md)。
后续模型复跑、新增观测接入和继续环境准备均不执行，已有数据与环境记录保留。

平台完成了两组真实服务协议检查、官方 agent 适配检查，以及固定 20 题的 SWE 环境验证。
四组 agent 工作环境与独立评分环境共 100 个，Python、依赖、源码与快照身份核对全部通过。
Xarray 和 Matplotlib 使用 Miniconda，两题均通过原始版本与参考修复验证。
首对 BFCL 四题功能任务与 SWE 20 题均完成真实执行和独立评分。
BFCL 两组均 0/4，SWE 为 `base` 0/20、`agent_offload` 1/20；机制改善伴随排队和超时增加，尚不能证明稳定性能收益。
详见 [首对报告](agent-benchmark-first-pair-pilot.md)，其中按相同预算的正确完成数优先解释结果。
按用户指定的窗口吞吐口径，首对共同前 30 分钟输出吞吐 +27.6%，各自全程 +8.6%，客户端完整响应 +6.8%；均不等同于正确解题效率提升。
其余 `agent`、`offload` 共 40 个环境也已准备完成，身份核对通过；合计 100 个任务环境就绪。
四组的 BFCL 功能任务已完成执行和评分，均为 0/4；SWE 四组 20 题也已全部完成，`agent`、`offload` 均为 0/20。
四组共 80 个 SWE 终态和评分完整，第二对服务已停止；详见 [四组试跑报告](agent-benchmark-four-mode-pilot.md)。
`agent` 前 30 分钟输出吞吐相对 base 为 +18.1%，各自全程为 -3.8%；`offload` 分别为 -9.4% 和 -27.8%，并观察到累计 2277 秒有排队但无生成进展的区间。
共享测量预算累计使用 7821.614 秒，约 2.17 小时；所有测量区间已结束，正式全量和重复矩阵未启动且已取消。
停止准备时，500 题中已有 46 题通过原始版本与参考修复的 F2P/P2P 验证，6 题参考验证失败，6 题下载失败，6 题准备取消，436 题尚未进入准备。
剩余 17 道 Astropy 题的安装与动态验证已完成，其中 16 题通过、1 题存在冻结测试名歧义，详见 [Astropy 环境验证](agent-benchmark-astropy-environment-validation.md)。
最新 Django 验证新增 6 题通过，并修复评分脚本的任务本地 locale 初始化；受影响的四道试跑题和四组保存预测复核后成绩一致，详见 [Django 复核](agent-benchmark-django-environment-validation.md)。
四组试跑使用的 100 个工作/评分环境与已有模型成绩另行保留；新增全量评分环境尚未用于模型运行。
原生逐请求追踪已通过两套源码的无模型导出夹具检查，取消路径缺少 span；字段与覆盖率限制保留于 [已取消的下一阶段提案](agent-benchmark-next-stage-proposal.md)，没有接入新的模型运行。
另用真实调度器复现了回载后队头容量不足阻挡后续就绪请求的机制；启用 agent 容量预留后同一夹具可以完成，详见 [停滞最小诊断](agent-benchmark-offload-stall-diagnosis.md)。

## 用户已确定的范围

- 使用 BFCL 多轮 Base + Long Context、SWE-bench Verified/test 和 mini-swe-agent。
- 沿用 Qwen2.5-14B-Instruct、BF16、单 A800、32,768 上下文、显存比例 0.5、批 token 预算 8192。
- offload 组使用 100 GiB CPU KV；比较原生 `base` 与 TokenCake 的三个配置。
- 先比较 `agent_offload` 与 `base`，完成后再运行 `agent`、`offload`。
- 环境、缓存、代码、任务和结果放在数据盘；使用本地执行和评分，不依赖 Docker 服务。
- 提高测试并发，增加内存压力。
- 两张 GPU 并行，GPU 0 跑 `base`、GPU 1 跑 `agent_offload`，每组 16 并发。
- uv 无法支持的环境使用服务器已有 Miniconda，新增前缀和缓存均放数据盘。
- 试跑结束即停止，取消正式全量实验，完成既有结果分析。

## 已确定的首轮参数

沿用此前首轮设置，并按最新授权更新为双 GPU 并行和 Miniconda 兼容后端。

| 项目 | 设置 |
| --- | --- |
| 应用并发 | 每组 16，两组最多 32 个应用同时在途 |
| GPU 与顺序 | 首先 GPU 0 `base` 与 GPU 1 `agent_offload` 并行；比较后再运行其他组 |
| 试跑到达 | 固定任务顺序的闭环并发，空出工作位后释放下一题；记录实际到达时间 |
| CPU 使用 | 客户端和工具的 OMP、OpenBLAS、MKL 等库线程数设为 1，避免 16 个任务各自展开大量线程 |
| 试跑测量预算 | BFCL 与 SWE 共用 12 小时；并行区间取并集，环境准备、模型加载与离线评分单独计时 |
| SWE 选择规则 | 对 `20260908:` 加 instance_id 计算 SHA256，按哈希升序取前 20 题 |
| BFCL 功能样例 | 每类以同一规则取 2 题；完整选定数据仍为两类各 200 题 |
| mini 提示 | 冻结源码中的官方 `swebench_xml.yaml`，本地目录占位改为 `.` |

容器内存限制已核实为 **240 GiB**，CPU 配额为 **36 核**；不能按宿主机约 1 TiB 内存估算余量。
试跑监测已记录 cgroup 内存、OOM、内存/CPU pressure、GPU 显存和 KV/队列指标；两对共享 cgroup 内存采样峰值均约 239.99 GiB，未发生 OOM。
两组共享上述 CPU 和内存限制；cgroup 样本不能按服务重复相加，也不能直接归因给某一组。
SWE 20 题试跑的独立任务并发上限是 20；BFCL 四题功能样例也不能作为 16 并发压测成绩。
正式应用 QPS、到达分布、重复三次的负载点及全量预算未冻结，相关工作已取消。

首轮压力设置的准备证据：两组真实服务协议检查时，日志均显示 GPU KV 容量为 **48,800 token**。
离线复用冻结的官方初始提示构造与模型 tokenizer，得到以下统计；统计过程没有模型请求和工具调用。

| 初始输入 | 任务数 | 中位数 token | P95 token | 最大 token |
| --- | --- | --- | --- | --- |
| BFCL Multi-turn Base | 200 | 6,715 | 8,905.1 | 9,001 |
| BFCL Multi-turn Long Context | 200 | 6,715 | 8,905.1 | 9,001 |
| SWE Verified | 500 | 1,697 | 2,738.5 | 9,110 |

全部初始输入加 4,096 输出预算均未超过 32,768 上下文。
16 条独立序列平均分配 48,800 token 相当于每条 3,050 token，但共享前缀和准入策略会改变实际占用，不能由这个算式认定已达到内存压力。
上述统计仅覆盖第一轮输入；实际试跑的后继历史、工具输出与 KV 压力另见 [四组报告](agent-benchmark-four-mode-pilot.md)。
准备阶段证据见 [900 题初始输入统计](/root/autodl-tmp/tokencake-agent-bench/manifests/initial-context-inventory-01/summary.json)和两组 [服务日志目录](/root/autodl-tmp/tokencake-agent-bench/results/serving-smoke-03)。

已确认机器可读清单：
[`confirmed-parallel-20260908-01/manifest.json`](/root/autodl-tmp/tokencake-agent-bench/manifests/confirmed-parallel-20260908-01/manifest.json)。
其中包括全部任务 ID、数据哈希、模型身份、依赖锁、资源限制、双 GPU 映射和兼容环境管理器，状态为 `confirmed`；原候选清单保留。
版本、模型文件和已安装依赖的核验已通过。

## 已取得的官方输入

| 项目 | Git 提交或数据范围 |
| --- | --- |
| BFCL / gorilla | `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` |
| mini-swe-agent | `04d809ceab9df28f9adaed044884180159172930`，安装版本 2.4.6 |
| SWE-bench 判分 | `02e7a74ffd0b707aab73d203fe87bdc7c76afc8e`，安装版本 5.0.2 |
| 官方 swe-bench-tasks | `3d07b464b7b311a0cbfb5ed5b2d8a3b96f84a33d` |
| 原生 vLLM 基线 | `0b3ba88f165976e77ca5e6a7a3f5bba4562b80af`，已核验独立检出无跟踪文件修改 |
| TokenCake 被测代码 | `e93753cae3709f2805f998a4e368ef61cf48bf63` |
| BFCL 文件 | 当前冻结源码中的 V4 `multi_turn_base`、`multi_turn_long_context`，各 200 题 |
| SWE 数据 | 官方 task 仓库中 `SWE-bench/SWE-bench_Verified`、`test`，共 500 题 |

Hugging Face 直接访问失败，镜像 API 返回 403，因此从当前 SWE-bench 官方代码指向的 GitHub task 仓库获取任务内容。
数据来源及其 Git revision 已记录，不将其伪称为一次成功的 Hugging Face revision 下载。
模型的 8 个权重分片和配置/tokenizer 文件均已计算 SHA256。
当前服务环境使用已有预编译扩展，服务 Git 身份与 Python 包版本分别记录。

候选 SWE 20 题的实际顺序如下：

```text
sphinx-doc__sphinx-10449
django__django-16502
django__django-15973
sympy__sympy-22714
sympy__sympy-15976
django__django-13933
astropy__astropy-12907
pylint-dev__pylint-7080
django__django-12419
django__django-11211
pydata__xarray-3677
sphinx-doc__sphinx-7462
sympy__sympy-21379
django__django-11820
django__django-15814
scikit-learn__scikit-learn-25102
sympy__sympy-16766
matplotlib__matplotlib-22865
sympy__sympy-22080
django__django-13794
```

BFCL 功能样例为 `multi_turn_base_198`、`multi_turn_base_103`、`multi_turn_long_context_97`、`multi_turn_long_context_96`。
抽样规则只使用 ID，不读取题目难度、参考补丁、测试表现或模型成功情况。

## 验证证据

39 项测试已通过：9 项传输测试、20 项执行/评分/汇总测试、5 项官方 agent 和评分适配测试、5 项窗口吞吐测试。
历史环境适配覆盖真实 locale 编译/快照复位和 Conda 管理器导入隔离；最新 20 项执行与评分相关测试已重新运行通过。
新增测试验证双组执行确实重叠、预算只计一个区间、任一控制器异常时取消另一组并保留全部任务终态。
执行测试包含 20 个真实子进程任务在并发上限 16 下完成的检查，但这些夹具不调用模型，不计为 benchmark 压测。
本轮所有修改文件的 pre-commit 检查已通过。
BFCL 首次真实执行发现工具状态中的 `set` 无法直接写成 JSON；现复用官方序列化函数，增加含 MessageAPI 状态的回归检查，重跑两组各四题均正常保存。
首次失败记录保留在 `pilot-bfcl-parallel-first-pair-01`，修复后的运行使用 `pilot-bfcl-parallel-first-pair-02`。
完整索引见 [平台验证记录](/root/autodl-tmp/tokencake-agent-bench/results/platform-validation-index-20260908-03.json)。
新指标回归确认失败不增加正确曲线、任务晚到时间计入测量时钟、未测时段不外推、超时输出长度保持未知，以及缺失计数器不按零处理。
评分补齐冻结官方的补丁应用回退与部分应用复位，首对 SWE 保存预测已统一重新评分。
新增窗口吞吐测试覆盖多 engine 计数、截止后生成量排除、被总量增长掩盖的单 engine 重置、缺失采样与未覆盖完整窗口。
新版分析已逐项复现用户提供的六个输出 token 校验值；原始报告保留，最新版本为 `report-throughput-02.json`。
独立只读原始文件的核算再次复现全部六值，输入哈希保持一致，见 [吞吐独立复核](/root/autodl-tmp/tokencake-agent-bench/results/pilot-first-pair-throughput-readonly-audit-20260908-01.json)。
评分 locale 修复后，20 项相关测试重新通过；全部 500 份脚本的测试命令及补丁保持一致。
试跑受影响的四道 Django 题在新环境中验证通过，四组共 16 份保存预测的评分状态和通过数均未改变，见 [Django 复核](agent-benchmark-django-environment-validation.md)。

| 检查 | 已观察到的结果 | 适用范围 |
| --- | --- | --- |
| 真实服务协议 | 两组均完成生成、真实工具、事件及后继生成；输出均为 `4`、`4`；本地与服务端输入 token 数一致 | 验证服务接入，不作为 benchmark 成绩 |
| GPU 与系统监测 | 两组观测正常，没有发现其他 GPU 计算进程或 CPU affinity 违规；检查后服务已停止 | 验证监测路径，不代表高并发压力已测量 |
| 官方 BFCL 适配 | 官方与适配后的提示和工具状态一致，官方多轮 checker 接受参考动作 | 使用测试夹具响应，不是模型准确率 |
| 官方 mini 适配 | XML 动作经真实 shell 执行，提交 marker 与事件完整 | 验证动作和提交流程 |
| Sphinx 10449 | Python 3.9.20；原始版本目标测试失败，参考修复后 31 项测试通过 | 该题本地环境验证 |
| Django 12419 | Python 3.6.13；原始版本目标测试失败，参考修复后通过 | 该题本地环境及快照复位验证 |
| SWE 20 题 | 原始版本的目标失败与 P2P 通过均得到验证，全部参考修复的官方判分为 resolved | 环境验证，不是模型解决率 |
| 两组工作环境与评分环境 | 20 题各 3 个环境，共 60 个；实际 Python、依赖、Conda 包版本/build、原始提交与准备后源码一致，快照哈希正确，跟踪文件干净 | 完整 20 题执行准备就绪 |
| SWE 配方清点 | 全部 500 份配方可解析；500 份评分脚本中的测试补丁 heredoc 保持原样 | 静态核验，不等于 500 个环境都已安装 |

主要原始证据：

- [全部 20 题环境验证](/root/autodl-tmp/tokencake-agent-bench/grading/confirmed-environment-validation-summary-02.json)与[60 个环境身份核对](/root/autodl-tmp/tokencake-agent-bench/grading/confirmed-prepared-environment-audit-01.json)。
- [双 GPU SWE 首轮运行配置](/root/autodl-tmp/tokencake-agent-bench/results/pilot-swe-parallel-first-pair-01/experiment.json)。
- [首对最新窗口指标](/root/autodl-tmp/tokencake-agent-bench/results/pilot-swe-parallel-first-pair-01/report-throughput-02.json)和[重新评分结果](/root/autodl-tmp/tokencake-agent-bench/grading/pilot-swe-parallel-first-pair-02/scores.json)。
- [窗口吞吐与执行回归](/root/autodl-tmp/tokencake-agent-bench/results/window-throughput-tests-20260908.xml)。
- [剩余两组 SWE 运行配置](/root/autodl-tmp/tokencake-agent-bench/results/pilot-swe-parallel-second-pair-01/experiment.json)和[BFCL 执行评分报告](/root/autodl-tmp/tokencake-agent-bench/results/pilot-bfcl-parallel-second-pair-01/report-throughput-02.json)。
- [四组 SWE 完整对照](/root/autodl-tmp/tokencake-agent-bench/results/pilot-swe-four-mode-01.json)、[第二对窗口分析](/root/autodl-tmp/tokencake-agent-bench/results/pilot-swe-parallel-second-pair-01/report.json)及[第二对评分](/root/autodl-tmp/tokencake-agent-bench/grading/pilot-swe-parallel-second-pair-01/scores.json)。
- [剩余两组 40 个环境身份核对](/root/autodl-tmp/tokencake-agent-bench/grading/remaining-prepared-environment-audit-01.json)。
- [指标回归](/root/autodl-tmp/tokencake-agent-bench/results/quality-report-tests-20260908.xml)与[官方补丁应用回退回归](/root/autodl-tmp/tokencake-agent-bench/results/grading-patch-fallback-tests-20260908.xml)。
- [双组并行与异常清理回归](/root/autodl-tmp/tokencake-agent-bench/results/parallel-execution-tests-20260908.xml)和[包含 BFCL 状态序列化的适配回归](/root/autodl-tmp/tokencake-agent-bench/results/adapter-tests-20260908-04.xml)。
- [真实服务检查结果](/root/autodl-tmp/tokencake-agent-bench/results/serving-smoke-03/result.json)。
- [Sphinx 原始版本评分](/root/autodl-tmp/tokencake-agent-bench/grading/environment-validation-02/sphinx-doc__sphinx-10449/base/report.json)与[参考修复评分](/root/autodl-tmp/tokencake-agent-bench/grading/environment-validation-02/sphinx-doc__sphinx-10449/gold/report.json)。
- [Django 原始版本评分](/root/autodl-tmp/tokencake-agent-bench/grading/environment-validation-04/django__django-12419/base/report.json)与[参考修复评分](/root/autodl-tmp/tokencake-agent-bench/grading/environment-validation-04/django__django-12419/gold/report.json)。
- [500 题配方及补丁核验清单](/root/autodl-tmp/tokencake-agent-bench/manifests/environment-inventory-02/summary.json)。
- [候选 20 题环境验证汇总](/root/autodl-tmp/tokencake-agent-bench/grading/candidate-environment-validation-summary-01.json)，包括每题原始版本和参考修复的日志、检查与环境来源。
- [新增环境回归测试](/root/autodl-tmp/tokencake-agent-bench/results/environment-regression-tests-20260908.xml)，覆盖镜像系统安装命令隔离、官方截断测试名匹配及缺失/跳过测试的拒绝。
- [两组与评分环境的身份核对](/root/autodl-tmp/tokencake-agent-bench/grading/candidate-prepared-environment-audit-01.json)，完整保留 20 题分母、18 题就绪状态和剩余两题。
- [指定提交缓存回归测试](/root/autodl-tmp/tokencake-agent-bench/results/repository-cache-regression-tests-20260908.xml)，用真实 Git 检出确认使用指定原始内容，并拒绝发生变动的缓存 ref。
- [官方 agent 适配测试记录](/root/autodl-tmp/tokencake-agent-bench/results/adapter-tests-20260908-02.xml)。
- [客户端、评分与服务依赖身份](/root/autodl-tmp/tokencake-agent-bench/manifests/dependencies-20260908/identity.json)。

早期失败的模型启动、tox、Python 3.6 editable 安装和环境批次记录仍保留在数据盘；修复后的验证使用新输出目录。
批量准备时将 Git fetch 的 HTTP 版本限制在该条命令内，解决了本机默认连接超时；没有修改全局网络配置。
之后仍出现 GitHub TLS 断连，因此从已取得的仓库创建了 20 个指定 base commit 缓存；后续准备核对缓存身份，再建立独立检出。
缓存重试完成了 `base` 的剩余 15 个环境；先前已准备好的 3 个环境继续保留，失败或中断目录另行归档。
准备记录见 [首批两组准备](/root/autodl-tmp/tokencake-agent-bench/grading/candidate-agent-environment-preparation-01/campaign.json)、[base 缓存重试](/root/autodl-tmp/tokencake-agent-bench/grading/candidate-agent-environment-preparation-02/base/summary.json)和[失败目录归档索引](/root/autodl-tmp/tokencake-agent-bench/grading/candidate-agent-environment-preparation-01/base/archive-index.json)。
Pylint 的历史模块发现需要显式使用任务仓库的 Python 搜索路径，修正后原始版本和参考修复均符合预期。
Sphinx 7462 的官方参考评分原本已通过；环境检查器现沿用官方对截断参数化测试名的匹配规则，避免误报。

## 已授权的 Miniconda 依赖兼容方案

此前 uv 安装失败题为 `pydata__xarray-3677` 和 `matplotlib__matplotlib-22865`。
它们分别遇到 `antlr-python-runtime==4.11.1` 和 `nbconvert-core==7.16.6` 无法按原名从 PyPI 获取；配方还包含其他 Conda 和原生依赖。
不将简单改名或删去依赖当作完整复现。

使用服务器 `/root/miniconda3/bin/conda`（24.4.0）复现官方冻结 Conda 配方；任务目录仍在数据盘，解释器入口仍为各题的 `.venv/bin/python`。
官方 pip 依赖和项目安装继续经 uv；客户端、评分服务和模型服务继续使用已有 uv 环境。
使用独立前缀、复制安装和既有完整快照复位，随后重新验证原始版本与参考修复。
镜像中的系统包安装另行核对并记录本地提供方式，不能直接对共享主机运行镜像的 apt upgrade。

早期 micromamba 2.9.0 仅用于依赖解析；实际创建前缀使用用户指定的 Miniconda。
Xarray 原始版本与参考修复各解析 22 项测试，全部环境检查通过，参考修复判定为 resolved。
Matplotlib 原始版本 3 项目标失败、57 项通过；参考修复后 60 项全部通过。
Conda 缓存冲突和下载校验失败记录保留；安装器已串行化缓存写入，两题均支持从通过原冻结校验值的本地包归档安装，避免重复获取错误缓存内容。

| 任务 | 精确 Python | Conda 包数 | 预计 Conda 下载量 | 官方 pip 条目 |
| --- | --- | --- | --- | --- |
| Xarray 3677 | 3.10.15 | 323 | 约 370 MiB | 13 |
| Matplotlib 22865 | 3.11.11 | 326 | 约 424 MiB | 18 |

下载量不包含 pip 包、项目编译和系统依赖，也不等于安装后的磁盘占用。
证据见 [Xarray 依赖解析](/root/autodl-tmp/tokencake-agent-bench/manifests/native-environment-plan-01/pydata__xarray-3677/result.json)与[Matplotlib 依赖解析](/root/autodl-tmp/tokencake-agent-bench/manifests/native-environment-plan-01/matplotlib__matplotlib-22865/result.json)；同目录保存原始配方、命令和完整 solver 输出。
所有原配方 Conda 包的版本和 build 均与解析结果一致；同目录还保存带包 URL 和校验值的 `conda-explicit.txt`、原始 pip 条目及 `locks.json`。
500 题中需要镜像系统安装命令的配方另有 [静态清点](/root/autodl-tmp/tokencake-agent-bench/manifests/environment-inventory-03/summary.json)，该清点不代表已安装系统依赖。

用户已明确授权 uv 无法支持时使用服务器 Miniconda，此授权覆盖这里的兼容后端。
Xarray 证据见 [完整环境验证](/root/autodl-tmp/tokencake-agent-bench/grading/miniconda-validation-01/pydata__xarray-3677/validation.json)。

## 收尾状态与保留的环境准备记录

1. 四组试跑、保存预测评分与现有日志分析已完成；最终报告同时呈现收益、回退和缺失指标。
2. 试跑存在上下文溢出、无效提交、超时与 offload 停滞；历史停滞根因仍未确定，作为结果限制保留。
3. 正式全量、负载校准、模型复跑和进一步环境准备已取消，没有遗留 benchmark 或 GPU 计算进程。

以下为停止准备时的历史进度，依据已保存的原始/参考测试报告汇总，保留全部 500 个任务 ID；不代表模型成绩或后续待办：

| 状态 | 题数 | 含义 |
| --- | ---: | --- |
| `validated` | 46 | 原始版本规定的 F2P 失败、P2P 通过，参考修复满足冻结官方评分规则 |
| `reference_validation_failed` | 6 | 解释器和依赖已安装，但原始测试或测试名匹配与冻结预期列表不符 |
| `environment_error` | 6 | 本批 GitHub 仓库下载连接超时，未完成环境准备 |
| `validation_cancelled` | 6 | 因下载连续失败停止准备批次，保留取消及未启动记录 |
| `not_attempted` | 436 | 尚未进入逐题准备批次 |

较早新增的以下四题按已冻结的完整清单顺序准备，使用 Python 3.9.20、uv 和各自冻结的 28 项 Python 依赖。
它们的解释器、依赖版本、准备提交、原始提交、源码复位与快照哈希核对均通过。

| 新增任务 | F2P / P2P 数量 | 参考修复后，已解析测试的通过 / 失败数 |
| --- | ---: | ---: |
| `astropy__astropy-13033` | 1 / 20 | 21 / 1 |
| `astropy__astropy-13236` | 2 / 644 | 646 / 2，另有 1 项预期失败 |
| `astropy__astropy-13398` | 4 / 68 | 72 / 1 |
| `astropy__astropy-13453` | 1 / 9 | 10 / 0 |

`validated` 不表示评分脚本启动的每项测试都通过。
上述残留失败在原始版本也存在，均不属于冻结 F2P/P2P 集合；日志包含闰秒数据过期、类型和异常断言错误，均已保留。
这些环境仍使用本地原生库，不能据此声称与官方容器逐项等价；新增四题尚未建立四组 agent 副本或产生模型成绩。
证据见 [500 题最新进度](/root/autodl-tmp/tokencake-agent-bench/grading/full-environment-validation-progress-20260908-05.json)、[较早四题与追踪身份核验](/root/autodl-tmp/tokencake-agent-bench/results/post-pilot-validation-20260908-01.json)及[该四题评分集合之外的失败清单](/root/autodl-tmp/tokencake-agent-bench/grading/full-environment-validation-20260908-01/test-scope-audit.json)。
其后 17 道 Astropy 题的结果、冻结依赖构建、子模块缓存与所有重试见 [Astropy 环境验证](agent-benchmark-astropy-environment-validation.md)。
最新 Django 批次新增 6 题通过，10554、10999 的原始测试仍无法满足规定的逐项验证；具体日志、原生依赖身份与下载失败见 [Django 环境验证](agent-benchmark-django-environment-validation.md)。

全量数据的三题历史解释器和冻结依赖已通过 Miniconda 安装；安装身份核对通过，但三题均为 `reference_validation_failed`：

| 任务 | 官方 Python | 当前验证问题 |
| --- | --- | --- |
| `django__django-10097` | 3.5.6 | 原始版本 F2P 已通过，原始与参考版本均被判 resolved；完整参考测试仍有大量失败 |
| `django__django-7530` | 3.5.6 | 原始版本 F2P 已通过；实际 error 测试未在冻结评分集合中 |
| `psf__requests-1724` | 2.7.18 | 原始版本六项 F2P 已通过；实际失败测试被列为 P2P，参考修复后 88 项全通过 |

500 题冻结输入与官方加载器逐字段一致，未发现适配层抄录错位；这不能代替本地与官方环境的一致性验证。
没有换版本、改评分分类、换题或剔除它们；详见 [历史环境验证报告](agent-benchmark-legacy-environment-validation.md)。
这三题不在当前固定 20 题中，不改变已完成的试跑成绩。
新增的第四题验证失败为 Astropy 14369：冻结 P2P 截断名称同时匹配原始版本的通过项和 F2P 失败项，官方匹配器无法确定其原始状态；没有改变匹配规则或删去该题。
停止准备前检查时数据盘剩余约 109 GiB；正式 500 题四组及评分环境的存储方案未实施，不能假定所有副本已经就绪。
未验证任务的原生依赖差异保留为历史准备缺口，不继续安装或验证。

具体命令和适配说明见 [运行入口](../agent_bench/README.md)。

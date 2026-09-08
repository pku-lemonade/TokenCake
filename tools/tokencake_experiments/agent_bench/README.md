# 真实 Agent Benchmark 运行入口

实现依据：[实验计划](../reports/agent-benchmark-implementation-plan.md)。
本目录包含 BFCL 多轮、mini-swe-agent 与本地 SWE 评分适配；不包含固定轨迹回放或人工工具等待。
当前进度与验证证据见 [实施状态](../reports/agent-benchmark-platform-status.md)。

2026-09-08 最终状态：四组 BFCL 4 题功能试跑与 SWE 20 题试跑均完成，服务已停止。
用户要求停在试跑并取消正式全量；不继续环境准备、模型复跑或新增观测接入。
结论见 [试跑最终分析](../reports/agent-benchmark-pilot-final-analysis.md)，下方命令仅作已有实现的手动复现参考，不表示仍有待启动的实验。

## 目录与环境

全部实验环境和运行数据放在 `/root/autodl-tmp/tokencake-agent-bench`。

| 路径 | 用途 |
| --- | --- |
| `sources/` | BFCL、mini-swe-agent、SWE-bench、官方 SWE task 仓库 |
| `.venv/bin/python` | 独立客户端 Python 3.12、官方 agent 与 tokenizer |
| `environments/grading/.venv/bin/python` | 独立官方 SWE 评分环境 |
| `manifests/` | Git SHA、模型哈希、依赖清单、候选和确认后的任务配置 |
| `environments/<campaign>/<mode>/<instance_id>/` | 每组每题独立的 agent 代码和依赖 |
| `environments/<campaign>/grading/<instance_id>/` | 独立评分目录 |
| `results/`、`grading/` | 预测、事件、服务日志、系统指标与评分 |
| `cache/`、`tmp/` | uv、模型、内核编译及临时数据 |

服务沿用仓库的 `.venv/bin/python`。`base` 从独立原生检出目录加载代码，其他组从当前仓库加载。
客户端和评分依赖分开，避免当前 SWE 评分包与 tokenizer 依赖版本互相覆盖。
BFCL 通过冻结源码路径导入所选多轮模块，没有安装与本轮无关的所有厂商 handler 和检索环境。
依赖记录位于 `manifests/dependencies-20260908/{client,grading,server}.txt`。

## 实现分工

| 文件 | 责任 |
| --- | --- |
| `transport.py` | 请求 UUID、预算、有限重试、真实工具事件与逐请求日志 |
| `bfcl.py` | 复用官方 Qwen 提示、最终多轮循环、工具执行和多轮检查器 |
| `mini.py` | 复用官方 XML 动作配置、解析、agent 循环、LocalEnvironment 与提交 |
| `inputs.py` | 数据、代码、模型和依赖身份记录及运行前核验 |
| `swe_local.py` | 官方历史环境配方、本地安装、完整状态复位与官方判分 |
| `environments.py` | 批量准备 SWE 环境，验证原始版本的 F2P/P2P 和参考修复 |
| `worker.py` | 每个应用独立进程、终态、轨迹和子进程清理 |
| `service.py`、`observe.py` | 全新模型服务、Prometheus、GPU 与 cgroup 压力观测 |
| `experiment.py`、`parallel.py`、`budget.py` | 双 GPU 并行或同卡顺序比较、并发释放、全任务终态与共享预算 |
| `score.py`、`report.py`、`report_metrics.py`、`throughput.py` | 独立离线评分、固定预算正确完成数、配对延迟、窗口吞吐和机制指标汇总 |
| `smoke.py` | 真实服务协议检查，结果不作为 benchmark 成绩 |

模型输入只含任务所需信息。SWE 参考补丁、测试补丁、评分脚本和 F2P/P2P 不传给 worker。
mini 的官方 XML 提示中，将容器路径 `/testbed` 替换为 `.`，使不同物理目录中的各组获得相同初始提示。
本轮采用真实生成历史；温度为零仍可能产生不同轨迹，报告记录实际工作量。

## 本地 SWE 环境适配

每个任务从官方 Dockerfile 的冻结环境导出中读取精确 Python 和包版本，用 uv 创建独立虚拟环境。
Python 包使用对应版本的 PyPI 分发；原生库来自主机、wheel 和所需历史解释器分发。
这属于自定义本地执行环境。固定 20 题全部通过原始版本与参考修复验证，不能据此认为全部 500 题已与官方镜像等价。
Xarray 3677 和 Matplotlib 22865 的冻结环境含有不能按原名从 PyPI 安装的 Conda 包。
用户已授权使用服务器 Miniconda，准备入口增加 `--environment-manager miniconda`，使用数据盘独立前缀和复制安装。
安装前校验精确包锁，安装后核对官方 Conda 版本/build 并保存激活变量，再安装官方 pip 条目与项目。
Xarray 和 Matplotlib 均已通过原始版本与参考修复验证。Conda 共享缓存写入加锁，试跑 agent 执行仍并行。
两题均支持从 `cache/conda-archives/<task>/` 的已校验本地包安装；原始下载 URL、版本/build 和校验值均保留。

安装步骤保留官方项目设置和依赖目标。普通安装命令转为 `uv pip install --python <task>/.venv/bin/python`。
uv 不支持的 `--no-use-pep517` 构建开关在翻译时移除，因此需要验证构建与参考测试结果。
Python 3.6/3.7 的解释器可从官方配方对应的 Anaconda 二进制获取，再交由 uv 创建虚拟环境。
Python 3.6 的旧 setuptools 不支持现代 editable 构建，兼容入口是 `uv run --no-project --python <task>/.venv/bin/python <task>/.venv/bin/python -m pip install ...`，使用任务环境内的固定旧版安装器。
Python 2.7/3.5 使用显式指定的 Miniconda 后端和 `<task>/.venv/bin/python -m pip`，保留官方冻结的 pip 版本；这些解释器不交给 uv 查询。
Conda 管理进程清空任务 `PYTHONPATH`，防止历史 Requests 源码被管理器自身导入；实际任务仍使用自己的源码路径。
激活变量保留任务前缀的 `lib/pkgconfig`，使 CFFI 构建找到该前缀内已冻结的 libffi。
依赖清单在任务目录中采集，避免 pip 将实验仓库的本地包元数据误列为任务依赖。
没有调用系统 Python 或裸 `pip` 命令。

`tox --current-env` 的解释器软链接在 uv 虚拟环境下会丢失依赖路径，适配层以指向同一任务解释器的 exec wrapper 保留其环境。
官方测试命令、目标测试集合和嵌入的测试补丁保持原样；评分沿用官方实际测试退出码记录、日志解析和 resolved 规则。
原始版本的检查沿用官方对截断参数化测试名的匹配规则，同时要求目标失败确实出现在日志中，P2P 也不能仅被跳过。
任务进程显式设置 `PYTHONPATH` 为自己的仓库路径，避免框架搜索路径影响 Pylint 等历史项目的模块发现。
镜像安装中的 apt 等系统包命令单独记录，不直接在共享主机上执行镜像的系统升级。
两题 Python 3.5 配方要求的 `en_US.UTF-8` 使用现有 glibc 输入编译到任务 `.venv/lib/locale`，通过 `LOCPATH` 启用并纳入快照，不修改主机 `/etc/locale.gen`。
环境准备还读取官方 `eval.sh` 的 locale 初始化要求；111 道 Django 题的需求仅出现在评分脚本中。
本地翻译移除已由任务快照提供的系统 locale 生成命令，保留官方 locale 导出、测试命令和补丁 heredoc。
缺少任务本地 locale 的旧环境需要按冻结配方重新准备，不能修改历史快照后继续使用旧哈希。
相关修复、500 份脚本核对及试跑保存预测复核见 [Django 验证报告](../reports/agent-benchmark-django-environment-validation.md)。
这部分原生依赖仍须在对应题的本地环境中落实并记录，不能通过省略依赖将失败题算成验证通过。

准备完成后保存 `repo` 和 `.venv` 的完整快照，包括被 Git 忽略的编译文件。
每次评分恢复这份快照并清空任务临时目录和缓存，避免参考补丁、前组编译产物和安装变更污染后续评分。
快照必须属于本工具创建且哈希未变的目录；旧版缺少快照的验证目录不能直接作为新评分环境。
uv 安装使用复制模式，避免不同任务通过依赖硬链接共享可修改内容。

`environments.py --prepare-only` 用于准备各比较组的独立 agent 工作环境；成功状态为 `prepared`，不将其标成参考测试已验证。
原始版本与参考修复仍在独立评分环境中验证。各组从精确 base commit 重新安装，避免复制虚拟环境后残留指向评分目录的 editable 路径。
批量入口可以向已有组目录补充尚不存在的任务目录；已有选中任务会被拒绝，失败准备需先归档并使用新的输出目录。
`cache/swe-repositories/<instance_id>/` 可保存已取得的指定 base commit。安装器核对来源身份与固定 ref 后，从缓存 fetch 同一提交并建立独立工作树。
缓存不包含参考补丁或模型提交；本地缓存命中仅影响准备阶段，不计入模型测量耗时。

Python 2.7/3.5 的三道正式任务已完成历史解释器和冻结依赖安装，原始版本与参考补丁的测试验证结果单独保留。
它们不在试跑名单中；原始 F2P/P2P 与冻结评分列表不符的问题仍须核实，不能自动换解释器、改评分集合或删题。
具体安装和原始/参考测试证据见 [历史环境验证报告](../reports/agent-benchmark-legacy-environment-validation.md)。

## 命令示例

以下命令从仓库根目录运行。首轮清单已保存到 `manifests/confirmed-parallel-20260908-01/manifest.json`。
实验驱动仍拒绝 `status=candidate`；冻结输入始终使用新目录，保留原始记录。
当前已验证的任务环境根目录为 `environments/candidate-20260908-03`，名称保留准备阶段的历史记录；可配合已确认清单使用。
评分目录中的 Xarray 和 Matplotlib 使用指向其已验证物理前缀的目录链接，保持 Conda 内部路径正确。
locale 修复后的完整 20 题评分入口为 `environments/pilot-grading-locale-20260908-01`，其中四道 Django 指向新验证环境，其余指向既有独立评分环境。
目录映射与快照核验见 [评分环境索引](/root/autodl-tmp/tokencake-agent-bench/manifests/pilot-grading-locale-20260908-01.json)；这不修改已保存的 agent 运行环境或测量数据。
每次测量前，运行器从哈希校验过的快照恢复各组工作环境。

```bash
export TC_BENCH_ROOT=/root/autodl-tmp/tokencake-agent-bench
export TC_BENCH_CLIENT="$TC_BENCH_ROOT/.venv/bin/python"

# 用户已确定的首轮输入；输出目录必须是新的目录。
"$TC_BENCH_CLIENT" -m tools.tokencake_experiments.agent_bench.inputs \
  --platform "$TC_BENCH_ROOT" \
  --output "$TC_BENCH_ROOT/manifests/new-confirmed-parallel" \
  --seed 20260908 --workers 16 --status confirmed --execution parallel_pair \
  --dependencies "$TC_BENCH_ROOT/manifests/dependencies-20260908/identity.json"

# 每个 mode/task 和独立 grading/task 分别准备，directory 必须不存在。
"$TC_BENCH_CLIENT" -m tools.tokencake_experiments.agent_bench.swe_local prepare \
  --platform "$TC_BENCH_ROOT" \
  --task-source "$TC_BENCH_ROOT/sources/swe-bench-tasks/tasks/django__django-12419" \
  --directory "$TC_BENCH_ROOT/environments/pilot/agent_offload/django__django-12419"

# 批量环境验证不调用模型，不消耗试跑测量预算。
# setup-jobs 是安装/参考测试并发，与计划的模型应用并发 16 分开。
"$TC_BENCH_CLIENT" -m tools.tokencake_experiments.agent_bench.environments \
  --platform "$TC_BENCH_ROOT" \
  --manifest "$TC_BENCH_ROOT/manifests/candidate-20260908-02/manifest.json" \
  --directory "$TC_BENCH_ROOT/environments/new-validation/grading" \
  --output "$TC_BENCH_ROOT/grading/new-validation" \
  --setup-jobs 2 --subset pilot

# 验证配方后，为一个比较组批量准备独立工作环境。
# 需要补齐全部 20 题才能运行对应 SWE 试跑。
"$TC_BENCH_CLIENT" -m tools.tokencake_experiments.agent_bench.environments \
  --platform "$TC_BENCH_ROOT" \
  --manifest "$TC_BENCH_ROOT/manifests/candidate-20260908-02/manifest.json" \
  --directory "$TC_BENCH_ROOT/environments/new-pilot/agent_offload" \
  --output "$TC_BENCH_ROOT/grading/new-pilot-agent-offload-preparation" \
  --setup-jobs 2 --subset pilot --prepare-only

# 确认后的首轮两组。SWE 需要先准备完各组的全部 20 题环境。
# bfcl/pilot 仅为四题功能验证，完整类别使用 bfcl/full。
"$TC_BENCH_CLIENT" -m tools.tokencake_experiments.agent_bench.experiment \
  --platform "$TC_BENCH_ROOT" \
  --manifest "$TC_BENCH_ROOT/manifests/confirmed-parallel-20260908-01/manifest.json" \
  --output "$TC_BENCH_ROOT/results/new-pilot-swe-first-pair" \
  --benchmark swe --subset pilot --modes agent_offload base \
  --environment-root "$TC_BENCH_ROOT/environments/candidate-20260908-03" \
  --budget-seconds 43200 \
  --budget-ledger "$TC_BENCH_ROOT/results/parallel-pilot-budget.jsonl"

# 在独立进程和评分目录中读取保存的实际提交。
"$TC_BENCH_CLIENT" -m tools.tokencake_experiments.agent_bench.score \
  --platform "$TC_BENCH_ROOT" \
  --experiment "$TC_BENCH_ROOT/results/new-pilot-swe-first-pair" \
  --environment-root "$TC_BENCH_ROOT/environments/pilot-grading-locale-20260908-01" \
  --output "$TC_BENCH_ROOT/grading/new-pilot-swe-first-pair"

"$TC_BENCH_CLIENT" -m tools.tokencake_experiments.agent_bench.report \
  --experiment "$TC_BENCH_ROOT/results/new-pilot-swe-first-pair" \
  --scores "$TC_BENCH_ROOT/grading/new-pilot-swe-first-pair/scores.json" \
  --output "$TC_BENCH_ROOT/results/new-pilot-swe-first-pair/report.json"
```

BFCL 与 SWE 的所有试跑命令使用同一个 budget ledger，共用 12 小时测量预算。
双 GPU 首对固定为 GPU 0 `base`、GPU 1 `agent_offload`，服务就绪后同步释放任务；每组最多 16 个应用。
重叠的测量时间只计一次。CPU 配额和 cgroup 内存由两组共享，报告显式标注其观测范围。
服务加载、环境准备和离线评分不计入模型测试预算。达到预算后停止新任务，记录取消与有界清理耗时。
清理过程最多为在途工具事件留出 110 秒；这部分真实时间仍保留，不通过删除超时任务美化结果。
程序异常中止、留下未闭合的计时段时，先依据日志核对预算，再恢复测量。

当前运行器提供候选试跑用的固定顺序闭环并发。它没有冻结正式 QPS、开放到达分布和三次重复矩阵，不能直接据此宣称正式压测完成。

## 结果解释

`terminal.json` 为每个计划任务的运行终态；`journal.jsonl` 记录每次实际模型尝试、响应、工具和事件。
`scores.json` 保留全部任务及错误类别；空提交与未提交不会算作成功。
`report.json` 同时给出完整任务分母的质量、含失败任务的 E2E、实际 token、重试、工具时间和按标签保存的服务计数差值。
无法观测的指标保留缺失状态；2 秒采样的峰值是采样峰值，不声称捕获瞬时最大值。

主指标为 15、30、60 分钟相同预算内的官方正确完成数，时钟从该组开始测量计起。
失败早退不能增加正确完成曲线；固定 20 题提前结束后保持最终结果，不自动加入新任务或外推吞吐。
双方共同成功题才进入配对 E2E，并同时保留单方成功和双方失败；共同成功集合为空时结果为不可用。
长度分组包括成功 HTTP 响应和输出长度未知的失败尝试；响应成功不代表对应任务通过评分。
非流式日志只记录完整 HTTP 请求耗时，按长度分组的 TTFT 和生成 TPOT 明确为不可用；服务端聚合计时只作诊断。
评分目录保存评分适配代码副本及哈希；补丁应用与冻结官方回退策略一致，重试前复位部分应用状态。

`output_throughput` 按用户指定方法提供三个口径：服务端共同前 1800 秒、服务端各自全程、客户端完整响应。
共同起点读取匹配实验路径和模式顺序的计时账本 start 事件；只选截止前最后一个有效采样，并记录采样滞后。
服务端分子为当前模型各 engine 的 `vllm:generation_tokens_total` 与初始值的差，不使用关闭前 `metrics.prom`。
每组单 GPU，分母为一次窗口墙钟；不会除以并发数或逐请求耗时总和。
客户端计数仅包含完整响应的 usage，服务端还可能包含已中断请求的生成量；两者均可能来自最终失败的任务。
报告保留采样缺失和逐 engine 计数器下降诊断；重置、缺失计时或未覆盖完整 30 分钟时，相应速率明确为不可用。
首对的新版分析为 `report-throughput-02.json`，原 `report.json` 保留；后续新运行的 `report.json` 直接包含新口径。

提高并发后的压力依据包括 GPU KV 使用率、等待队列、抢占/暂缓、CPU KV 传输计数，以及 cgroup 内存、OOM 事件和压力文件。
工具事件确认不等于发生 KV 保存；真实保存、CPU 命中与传输开销需要分别核对。
闭环运行保证任务顺序和并发规则相同，实际到达时刻仍会随各组任务完成速度变化。

## 检查

```bash
TC_AGENT_BENCH_ROOT="$TC_BENCH_ROOT" \
"$TC_BENCH_CLIENT" -m pytest \
  tests/tokencake/test_agent_benchmark_transport.py \
  tests/tokencake/test_agent_benchmark_execution.py \
  --confcutdir=tests/tokencake -q

TC_AGENT_BENCH_ROOT="$TC_BENCH_ROOT" \
BFCL_PROJECT_ROOT="$TC_BENCH_ROOT/tmp/bfcl-tests" \
MSWEA_SILENT_STARTUP=1 \
MSWEA_GLOBAL_CONFIG_DIR="$TC_BENCH_ROOT/cache/mini" \
PYTHONPATH="$TC_BENCH_ROOT/sources/gorilla/berkeley-function-call-leaderboard:$PWD" \
"$TC_BENCH_CLIENT" -m pytest \
  tests/tokencake/test_agent_benchmark_adapters.py \
  --confcutdir=tests/tokencake -q
```

适配测试中的脚本化响应仅用于验证官方提示、解析与真实工具状态的一致性，不参与 benchmark 评分或性能汇总。

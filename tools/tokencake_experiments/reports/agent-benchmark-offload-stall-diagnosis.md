# offload 停滞的调度器最小诊断

日期：2026-09-08。范围：冻结源码的无模型调度夹具，以及已完成试跑的只读日志。

已复现一种队头容量不足导致全部等待请求无法前进的状态。
`offload` 允许多个请求回载 KV 后持有 GPU 块；队头的后续输入所需空间不足时，调度循环提前结束，后面已完成回载且空间足够的请求也无法被调度。
相同夹具启用 agent 调度及 `generation_reserve_mode="all"` 后，容量预留使两个请求都能完成。
**这证明一个具体机制可以发生，尚未证明原试跑的 2277 秒停滞全部由它造成。**

## 1. 夹具构造与观察

使用当前真实 Scheduler / AsyncScheduler、TokenCake connector 和 CPU offload 元数据管理器。
通过现有测试辅助函数建立请求，只读取本地模型配置，不加载权重、不执行模型或 KV 数据拷贝。
CPU 前缀元数据预先置为就绪，worker 的回载完成通知和生成 token 由夹具提供。

| 参数 | 设置 |
| --- | --- |
| GPU KV 块 | 21 个，其中 1 个为 null block，20 个可用 |
| 每块 token | 16 |
| 请求 A | 256 个输入 token，CPU 中已有 128 个 token 的前缀 |
| 请求 B | 160 个输入 token，CPU 中已有 128 个 token 的前缀 |
| 生成上限 | 每个请求 16 token |
| 调度批 token 预算 | 512 |
| CPU KV 元数据容量 | 64 块 |
| 调度实现 | 分别测试同步及异步版本 |
| 通知时机 | 回载发布后本步完成，或延迟一个调度步完成 |

首轮 `offload` 同时为 A、B 各分配 8 个块用于回载，剩余 4 个。
收到两个请求的完成通知后，出现如下状态：

| 请求 | 已持有 GPU 块 | 后续输入还需新增块 | 此时的表现 |
| --- | ---: | ---: | --- |
| 队头 A | 8 | 8 | 完成回载后被提升为 WAITING，但分配失败 |
| 后续 B | 8 | 2 | 完成通知已收到，仍停在 WAITING_FOR_REMOTE_KVS，未获遍历与调度 |

连续最后 10 个调度步中，队列顺序、持有块、4 个空闲块和零输出保持不变，已没有模拟传输任务待完成。
运行共 40 步仍无请求完成；取消 A 后，B 在下一步立即获得 32 个 prefill token 的调度。
此处“40 步”是有限夹具的观测范围，不是墙钟时间或 benchmark 性能数据。

启用 agent 调度后，第一步只允许 A 回载，B 暂缓且不持有 GPU 块。
预留的容量允许 A 继续执行并释放资源，随后 B 执行；两者均按夹具输出达到 16 token 上限。

| 控制条件 | offload | agent_offload |
| --- | --- | --- |
| 等长输入：A、B 均为 256 token，同步调度、立即完成通知 | 两个请求均完成 | 两个请求均完成 |
| 不等长输入，同步调度、立即通知 | 40 步仍停滞；取消 A 后 B 获调度 | 两个请求均完成 |
| 不等长输入，同步调度、延迟一步通知 | 同上 | 两个请求均完成 |
| 不等长输入，异步调度、立即通知 | 同上 | 两个请求均完成 |
| 不等长输入，异步调度、延迟一步通知 | 同上 | 两个请求均完成 |

共核对 10 个配置实例，其中 4 个复现停滞；等长控制也完整保留。
上述完成指调度夹具按指定 token 序列正常结束，不代表任何 SWE 或 BFCL 题目答对。

## 2. 对应源码路径

1. 回载分支将 `num_new_tokens` 设为 0，仅为已有 CPU KV 分配 GPU 块，随后请求进入 `WAITING_FOR_REMOTE_KVS`。
2. 回载通知进入 `finished_recving_kv_req_ids`；等待队列在后续遍历时才提升请求状态。
3. 队头调用 `allocate_slots` 返回 None 后直接 `break`，该轮不会继续访问后面的 B。
4. 此时没有 RUNNING 请求，现有运行队列抢占路径不能释放 A、B 持有的块。
5. agent 调度在 CPU 查询和准入阶段计入已承诺容量，避免在本夹具中同时回载两者。

相关位置：

- [异步回载与新 token 数](/root/autodl-tmp/code/TokenCake/vLLM-TokenCake-upstream/vllm/v1/core/sched/scheduler.py:843)。
- [分配失败后的循环退出](/root/autodl-tmp/code/TokenCake/vLLM-TokenCake-upstream/vllm/v1/core/sched/scheduler.py:967)。
- [回载完成通知接收](/root/autodl-tmp/code/TokenCake/vLLM-TokenCake-upstream/vllm/v1/core/sched/scheduler.py:2447)。
- [完整输入容量检查](/root/autodl-tmp/code/TokenCake/vLLM-TokenCake-upstream/vllm/v1/core/kv_cache_manager.py:346)。
- [agent 调度的 CPU 查询前容量检查](/root/autodl-tmp/code/TokenCake/vLLM-TokenCake-upstream/vllm/tokencake/scheduling.py:575)。

原生 `scheduler_reserve_full_isl=True` 的检查也在夹具中保留。
它在每次分配时检查完整输入是否能装下，但不会为先前已开始回载的请求保留尚未分配的全部空间；所以不能单凭这个开关排除上述状态。
本次未修改被测调度逻辑。

## 3. 与实际试跑的关系及下一步

原 `offload` 服务日志确认启用了异步调度，所以同时验证了 AsyncScheduler。
实际试跑存在长时间运行数为零、等待数大于零、KV 占用不降、生成和已完成传输计数不增长，并在客户端取消后恢复的区间。
这些现象与本夹具相容；但聚合日志缺少每个请求的持有块、分配需求、队列顺序和完成通知集合，仍不足以确认它就是历史运行的根因。
真实 GPU 传输异常、通知交付异常及其他容量阻塞仍不能由夹具排除。

下一阶段可优先核对以下同一时刻的状态：

- 调度批次的运行数、waiting/skipped 顺序、队头 ID 和请求状态。
- 可分配 GPU 块数、队头持有块数与本次申请需求、分配是否返回 None。
- 后续回载请求的持有块、是否已收到完成通知，以及是否本可获得所需空间。
- 对应传输 job 的发布、worker 完成、scheduler 接收和请求状态提升记录。
- 取消的是哪个请求、释放了多少块、紧接着哪个请求恢复执行。

这些记录应只在状态变化及停滞诊断时采集，测量和报告其开销。
普通请求 OpenTelemetry span 可以补充逐请求 TTFT/TPOT，但不能独自提供上述分配与队列状态。
新增真实服务观察及原 20 题复跑仍按 [下一阶段方案](agent-benchmark-next-stage-proposal.md)确认；任何后续修复均应保留原版本结果，并另记源码身份。

## 4. 可复核证据

- [等长控制数据](/root/autodl-tmp/tokencake-agent-bench/results/offload-admission-fixture-20260908-01.json)与[源码](/root/autodl-tmp/tokencake-agent-bench/results/offload-admission-fixture-20260908-01.py)。
- [同步不等长数据](/root/autodl-tmp/tokencake-agent-bench/results/offload-admission-fixture-20260908-02.json)与[源码](/root/autodl-tmp/tokencake-agent-bench/results/offload-admission-fixture-20260908-02.py)。
- [异步不等长数据](/root/autodl-tmp/tokencake-agent-bench/results/offload-admission-fixture-20260908-03.json)与[源码](/root/autodl-tmp/tokencake-agent-bench/results/offload-admission-fixture-20260908-03.py)。
- [10 个配置实例断言核对、输入哈希及 9 份源码快照](/root/autodl-tmp/tokencake-agent-bench/results/offload-admission-fixture-verification-20260908-01/verification.json)。
- [原 offload 服务日志](/root/autodl-tmp/tokencake-agent-bench/results/pilot-swe-parallel-second-pair-01/offload/service/server.log:13)。

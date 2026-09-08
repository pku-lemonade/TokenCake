# Django 环境验证与试跑评分复核

日期：2026-09-08。本轮继续验证环境和复核已保存的模型预测，没有产生新的模型运行成绩。

范围更新：用户已要求试跑后停止并取消正式全量；本文保留停止前的验证记录，不继续准备环境。

新增 6 道 Django 题通过原始版本与参考修复验证，2 道仍无法满足冻结的原始测试预期。
试跑中的 4 道受 locale 初始化影响的题已在新环境中重新验证，四组共 16 份保存预测的评分状态与通过结果均未改变。
现有 SWE 20 题成绩仍为 `base` 0/20、`agent` 0/20、`offload` 0/20、`agent_offload` 1/20。

## 1. 评分环境适配修复

全量 500 份冻结脚本中，111 道 Django 题的 `eval.sh` 含有以下初始化，而其 Dockerfile 项目安装段未列出对应步骤：

```bash
sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen
export LANG=en_US.UTF-8
export LANGUAGE=en_US:en
export LC_ALL=en_US.UTF-8
```

此前适配只在 Dockerfile 安装段识别 locale 初始化，因此这些评分脚本仍调用宿主命令。
本机缺少 `locale-gen`，Python 3.6 的测试输出可能退回 ASCII；Django 10880、10914 在原始或参考版本中出现编码异常。

修复后，环境配方同时记录评分脚本中的 locale 需求及脚本 SHA256，准备时用 `localedef` 在任务的 `.venv/lib/locale` 下生成 `en_US.UTF-8`。
生成结果进入完整环境快照，评分通过 `LOCPATH` 使用它，保留官方 `LANG`、`LANGUAGE`、`LC_ALL` 导出。
本地评分脚本移除已识别的宿主初始化命令；缺少已准备的 locale 时明确报错，要求使用新环境。
未识别的初始化命令仍需明确适配，不直接执行。

静态检查确认全部 500 题的测试命令和补丁 heredoc 保持一致，locale 导出也均保留。
新增回归实际生成、保存和恢复 locale，执行 UTF-8 输出及非零退出的 shell 夹具，确认测试退出码和 heredoc 字面内容保持正确。
20 项执行及评分相关测试通过；测试命令与记录见 [回归 XML](/root/autodl-tmp/tokencake-agent-bench/results/evaluation-locale-tests-20260908-03.xml)。
实现见 [本地环境与评分适配](../agent_bench/swe_local.py)，全量静态证据见 [脚本核对](/root/autodl-tmp/tokencake-agent-bench/results/evaluation-locale-scope-audit-20260908-01.json)。

旧试跑和中断批次中的 `sed` 曾作用于 `/etc/locale.gen`；当前该文件的 `en_US.UTF-8` 行已启用，但 `locale -a` 仍只有 `C`、`C.utf8`、`POSIX`。
缺少这些执行之前的文件快照，不能把该行状态归因到某一批，也没有猜测并覆盖宿主原配置。
本轮核对记录了宿主文件哈希；修正后的评分使用任务本地 locale。

## 2. 新增任务的动态验证

这批任务来自冻结完整清单中接下来的 20 个未尝试 ID，使用原定 Python 3.6.13 和冻结的 40 项 Python 依赖。
环境准备并发为 2，与模型 benchmark 的每组 16 并发分别记录。

| 任务后缀，均为 django__django | F2P / P2P | 当前结果 | 参考版本解析结果 |
| --- | ---: | --- | --- |
| 10554 | 2 / 23 | 原始测试无法验证 | 25 通过、2 跳过 |
| 10880 | 1 / 55 | 验证通过 | 56 通过 |
| 10914 | 1 / 98 | 验证通过 | 99 通过、1 跳过 |
| 10973 | 5 / 0 | 验证通过 | 5 通过 |
| 10999 | 2 / 10 | 原始测试无法验证 | 12 通过 |
| 11066 | 1 / 3 | 验证通过 | 4 通过 |
| 11087 | 1 / 41 | 验证通过 | 42 通过、1 跳过 |
| 11095 | 1 / 19 | 验证通过 | 20 通过 |

10554 的原始日志在失败报告过程中出现 `DatabaseError`，相邻测试名称连接在同一行。
测试命令退出码为 1，但官方解析器未得到相应失败状态；冻结评分器的退出码一致性检查因此不接受该原始日志。
10999 的原始日志有 5 个子测试失败，但冻结解析器记录的短名称及连接输出不能与规定的完整 F2P 名称逐项对应。
两题参考版本均被官方规则判为 resolved，但仍不能证明原始版本的全部规定 F2P/P2P 表现。
没有更改官方匹配器、拆改测试名称或删除这些题。

| 准备批次 | 保存的终态 |
| --- | --- |
| 07，首次 20 题 | 3 验证通过、4 参考验证失败、13 取消；发现评分 locale 缺口后停止 |
| 08，修复后 20 题及 4 道试跑复核题 | 6 验证通过、2 参考验证失败、6 下载失败、10 取消 |
| 09，缓存可用的 4 道试跑题 | 4 验证通过 |

08 中连续 6 次 GitHub 仓库获取失败，日志包含端口 443 连接超时；独立 bare clone 也出现 TLS 断连，有时限的 HTTPS 检查未能连接。
因此停止剩余下载并保留终态，再通过已有指定提交缓存完成试跑复核；模型测量预算没有因此扣除或修改。
下载失败和取消状态都保留在 500 题清单中，不视为模型失败，也不视为验证通过。
证据见 [07 中断记录](/root/autodl-tmp/tokencake-agent-bench/grading/full-environment-validation-20260908-07/intervention.json)、[08 汇总](/root/autodl-tmp/tokencake-agent-bench/grading/full-environment-validation-20260908-08/summary.json)及[08 中断记录](/root/autodl-tmp/tokencake-agent-bench/grading/full-environment-validation-20260908-08/intervention.json)。

## 3. 对现有试跑评分的影响

`django__django-11211`、`11820`、`12419`、`13794` 使用新建的任务本地 locale 环境再次验证。
四题分别为 F2P/P2P 1/76、2/61、1/0、3/24，所有规定检查通过。
解释器、冻结依赖、原始及准备提交、源码复位、快照哈希和 UTF-8 运行环境核对通过。
其中 13794 的 `backports.zoneinfo` 与配方的 `backports-zoneinfo` 是同一规范化包名，版本均为 0.2.1；较早身份检查遗漏点号规范化产生的误报已单独保留并更正，没有更改安装版本。

在这些新评分环境中重新处理四组对应的 16 份保存预测：

| 处理结果 | 数量 |
| --- | ---: |
| 非空补丁进入测试，未解决 | 2 |
| 完整官方补丁应用策略仍失败 | 4 |
| 空补丁 | 3 |
| 生成阶段已失败 | 7 |

16 份预测的评分状态与 resolved 值均未改变；原始预测文件哈希保持一致，评分结束后恢复准备快照。
表中数量按复核 JSON 的 `after_status` 重新汇总，纠正此前将前两行均写为 3 的文档计数错误；原始评分没有变更。
其余题沿用已完成的评分；合并后四组完整的 20 题分母与唯一成功题均保持原结果。
这项复核没有重跑 agent，也不为历史请求补造 TTFT/TPOT。
见 [预测复核及合并评分](/root/autodl-tmp/tokencake-agent-bench/grading/pilot-swe-locale-recheck-20260908-01/summary.json)和[最终环境身份核对](/root/autodl-tmp/tokencake-agent-bench/grading/full-environment-identity-audit-20260908-05.json)。

## 4. 当前全量准备状态

| 状态 | 题数 |
| --- | ---: |
| validated | 46 |
| reference_validation_failed | 6 |
| environment_error | 6 |
| validation_cancelled | 6 |
| not_attempted | 436 |

仍保留全部 500 个 ID；`validated` 指规定的 F2P/P2P 验证通过，不表示所有被启动测试都通过，也不是模型成绩。
较早的 Astropy 和历史解释器验证问题继续保留。
最新索引见 [500 题准备进度](/root/autodl-tmp/tokencake-agent-bench/grading/full-environment-validation-progress-20260908-05.json)。
全量 BFCL/SWE 模型评测和正式矩阵未执行且已按用户要求取消，现有吞吐、质量与机制结论见 [试跑最终分析](agent-benchmark-pilot-final-analysis.md)。

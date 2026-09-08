# 全量 Astropy 环境验证

日期：2026-09-08。范围：已冻结 SWE Verified/test 清单中的 Astropy 任务环境，不包含模型请求。

本批继续验证剩余 17 道 Astropy 题，16 题通过原始 F2P 失败、P2P 通过及参考修复 resolved 的检查。
另一题 `astropy__astropy-14369` 保留为 `reference_validation_failed`，其冻结 P2P 测试名存在截断匹配歧义。
此前 5 道 Astropy 题已完成同样验证，因此该项目共 22 题中 21 题通过、1 题尚未通过验证。
全量 500 题当前为 **40 题通过、4 题验证失败、456 题尚未尝试**；完整集合未变，新增任务尚未产生模型成绩。

## 1. 本批任务与结果

| 任务后缀，均为 `astropy__astropy-` | Python | F2P / P2P 数量 | 最终验证 |
| --- | --- | ---: | --- |
| 13579 | 3.9.20 | 1 / 40 | 通过 |
| 13977 | 3.9.20 | 20 / 322 | 通过 |
| 14096 | 3.9.20 | 1 / 426 | 通过 |
| 14182 | 3.9.20 | 1 / 9 | 通过 |
| 14309 | 3.9.20 | 1 / 141 | 通过，保留首次网络失败 |
| 14365 | 3.9.20 | 1 / 8 | 通过，保留首次网络失败 |
| 14369 | 3.9.20 | 3 / 732 | 原始 P2P 截断名称匹配有歧义 |
| 14508 | 3.9.20 | 1 / 174 | 通过 |
| 14539 | 3.9.20 | 2 / 46 | 通过 |
| 14598 | 3.9.20 | 1 / 175 | 通过 |
| 14995 | 3.9.20 | 1 / 179 | 通过 |
| 7166 | 3.6.13 | 1 / 6 | 通过，使用冻结构建工具生成的依赖 wheel |
| 7336 | 3.6.13 | 1 / 339 | 同上 |
| 7606 | 3.6.13 | 1 / 240 | 同上 |
| 7671 | 3.6.13 | 1 / 3 | 同上 |
| 8707 | 3.9.20 | 1 / 11 | 通过，使用冻结 pip 安装旧式 editable 项目 |
| 8872 | 3.9.20 | 1 / 80 | 同上 |

“通过”对应规定的 F2P/P2P 检查，不表示脚本启动的所有测试都成功，也不证明本地系统库与官方容器逐项等价。
身份审计逐题核对解释器、冻结依赖版本、准备提交及其原始父提交、源码复位和快照 SHA256，已安装的本批环境均通过。
测试日志中评分集合之外的失败、原始失败和网络失败均有独立记录。

## 2. 兼容安装的具体处理

四道 Python 3.6 任务在 uv 安装依赖时失败：其隔离构建使用的 setuptools 不再提供旧 MarkupSafe 所需的 `Feature`。
在数据盘新建构建环境，使用此前已核验的 Miniconda Python 3.6.13，先安装原配方的 pip 21.2.2、setuptools 38.2.4、wheel 0.37.1。
通过 `uv run` 启动该环境的 `.venv/bin/python -m pip` 构建和安装原 44 项依赖，全部版本核对一致。
其中生成的 9 个 wheel 按文件 SHA256 归档，再通过仅作用于这批准备进程的 `UV_FIND_LINKS` 提供给 uv；没有替换包版本。
项目安装仍使用适配器原有的 Python 3.6 旧式 pip 路径。

两道 Python 3.9 旧项目没有 `pyproject.toml`，其原配方 setuptools 为 58.0.0、pip 为 24.2。
uv 隔离构建使用的新后端缺少 `pkg_resources`，因此在有记录的准备 worker 中复用旧式安装路径：由 `uv run` 调用任务解释器和已经安装的冻结 pip。
Python 版本与 64 项冻结依赖保持一致，执行 wrapper 与环境配置均在运行前保存；这不是对模型服务的修改。

旧项目还依赖 Git 子模块 `astropy_helpers`。
分别从六题原始提交读取 gitlink，已在本地缓存核对全部六个指定提交及其 tree；准备进程通过作用域内的 Git URL 配置获取缓存内容。
共享主机的全局 Git 配置没有改动，子模块仍检出原始任务规定的提交。

## 3. 原始失败与重试

| 批次 | 工作及保存状态 |
| --- | --- |
| `full-environment-validation-20260908-02` | 首次 17 题：8 题通过、1 题测试名歧义、2 题 GitHub 连接失败、6 题构建兼容失败 |
| `...-03` | 两题源码下载重试，均验证通过 |
| `...-04` | 四题使用冻结构建 wheel 重试：7336、7606 通过；7166 子模块网络失败；7671 在切换本地缓存前记录为取消 |
| `...-05` | 8707、8872 使用冻结 pip 与子模块缓存，均验证通过 |
| `...-06` | 7166、7671 使用相同依赖及指定子模块缓存，均验证通过 |

批次 04 在明确发生子模块 TLS/连接失败、且精确提交缓存已就绪后，由控制器有记录地中断。
清理与汇总期间已完成的两题成功结果保留，后续仅重试失败和取消的两题；没有覆盖旧目录，也没有把取消写成通过。
这些均属环境准备与参考测试，不计入模型试跑的共享测量预算。

## 4. 14369 的测试名歧义

冻结的 P2P 列表包含截断名称：

```text
astropy/units/tests/test_format.py::test_cds_grammar_fail[km
```

原始日志里以它开头的五个参数化测试中，四项通过；另一项 `test_cds_grammar_fail[km/s.Mpc-1]` 失败，且该完整名称正是冻结 F2P 项。
冻结官方 `_resolve_case` 在多个前缀匹配的结果不一致时返回不可确定，因此原始 P2P 检查不能通过。
参考修复后这五项全部通过，官方参考评分为 resolved，738 项已解析测试均通过。
这不是“原始日志没运行测试”，也不能仅凭参考修复通过就绕过原始 P2P 检查。
本次保留验证失败状态，未更改测试名称、匹配器或评分集合。

## 5. 证据与复现入口

- [500 题最新进度及全部 ID](/root/autodl-tmp/tokencake-agent-bench/grading/full-environment-validation-progress-20260908-03.json)。
- [首批与网络重试的身份/测试审计](/root/autodl-tmp/tokencake-agent-bench/grading/full-environment-identity-audit-20260908-02.json)及[兼容重试审计](/root/autodl-tmp/tokencake-agent-bench/grading/full-environment-identity-audit-20260908-03.json)。
- [17 题准备清单及源码快照](/root/autodl-tmp/tokencake-agent-bench/manifests/full-environment-preparation-20260908-02/preparation.json)。
- [冻结构建工具的安装验证](/root/autodl-tmp/tokencake-agent-bench/environments/legacy-build-probe-20260908-01/result.json)和[44 项依赖身份](/root/autodl-tmp/tokencake-agent-bench/environments/legacy-build-probe-20260908-01/identity-audit.json)。
- [构建 wheel 的来源与哈希](/root/autodl-tmp/tokencake-agent-bench/cache/frozen-legacy-wheels-20260908-01/identity.json)及[六题子模块提交](/root/autodl-tmp/tokencake-agent-bench/cache/astropy-helpers-20260908-01.json)。
- [批次 04 中断原因](/root/autodl-tmp/tokencake-agent-bench/grading/full-environment-validation-20260908-04/intervention.json)及[完整终态汇总](/root/autodl-tmp/tokencake-agent-bench/grading/full-environment-validation-20260908-04/summary.json)。
- [Python 3.9 旧式安装 worker](/root/autodl-tmp/tokencake-agent-bench/results/legacy-project-worker-20260908-01.py)及[运行前配置](/root/autodl-tmp/tokencake-agent-bench/manifests/full-environment-preparation-20260908-05/preparation.json)。
- [指定 wheel/子模块缓存的准备驱动](/root/autodl-tmp/tokencake-agent-bench/results/legacy-wheel-validation-20260908-02.py)及[运行前配置](/root/autodl-tmp/tokencake-agent-bench/manifests/full-environment-preparation-20260908-06/preparation.json)。

复现必须使用新输出和环境目录，核对上述缓存与源码身份，并保留所有失败记录。
上述驱动用于环境准备；正式模型运行仍需完成其余环境和四组工作目录的准备，并确认正式负载与预算。

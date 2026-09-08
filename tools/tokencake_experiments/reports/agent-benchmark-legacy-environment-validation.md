# SWE 历史环境兼容验证

日期：2026-09-08。对应 [实现计划](agent-benchmark-implementation-plan.md)和[实施状态](agent-benchmark-platform-status.md)。

正式 500 题中的三道 Python 2.7/3.5 任务已完成 Miniconda 安装，解释器、Conda 版本/build、官方 pip 条目、源码复位与快照哈希均核对通过。
但三题的原始版本均未满足冻结 F2P/P2P 列表的验证要求，状态保留为 `reference_validation_failed`。
这些是环境验证，未调用模型，不计入吞吐量、解决率或试跑测量预算；三题均不在已完成的固定 20 题中。

## 安装与验证结果

| 任务 | Python | Conda 精确包数 / 导出的 pip 条目数 | 原始版本 / 参考补丁的 resolved | 环境验证问题 |
| --- | --- | ---: | --- | --- |
| `django__django-10097` | 3.5.6 | 19 / 24 | true / true | 原始版本的 438 项 F2P 已全部通过；完整测试在参考补丁下仍有大量失败 |
| `django__django-7530` | 3.5.6 | 19 / 23 | true / true | 冻结 F2P 在原始版本已通过，实际错误的测试未列入评分集合 |
| `psf__requests-1724` | 2.7.18 | 44 / 1 | false / true | 六项 F2P 在原始版本全通过；真正由失败变通过的测试被列在 P2P 中 |

这里的 resolved 是冻结官方评分函数对指定测试集合的判定，不表示整个测试命令全部通过，也不是 agent 成绩。
原始版本指不应用模型或参考修复、按冻结脚本应用测试补丁后的版本。

`django__django-10097` 两次各运行 12,317 项测试。原始版本为 57 failure、253 error；参考补丁为 51 failure、253 error，两次均有 884 skipped、4 expected failure。
冻结列表中的 438 项 F2P 与 1,427 项 P2P 在两次运行均被判通过，因此原始版本也被判 resolved。
原始日志的错误包括 SQLite 表重命名相关错误、缺失模板及应用注册问题；尚未证明这些差异在官方容器环境中也会出现。

`django__django-7530` 两次各运行 63 项测试。原始版本有一个 error，参考补丁后全部通过。
出错的是 `MakeMigrationsTests.test_makemigrations_consistency_checks_respect_routers`，与测试补丁及参考修复对应。
冻结 F2P 却为 `SquashMigrationsTests.test_squashmigrations_initial_attribute`，它在两次运行都通过。

`psf__requests-1724` 原始版本为 87 passed、1 failed，参考补丁为 88 passed。
实际由失败变通过的是 `RequestsTestCase.test_unicode_method_name`，冻结列表将其放在 P2P。
本次执行保留了测试原有的 HTTPBin 地址，没有替换服务或改变断言。

## 数据来源核对

使用冻结 SWE-bench 的 `swebench.task.repo.load_task` 重新读取全部 500 题，与确认清单逐字段比较：任务 ID、repo、base commit、version、问题文本、修复补丁、测试补丁、评分脚本、F2P、P2P 全部相等。
官方 task 检出仍为 `3d07b464b7b311a0cbfb5ed5b2d8a3b96f84a33d`，任务文件无跟踪修改；确认数据文件哈希保持不变。
这排除了适配层抄录这些字段时发生错位，不能单独证明本地环境与官方镜像等价，也不能据此擅自修正冻结测试分类。

证据：[500 题官方加载器一致性核对](/root/autodl-tmp/tokencake-agent-bench/manifests/official-task-loader-audit-20260908-01.json)。

## 兼容实现与实际修复

- 显式选择服务器 Miniconda，在数据盘建立独立前缀；保留冻结 Python 和 Conda 包版本/build，不交给 uv 管理 Python 2.7/3.5。
- 使用任务 `.venv/bin/python -m pip` 及原有 pip 版本安装导出的依赖与项目。
- Conda 管理进程清空任务 `PYTHONPATH`，避免其自身 Python 导入历史 Requests 源码；任务执行仍使用自己的源码路径。
- 将任务前缀的 `lib/pkgconfig` 放入构建查找路径，使 CFFI 使用已经安装的 libffi 头文件；未换依赖版本。
- 用主机现有 glibc 输入将 `en_US.UTF-8` 编译到任务虚拟环境，通过 `LOCPATH` 使用并纳入快照；不修改共享主机的 locale 配置。
- 从已有干净检出建立校验过的指定提交缓存，解决重试中的 GitHub TLS fetch 失败；新环境仍拥有独立源码目录。
- 依赖冻结在任务目录执行并为旧 pip 加 `--all`，避免误收录实验仓库的本地包元数据，同时记录 pip、setuptools 和 wheel。

准备尝试 01 的 CFFI 和 Conda 导入失败、尝试 02 的 Git 网络失败均保留。
最终评分使用尝试 02 的 Django 7530 与尝试 03 的另两题，各验证目录保存当次适配代码副本。
依赖冻结的目录与 `--all` 修正发生在上述准备之后；最新审计在相同任务目录重新只读采集完整依赖，原始 `installed.txt` 保留。
早期审计 01 将默认 pip freeze 不展示的安装器和 Conda 命名空间包误判为缺失，已由审计 02 更正；环境和评分数据均未因此修改。

## 证据与进入正式评测的条件

- [最终安装身份及原始/参考评分审计](/root/autodl-tmp/tokencake-agent-bench/grading/legacy-compat-audit-20260908-02.json)，附实际解释器、全部包核对、快照及报告哈希。
- [Django 7530 验证](/root/autodl-tmp/tokencake-agent-bench/grading/legacy-compat-validation-02/django__django-7530/validation.json)。
- [Django 10097 验证](/root/autodl-tmp/tokencake-agent-bench/grading/legacy-compat-validation-03/django__django-10097/validation.json)。
- [Requests 1724 验证](/root/autodl-tmp/tokencake-agent-bench/grading/legacy-compat-validation-03/psf__requests-1724/validation.json)。

原始版本的目标失败、P2P 通过和参考修复判定必须一起核实；不能只凭安装完成或参考补丁的 resolved 标记为环境有效。
三题继续保留在冻结的 500 题名单中，尚未修改评分分类、放宽验证或删除任务。
全量运行前仍需查明本地测试表现与冻结列表的差异，并确认处理方案；本轮不将这些验证结果替换为模型失败或模型成功。

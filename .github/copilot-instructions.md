## 快速指南 — 给 AI 编程代理的实用提示

下面是能让你快速在本仓库中高效工作的要点。请聚焦可执行改动（修复 bug、改进模块、补充测试），并在提交前运行测试与格式化。只记录可从代码/配置中直接发现的约定与示例。

- 项目总体：这是 PatchCore 异常检测实现（论文: Roth et al. 2021）。核心实现位于 `src/patchcore/`，入口脚本在 `bin/`。训练/评估面向 MVTec AD 数据集。

- 关键文件（引用示例）
  - 入口/运行：`bin/run_patchcore.py`（训练）、`bin/load_and_evaluate_patchcore.py`（加载并评估）
  - 核心实现：`src/patchcore/patchcore.py`（PatchCore class、PatchMaker）、`src/patchcore/backbones.py`（backbone loader）、`src/patchcore/sampler.py`（coreset / sampler 实现）
  - 工具与公用：`src/patchcore/common.py`, `src/patchcore/utils.py`, `src/patchcore/metrics.py`
  - 测试：`test/` 下为 pytest 测试用例（例如 `test/test_patchcore.py` 展示了最小化运行/保存/加载行为的示例）。

- 运行与开发工作流（可复制的命令）
  - 安装依赖：`pip install -r requirements.txt`（开发依赖见 `requirements_dev.txt`）或 `pip install -e .`（在虚拟环境中）
  - 本地单元测试：`pytest -q`（或使用 tox：`tox -e py38`）
  - 快速训练示例（来自 README）：
    ```bash
    env PYTHONPATH=src python bin/run_patchcore.py results \ 
      patch_core -b wideresnet50 -le layer2 -le layer3 sampler -p 0.1 approx_greedy_coreset \ 
      dataset --resize 256 --imagesize 224 -d bottle mvtec /path/to/mvtec
    ```
  - 评估已保存模型（来自 README）：
    ```bash
    python bin/load_and_evaluate_patchcore.py --gpu 0 --seed 0 outdir \ 
      patch_core_loader -p /path/to/model_folder/models/mvtec_bottle dataset -d bottle mvtec /path/to/mvtec
    ```

- 项目约定与陷阱（对自动改动很重要）
  - GPU 设备管理：`bin/*` 中显式使用 torch.cuda.device 上下文管理器来避免“GPU 内存泄漏”问题；在修改运行逻辑时保留设备上下文管理（或理解其副作用）。
  - Backbones 注册：`src/patchcore/backbones.py` 使用字符串到表达式的映射并通过 `eval` 构建模型；新增 backbone 时只需在该字典添加条目（注意安全性与依赖）。
  - 模型保存格式：PatchCore 将参数写为 `patchcore_params.pkl`，且 nearest-neighbour 索引为 `.faiss` 文件，路径与命名在 `models/.../` 下（见 README 中的示例结构）。如果调整保存/加载，请同步更新 `save_to_path`/`load_from_path` 与 `bin/load_and_evaluate_patchcore.py`。
  - 采样器（sampler）与非确定性：`ApproximateGreedyCoresetSampler` 中使用随机起点，会影响可复现性；测试中通常用固定随机 seed（见 tests）。

- 要点：修改核心接口时要同时更新
  - `bin/` 中的 glue-code（参数传递与选项）
  - 对应单元测试（`test/test_patchcore.py`）以覆盖训练/预测/保存/加载流程
  - `requirements.txt` 或 `requirements_dev.txt`（若引入新依赖）

- 开发风格与工具链
  - 格式化/静态检查：项目提供 `tox.ini`，包含 `black` 和 `flake8` 检查（`tox -e black` / `tox -e flake8`）。保持与现有 style 一致。
  - 包安装：`setup.py` 与 `pyproject.toml` 均存在；`setup.py` 会根据 `requirements.txt` 填充 `install_requires`。

- 快速示例（当你要实现/修改某个功能）
  1. 在 `src/patchcore/` 中实现功能并添加类型提示（仓库包含 `py.typed`）。
  2. 在 `test/test_patchcore.py` 添加或修改最小单元测试（参考已有 dummy dataloader helpers）。
  3. 运行 `pytest -q`，修复失败后提交。

若这些说明有遗漏（例如需要更多命令示例，或更细粒度的模块调用序列图），告诉我你希望增加的部分，我会迭代更新本文件。

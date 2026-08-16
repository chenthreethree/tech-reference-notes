# 项目总览

## 1. 项目到底解决什么问题

项目研究的是电商账号关系风险识别：一个账号自身的订单、退款等行为可能不明显，但多个账号可能通过设备、IP、地址、订单形成可疑关系。项目把业务事件转换为“账号—实体”异构图，用 GraphSAGE 聚合邻域信息并输出账号风险概率，同时保留 HGB 作为非图神经网络基线。

当前仓库是研究原型，不是生产反欺诈系统：

- 数据是与 通用电商 事件字段兼容的受控离线生成数据，不是真实业务数据，也不是当前代码实时调用 通用电商 生成的数据（`v2_strict/generate.py:156-177`、`final_stage/holdout.py:309-356`、`README.md:71-75`）。
- GraphSAGE 输出账号概率，不直接输出“真实团伙”；高风险关联组件由预测结果和有效关系做后处理得到（`gnn_platform/final_service.py:169-248`）。
- 系统只提供复核线索，不自动封号、不修改订单（`README.md:37-39`、`README.md:115-117`）。

## 2. 整体架构

```text
受控事件生成 / 外部CSV
        │
        ▼
事件标准化（events.jsonl）
        │
        ├── 7维行为特征 ──────────────┐
        ├── 7维账号图统计特征 ── HGB基线 ── 风险概率
        │                            │
        └── 账号/设备/IP/地址/订单图   │
                    │                │
                    ▼                │
              25维节点特征            │
                    │                │
              GraphSAGE 3×64 ────────┘
                    │
                    ▼
        冻结阈值 → 账号风险预测
                    │
        关系图、边遮蔽、关联组件
                    │
                    ▼
                 人工复核
```

核心图构建在 `v2_strict/graph.py::build_graph_split`；模型定义在 `model_stage/models.py`；训练在 `model_stage/run.py::run_seed`；Final Holdout 一次性评价在 `final_stage/evaluate.py::evaluate_once`。

## 3. 数据从哪里来

### 3.1 V2 开发数据

`v2_strict/generate.py::generate_dataset` 直接在 Python 中生成 通用电商 风格事件，划分为：

- train：768账号；
- validation：192账号；
- calibration：192账号；
- 四个开发测试场景：各96账号，共384账号；
- public NAT 压力测试：410账号，独立于主指标。

数值来自 `data/v2_strict/dataset_summary.json`。每个普通 Actor 有6个账号；风险/正常身份在事件生成前分配，标签只写入 `labels.csv`（`v2_strict/generate.py:105-146`、`275-325`）。

### 3.2 Final Holdout

`final_stage/holdout.py::generate_holdout` 生成1,500账号，其中120风险、1,380正常，风险比例8%；共有6,650条事件。数据在模型架构选择前封存并做实体重叠、ID模式、图组件和哈希审计（`final_stage/seal.py::seal_holdout`）。

### 3.3 数据接入端 上传数据

工作台接受 UTF-8 CSV，最大20MB、10万行（`workbench/engine.py:35-47`、`88-108`）。它支持字段映射、质量检查、构图和冻结模型兼容性推理。上传数据没有真实标签时，不能计算准确率；工作台创建的全0占位标签仅为复用 `GraphSplit` 数据结构，不参与推理（`workbench/engine.py:331-367`）。

## 4. 输入与输出

### 离线训练输入

- `events.jsonl`：事件类型、账号、设备、IP、订单、时间、地址摘要、支付/退款/优惠信息；
- `labels.csv`：`user_id` 与 `is_abuse`；
- `actors.csv`：仅用于场景/角色评价与隔离审计，不是模型特征；
- `manifest.json`：数据版本、编码表和生成边界。

### 模型输入

- HGB-behavior：7维行为特征；
- HGB-structure-only：5维简单结构统计；
- HGB-behavior+graph-stat：14维账号统计特征；
- GraphSAGE：25维全节点特征 + 双向加权边。具体字段见 `v2_strict/graph.py:15-55`。

### 模型输出

- 每个账号一个0到1的风险概率；
- 使用独立 calibration 集确定的固定阈值得到0/1预测；
- Final Holdout 评价输出混淆矩阵、Precision、Recall、F1、ROC-AUC、PR-AUC、FPR、Recall@FPR；
- 网页输出局部关系、边遮蔽敏感度、候选关联组件和人工复核记录。

## 5. 从数据到结果的完整流程

1. 生成数据：`v2_strict/generate.py::generate_dataset` 或 `final_stage/holdout.py::generate_holdout`。
2. 隔离审计：`v2_strict/audit.py::run_integrity_audit` 检查跨划分实体、跨Actor关系、模式化ID和禁用字段。
3. 图与特征：`v2_strict/graph.py::build_graph_split` 读取事件，生成账号特征、节点特征、关系边和审计信息。
4. 训练 HGB：`model_stage/run.py:320-357` 对不同特征集合调用 `HistGradientBoostingClassifier.fit`。
5. 训练 GNN：`model_stage/run.py::train_gnn` 使用加权 BCE、AdamW 和 validation checkpoint。
6. 定阈值：`v2_strict/train.py::best_threshold` 在独立 calibration 集候选分位数上最大化F1，以Precision破平局。
7. 有限选型：比较2层/3层 GraphSAGE、关系感知GraphSAGE和R-GCN；`model_stage/select.py::select` 冻结 GraphSAGE 3×64、seed 46及阈值。
8. 一次性评价：`final_stage/evaluate.py::evaluate_once` 验证模型/数据哈希，在封存 Final Holdout 上评价一次并写完成标记。
9. 只读展示端展示：`gnn_platform/final_service.py::FinalResearchService` 读取封存结果；常规页面不重新训练。
10. 数据接入端兼容推理：`workbench/engine.py::score_dataset` 对上传事件构图，加载冻结 GraphSAGE/HGB，输出线索；人工结论写 SQLite，不反馈模型。

## 6. 模型与算法清单

| 方法 | 项目角色 | 真实实现 |
|---|---|---|
| HGB-behavior | 只看7维行为特征的基线 | `model_stage/run.py::_hgb`、`hgb_features` |
| HGB-structure-only | 判断简单度数统计能否解释GNN优势 | 同上，输入5维 `STRUCTURE_ONLY_FEATURES` |
| HGB-behavior+graph-stat | 最强传统基线，输入14维 | 同上，输入 `ACCOUNT_FEATURES` |
| GraphSAGE 2×64 | 基础GNN候选 | `model_stage/models.py::GraphSAGENet` |
| GraphSAGE 3×64 | 最终冻结研究模型 | 同上，3个 `SageLayer` |
| Relation-aware GraphSAGE | 每类关系独立变换和门控的对照 | `RelationAwareGraphSAGE` |
| R-GCN | 每类关系独立线性变换的对照 | `RGCN` |
| 边遮蔽 | 删除单边后重新推理的局部敏感度 | `gnn_platform/final_service.py`、`workbench/engine.py::account_detail` |
| 高风险关联组件 | 有效实体连通 + 至少2个高风险种子 | `FinalResearchService::_build_groups` |

Isolation Forest、XGBoost、逻辑回归、SVM、大语言模型/大语言模型 调用：未在当前仓库代码中找到。

## 7. 模块之间如何连接

| 模块 | 上游 | 下游 |
|---|---|---|
| `v2_strict` | 生成配置 | `model_stage`、审计、测试 |
| `model_stage` | V2的train/validation/calibration/test | 冻结模型、选型表 |
| `final_stage` | 已封存Holdout + 模型锁 | 最终指标与预测CSV |
| `gnn_platform` | Final Holdout、最终指标、模型锁 | 只读展示端只读研究网页 |
| `workbench` | 用户CSV + 冻结模型 | 数据接入端检测、关系证据、SQLite复核审计 |

## 8. 最终输出与使用方式

- 最终研究模型：`models/model_stage/seed46/graphsage_3x64.pt`；
- 模型锁：`results/model_stage/final_model_lock.json`；
- 一次性结果：`results/final_holdout/final_metrics.json`；
- 逐账号预测：`results/final_holdout/predictions.csv`；
- 只读展示端只读展示：`bash scripts/gnnctl.sh start`；
- 数据接入端上传/复核：`bash scripts/workbenchctl.sh start`。


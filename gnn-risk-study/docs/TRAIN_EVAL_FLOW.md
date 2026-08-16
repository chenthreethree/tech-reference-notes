# 训练与评估流程

## 0. 先区分三套数据

| 数据 | 用途 | 规模 | 代码/文件 |
|---|---|---:|---|
| V2 strict | 训练、验证、定阈值、开发选型 | train 768；val 192；cal 192；开发test 384 | `v2_strict/generate.py::generate_dataset`、`data/v2_strict/dataset_summary.json` |
| Public NAT | 高连接IP压力测试，不进主指标 | 410账号 | `v2_strict/generate.py::_generate_public_nat` |
| Final Holdout | 冻结后一次性最终评价 | 1,500账号，风险120（8%） | `final_stage/holdout.py::generate_holdout` |

V2的四个“test”后来已用于选架构，因此是开发测试集，不是最终无偏测试；Final Holdout 才是最终一次性评价。证据：`README.md:75`、`final_stage/evaluate.py:145-169`。

## 1. 数据来源与生成

### 1.1 V2

`v2_strict/generate.py::generate_dataset` 使用固定主seed `20260807`，通过 `OpaqueAllocator` 生成不透明ID和全局唯一私网IP。每对Actor包括一个正常Actor和一个风险Actor，二者使用相同账号数、拓扑边和实体类型序列，但角色与行为组合不同（`_generate_split:446-488`）。

正常画像有 family、campus、corporate、groupbuy、dormitory、parcel_address、device_churn、reasonable_refund；风险难度有 simple、complex、weak、hard（`v2_strict/generate.py:27-46`）。

### 1.2 标签

标签在事件生成前由 `_generate_actor(..., is_abuse=...)` 确定：正常为0、风险为1，写入 `labels.csv`；事件JSON不含标签（`v2_strict/generate.py:296-325`）。`actors.csv`保存Actor、场景、角色，仅用于评价/审计。

这不是人工后验标注，也不是真实执法/损失标签，而是生成器已知身份的实验真值。

### 1.3 Final Holdout

`final_stage/holdout.py::generate_holdout` 生成：342个公共网络正常账号、173×6个困难正常账号、20×6个风险账号，合计1,500，风险比例8%（`final_stage/holdout.py:325-344`）。

## 2. 数据划分协议

```text
train：拟合HGB参数和GNN权重
validation：仅选择GNN checkpoint / early stopping
calibration：训练结束后选择分类阈值
V2四场景test：开发期模型比较
Final Holdout：冻结模型后一次性最终评价
```

实现证据：`model_stage/run.py:415-428`、`model_stage/select.py:117-155`、`final_stage/guard.py:9-26`。

实体隔离由 `v2_strict/audit.py::run_integrity_audit` 和 `final_stage/seal.py::seal_holdout` 检查；已有结果 `results/v2_strict/integrity_audit.json` 与 `results/final_holdout/integrity_audit.json` 均记录通过。

## 3. 数据预处理与特征

入口：`v2_strict/graph.py::build_graph_split`。

### 3.1 7维行为特征

1. 注册到领券秒数；
2. 领券到下单秒数；
3. 事件数；
4. 订单数；
5. 优惠订单数；
6. 退款数；
7. 退款比例。

计算代码：`v2_strict/graph.py:215-233`。

### 3.2 7维账号图统计

1. 最大设备关联账号数；
2. 最大IP关联账号数；
3. 最大地址关联账号数；
4. 核销时间桶关联账号数（主实验不启用核销边时恒为1）；
5. 共享实体类型数；
6. 实体关系数；
7. 订单实体数。

计算代码：`v2_strict/graph.py:234-243`。

### 3.3 25维GNN节点特征

14维账号特征 + 6维节点类型one-hot + 5维节点元数据：加权度、超高连接比例、订单金额log、优惠比例、是否退款。非账号节点的14维账号部分为0，订单节点通过元数据承载金额/优惠/退款（`v2_strict/graph.py:311-331`）。

训练集均值和标准差用于所有split标准化；标准差过小设为1（`model_stage/run.py:152-157`）。

## 4. 图结构

节点：account、device、ip、address、order；代码也定义 redemption 类型，但主实验 `include_redemption=False`，所以主图无核销节点。

边：account-device、account-ip、account-address、account-order、order-address。构图时生成双向边（`v2_strict/graph.py:262-310`）。模型层不使用方向差异。

高连接降权：基础权重为 `1/sqrt(max(1, degree-1))`；超过类型上限后再除以 `degree/limit`（`v2_strict/graph.py::_relation_weight`）。公共优惠码原文不会直接成为节点（`v2_strict/graph.py:278-295`）。

## 5. 模型输入

| 模型 | 输入 |
|---|---|
| HGB-behavior | `data.behavior_features`，7维 |
| HGB-structure-only | `data.structure_features`，5维 |
| HGB-behavior+graph-stat | `data.account_features`，14维 |
| 所有GNN | 同一25维节点特征、同一device/IP/address/order关系 |

证据：`model_stage/run.py::hgb_features`、`tests/test_model_stage.py:29-40`。

## 6. 模型初始化与超参数

### HGB

- learning_rate=0.08；
- max_iter=250；
- max_leaf_nodes=15；
- l2_regularization=1.0；
- random_state=seed。

位置：`model_stage/run.py::_hgb`。

### GraphSAGE

- 候选：2层或3层；最终3层；
- 每层hidden=64；
- dropout=0.2；
- 聚合：边权加权均值；
- 激活：ReLU；
- 输出：Linear(64,1)产生logit；
- 无LayerNorm。

位置：`model_stage/models.py::SageLayer`、`GraphSAGENet`。

## 7. 损失函数与优化

`train_gnn` 只对前 `train.account_count` 个账号节点计算损失；实体节点参与消息传播但没有分类标签。损失是：

```text
BCEWithLogitsLoss(pos_weight = max(1, 正样本数/负样本数的倒数))
```

准确说代码为 `negatives / positives`，用于类别不平衡补偿。优化器是 AdamW，学习率0.01，weight_decay=1e-3（`model_stage/run.py:158-165`）。

## 8. 训练与checkpoint

每个epoch：清梯度 → 整图前向 → 账号节点BCE → 反向 → AdamW更新。第10轮后每5轮在validation上评分；validation内部临时找最佳F1阈值，只用于比较checkpoint。连续60轮无改进提前停止（`model_stage/run.py:171-203`）。默认最多300 epoch、5个seed 42～46（`model_stage/run.py:583-595`）。

注意：validation临时阈值不是最终阈值；最终阈值来自独立calibration。

## 9. 阈值选择

`v2_strict/train.py::best_threshold`：对calibration分数的2%到98%分位构造97个候选，选择F1最大者；F1相同时Precision高者优先。最终3×64 seed46阈值为 `0.28877802670002023`，来源写入model payload和model lock。

阈值是在模型训练完成后选择的，Final Holdout不参与。

## 10. 模型保存

### HGB `.joblib`

保存sklearn模型、阈值、阈值来源、特征名和模型阶段版本（`model_stage/run.py:329-336`）。

### GNN `.pt`

保存state_dict、model_name、阈值、25维特征名、关系名、训练均值/标准差、版本和训练元数据（`model_stage/run.py:372-385`）。

最终冻结文件：`models/model_stage/seed46/graphsage_3x64.pt`；SHA-256记录于 `results/model_stage/final_model_lock.json`。

## 11. 模型选择

只比较有限四种GNN。`model_stage/select.py::select` 的优先级是PR-AUC和Recall@FPR=5%、弱关联召回、组织共享网络与校园网络误报、稳定性和成本。满足相对2×64约束后选择PR-AUC最高者，得到3×64。

代表seed不是挑最好结果，而是选七项开发指标到五seed均值标准化距离最近者，得到seed46（`model_stage/select.py::_representative_seed`）。

## 12. 推理

`model_stage/run.py::graph_scores`：使用训练集均值方差标准化目标图，整图前向，取前N个账号logit，sigmoid变成概率。概率 `>= threshold` 判为高风险。

数据接入端的 `workbench/engine.py::_score_graph` 同时加载冻结GNN和HGB，对新上传图执行兼容性推理，不重新训练。

## 13. 指标公式与代码

实现：`v2_strict/train.py::metrics`。

- TP：风险且预测风险；FP：正常但预测风险；TN：正常且预测正常；FN：风险但预测正常；
- Precision = TP/(TP+FP)：报警中有多少是真的；
- Recall = TP/(TP+FN)：真实风险找回多少；
- F1 = Precision与Recall调和平均；
- FPR = FP/(FP+TN)：正常账号被误报比例；
- ROC-AUC：跨所有阈值的TPR/FPR排序能力；
- PR-AUC：Precision-Recall曲线面积，低基率风险任务更值得关注；
- Recall@FPR=1%/5%：限制误报率后的最大召回，属于排序诊断，不替换冻结阈值。

## 14. Final Holdout一次性评价

`final_stage/evaluate.py::evaluate_once`：

1. 要求显式flag；
2. 检查模型锁；
3. 检查完成marker不存在；
4. 验证Holdout哈希；
5. 验证最终模型artifact哈希；
6. 加载两个HGB和2×64/3×64；
7. 使用各自V2 calibration阈值评价；
8. 保存结果与逐账号预测；
9. 写 `EVALUATION_COMPLETE.json`，阻止重跑。

最终文件：

- `results/final_holdout/final_metrics.json`；
- `results/final_holdout/predictions.csv`；
- `results/final_holdout/EVALUATION_COMPLETE.json`。

## 15. 可复现边界

数据生成、划分、模型选择和最终评价均保留明确的配置与完整性校验。公开材料只描述方法边界，不包含本地环境和外部历史数据依赖。

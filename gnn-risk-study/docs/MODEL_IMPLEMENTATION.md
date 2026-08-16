# 模型实现

## 1. HGB-behavior

- 为什么需要：建立“只看账号行为、不看关系结构”的传统机器学习基线。
- 输入：7维 `BEHAVIOR_FEATURES`（`v2_strict/graph.py:16-24`）。
- 输出：`predict_proba(... )[:,1]` 风险概率。
- 训练目标：用0/1实验标签做二分类；sklearn内部优化细节由库实现，仓库只配置超参数。
- 代码：`model_stage/run.py::_hgb`、`hgb_features`、`run_seed:320-357`。
- 超参数：learning_rate 0.08、250 iterations、15 leaf nodes、L2=1.0。
- 实际调用：V2 train拟合、calibration定阈值、开发test比较；Final Holdout中作为对照（`final_stage/evaluate.py::_load_hgb`）。
- 与其他模型关系：不使用设备/IP/地址共享统计，是最弱语义基线。
- 为什么不是其他替代方案：仓库没有形成随机森林、XGBoost、逻辑回归等对照，因此只能说“选择HGB作为本项目基线”，不能说代码证明HGB优于所有传统模型。

## 2. HGB-structure-only

- 为什么需要：排查GraphSAGE是否只学会了简单度数捷径。
- 输入：5维 `STRUCTURE_ONLY_FEATURES`：非核销实体数、最大设备/IP/地址共享数、两跳账号数（`v2_strict/graph.py:43-49`）。
- 输出/目标/超参数：与其他HGB相同。
- 代码：`model_stage/run.py::hgb_features`。
- 实际结果：V2上ROC-AUC/PR-AUC都是0.50，阈值0.5时全判风险，F1 66.67%（`results/v2_strict/main_no_redemption/aggregate.json`）。
- 意义：说明当前V2的GraphSAGE优势不能仅由这5个粗结构统计解释；不代表所有图统计模型都无效。

## 3. HGB-behavior+graph-stat

- 为什么需要：这是最强传统基线，让HGB同时看7维行为和7维人工图统计。
- 输入：14维 `ACCOUNT_FEATURES`（`v2_strict/graph.py:16-34`）。
- 输出：账号风险概率。
- 实际调用：训练与保存同上；数据接入端把它作为辅助概率（`workbench/engine.py::_model_payloads`、`_score_graph`）。
- 与GraphSAGE差别：它知道“最大共享设备账号数”等汇总数值，但看不到每个具体邻居是谁、邻居还连接谁以及多轮关系组合；GraphSAGE在完整二部图上传播消息。
- 局限：14维中不包含关系排列，且人工特征可能丢信息；Final Holdout上PR-AUC 47.12%略高于3×64的46.45%，说明它仍有价值。

## 4. GraphSAGE 2×64

- 为什么需要：V2最初冻结的基础GNN；用于验证两轮邻域聚合是否优于HGB。
- 输入：25维节点特征、双向加权边；关系集合device/IP/address/order。
- 输出：每个节点logit，评价时只取账号节点；sigmoid为风险概率。
- 训练目标：账号节点二分类，`BCEWithLogitsLoss`带类别权重。
- 代码：`model_stage/models.py::SageLayer`、`GraphSAGENet`；`make_model("graphsage_2x64")`。
- 超参数：2层、hidden64、dropout0.2、ReLU、AdamW(lr=0.01, weight_decay=1e-3)、最多300 epoch。
- 实际调用：V2、Final Holdout对照；不是最终选定模型。
- “2跳”准确解释：账号—设备是1条边，设备—另一个账号是第2条边；两层聚合最多传播两条图边，不等于“经过两个账号”。

## 5. GraphSAGE 3×64（最终冻结研究模型）

- 为什么需要：在相同输入、训练流程和5个seed下，开发集的PR-AUC、F1、低FPR召回及公司误报等综合表现优于2×64和两个关系模型。
- 输入/输出/训练目标：与2×64相同。
- 代码：`model_stage/models.py::GraphSAGENet`，`make_model`传 `layers=3`。
- 关键超参数：3层，每层64维，dropout0.2；参数20,033；最终seed46；阈值0.288778。
- 模型文件：`models/model_stage/seed46/graphsage_3x64.pt`；锁在 `results/model_stage/final_model_lock.json`。
- 实际调用：只读展示端展示冻结Final预测和真实边遮蔽；数据接入端对上传图推理。
- 3×64是什么意思：连续3个SAGE层，每层输出64维节点表示。64维是模型学习的潜在表示，不是64个人工业务字段。
- 为何不是更深：仓库只有限比较到3层；没有4层/更多层结果。不能声称3层理论最优，只能说它在预先限定候选中最好。

## 6. Relation-aware GraphSAGE

- 为什么需要：验证把设备/IP/地址/订单分别建参数是否比普通GraphSAGE更好。
- 输入：同一25维节点输入和同一图，但使用 `edge_type`。
- 实现：每层每种关系独立Linear，并学习4个softmax标量门控（`model_stage/models.py::RelationAwareSageLayer`）。
- 输出/目标：账号二分类概率，训练流程同普通GNN。
- 超参数：2层64维、dropout0.2；29,193参数。
- 结果关系：开发集F1 79.62%，低于3×64的83.60%；未选中。
- 局限：标量门控很简化，不代表所有关系感知GNN都不如普通GraphSAGE。

## 7. R-GCN

- 为什么需要：作为经典多关系图卷积对照。
- 输入：同一节点特征和edge_type。
- 实现：每类关系独立无bias线性变换，按总加权度归一后累加自身项（`model_stage/models.py::RGCNLayer`）。
- 输出/目标：同上。
- 超参数：2层64维、dropout0.2；28,801参数。
- 结果关系：开发集F1 80.86%，低于3×64；未选中。
- 不能外推：这是仓库内的紧凑自实现R-GCN，不是完整工业R-GCN调优结论。

## 8. 高风险关联组件（不是模型）

- 目的：把高风险账号按真实共享实体整理为人工调查候选。
- 输入：GraphSAGE高风险预测、device/IP/address实体；不使用标签、actor_id、risk_events。
- 算法：过滤度数超限实体；在所有账号上做连通分量；保留至少2个GNN高风险账号的组件（`gnn_platform/final_service.py::_build_groups`）。
- 输出：组件成员、支撑实体、核心账号、组件分数。
- 边界：不是GraphSAGE直接输出的团伙，更不是真实团伙标签。

## 9. 边遮蔽（不是GNNExplainer）

- 目的：解释某个账号对哪些已存在边敏感。
- 过程：删除一条关系边，重新构图/推理，记录原概率、遮蔽后概率和差值（`workbench/engine.py::account_detail:677-697`）。
- 边界：是局部敏感度，不是因果证明；仓库未使用PyG的GNNExplainer。


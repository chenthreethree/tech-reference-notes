# 结果与限制

## 1. 最终可对外说明的核心结果

### Final Holdout（最重要）

配置：1,500账号，120风险、1,380正常，风险比例8%；冻结seed46和V2 calibration阈值后只评价一次。

| 模型 | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | TP/FP/TN/FN | 结果文件 | 计算代码 |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| HGB-behavior | 11.51% | 67.50% | 19.66% | 74.15% | 43.27% | 45.14% | 81/623/757/39 | `results/final_holdout/final_metrics.json` | `final_stage/evaluate.py` + `v2_strict/train.py::metrics` |
| HGB-behavior+graph-stat | 11.70% | 74.17% | 20.20% | 77.40% | 47.12% | 48.70% | 89/672/708/31 | 同上 | 同上 |
| GraphSAGE 2×64 | 25.19% | 85.00% | 38.86% | 88.07% | 45.44% | 21.96% | 102/303/1077/18 | 同上 | 同上 |
| GraphSAGE 3×64 | 32.38% | 85.00% | 46.90% | 92.77% | 46.45% | 15.43% | 102/213/1167/18 | 同上 | 同上 |

相对HGB行为+图统计，3×64：F1 +26.69个百分点，Recall +10.83个百分点，FPR -33.26个百分点，但PR-AUC -0.67个百分点。代码直接生成该差值（`final_stage/evaluate.py:160-165`）。

这是技术交流时最应该说的结果，因为它来自低基率、封存、一次性Holdout。必须同时说FPR仍为15.43%，离生产要求较远。

## 2. V2严格隔离复现

配置：主测试384账号、风险比例50%，5个seed 42～46；GraphSAGE有随机性，HGB在当前实现和数据下五seed结果相同。

| 模型 | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | 结果文件 |
|---|---:|---:|---:|---:|---:|---:|---|
| HGB-behavior | 72.96% | 74.48% | 73.71% | 81.46% | 81.89% | 27.60% | `results/v2_strict/main_no_redemption/aggregate.json` |
| HGB-structure-only | 50.00% | 100.00% | 66.67% | 50.00% | 50.00% | 100.00% | 同上 |
| HGB-behavior+graph-stat | 71.78% | 75.52% | 73.60% | 80.86% | 82.40% | 29.69% | 同上 |
| GraphSAGE 2×64 | 77.69% | 80.83% | 79.18%±1.34% | 89.47% | 90.49% | 23.33% | 同上 |

GraphSAGE 2×64相对HGB行为+图统计F1 +5.57个百分点；相对简单结构HGB +12.51个百分点。来源：`results/v2_strict/v1_v2_comparison.json`。

核销关系消融中2×64 F1 79.64%，仅比主图高0.46个百分点，不能据此证明核销边重要（`results/v2_strict/redemption_ablation/aggregate.json`）。

## 3. 架构比较（开发数据，不是最终测试）

| 模型 | F1 mean±std | PR-AUC mean | FPR mean | 参数量 | CPU ms/账号 | 结果文件 |
|---|---:|---:|---:|---:|---:|---|
| GraphSAGE 2×64 | 79.18%±1.34% | 90.49% | 23.33% | 11,713 | 0.0083 | `results/model_stage/architecture/aggregate.json` |
| GraphSAGE 3×64 | 83.60%±0.93% | 94.63% | 14.90% | 20,033 | 0.0127 | 同上 |
| Relation-aware GraphSAGE | 79.62%±1.97% | 90.66% | 19.69% | 29,193 | 0.0248 | 同上 |
| R-GCN | 80.86%±0.96% | 91.14% | 20.94% | 28,801 | 0.0221 | 同上 |

这张表用于解释为什么冻结3×64，不能把83.60%当成项目最终F1。

## 4. 关系消融

| 图关系配置 | F1 mean | PR-AUC mean | FPR mean | 结果文件 |
|---|---:|---:|---:|---|
| 完整图 | 83.60% | 94.63% | 14.90% | `results/model_stage/relation_ablation/aggregate.json` |
| 去设备边 | 78.36% | 89.60% | 26.88% | 同上 |
| 去IP边 | 78.85% | 90.40% | 27.40% | 同上 |
| 去地址边 | 78.03% | 90.55% | 22.19% | 同上 |
| 去订单边 | 83.59% | 94.50% | 19.17% | 同上 |

设备/IP/地址边删除后性能明显下降；订单边删除后F1几乎不变但FPR变差。`without_time_decay` 明确是“不适用”，因为主图没有时间衰减，不能虚构时间消融。

## 5. Final Holdout 场景结果

GraphSAGE 3×64风险召回：simple 100.00%、complex 83.33%、hard 93.33%、weak 63.33%。主要正常误报：same_product_cluster 66.67%、corporate 40.74%、dormitory 27.45%。来源：`results/final_holdout/final_metrics.json` 的 `scenario_evaluation`。

这说明模型最难的是弱关联风险和正常同商品聚集/组织共享网络关系。

## 6. 数据完整性结果

| 检查 | 结果 | 证据 |
|---|---|---|
| V2跨split customer/device/IP/address/order/payment/refund/actor交集 | 0 | `results/v2_strict/integrity_audit.json` |
| V2非预期跨Actor实体 | 0 | 同上 |
| V2模式化ID | 0 | 同上 |
| 标签/risk_events进入模型输入 | False | 同上、`tests/test_v2_strict.py` |
| Final与V1/V2实体重叠 | 全部0 | `results/final_holdout/integrity_audit.json` |
| Final节点/无向边/组件 | 10,329 / 12,290 / 196 | 同上 |
| Final重复评价 | marker禁止 | `results/final_holdout/EVALUATION_COMPLETE.json`、`final_stage/guard.py` |

## 7. 公开结论与限制

- 所有指标来自受控离线生成数据，不能替代真实业务外部验证。
- 最终模型只是在有限候选中的最佳研究模型，不代表整个模型空间的最优方案。
- 固定阈值下，GNN 的 F1、召回率和误报率优于当前 HGB 基线，但 PR-AUC 略低，不能声称所有指标全面领先。
- 弱关联风险召回和复杂正常共享关系误报仍是主要问题。
- 当前系统定位为研究原型和人工复核辅助，不执行自动处置。
- 后续工作应优先增加脱敏业务数据验证、概率校准、低误报约束和更完整的传统模型基线。

完整流程
Workflow 1   fetch_reports.yml        研报下载+倍数提取      已有，不动
Workflow 3   extract_data.yml         传图→Claude读数→JSON   新增
Workflow 2   validate_data.yml        读JSON→链路层校验       新增
Workflow 4   run_valuation.yml        读JSON→Claude估值       新增
Workflow 5   push_fangtang.py         推送微信               已有，不动

调试时的重跑策略：
改 prompt → 只重跑 Workflow 4
数字读错 → 重跑 Workflow 3，再重跑 4
研报倍数变了 → 重跑 Workflow 1，再重跑 4
校验规则变了 → 重跑 Workflow 2（零 token），再重跑 4

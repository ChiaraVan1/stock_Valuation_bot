"""
validate_data.py — Workflow 2
读取 workflow 3 输出的 {code}_data.json，
自动填充 BalanceSheet 参数，运行链路层校验。
零 token 消耗，纯本地 Python。
"""

import argparse
import json
import sys
from pathlib import Path

from validation import (
    BalanceSheet, Scenarios,
    validate_balance_sheet, assert_scenario_monotonic, ScenarioMonotonicError,
)

OUTPUT_DIR = Path("output")


# ────────────────────────────────────────────────────────────
# 从 JSON 构建 BalanceSheet
# ────────────────────────────────────────────────────────────

def build_balance_sheet(data: dict) -> BalanceSheet:
    bs = data.get("balance_sheet", {})
    return BalanceSheet(
        cash=bs.get("cash") or 0.0,
        cash_prior=bs.get("cash_prior"),
        trading_assets=bs.get("trading_assets") or 0.0,
        notes_receivable=bs.get("notes_receivable") or 0.0,
        accounts_receivable=bs.get("accounts_receivable") or 0.0,
        inventory=bs.get("inventory") or 0.0,
        current_due_noncurrent=bs.get("current_due_noncurrent") or 0.0,
        other_current=bs.get("other_current") or 0.0,
        long_term_equity=bs.get("long_term_equity") or 0.0,
        other_equity_instruments=bs.get("other_equity_instruments") or 0.0,
        investment_property=bs.get("investment_property") or 0.0,
        fixed_assets=bs.get("fixed_assets") or 0.0,
        construction_in_progress=bs.get("construction_in_progress") or 0.0,
        right_of_use=bs.get("right_of_use") or 0.0,
        intangibles=bs.get("intangibles") or 0.0,
        goodwill=bs.get("goodwill") or 0.0,
        other_noncurrent=bs.get("other_noncurrent") or 0.0,
        total_assets=bs.get("total_assets") or 0.0,
        total_liabilities=bs.get("total_liabilities") or 0.0,
        minority_interest=bs.get("minority_interest") or 0.0,
        total_equity=bs.get("total_equity") or 0.0,
        # 有息负债合计（各科目之和）
        interest_bearing_debt=(
            (bs.get("short_term_loans") or 0.0)
            + (bs.get("current_portion_lt_debt") or 0.0)
            + (bs.get("long_term_loans") or 0.0)
            + (bs.get("bonds_payable") or 0.0)
            + (bs.get("lease_liabilities") or 0.0)
        ),
    )


# ────────────────────────────────────────────────────────────
# 从 JSON 构建三情景（如果 profit_history 足够）
# ────────────────────────────────────────────────────────────

def build_scenarios(data: dict) -> Scenarios | None:
    history = data.get("profit_history", [])
    if len(history) < 3:
        return None

    profits = sorted(history, key=lambda x: x["year"])
    values = [p["net_profit"] for p in profits]

    # 悲观：近 3 年均值
    bearish = sum(values[-3:]) / 3
    # 中性：全周期均值
    neutral = sum(values) / len(values)
    # 乐观：最高 3 年均值
    top3 = sorted(values, reverse=True)[:3]
    optimistic = sum(top3) / 3

    return Scenarios(
        bearish=round(bearish, 2),
        neutral=round(neutral, 2),
        optimistic=round(optimistic, 2),
        bearish_note=f"近3年均值（{profits[-3]['year']}—{profits[-1]['year']}）",
        neutral_note=f"全周期{len(values)}年均值（{profits[0]['year']}—{profits[-1]['year']}）",
        optimistic_note=f"历史最高3年均值",
    )


# ────────────────────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Workflow 2 — 链路层校验")
    parser.add_argument("--code", required=True, help="股票代码，如 002223")
    args = parser.parse_args()

    json_path = OUTPUT_DIR / f"{args.code}_data.json"
    if not json_path.exists():
        print(f"❌ 未找到数据文件: {json_path}")
        print("  请先运行 Workflow 3（extract_data.py）生成 JSON")
        sys.exit(1)

    data = json.loads(json_path.read_text(encoding="utf-8"))
    meta = data.get("meta", {})
    print(f"\n🔍 Workflow 2 — 链路层校验 | {meta.get('stock_name')}（{args.code}）{meta.get('report_year')}年报")
    print("=" * 60)

    passed_all = True
    report_lines = []

    # ── 1. 资产负债表校验 ──────────────────────────────────
    print("\n【1】资产负债表合理性 + 货币资金骤降检测")
    bs_obj = build_balance_sheet(data)
    result = validate_balance_sheet(bs_obj)

    if result.errors:
        passed_all = False
        print("  ❌ 阻断性错误：")
        for e in result.errors:
            print(f"    {e}")
            report_lines.append(f"[ERROR] {e}")

    if result.warnings:
        print("  ⚠️  警告：")
        for w in result.warnings:
            # 只打印前150字，完整内容写入报告
            print(f"    {w[:150]}{'...' if len(w) > 150 else ''}")
            report_lines.append(f"[WARNING] {w}")

    if result.cash_merged is not None:
        print(f"\n  → 现金类资产合并结果: {result.cash_merged:.2f} 亿")
        print(f"  → {result.cash_merge_detail.split(chr(10))[0]}")
        report_lines.append(f"[CASH_MERGED] {result.cash_merged:.2f}")

    # 读取 Workflow 3 Claude 的骤降判断（与链路层校验交叉比对）
    bs_raw = data.get("balance_sheet", {})
    drop_confirmed = bs_raw.get("cash_drop_confirmed")
    drop_note      = bs_raw.get("cash_drop_note", "")
    if drop_confirmed is True:
        print(f"\n  ℹ️  Workflow 3 Claude 判断：货币资金骤降确认为定存重分类")
        print(f"     {drop_note}")
        if result.cash_merged is None:
            print(f"  ⚠️  链路层未触发骤降规则，但 Claude 判断存在定存重分类，请人工核查")
            report_lines.append(f"[CASH_DROP_MISMATCH] Claude确认但链路层未触发，需人工核查")
    elif drop_confirmed is False:
        print(f"\n  ℹ️  Workflow 3 Claude 判断：货币资金骤降为真实流出（非定存重分类）")
        print(f"     {drop_note}")
        report_lines.append(f"[CASH_DROP_REAL_OUTFLOW] {drop_note}")

    if result.passed and not result.warnings:
        print("  ✅ 全部通过")

    # ── 2. 异常科目检查 ────────────────────────────────────
    abnormal = data.get("abnormal_items", [])
    if abnormal:
        print(f"\n【2】异常科目（Claude 提取时已标记 {len(abnormal)} 项）")
        for item in abnormal:
            print(f"  ⚠️  {item['item']}: {item.get('prior')} → {item.get('current')} 亿"
                  f"（{item.get('yoy', 0)*100:.0f}%）— {item.get('note', '')}")
            report_lines.append(f"[ABNORMAL] {item['item']}: {item.get('note', '')}")
    else:
        print("\n【2】异常科目：无")

    # ── 3. 三情景单调性预检 ────────────────────────────────
    print("\n【3】三情景单调性预检")
    scenarios = build_scenarios(data)
    if scenarios:
        try:
            assert_scenario_monotonic(scenarios)
            print(f"  ✅ 通过（悲观{scenarios.bearish} < 中性{scenarios.neutral} < 乐观{scenarios.optimistic}）")
            print(f"     {scenarios.bearish_note} / {scenarios.neutral_note}")
            report_lines.append(f"[SCENARIOS_OK] {scenarios.bearish} / {scenarios.neutral} / {scenarios.optimistic}")
        except ScenarioMonotonicError as e:
            passed_all = False
            print(f"  ❌ 单调性违反：{str(e)[:100]}")
            report_lines.append(f"[SCENARIOS_ERROR] {str(e)}")
    else:
        print("  ⚠️  利润历史数据不足（<3年），跳过预检")
        report_lines.append("[SCENARIOS_SKIP] 利润历史数据不足")

    # ── 4. 分红率完整性检查 ────────────────────────────────
    print("\n【4】分红记录完整性")
    dividends = data.get("dividend_history", [])
    profits = data.get("profit_history", [])
    if dividends and profits:
        div_years = {d["year"] for d in dividends}
        profit_years = {p["year"] for p in profits}
        missing = profit_years - div_years
        if missing:
            print(f"  ⚠️  以下年份有利润但缺分红记录: {sorted(missing)}")
            report_lines.append(f"[DIV_MISSING] {sorted(missing)}")
        else:
            print(f"  ✅ 分红记录覆盖所有利润年份（{len(dividends)}年）")
    else:
        print("  ⚠️  分红或利润数据为空，跳过")

    # ── 输出校验报告 ───────────────────────────────────────
    print("\n" + "=" * 60)
    report_path = OUTPUT_DIR / f"{args.code}_validation_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"校验报告已保存至: {report_path}")

    if not passed_all:
        print("\n❌ 校验未通过，建议修正后再运行 Workflow 4")
        sys.exit(1)
    else:
        print("\n✅ 校验通过，可继续运行 Workflow 4")


if __name__ == "__main__":
    main()

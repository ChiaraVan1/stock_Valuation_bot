"""
validation.py  —  链路层数值校验模块
挂载点：在 valuation.py 调用 Claude 估值之前执行
包含：
  1. validate_balance_sheet()   资产负债表合理性 + 货币资金骤降检测
  2. assert_scenario_monotonic() 三情景单调性断言
"""

from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ── 数据结构 ─────────────────────────────────────────────────

@dataclass
class BalanceSheet:
    """
    从 Claude 视觉识别结果或手工输入中结构化的资产负债表关键科目。
    单位：亿元。None 表示未找到该科目。
    """
    # 流动资产
    cash: float                          # 货币资金
    trading_assets: float = 0.0          # 交易性金融资产
    other_current: float = 0.0           # 其他流动资产（含结构性存款）
    notes_receivable: float = 0.0        # 应收票据
    accounts_receivable: float = 0.0     # 应收账款
    inventory: float = 0.0               # 存货
    current_due_noncurrent: float = 0.0  # 一年内到期非流动资产

    # 非流动资产
    long_term_equity: float = 0.0        # 长期股权投资
    other_equity_instruments: float = 0.0# 其他权益工具投资
    investment_property: float = 0.0     # 投资性房地产
    fixed_assets: float = 0.0            # 固定资产
    construction_in_progress: float = 0.0# 在建工程
    right_of_use: float = 0.0            # 使用权资产
    intangibles: float = 0.0             # 无形资产
    goodwill: float = 0.0                # 商誉
    other_noncurrent: float = 0.0        # 其他非流动资产（含长期定存）

    # 合计与权益
    total_assets: float = 0.0            # 资产总计
    total_liabilities: float = 0.0       # 负债合计
    minority_interest: float = 0.0       # 少数股东权益
    total_equity: float = 0.0            # 股东权益合计

    # 上期对比（用于骤降检测）
    cash_prior: Optional[float] = None   # 上期货币资金

    # 有息负债（用于净现金计算）
    interest_bearing_debt: float = 0.0


@dataclass
class ValidationResult:
    passed: bool = True
    warnings: list = field(default_factory=list)   # 警告：不阻断，但须在报告中标注
    errors: list = field(default_factory=list)     # 错误：阻断估值输出
    cash_merged: Optional[float] = None            # 合并后的现金类资产（触发骤降规则时赋值）
    cash_merge_detail: str = ""                    # 合并说明，供报告直接引用


# ── 1. 资产负债表校验 ─────────────────────────────────────────

# 合理性规则阈值（可按行业调整）
_RULES = {
    "fixed_assets_ratio_max": 0.60,   # 固定资产 / 总资产 上限
    "cash_ratio_min": 0.05,           # 货币资金 / 总资产 下限（极端情况警告）
    "cash_ratio_max": 0.80,           # 货币资金 / 总资产 上限（含定存合并后）
    "cash_drop_threshold": 0.30,      # 货币资金骤降触发阈值（下降比例）
    "goodwill_ratio_warn": 0.15,      # 商誉 / 总资产 超过此值发出警告
    "single_item_max_ratio": 1.0,     # 任何单科目不超过总资产（恒等式）
}

# 定期存款折算率
_DEPOSIT_DISCOUNT = 0.90


def validate_balance_sheet(bs: BalanceSheet) -> ValidationResult:
    """
    对资产负债表做三类检查：
      A. 数量级合理性（固定资产占比、单科目上限）
      B. 货币资金骤降识别 + 自动合并定期存款
      C. 商誉风险预警

    返回 ValidationResult，调用方根据 .passed 决定是否继续估值流程。
    严重错误（passed=False）应阻断输出，警告（warnings）写入报告标注。
    """
    result = ValidationResult()

    if bs.total_assets <= 0:
        result.errors.append("total_assets 为零或负数，无法进行校验，请重新提取数据。")
        result.passed = False
        return result

    ta = bs.total_assets

    # ── A. 数量级合理性 ──────────────────────────────────────
    checks = [
        ("固定资产",          bs.fixed_assets,          _RULES["fixed_assets_ratio_max"]),
        ("货币资金",          bs.cash,                   _RULES["single_item_max_ratio"]),
        ("存货",              bs.inventory,              _RULES["single_item_max_ratio"]),
        ("应收账款",          bs.accounts_receivable,    _RULES["single_item_max_ratio"]),
        ("商誉",              bs.goodwill,               _RULES["single_item_max_ratio"]),
        ("其他非流动资产",    bs.other_noncurrent,       _RULES["single_item_max_ratio"]),
    ]
    for name, val, limit in checks:
        ratio = val / ta
        if ratio > limit:
            result.errors.append(
                f"[数量级异常] {name} {val:.2f}亿 占总资产 {ratio:.1%}，"
                f"超过上限 {limit:.0%}，请核查单位换算是否有误。"
            )
            result.passed = False

    # 固定资产专项
    fa_ratio = bs.fixed_assets / ta
    if fa_ratio > _RULES["fixed_assets_ratio_max"]:
        # 已在上面 checks 里覆盖，此处不重复
        pass

    # ── B. 货币资金骤降检测 ──────────────────────────────────
    if bs.cash_prior is not None and bs.cash_prior > 0:
        drop_ratio = (bs.cash_prior - bs.cash) / bs.cash_prior
        if drop_ratio > _RULES["cash_drop_threshold"]:
            # 触发骤降规则，尝试合并定期存款
            #   候选承接科目：other_noncurrent（长期定存）、other_current（结构性存款）、
            #                  current_due_noncurrent（一年内到期非流动资产）
            noncurrent_deposit = bs.other_noncurrent   # 视为长期定存，折 90%
            current_deposit    = bs.other_current       # 视为结构性存款，折 90%
            short_deposit      = bs.current_due_noncurrent  # 折 100%

            merged_cash = (
                bs.cash * 1.00
                + short_deposit * 1.00
                + current_deposit * _DEPOSIT_DISCOUNT
                + noncurrent_deposit * _DEPOSIT_DISCOUNT
                + bs.trading_assets * 1.00
            )

            detail = (
                f"货币资金较上期下降 {drop_ratio:.1%}（{bs.cash_prior:.2f}亿→{bs.cash:.2f}亿），"
                f"触发骤降识别规则。\n"
                f"  合并口径（现金类资产）= "
                f"货币资金 {bs.cash:.2f} × 100%"
                f" + 一年内到期非流动资产 {short_deposit:.2f} × 100%"
                f" + 其他流动资产（结构性存款）{current_deposit:.2f} × {_DEPOSIT_DISCOUNT:.0%}"
                f" + 其他非流动资产（长期定存）{noncurrent_deposit:.2f} × {_DEPOSIT_DISCOUNT:.0%}"
                f" + 交易性金融资产 {bs.trading_assets:.2f} × 100%"
                f" = {merged_cash:.2f}亿\n"
                f"  ⚠️ 两轨净现金均须使用合并口径，不得单独使用货币资金科目。"
            )
            result.cash_merged = merged_cash
            result.cash_merge_detail = detail
            result.warnings.append(f"[货币资金骤降] {detail}")
            logger.warning(detail)
    else:
        # 无上期数据时，仍检查货币资金占比是否异常低
        cash_ratio = bs.cash / ta
        if cash_ratio < _RULES["cash_ratio_min"]:
            result.warnings.append(
                f"[现金偏低] 货币资金 {bs.cash:.2f}亿 仅占总资产 {cash_ratio:.1%}，"
                f"建议人工核查是否存在大额定期存款未被识别。"
            )

    # ── C. 商誉风险 ──────────────────────────────────────────
    goodwill_ratio = bs.goodwill / ta
    if goodwill_ratio > _RULES["goodwill_ratio_warn"]:
        result.warnings.append(
            f"[商誉风险] 商誉 {bs.goodwill:.2f}亿 占总资产 {goodwill_ratio:.1%}，"
            f"超过 {_RULES['goodwill_ratio_warn']:.0%} 警戒线，"
            f"须在报告中测算全额减值对每股净资产的影响。"
        )

    return result


# ── 2. 三情景单调性断言 ───────────────────────────────────────

@dataclass
class Scenarios:
    bearish: float    # 悲观盈利力（亿元）
    neutral: float    # 中性盈利力
    optimistic: float # 乐观盈利力
    # 可选：每个情景的取值依据说明，用于报告
    bearish_note: str = ""
    neutral_note: str = ""
    optimistic_note: str = ""


class ScenarioMonotonicError(ValueError):
    """三情景不满足单调性时抛出，阻断估值输出"""
    pass


def assert_scenario_monotonic(s: Scenarios) -> None:
    """
    断言 悲观 < 中性 < 乐观。
    违反时抛出 ScenarioMonotonicError，调用方须捕获并触发重新核算，
    不得在情景排列不单调的情况下继续输出估值结果。

    使用方式：
        try:
            assert_scenario_monotonic(scenarios)
        except ScenarioMonotonicError as e:
            # 记录错误，要求 Claude 重新给出情景数值
            raise
    """
    errors = []
    if s.bearish >= s.neutral:
        errors.append(
            f"悲观值 {s.bearish:.2f} ≥ 中性值 {s.neutral:.2f}：\n"
            f"  → 悲观应取「近3年均值」与「当前低谷连续年份均值」中更低者，\n"
            f"    中性应为全周期均值，不得高于或等于乐观值。"
        )
    if s.neutral >= s.optimistic:
        errors.append(
            f"中性值 {s.neutral:.2f} ≥ 乐观值 {s.optimistic:.2f}：\n"
            f"  → 乐观应为峰值区间均值，必须是三者最高值。"
        )
    if errors:
        msg = "【三情景单调性校验失败，估值输出已阻断】\n" + "\n".join(errors)
        logger.error(msg)
        raise ScenarioMonotonicError(msg)


# ── 快速测试（直接运行此文件时执行）────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # --- 测试 A：正常情况 ---
    print("=== 测试 A：鱼跃医疗 2025年报（正常路径）===")
    bs = BalanceSheet(
        cash=5.79, cash_prior=66.19,
        trading_assets=0.0,
        other_current=6.72,
        current_due_noncurrent=2.62,
        other_noncurrent=60.14,
        accounts_receivable=7.09, notes_receivable=0.09,
        inventory=14.78,
        long_term_equity=21.39,
        fixed_assets=18.45,
        goodwill=10.77,
        intangibles=4.60,
        total_assets=156.13,
        total_liabilities=27.29,
        minority_interest=1.14,
        total_equity=128.84,
        interest_bearing_debt=0.14,
    )
    r = validate_balance_sheet(bs)
    print(f"passed={r.passed}")
    for w in r.warnings:
        print(f"  WARNING: {w[:120]}...")
    if r.cash_merged:
        print(f"  合并现金类资产: {r.cash_merged:.2f}亿")
    print()

    # --- 测试 B：单位换算错误（固定资产过大）---
    print("=== 测试 B：单位换算错误模拟 ===")
    bs_bad = BalanceSheet(
        cash=5.79, total_assets=156.13,
        fixed_assets=184.5,  # 漏掉÷10，数量级异常
    )
    r2 = validate_balance_sheet(bs_bad)
    print(f"passed={r2.passed}")
    for e in r2.errors:
        print(f"  ERROR: {e}")
    print()

    # --- 测试 C：三情景单调性 ---
    print("=== 测试 C：三情景单调性 ===")
    # 正常
    try:
        assert_scenario_monotonic(Scenarios(16.44, 17.53, 19.96))
        print("  正常情景：通过 ✓")
    except ScenarioMonotonicError as e:
        print(f"  意外失败：{e}")

    # 违反（悲观 > 中性）
    try:
        assert_scenario_monotonic(Scenarios(18.0, 17.53, 19.96))
        print("  异常情景：未被拦截（错误！）")
    except ScenarioMonotonicError as e:
        print(f"  异常情景正确拦截 ✓\n  {str(e)[:100]}...")

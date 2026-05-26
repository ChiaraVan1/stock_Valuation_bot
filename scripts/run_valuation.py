"""
run_valuation.py — Workflow 4
读取 workflow 3 输出的 {code}_data.json 和 workflow 1 输出的研报倍数，
以纯文字方式调用 Claude 做双轨估值，不传图，大幅节省 token。
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from ai_client import CLAUDE_API_KEY, CLAUDE_BASE_URL, CLAUDE_MODEL

OUTPUT_DIR     = Path("output")
QUARTERLY_DIR  = Path("quarterly_reports")
MANUAL_RPT_DIR = Path("manual_reports")

# ────────────────────────────────────────────────────────────
# 复用 valuation.py 中完整的 VALUATION_PROMPT
# （直接 import，保持单一来源）
# ────────────────────────────────────────────────────────────

from valuation import VALUATION_PROMPT


# ────────────────────────────────────────────────────────────
# 读取辅助文件
# ────────────────────────────────────────────────────────────

def load_report_md(code: str) -> str:
    path = OUTPUT_DIR / f"{code}_valuation.md"
    if not path.exists():
        return "[fallback-用户跳过] 未找到研报倍数文件"
    content = path.read_text(encoding="utf-8")
    print(f"已读取研报倍数: {path}（{len(content)}字符）")
    return content


def load_validation_report(code: str) -> str:
    path = OUTPUT_DIR / f"{code}_validation_report.txt"
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8")
    print(f"已读取校验报告: {path}")
    return content


def load_quarterly_text(code: str) -> str:
    """读取季报 PDF 文字（不传图，只用文字）"""
    import pdfplumber
    q_dir = QUARTERLY_DIR / code
    if not q_dir.exists():
        return ""

    pdf_files = list(q_dir.glob("*.pdf"))
    if not pdf_files:
        return ""

    parts = []
    for pdf_path in pdf_files:
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages[:20]:
                    text += page.extract_text() or ""
            parts.append(f"=== 季报: {pdf_path.name} ===\n{text[:8000]}")
        except Exception as e:
            print(f"  ⚠️  季报提取失败: {e}")

    if not parts:
        return ""

    return """---

## 📋 季报参考层（不替代年报，仅用于验证假设和催化剂判断）

> - 可用于验证：收入增速、存货变化、应收账款质量、现金流健康度
> - 可用于判断：新产品认证/海外准入/重大合同等催化剂
> - 不可从季报获取：分板块收入和毛利率
> - 不可从季报调整：PE/PEG 倍数的精确数字

""" + "\n\n".join(parts) + "\n---"


def load_manual_reports(code: str) -> str:
    import pdfplumber
    rpt_dir = MANUAL_RPT_DIR / code
    if not rpt_dir.exists():
        return ""
    pdf_files = list(rpt_dir.glob("*.pdf"))
    if not pdf_files:
        return ""
    parts = []
    for pdf_path in pdf_files:
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages[:30]:
                    text += page.extract_text() or ""
            parts.append(f"=== 手动研报: {pdf_path.name} ===\n{text[:12000]}")
        except Exception:
            pass
    return "\n\n".join(parts)


# ────────────────────────────────────────────────────────────
# 构建发送给 Claude 的文字 prompt
# ────────────────────────────────────────────────────────────

def build_user_prompt(
    data: dict,
    report_table: str,
    validation_note: str,
    quarterly_text: str,
    price: str,
) -> str:
    meta = data.get("meta", {})
    stock_name  = meta.get("stock_name", "")
    stock_code  = meta.get("stock_code", "")
    report_year = meta.get("report_year", "")
    stock_price = price or str(meta.get("stock_price") or "未提供")

    # 把 JSON 数据格式化为人类可读的文字，便于 Claude 理解
    lines = [
        f"标的：{stock_name}（{stock_code}）",
        f"报告期：{report_year}年报",
        f"当前股价：{stock_price}元",
        f"总股本：{meta.get('shares_total')}亿股",
        "",
        "---",
        "## 财务数据（已由 Workflow 3 从年报截图提取，单位：亿元）",
        "",
        "### 资产负债表",
    ]

    bs = data.get("balance_sheet", {})
    bs_fields = [
        ("货币资金（当期）", "cash"),
        ("货币资金（上期）", "cash_prior"),
        ("交易性金融资产", "trading_assets"),
        ("应收票据", "notes_receivable"),
        ("应收账款", "accounts_receivable"),
        ("应收款项融资", "notes_receivable_financing"),
        ("其他应收款", "other_receivables"),
        ("存货", "inventory"),
        ("一年内到期非流动资产", "current_due_noncurrent"),
        ("其他流动资产", "other_current"),
        ("长期股权投资", "long_term_equity"),
        ("其他权益工具投资", "other_equity_instruments"),
        ("投资性房地产", "investment_property"),
        ("固定资产", "fixed_assets"),
        ("在建工程", "construction_in_progress"),
        ("使用权资产", "right_of_use"),
        ("无形资产", "intangibles"),
        ("商誉", "goodwill"),
        ("长期待摊费用", "long_term_deferred_expenses"),
        ("递延所得税资产", "deferred_tax_assets"),
        ("其他非流动资产", "other_noncurrent"),
        ("资产总计", "total_assets"),
        ("负债合计", "total_liabilities"),
        ("短期借款", "short_term_loans"),
        ("一年内到期非流动负债", "current_portion_lt_debt"),
        ("长期借款", "long_term_loans"),
        ("应付债券", "bonds_payable"),
        ("租赁负债", "lease_liabilities"),
        ("少数股东权益", "minority_interest"),
        ("股东权益合计", "total_equity"),
        ("归母净资产", "net_assets_parent"),
    ]
    for label, key in bs_fields:
        val = bs.get(key)
        if val is not None:
            lines.append(f"- {label}: {val}")

    inc = data.get("income_statement", {})
    lines += [
        "",
        "### 利润表",
        f"- 总营收: {inc.get('revenue')}",
        f"- 归母净利润: {inc.get('net_profit_parent')}",
        f"- 上期归母净利润: {inc.get('net_profit_parent_prior')}",
        f"- 整体净利润率: {inc.get('net_profit_margin')}",
    ]

    profit_history = data.get("profit_history", [])
    if profit_history:
        lines += ["", "### 近年归母净利润（亿元）"]
        for p in sorted(profit_history, key=lambda x: x["year"]):
            lines.append(f"- {p['year']}: {p['net_profit']}")

    dividends = data.get("dividend_history", [])
    if dividends:
        lines += ["", "### 历年分红记录（亿元，不含回购）"]
        for d in sorted(dividends, key=lambda x: x["year"]):
            buyback = d.get("buyback", 0)
            note = d.get("note", "")
            lines.append(
                f"- {d['year']}: 分红{d['dividend']}"
                + (f" 回购{buyback}" if buyback else "")
                + (f"（{note}）" if note else "")
            )

    segments = data.get("segment_data", [])
    if segments:
        lines += ["", "### 分产品数据"]
        for s in segments:
            gm = f"毛利率{s['gross_margin']*100:.1f}%" if s.get("gross_margin") is not None else s.get("gross_margin_note", "毛利率未披露")
            yoy = f" 同比{s['yoy']*100:.1f}%" if s.get("yoy") is not None else ""
            lines.append(f"- {s['name']}: 营收{s['revenue']}亿 {gm}{yoy}")

    abnormal = data.get("abnormal_items", [])
    if abnormal:
        lines += ["", "### ⚠️ 异常科目（Claude 提取时已标记）"]
        for a in abnormal:
            lines.append(f"- {a['item']}: {a.get('prior')}→{a.get('current')}亿 {a.get('note','')}")

    if validation_note:
        lines += ["", "---", "## ⚙️ 链路层预校验结果（Workflow 2 已执行）", ""]
        for line in validation_note.split("\n"):
            tag = line.split("]")[0].lstrip("[") if "]" in line else ""
            content = line.split("]", 1)[-1].strip() if "]" in line else line
            if tag == "ERROR":
                lines.append(f"- ❌ {content}")
            elif tag == "WARNING":
                lines.append(f"- ⚠️ {content}")
            elif tag == "CASH_MERGED":
                lines.append(f"- 合并现金类资产: **{content}亿**（两轨净现金须使用此值）")
            elif tag == "SCENARIOS_OK":
                vals = content.split("/")
                if len(vals) == 3:
                    lines.append(f"- ✅ 三情景参考值: 悲观{vals[0].strip()} / 中性{vals[1].strip()} / 乐观{vals[2].strip()}")
            elif tag == "ABNORMAL":
                lines.append(f"- ⚠️ 异常科目: {content}")

    lines += [
        "",
        "---",
        "## 研报估值倍数（供轨道二使用）",
        "",
        report_table,
    ]

    if quarterly_text:
        lines += ["", quarterly_text]

    lines += [
        "",
        "---",
        "请根据以上数据，完成完整的双轨估值分析，输出完整报告（Markdown格式）。",
        "所有数据均已从截图提取完毕，直接使用上方数字，不要说「截图中显示」之类的表述。",
    ]

    return "\n".join(lines)


# ────────────────────────────────────────────────────────────
# 三情景单调性校验（复用 valuation.py 中的函数）
# ────────────────────────────────────────────────────────────

from valuation import _check_scenario_monotonic


# ────────────────────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Workflow 4 — 读JSON做估值（不传图）")
    parser.add_argument("--code",  required=True,  help="股票代码，如 002223")
    parser.add_argument("--price", required=False, default="", help="当前股价（可选）")
    args = parser.parse_args()

    json_path = OUTPUT_DIR / f"{args.code}_data.json"
    if not json_path.exists():
        raise FileNotFoundError(f"未找到数据文件: {json_path}，请先运行 Workflow 3")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    meta = data.get("meta", {})
    print(f"\n🚀 Workflow 4 — 估值（不传图）| {meta.get('stock_name')}（{args.code}）{meta.get('report_year')}年报")

    # 读取辅助文件
    report_table    = load_report_md(args.code)
    validation_note = load_validation_report(args.code)
    quarterly_text  = load_quarterly_text(args.code)
    manual_text     = load_manual_reports(args.code)

    if manual_text:
        report_table += (
            "\n\n## 手动补充研报（请额外提取估值倍数合并入上表）\n\n"
            + manual_text
        )

    # 构建纯文字 prompt
    user_prompt = build_user_prompt(
        data=data,
        report_table=report_table,
        validation_note=validation_note,
        quarterly_text=quarterly_text,
        price=args.price,
    )

    print(f"\n→ 调用Claude（纯文字，无图片）...")
    print(f"  prompt 长度: {len(user_prompt)} 字符")

    client = OpenAI(api_key=CLAUDE_API_KEY, base_url=CLAUDE_BASE_URL)
    resp = client.chat.completions.create(
        model=CLAUDE_MODEL,
        max_tokens=16000,
        temperature=0.1,
        messages=[
            {"role": "system", "content": VALUATION_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]
    )

    report   = resp.choices[0].message.content
    provider = "claude-text-only"
    print(f"  ✅ 完成（{len(report)}字）")

    # 三情景单调性校验
    print("\n--- 三情景单调性校验 ---")
    report = _check_scenario_monotonic(report)

    # 保存报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path  = OUTPUT_DIR / f"{args.code}_{timestamp}_估值报告.md"
    header    = f"> 本报告由 **{provider}** 生成 | {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    out_path.write_text(header + report, encoding="utf-8")

    print(f"\n✅ 完成！报告: {out_path}")
    print(f"\n{'='*60}")
    print(report[:2000])
    print("\n... （完整报告见输出文件）")


if __name__ == "__main__":
    main()

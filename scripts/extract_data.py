"""
extract_data.py — Workflow 3
传图给 Claude，输出结构化财务数据 JSON。
后续 workflow 2（链路层校验）和 workflow 4（估值）均读取此 JSON，不再传图。
"""

import argparse
import base64
import json
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from ai_client import CLAUDE_API_KEY, CLAUDE_BASE_URL, CLAUDE_MODEL
from valuation import VALUATION_PROMPT  # 单一来源，动态截取提取规则

DOCS_DIR   = Path("Supporting Documents for Valuation")
OUTPUT_DIR = Path("output")

# ────────────────────────────────────────────────────────────
# 读图 Prompt
# ────────────────────────────────────────────────────────────

def _get_extraction_rules() -> str:
    """
    从 VALUATION_PROMPT 动态截取「数据提取通用规则」一节。
    确保 Workflow 3 的提取规则与 Workflow 4 的估值规则完全同源，
    不需要手动同步维护两份规则。
    """
    start_marker = "## 数据提取通用规则（两轨共用）"
    end_marker   = "## 【轨道一】分红累加 + 清算价值"
    start = VALUATION_PROMPT.find(start_marker)
    end   = VALUATION_PROMPT.find(end_marker)
    if start == -1 or end == -1:
        raise ValueError("VALUATION_PROMPT 中未找到数据提取通用规则，请检查边界标记")
    return VALUATION_PROMPT[start:end].strip()


EXTRACT_SYSTEM = "只输出JSON，不输出任何其他文字、解释或Markdown代码块。"

EXTRACT_PROMPT = """你是A股财务数据提取专员。从以下年报截图中提取所有财务数据，输出严格JSON。

---

## 数据提取通用规则（与估值分析师完全一致，必须遵守）

{_get_extraction_rules()}

### 补充规则（Workflow 3 专用）
- 货币资金骤降时，须在 cash_drop_confirmed 字段标注 true（定存重分类）或 false（真实流出），
  并在 cash_drop_note 中说明承接科目及金额或流出原因
- 找不到的字段填 null，不得使用训练数据中的历史数字填充

### 分红记录规则
- 中期分红归属对应利润年度（以利润产生年份为准，不以实施年份为准）
- 年度分红总额 = 末期分红 + 中期分红（均不含回购）
- 已公告未实施分红：若截至报告期末已通过决议，须归属至对应利润年度
- 回购金额须从分红总额中剔除，在 buyback 字段单独列示

---

## 输出格式（严格JSON，所有金额单位为亿元）

{
  "meta": {
    "stock_code": "股票代码",
    "stock_name": "股票名称",
    "report_year": 年份整数,
    "stock_price": 当前股价或null,
    "shares_total": 总股本亿股
  },
  "balance_sheet": {
    "cash": 货币资金,
    "cash_prior": 上期货币资金,
    "cash_drop_confirmed": true或false或null,
    "cash_drop_note": "骤降原因及承接科目说明或null",
    "trading_assets": 交易性金融资产,
    "notes_receivable": 应收票据,
    "accounts_receivable": 应收账款,
    "notes_receivable_financing": 应收款项融资或null,
    "prepayments": 预付款项或null,
    "other_receivables": 其他应收款或null,
    "inventory": 存货,
    "current_due_noncurrent": 一年内到期非流动资产或null,
    "other_current": 其他流动资产,
    "long_term_equity": 长期股权投资,
    "other_equity_instruments": 其他权益工具投资或null,
    "investment_property": 投资性房地产或null,
    "fixed_assets": 固定资产,
    "construction_in_progress": 在建工程或null,
    "right_of_use": 使用权资产或null,
    "intangibles": 无形资产,
    "goodwill": 商誉,
    "long_term_deferred_expenses": 长期待摊费用或null,
    "deferred_tax_assets": 递延所得税资产或null,
    "other_noncurrent": 其他非流动资产,
    "total_assets": 资产总计,
    "total_liabilities": 负债合计,
    "short_term_loans": 短期借款或null,
    "current_portion_lt_debt": 一年内到期非流动负债或null,
    "long_term_loans": 长期借款或null,
    "bonds_payable": 应付债券或null,
    "lease_liabilities": 租赁负债或null,
    "minority_interest": 少数股东权益,
    "total_equity": 股东权益合计,
    "net_assets_parent": 归母净资产
  },
  "income_statement": {
    "revenue": 总营收,
    "revenue_prior": 上期营收或null,
    "net_profit_parent": 归母净利润,
    "net_profit_parent_prior": 上期归母净利润或null,
    "net_profit_margin": 净利润率小数
  },
  "profit_history": [
    {"year": 年份, "net_profit": 归母净利润}
  ],
  "dividend_history": [
    {
      "year": 年份,
      "dividend": 现金分红不含回购,
      "buyback": 回购金额,
      "note": "说明（含中期/末期/已公告未实施等）"
    }
  ],
  "segment_data": [
    {
      "name": "板块名称",
      "revenue": 营收,
      "gross_margin": 毛利率小数或null,
      "gross_margin_note": "若null说明原因，如年报注明无分产品毛利率",
      "yoy": 同比增速小数或null
    }
  ],
  "abnormal_items": [
    {
      "item": "科目名称",
      "current": 本期值,
      "prior": 上期值,
      "yoy": 同比增速小数,
      "note": "异常原因说明（如路产收购并入、定存重分类等）"
    }
  ]
}

---

其他注意事项：
- segment_data 过滤合计行、汇总行，只保留独立业务板块
- dividend_history 中 buyback 若无则填 0
- profit_history 尽量覆盖近 11 年（2015-2025），有几年填几年
- 若某科目在截图中完全找不到，填 null，不得用历史数据填充
"""


# ────────────────────────────────────────────────────────────
# 加载截图
# ────────────────────────────────────────────────────────────

def load_images(code: str, name: str = "") -> list:
    img_dir = DOCS_DIR / code
    if not img_dir.exists() and name:
        img_dir = DOCS_DIR / name
    if not img_dir.exists():
        raise FileNotFoundError(f"未找到截图文件夹: {DOCS_DIR}/{code}")

    supported = {".png", ".jpg", ".jpeg", ".webp"}
    img_files = [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in supported]
    if not img_files:
        raise FileNotFoundError(f"文件夹中没有图片: {img_dir}")

    print(f"找到 {len(img_files)} 张截图")
    blocks = []
    for img_path in img_files:
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = img_path.suffix.lower().replace(".", "")
        media_type = f"image/{'jpeg' if ext == 'jpg' else ext}"
        blocks.append({
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{b64}"}
        })
        print(f"  ✅ {img_path.name}")
    return blocks


# ────────────────────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Workflow 3 — 传图提取财务数据JSON")
    parser.add_argument("--code",  required=True,  help="股票代码，如 002223")
    parser.add_argument("--name",  required=True,  help="股票名称，如 鱼跃医疗")
    parser.add_argument("--year",  required=True,  help="报告年度，如 2025")
    parser.add_argument("--price", required=False, default="", help="当前股价（可选）")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    if not CLAUDE_API_KEY:
        raise ValueError("CLAUDE_API_KEY 未设置")

    print(f"\n🚀 Workflow 3 — 传图提取数据 | {args.name}（{args.code}）{args.year}年报")

    # 加载截图
    image_blocks = load_images(args.code, args.name)

    # 构建用户消息
    text_block = {
        "type": "text",
        "text": (
            f"标的：{args.name}（{args.code}）\n"
            f"报告期：{args.year}年报\n"
            + (f"当前股价：{args.price}元\n" if args.price else "")
            + "\n请从以上截图中提取所有财务数据，严格按照JSON格式输出。"
        )
    }

    client = OpenAI(api_key=CLAUDE_API_KEY, base_url=CLAUDE_BASE_URL)
    print(f"\n→ 调用Claude读图（{len(image_blocks)}张截图）...")

    resp = client.chat.completions.create(
        model=CLAUDE_MODEL,
        max_tokens=4000,
        temperature=0,
        messages=[
            {"role": "system",  "content": EXTRACT_SYSTEM},
            {"role": "user",    "content": image_blocks + [text_block]}
        ]
    )

    raw = resp.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    # 验证 JSON 可解析
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析失败: {e}")
        # 保存原始输出便于排查
        err_path = OUTPUT_DIR / f"{args.code}_raw_extract.txt"
        err_path.write_text(raw, encoding="utf-8")
        print(f"  原始输出已保存至: {err_path}")
        raise

    # 写入 JSON 文件
    out_path = OUTPUT_DIR / f"{args.code}_data.json"
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ 完成！财务数据已保存至: {out_path}")

    # 简单预览
    bs = data.get("balance_sheet", {})
    inc = data.get("income_statement", {})
    print(f"\n【数据预览】")
    print(f"  总资产: {bs.get('total_assets')} 亿")
    print(f"  货币资金（当期/上期）: {bs.get('cash')} / {bs.get('cash_prior')} 亿")
    print(f"  归母净利润: {inc.get('net_profit_parent')} 亿")
    print(f"  历史利润年数: {len(data.get('profit_history', []))}")
    print(f"  分红记录年数: {len(data.get('dividend_history', []))}")
    print(f"  分产品板块数: {len(data.get('segment_data', []))}")
    abnormal = data.get("abnormal_items", [])
    if abnormal:
        print(f"  ⚠️ 异常科目: {[a['item'] for a in abnormal]}")


if __name__ == "__main__":
    main()

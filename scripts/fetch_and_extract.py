"""
双轨估值系统 - 研报自动获取 + 估值倍数提取
★ 改动：extract_valuation() 改用 Claude优先 / DeepSeek fallback
"""

import argparse
import json
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import pdfplumber

# ── 导入统一AI客户端 ────────────────────────────────────────
from ai_client import text_completion

OUTPUT_DIR  = Path("output")
REPORTS_DIR = Path("output/reports")

# ────────────────────────────────────────────────────────────
# 研报下载
# ────────────────────────────────────────────────────────────

def download_reports(code, count, begin, end):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"⬇️  下载研报 {code}，最多{count}篇 [{begin} ~ {end}]")
    cmd = [
        "python", "-m", "eastmoney", "d",
        "-t", "stock", "-c", code,
        "-s", str(count), "-o", str(REPORTS_DIR),
        "--begin", begin, "--end", end,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    all_pdfs = list(REPORTS_DIR.glob("**/*.pdf"))
    print(f"共找到 {len(all_pdfs)} 个PDF")
    return REPORTS_DIR


def extract_pdf_text(pdf_path, max_pages=30):
    text_parts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages[:max_pages]):
                page_text = page.extract_text() or ""
                table_text = ""
                for table in page.extract_tables():
                    for row in table:
                        table_text += " | ".join([str(c or "").strip() for c in row]) + "\n"
                if page_text or table_text:
                    text_parts.append(f"--- 第{i+1}页 ---\n{page_text}\n{table_text}")
    except Exception as e:
        return f"[提取失败: {e}]"
    return "\n".join(text_parts)[:12000]


# ────────────────────────────────────────────────────────────
# 估值倍数提取  ★ 核心改动：Claude优先
# ────────────────────────────────────────────────────────────

EXTRACT_SYSTEM = "只输出JSON，不输出任何其他文字、解释或Markdown代码块。"

EXTRACT_PROMPT = """你是券商研报分析员。从以下研报文本中提取估值信息。

输出格式（严格JSON，无其他内容）：
{
  "broker": "券商名称",
  "title": "报告标题",
  "date": "YYYY-MM-DD",
  "target_price": 数字或null,
  "pe_range": [悲观PE, 乐观PE] 或 null,
  "peg": 数字或null,
  "ev_ebitda": 数字或null,
  "pb": 数字或null,
  "rating": "评级",
  "notes": "不超过50字的备注，说明估值逻辑"
}

注意事项：
- pe_range 填写研报给出的估值区间下限和上限，若只有一个值则两位相同
- 目标价单位为元
- 评级如：买入 / 增持 / 中性 / 减持 等
- 若研报未披露某字段，填 null

研报内容：
"""


def extract_valuation(report_text: str) -> dict:
    """
    从单篇研报文本中提取估值倍数。
    ★ Claude优先，失败时fallback到DeepSeek。
    """
    messages = [{"role": "user", "content": EXTRACT_PROMPT + report_text}]
    try:
        raw, provider = text_completion(
            messages=messages,
            system=EXTRACT_SYSTEM,
            max_tokens=500,
            temperature=0,
            task_label="研报倍数提取",
        )
        # 清理可能的markdown代码块包裹
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        result["_provider"] = provider   # 记录实际使用的模型，方便排查
        return result
    except Exception as e:
        return {"error": str(e)}


# ────────────────────────────────────────────────────────────
# 汇总输出
# ────────────────────────────────────────────────────────────

def generate_markdown_table(results: list, code: str) -> str:
    lines = [
        f"## 研报估值倍数汇总 — {code}",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "| 券商 | 报告标题 | 日期 | 目标价 | 悲观PE | 乐观PE | PEG | EV/EBITDA | PB | 评级 | 提取模型 |",
        "|------|---------|------|--------|--------|--------|-----|-----------|-----|------|---------|",
    ]

    valid = [r for r in results if "error" not in r]
    if not valid:
        lines.append(
            "| — | 未找到有效数据 | — | — | — | — | — | — | — | — | [fallback-无研报] |"
        )
    else:
        for r in valid:
            pe = r.get("pe_range") or [None, None]
            lines.append(
                f"| {r.get('broker','—')} "
                f"| {str(r.get('title','—'))[:20]} "
                f"| {r.get('date','—')} "
                f"| {r.get('target_price','—')} "
                f"| {pe[0] if pe else '—'} "
                f"| {pe[1] if pe else '—'} "
                f"| {r.get('peg','—')} "
                f"| {r.get('ev_ebitda','—')} "
                f"| {r.get('pb','—')} "
                f"| {r.get('rating','—')} "
                f"| {r.get('_provider','—')} |"
            )

    lines += ["", "### 悲观下限（供轨道二直接使用）", ""]
    if valid:
        pe_vals  = [r["pe_range"][0] for r in valid if r.get("pe_range") and r["pe_range"][0]]
        peg_vals = [r["peg"]         for r in valid if r.get("peg")]
        ev_vals  = [r["ev_ebitda"]   for r in valid if r.get("ev_ebitda")]
        lines.append(f"- **悲观PE下限**: {min(pe_vals)  if pe_vals  else '[fallback-倍数缺失]'}")
        lines.append(f"- **悲观PEG下限**: {min(peg_vals) if peg_vals else '[fallback-倍数缺失]'}")
        lines.append(f"- **悲观EV/EBITDA下限**: {min(ev_vals) if ev_vals else '[fallback-倍数缺失]'}")
    else:
        lines.append("- 使用申万行业历史熊市低位PE `[fallback-无研报]`")

    return "\n".join(lines)


# ────────────────────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code",        required=True)
    parser.add_argument("--months",      type=int, default=12)
    parser.add_argument("--max-reports", type=int, default=10)
    args = parser.parse_args()

    print(f"🚀 fetch_and_extract 启动 | code={args.code}")
    OUTPUT_DIR.mkdir(exist_ok=True)

    end_date   = datetime.now()
    begin_date = end_date - timedelta(days=args.months * 30)

    download_reports(
        args.code, args.max_reports,
        begin_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
    )

    pdf_files = list(REPORTS_DIR.glob("**/*.pdf"))
    print(f"\n待分析PDF：{len(pdf_files)} 个")

    if not pdf_files:
        out = OUTPUT_DIR / f"{args.code}_valuation.md"
        out.write_text(
            f"[fallback-无研报] {args.code} 未下载到研报",
            encoding="utf-8",
        )
        print("未找到PDF，已写入fallback标记")
        return

    results = []
    for i, pdf_path in enumerate(pdf_files[:args.max_reports]):
        print(f"\n[{i+1}/{min(len(pdf_files), args.max_reports)}] {pdf_path.name}")
        text   = extract_pdf_text(pdf_path)
        result = extract_valuation(text)
        results.append(result)
        if "error" in result:
            print(f"  ⚠️  提取失败: {result['error']}")
        else:
            print(
                f"  ✅ {result.get('broker')} | "
                f"PE:{result.get('pe_range')} | "
                f"目标价:{result.get('target_price')} | "
                f"by {result.get('_provider','?')}"
            )

    markdown = generate_markdown_table(results, args.code)
    out = OUTPUT_DIR / f"{args.code}_valuation.md"
    out.write_text(markdown, encoding="utf-8")
    (OUTPUT_DIR / f"{args.code}_raw.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n✅ 完成！输出: {out}")
    print("\n" + markdown)


if __name__ == "__main__":
    main()

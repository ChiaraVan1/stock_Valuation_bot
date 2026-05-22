"""
双轨估值系统 - 研报自动获取 + DeepSeek 估值倍数提取
"""

import argparse
import json
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import pdfplumber
from openai import OpenAI

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
OUTPUT_DIR = Path("output")
REPORTS_DIR = Path("output/reports")

def get_client():
    if not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY 未设置")
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

def download_reports(code, count, begin, end):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"⬇️ 开始下载研报，股票代码: {code}，最多{count}篇")
    cmd = [
        "python", "-m", "eastmoney", "d",
        "-t", "stock",
        "-c", code,
        "-s", str(count),
        "-o", str(REPORTS_DIR),
        "--begin", begin,
        "--end", end,
    ]
    print(f"执行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    print("=== 下载输出 ===")
    print(result.stdout)
    if result.stderr:
        print("=== 下载错误 ===")
        print(result.stderr)
    print(f"下载退出码: {result.returncode}")

    all_pdfs = list(REPORTS_DIR.glob("**/*.pdf"))
    print(f"下载后找到PDF数量: {len(all_pdfs)}")
    for p in all_pdfs:
        print(f"  - {p}")
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
    full_text = "\n".join(text_parts)
    return full_text[:12000]

EXTRACT_PROMPT = """你是券商研报分析员。从以下研报中提取估值信息，只输出JSON，不输出任何其他文字。

输出格式：
{
  "broker": "券商名称",
  "title": "报告标题",
  "date": "YYYY-MM-DD",
  "target_price": 数字或null,
  "pe_range": [悲观PE, 乐观PE]或null,
  "peg": 数字或null,
  "ev_ebitda": 数字或null,
  "pb": 数字或null,
  "rating": "评级",
  "notes": "不超过50字的备注"
}

研报内容：
"""

def extract_valuation(report_text, client):
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            max_tokens=500,
            temperature=0,
            messages=[
                {"role": "system", "content": "只输出JSON，不输出任何其他文字。"},
                {"role": "user", "content": EXTRACT_PROMPT + report_text}
            ]
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        return {"error": str(e)}

def generate_markdown_table(results, code):
    lines = [
        f"## 研报估值倍数汇总 — {code}",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "| 券商 | 报告标题 | 日期 | 目标价 | 悲观PE | 乐观PE | PEG | EV/EBITDA | PB | 评级 |",
        "|------|---------|------|--------|--------|--------|-----|-----------|-----|------|",
    ]
    valid = [r for r in results if "error" not in r]
    if not valid:
        lines.append("| — | 未找到有效数据 | — | — | — | — | — | — | — | [fallback-无研报] |")
    else:
        for r in valid:
            pe = r.get("pe_range") or [None, None]
            lines.append(
                f"| {r.get('broker','—')} | {str(r.get('title','—'))[:20]} "
                f"| {r.get('date','—')} | {r.get('target_price','—')} "
                f"| {pe[0] if pe else '—'} | {pe[1] if pe else '—'} "
                f"| {r.get('peg','—')} | {r.get('ev_ebitda','—')} "
                f"| {r.get('pb','—')} | {r.get('rating','—')} |"
            )

    lines += ["", "### 悲观下限（供轨道二直接使用）", ""]
    if valid:
        pe_vals = [r["pe_range"][0] for r in valid if r.get("pe_range") and r["pe_range"][0]]
        peg_vals = [r["peg"] for r in valid if r.get("peg")]
        ev_vals = [r["ev_ebitda"] for r in valid if r.get("ev_ebitda")]
        lines.append(f"- **悲观PE下限**: {min(pe_vals) if pe_vals else '[fallback-倍数缺失]'}")
        lines.append(f"- **悲观PEG下限**: {min(peg_vals) if peg_vals else '[fallback-倍数缺失]'}")
        lines.append(f"- **悲观EV/EBITDA下限**: {min(ev_vals) if ev_vals else '[fallback-倍数缺失]'}")
    else:
        lines.append("- 使用申万行业历史熊市低位PE `[fallback-无研报]`")
    return "\n".join(lines)

def main():
    print("🚀 脚本启动")
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", required=True)
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--max-reports", type=int, default=10)
    args = parser.parse_args()
    print(f"参数: code={args.code}, months={args.months}, max_reports={args.max_reports}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    end_date = datetime.now()
    begin_date = end_date - timedelta(days=args.months * 30)

    download_reports(
        args.code, args.max_reports,
        begin_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d")
    )

    pdf_files = list(REPORTS_DIR.glob("**/*.pdf"))
    print(f"找到 {len(pdf_files)} 个PDF待分析")

    if not pdf_files:
        print("未找到PDF，下载可能失败")
        out = OUTPUT_DIR / f"{args.code}_valuation.md"
        out.write_text(f"[fallback-无研报] {args.code} 未下载到研报", encoding="utf-8")
        return

    print(f"🤖 开始DeepSeek分析...")
    client = get_client()
    results = []

    for i, pdf_path in enumerate(pdf_files[:args.max_reports]):
        print(f"[{i+1}/{len(pdf_files)}] 处理: {pdf_path.name}")
        text = extract_pdf_text(pdf_path)
        print(f"  提取文字长度: {len(text)} 字符")
        result = extract_valuation(text, client)
        results.append(result)
        if "error" in result:
            print(f"  ⚠️ 提取失败: {result['error']}")
        else:
            print(f"  ✅ {result.get('broker')} | PE:{result.get('pe_range')} | 目标价:{result.get('target_price')}")

    markdown = generate_markdown_table(results, args.code)
    out = OUTPUT_DIR / f"{args.code}_valuation.md"
    out.write_text(markdown, encoding="utf-8")
    (OUTPUT_DIR / f"{args.code}_raw.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n✅ 全部完成！输出: {out}")
    print("\n--- 复制到Claude对话 ---\n")
    print(markdown)

if __name__ == "__main__":
    main()

"""
双轨估值系统 - 研报自动获取 + DeepSeek 估值倍数提取
用法: python scripts/fetch_and_extract.py --code 000001 --months 12
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pdfplumber
from openai import OpenAI

# ── 配置 ──────────────────────────────────────────────────────────────────────

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
OUTPUT_DIR = Path("output")
REPORTS_DIR = Path("output/reports")

# ── DeepSeek 客户端 ───────────────────────────────────────────────────────────

def get_client():
    if not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY 环境变量未设置")
    return OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )

# ── Step 1: 查询研报列表 ───────────────────────────────────────────────────────

def query_reports(code: str, months: int = 12) -> list[dict]:
    """调用 eastmoney CLI 查询研报列表"""
    end_date = datetime.now()
    begin_date = end_date - timedelta(days=months * 30)

    print(f"\n📋 查询 {code} 近{months}个月研报...")
    print(f"   时间范围: {begin_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")

    try:
        result = subprocess.run(
            [
                "python", "-m", "eastmoney", "q",
                "-t", "stock",
                "-c", code,
                "--begin", begin_date.strftime("%Y-%m-%d"),
                "--end", end_date.strftime("%Y-%m-%d"),
                "--format", "json"
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode != 0:
            print(f"⚠️  查询失败: {result.stderr}")
            return []

        reports = json.loads(result.stdout)
        print(f"✅ 找到 {len(reports)} 篇研报")
        return reports

    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError) as e:
        print(f"⚠️  查询出错: {e}")
        return []

# ── Step 2: 下载研报 PDF ──────────────────────────────────────────────────────

def download_reports(code: str, count: int, begin: str, end: str) -> Path:
    """下载研报到本地目录"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n⬇️  下载研报 PDF（最多{count}篇）...")

    cmd = [
        "python", "-m", "eastmoney", "d",
        "-t", "stock",
        "-c", code,
        "-s", str(count),
        "-o", str(REPORTS_DIR),
        "--begin", begin,
        "--end", end,
    ]
    print(f"   执行命令: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        print(f"⚠️  下载警告: {result.stderr}")
    else:
        print(f"✅ 下载完成，文件保存至 {REPORTS_DIR}")

    return REPORTS_DIR

# ── Step 3: PDF 文字提取 ──────────────────────────────────────────────────────

def extract_pdf_text(pdf_path: Path, max_pages: int = 30) -> str:
    """用 pdfplumber 提取研报文字（研报排版规整，效果较好）"""
    text_parts = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages_to_read = min(len(pdf.pages), max_pages)
            for i, page in enumerate(pdf.pages[:pages_to_read]):
                # 提取文本
                page_text = page.extract_text() or ""

                # 提取表格（估值表格通常在此）
                tables = page.extract_tables()
                table_text = ""
                for table in tables:
                    for row in table:
                        cells = [str(c or "").strip() for c in row]
                        table_text += " | ".join(cells) + "\n"

                if page_text or table_text:
                    text_parts.append(f"--- 第{i+1}页 ---\n{page_text}\n{table_text}")

    except Exception as e:
        return f"[提取失败: {e}]"

    full_text = "\n".join(text_parts)
    # 限制长度，避免超出 DeepSeek 上下文
    return full_text[:12000] if len(full_text) > 12000 else full_text

# ── Step 4: DeepSeek 提取估值倍数 ─────────────────────────────────────────────

EXTRACT_PROMPT = """你是一位专业的券商研报分析员。请从以下研报文字中提取估值相关信息。

要求：
1. 只提取明确出现的估值倍数，不要推断或捏造
2. 若某项未找到，填 null
3. 严格按 JSON 格式输出，不要加任何说明文字

输出格式（JSON）：
{
  "broker": "券商名称",
  "title": "报告标题",
  "date": "发布日期 YYYY-MM-DD",
  "target_price": 目标价数字或null,
  "pe_range": [悲观PE, 乐观PE] 或 null,
  "peg": PEG值或null,
  "ev_ebitda": EV/EBITDA值或null,
  "pb": PB值或null,
  "rating": "评级如买入/增持等",
  "notes": "与估值相关的简短备注，不超过50字"
}

研报内容：
"""

def extract_valuation(report_text: str, client: OpenAI) -> dict:
    """用 DeepSeek 从研报文字中提取估值倍数"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            max_tokens=500,
            temperature=0,   # 结构化提取用 0，减少幻觉
            messages=[
                {
                    "role": "system",
                    "content": "你是券商研报分析员，只输出 JSON，不输出任何其他文字。"
                },
                {
                    "role": "user",
                    "content": EXTRACT_PROMPT + report_text
                }
            ]
        )
        raw = response.choices[0].message.content.strip()
        # 去掉可能的 markdown 代码块
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)

    except json.JSONDecodeError as e:
        return {"error": f"JSON解析失败: {e}", "raw": raw[:200]}
    except Exception as e:
        return {"error": str(e)}

# ── 输出：生成 Markdown 汇总表 ────────────────────────────────────────────────

def generate_markdown_table(results: list[dict], code: str) -> str:
    """生成可直接粘贴进 Claude 对话的 Markdown 估值倍数表"""

    lines = [
        f"## 研报估值倍数汇总 — {code}",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "| 券商 | 报告标题 | 日期 | 目标价 | 悲观PE | 乐观PE | PEG | EV/EBITDA | PB | 评级 | 备注 |",
        "|------|---------|------|--------|--------|--------|-----|-----------|-----|------|------|",
    ]

    valid_results = [r for r in results if "error" not in r]

    if not valid_results:
        lines.append("| — | 未找到有效研报估值数据 | — | — | — | — | — | — | — | — | [fallback-无研报] |")
    else:
        for r in valid_results:
            pe_low = r.get("pe_range", [None, None])
            pe_low_val = pe_low[0] if pe_low else "—"
            pe_high_val = pe_low[1] if pe_low else "—"

            lines.append(
                f"| {r.get('broker','—')} "
                f"| {r.get('title','—')[:20]} "
                f"| {r.get('date','—')} "
                f"| {r.get('target_price','—')} "
                f"| {pe_low_val} "
                f"| {pe_high_val} "
                f"| {r.get('peg','—')} "
                f"| {r.get('ev_ebitda','—')} "
                f"| {r.get('pb','—')} "
                f"| {r.get('rating','—')} "
                f"| {r.get('notes','—')} |"
            )

    # 悲观下限汇总（供 Claude 直接使用）
    lines += ["", "### 悲观下限（供轨道二直接使用）", ""]
    if valid_results:
        pe_values = [r["pe_range"][0] for r in valid_results
                     if r.get("pe_range") and r["pe_range"][0] is not None]
        peg_values = [r["peg"] for r in valid_results if r.get("peg") is not None]
        ev_values = [r["ev_ebitda"] for r in valid_results if r.get("ev_ebitda") is not None]

        lines.append(f"- **悲观 PE 下限**: {min(pe_values) if pe_values else '[fallback-倍数缺失]'}")
        lines.append(f"- **悲观 PEG 下限**: {min(peg_values) if peg_values else '[fallback-倍数缺失]'}")
        lines.append(f"- **悲观 EV/EBITDA 下限**: {min(ev_values) if ev_values else '[fallback-倍数缺失]'}")
    else:
        lines.append("- 未找到有效数据，请使用申万行业历史熊市低位 PE `[fallback-无研报]`")

    return "\n".join(lines)

# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="研报自动获取 + 估值倍数提取")
    parser.add_argument("--code", required=True, help="股票代码，如 000001")
    parser.add_argument("--months", type=int, default=12, help="查询近N个月研报")
    parser.add_argument("--max-reports", type=int, default=10, help="最多下载N篇研报")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Step 1: 查询
    end_date = datetime.now()
    begin_date = end_date - timedelta(days=args.months * 30)
    reports = query_reports(args.code, args.months)

    if not reports:
        print("\n⚠️  未查询到研报，将使用 fallback")
        fallback_note = f"[fallback-无研报] 股票 {args.code} 近{args.months}个月无个股研报"
        output_path = OUTPUT_DIR / f"{args.code}_valuation.md"
        output_path.write_text(fallback_note, encoding="utf-8")
        print(f"📄 结果已保存: {output_path}")
        return

    # Step 2: 下载
    count = min(len(reports), args.max_reports)
    download_reports(
        args.code,
        count,
        begin_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d")
    )

    # Step 3 & 4: 提取 + DeepSeek 分析
    pdf_files = list(REPORTS_DIR.glob("*.pdf"))
    if not pdf_files:
        print("⚠️  未找到 PDF 文件，请检查下载是否成功")
        return

    print(f"\n🤖 开始 DeepSeek 分析（共 {len(pdf_files)} 个 PDF）...")
    client = get_client()
    results = []

    for i, pdf_path in enumerate(pdf_files[:args.max_reports]):
        print(f"   [{i+1}/{len(pdf_files)}] 处理: {pdf_path.name}")
        text = extract_pdf_text(pdf_path)
        result = extract_valuation(text, client)
        results.append(result)

        if "error" in result:
            print(f"   ⚠️  提取失败: {result['error']}")
        else:
            print(f"   ✅ {result.get('broker','?')} | PE: {result.get('pe_range')} | 目标价: {result.get('target_price')}")

    # 生成汇总表
    markdown = generate_markdown_table(results, args.code)
    output_path = OUTPUT_DIR / f"{args.code}_valuation.md"
    output_path.write_text(markdown, encoding="utf-8")

    # 同时保存原始 JSON
    json_path = OUTPUT_DIR / f"{args.code}_raw.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n✅ 完成！")
    print(f"   Markdown 表格: {output_path}")
    print(f"   原始 JSON:     {json_path}")
    print(f"\n--- 可直接复制到 Claude 对话的内容 ---\n")
    print(markdown)

if __name__ == "__main__":
    main()

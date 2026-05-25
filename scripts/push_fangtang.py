"""
估值报告推送方糖脚本
用法: python scripts/push_fangtang.py --code 002223 --name 鱼跃医疗 --price 29.08
"""
import os
import sys
import re
import glob
import argparse
import urllib.request
import urllib.parse
import json


def find_report(code: str) -> str | None:
    patterns = [
        f"output/{code}_*_估值报告.md",
        f"output/{code}_*.md",
    ]
    for pat in patterns:
        files = sorted(glob.glob(pat), reverse=True)
        if files:
            return files[0]
    return None


def extract_value(text: str, *patterns) -> str:
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1).strip()
    return "N/A"


def parse_report(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        text = f.read()

    d = {}

    # 股价
    d["price"] = extract_value(text,
        r"当前股价[：:]\s*([\d.]+)",
        r"股价[：:]\s*([\d.]+)\s*元",
    )

    # 轨道一
    d["t1_bearish"] = extract_value(text,
        r"每股内在价值[（(]悲观[)）][^\d]*([\d.]+)",
        r"悲观.*?每股[^\d]*([\d.]+)\s*元",
    )
    d["t1_neutral"] = extract_value(text,
        r"每股内在价值[（(]中性[)）][^\d]*([\d.]+)",
        r"中性.*?每股[^\d]*([\d.]+)\s*元",
    )
    d["t1_margin"] = extract_value(text,
        r"安全边际[（(]中性[)）][^\n]*([-+\d.]+%)",
        r"安全边际.*?中性.*?([-+\d.]+%)",
    )
    d["liquidation"] = extract_value(text,
        r"每股清算价值[^\d]*([\d.]+)",
    )

    # 轨道二
    d["sotp_bearish"] = extract_value(text,
        r"每股\s*SOTP[^\d]*([\d.]+)",
        r"SOTP.*?每股[^\d]*([\d.]+)\s*元",
        r"悲观.*?SOTP.*?([\d.]+)\s*元",
    )
    d["sotp_discount"] = extract_value(text,
        r"折价[^\d]?([\-\+]?\d+\.?\d*%)",
        r"[折溢]价.*?([\-\+]\d+\.?\d*%)",
        r"vs.*?股价.*?([\-\+]?\d+\.?\d*%)",
    )

    # 综合结论（取第一段）
    m = re.search(r"综合[判断结论]+[：:\n]+(.{20,150})", text)
    d["conclusion"] = m.group(1).strip()[:120] if m else ""

    # 主要风险（第一条）
    m = re.search(r"风险[提示因素]+.*?\n[-\-•*]\s*(.{10,80})", text)
    d["top_risk"] = m.group(1).strip() if m else ""

    return d


def build_message(code: str, name: str, d: dict) -> tuple[str, str]:
    """返回 (title, desp)，desp 支持 Markdown"""
    title = f"📊 {name}（{code}）估值报告"

    lines = [
        f"**股价：{d['price']} 元**",
        "",
        "## 轨道一｜分红清算",
        f"- 清算价值：**{d['liquidation']} 元**",
        f"- 内在价值（悲观/中性）：{d['t1_bearish']} / **{d['t1_neutral']} 元**",
        f"- 安全边际（中性）：{d['t1_margin']}",
        "",
        "## 轨道二｜SOTP",
        f"- 悲观每股：**{d['sotp_bearish']} 元**",
        f"- 对比股价：{d['sotp_discount']}",
        "",
    ]

    if d["conclusion"]:
        lines += ["## 综合结论", d["conclusion"], ""]

    if d["top_risk"]:
        lines += ["## 首要风险", f"> {d['top_risk']}", ""]

    lines.append("---")
    lines.append("*由 stock_Valuation_bot 自动生成*")

    return title, "\n".join(lines)


def push(sct_key: str, title: str, desp: str, dry_run: bool = False) -> None:
    if dry_run:
        print("=== DRY RUN：以下内容将推送到方糖 ===")
        print(f"标题：{title}")
        print(f"正文：\n{desp}")
        return

    url = f"https://sctapi.ftqq.com/{sct_key}.send"
    data = urllib.parse.urlencode({"title": title, "desp": desp}).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=15) as resp:
        body = json.loads(resp.read())
        if body.get("code") == 0:
            print(f"✅ 方糖推送成功：{title}")
        else:
            print(f"⚠️  方糖返回异常：{body}")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--price", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sct_key = os.environ.get("SCT_KEY", "")
    if not sct_key and not args.dry_run:
        print("❌ 未设置 SCT_KEY 环境变量")
        sys.exit(1)

    report_path = find_report(args.code)
    if not report_path:
        print(f"❌ 未找到 {args.code} 的估值报告，请确认 output/ 目录下存在报告文件")
        sys.exit(1)

    print(f"读取报告：{report_path}")
    d = parse_report(report_path)

    # 命令行传入的股价优先
    if args.price:
        d["price"] = args.price

    title, desp = build_message(args.code, args.name, d)
    push(sct_key, title, desp, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

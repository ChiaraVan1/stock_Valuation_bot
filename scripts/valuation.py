"""
双轨估值系统 - 完整版
Step1: 千问VL读截图 -> 结构化文字
Step2: DeepSeek跑双轨估值逻辑
"""

import argparse
import base64
import json
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from openai import OpenAI

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "")
DOCS_DIR = Path("Supporting Documents for Valuation")
OUTPUT_DIR = Path("output")
REPORTS_DIR = Path("output/reports")

VALUATION_PROMPT = """你是一位 A 股双轨估值分析师，同时运用两套独立估值体系对同一标的进行分析：

- 轨道一：分红累加 + 清算价值（价值投资视角，适合稳定分红公司）
- 轨道二：SOTP 分部加总（分部估值视角，适合多业务板块公司）

两套方法并行计算，最终输出交叉验证对比结论。

## 数据提取通用规则（两轨共用）

### 单位换算（最高优先级）
- 巨潮年报 PDF 原始单位为"元"，提取后必须统一换算：亿元 = 原始数值 ÷ 100,000,000
- 每个科目换算后须做数量级合理性检验：
  - 任何单一科目不得超过资产总计
  - 固定资产通常不超过总资产的 60%
  - 货币资金通常在总资产的 10%-40% 之间
- 若换算结果明显异常，须停止计算，重新核查原始数字

### 货币资金骤降识别规则
- 当货币资金较上期下降超过 30% 时，须检查交易性金融资产、其他流动资产、其他非流动资产是否承接
- 若确认为定期存款/结构性存款转移，须将其与货币资金合并为"现金类资产"，折算率不低于 90%，并在报告中单独说明

### 异常科目处理
- 某科目账面值较上期增幅超过 50%，须单独说明原因并重新评估折算率
- 提取不到的字段标注"未找到"，不得推测或捏造
- 所有数据注明来源和报告期

---

## 【轨道一】分红累加 + 清算价值

### 第一步：历史分红率
- 分红总额 = 历年现金派息金额之和，股票回购金额须全部剔除
- 若数据同时列示"含回购"与"不含回购"，必须使用"不含回购"口径
- 在报告中须注明："分红率计算已剔除回购金额 X 亿元"
- 分红率 = 历史累计现金分红（逐年加总，不含回购）÷ 历史累计归母净利润（逐年加总）
- 中期分红归属于对应利润年度，与同年末期分红合并

### 第二步：清算价值估算

逐科目折算：
- 货币资金 + 交易性金融资产：100%
- 定期存款/结构性存款（转入其他科目的）：90%
- 应收票据 + 应收账款：80%
- 存货：60%
- 长期股权投资：70%
- 固定资产：50%
- 在建工程：40%
- 使用权资产：30%
- 无形资产：30%
- 商誉：0%（强制归零）
- 其他资产：20%

负债扣除口径：扣除"负债合计"（含流动负债 + 非流动负债全部）

计算公式：
- 清算净值 = 资产折算合计 - 负债合计（100%）
- 归母清算净值 = 清算净值 x 归母权益占比（= 1 - 少数股东权益 / 股东权益合计）
- 每股清算价值 = 归母清算净值 / 总股本

### 第三步：正常盈利力判断（三情景）
- 悲观：近 3 年均值或历史最低合理值，取两者中更低者
- 中性：近 10-11 年全周期均值
- 乐观：历史峰值区间均值
- 必须验证：悲观值 < 中性值 < 乐观值

### 第四步：内在价值计算
内在价值总额（亿元）= 正常盈利力 x 10年 x 历史分红率 + 归母清算净值
每股内在价值 = 内在价值总额 / 总股本

对悲观、中性、乐观三个情景分别计算。

### 轨道一输出格式

数据汇总表：列示所有提取数据、换算过程、异常科目说明

轨道一结论表：
| 维度 | 数值 |
|------|------|
| 每股净资产（归母） | X 元 |
| 每股清算价值 | X 元 |
| 每股内在价值（悲观） | X 元 |
| 每股内在价值（中性） | X 元 |
| 每股内在价值（乐观） | X 元 |
| 当前股价 | X 元 |
| 安全边际（悲观） | X% |
| 安全边际（中性） | X% |

---

## 【轨道二】SOTP 分部加总

### 第一步：识别分部数据
从材料中提取每个独立业务板块的营收金额与毛利率（过滤汇总行、合计行、"其他业务"）

### 第二步：匹配估值方法与倍数
- 稳定盈利、增速 <15%：PE
- 净利增速 >=15% 的成长板块：PEG（需有分板块专项研报支撑，否则用PE+溢价上限2x）
- 重资产、周期性：EV/EBITDA
- 亏损/早期成长：PS

倍数取值：使用下方提供的研报估值倍数表中的悲观下限。若标注[fallback]则使用申万行业历史熊市低位PE。

### 第三步：SOTP 计算
- 板块净利润 = 板块营收 x 整体净利润率（若有分部毛利率则按毛利率加权分配）
- PE 板块估值 = 板块净利润 x 悲观 PE
- 净现金 = 货币资金 + 短期金融资产 - 有息负债
- SOTP 总估值 = 各板块估值之和 + 净现金
- 每股 SOTP = SOTP 总估值 / 总股本
- 隐含溢价 = (当前股价 - 每股 SOTP) / 每股 SOTP x 100%

### 轨道二输出格式
| 板块 | 营收(亿) | 段净利(亿) | 方法 | 悲观倍数 | 段估值(亿) |
|------|---------|-----------|------|---------|-----------|

并列示：SOTP悲观总估值、每股SOTP、隐含溢价、校验信息

---

## 【交叉验证】双轨对比与综合结论

| 维度 | 轨道一 | 轨道二 | 差异及原因 |
|------|--------|--------|-----------|
| 每股估值（中性/悲观） | X 元 | X 元 | — |
| 当前股价 | X 元 | X 元 | — |
| 隐含安全边际/溢价 | X% | X% | — |

差异归因分析（必填）、综合判断、估值局限性说明、风险提示均须输出。

---

## 特殊情况处理
- 商誉：轨道一强制归零，额外测算10%折算情景
- 货币资金骤降：合并定期存款，两轨口径保持一致
- 无分产品数据：轨道二降级为整体PE估值
- 亏损年份：纳入计算，不得剔除

请根据以下提取的财务数据完成完整的双轨估值分析，输出完整报告（Markdown格式）：
"""


def get_deepseek_client():
    if not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY 未设置")
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")


def get_qwen_client():
    if not QWEN_API_KEY:
        raise ValueError("QWEN_API_KEY 未设置")
    return OpenAI(
        api_key=QWEN_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )


def extract_images_with_qwen(code: str, name: str = "") -> str:
    """Step1: 千问VL读取所有截图，输出结构化财务数据文字"""
    img_dir = DOCS_DIR / code
    if not img_dir.exists() and name:
        img_dir = DOCS_DIR / name
    if not img_dir.exists():
        raise FileNotFoundError(f"未找到截图文件夹: {DOCS_DIR}/{code} 或 {DOCS_DIR}/{name}")

    supported = {".png", ".jpg", ".jpeg", ".webp"}
    img_files = [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in supported]

    if not img_files:
        raise FileNotFoundError(f"文件夹中没有图片: {img_dir}")

    print(f"找到 {len(img_files)} 张截图，开始千问VL识别...")
    client = get_qwen_client()

    all_extracted = []

    for i, img_path in enumerate(img_files):
        print(f"  [{i+1}/{len(img_files)}] 识别: {img_path.name}")
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = img_path.suffix.lower().replace(".", "")
        media_type = f"image/{'jpeg' if ext == 'jpg' else ext}"

        resp = client.chat.completions.create(
            model="qwen-vl-plus",
            max_tokens=1500,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{b64}"}
                        },
                        {
                            "type": "text",
                            "text": """请仔细识别这张财务截图中的所有数据，按以下格式输出：

1. 截图类型（如：资产负债表/利润表/归母净利润历史/分红记录/分产品营收/股本信息/股价信息等）
2. 所有数字数据，保持原始格式，注明单位
3. 表格数据请逐行列出

要求：只输出数据，不要分析，不要遗漏任何数字。"""
                        }
                    ]
                }
            ]
        )
        extracted = resp.choices[0].message.content
        all_extracted.append(f"=== 截图{i+1}: {img_path.name} ===\n{extracted}")
        print(f"    提取完成（{len(extracted)}字）")

    result = "\n\n".join(all_extracted)
    print(f"\n千问VL识别完成，共提取 {len(result)} 字符")
    return result


def download_reports(code: str, months: int, max_reports: int) -> str:
    """下载研报并返回估值倍数表文字"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    end_date = datetime.now()
    begin_date = end_date - timedelta(days=months * 30)

    print(f"下载研报...")
    cmd = [
        "python", "-m", "eastmoney", "d",
        "-t", "stock", "-c", code,
        "-s", str(max_reports),
        "-o", str(REPORTS_DIR),
        "--begin", begin_date.strftime("%Y-%m-%d"),
        "--end", end_date.strftime("%Y-%m-%d"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    print(result.stdout)

    pdf_files = list(REPORTS_DIR.glob("**/*.pdf"))
    print(f"找到 {len(pdf_files)} 个研报PDF")

    if not pdf_files:
        return "[fallback-无研报] 未下载到研报，请使用申万行业历史熊市低位PE"

    try:
        import pdfplumber
        client = get_deepseek_client()

        EXTRACT_PROMPT = """从以下研报中提取估值信息，只输出JSON，不输出任何其他文字。
格式：{"broker":"券商","title":"标题","date":"YYYY-MM-DD","target_price":数字或null,"pe_range":[悲观PE,乐观PE]或null,"peg":数字或null,"ev_ebitda":数字或null,"pb":数字或null,"rating":"评级"}
研报内容："""

        results = []
        for pdf_path in pdf_files[:max_reports]:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages[:20]:
                    text += page.extract_text() or ""
            text = text[:10000]

            resp = client.chat.completions.create(
                model="deepseek-v4-flash",
                max_tokens=300,
                temperature=0,
                messages=[
                    {"role": "system", "content": "只输出JSON。"},
                    {"role": "user", "content": EXTRACT_PROMPT + text}
                ]
            )
            raw = resp.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
            try:
                results.append(json.loads(raw))
            except Exception:
                pass

        if not results:
            return "[fallback-倍数缺失] 研报未披露估值倍数，请使用申万行业历史熊市低位PE"

        lines = [
            "## 研报估值倍数汇总（自动提取）",
            "",
            "| 券商 | 标题 | 日期 | 目标价 | 悲观PE | 乐观PE | EV/EBITDA | PB | 评级 |",
            "|------|------|------|--------|--------|--------|-----------|-----|------|",
        ]
        for r in results:
            pe = r.get("pe_range") or [None, None]
            lines.append(
                f"| {r.get('broker','—')} | {str(r.get('title','—'))[:20]} "
                f"| {r.get('date','—')} | {r.get('target_price','—')} "
                f"| {pe[0] if pe else '—'} | {pe[1] if pe else '—'} "
                f"| {r.get('ev_ebitda','—')} | {r.get('pb','—')} | {r.get('rating','—')} |"
            )

        valid = [r for r in results if r.get("pe_range") and r["pe_range"][0]]
        pe_min = min(r["pe_range"][0] for r in valid) if valid else None
        ev_vals = [r["ev_ebitda"] for r in results if r.get("ev_ebitda")]
        ev_min = min(ev_vals) if ev_vals else None

        lines += [
            "",
            "### 悲观下限（轨道二直接使用）",
            f"- 悲观PE下限: {pe_min if pe_min else '[fallback-倍数缺失]'}",
            f"- 悲观EV/EBITDA下限: {ev_min if ev_min else '[fallback-倍数缺失]'}",
        ]
        return "\n".join(lines)

    except Exception as e:
        return f"[fallback-提取失败: {e}] 请使用申万行业历史熊市低位PE"


def run_valuation(code: str, stock_name: str, report_year: str,
                  price: str, months: int, max_reports: int):
    """主流程：千问VL识图 + 研报倍数 -> DeepSeek完整估值"""

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Step1: 千问VL读取所有截图
    print(f"\n--- Step1: 千问VL识别截图 ---")
    financial_data = extract_images_with_qwen(code, stock_name)

    # 保存识别结果（调试用）
    raw_path = OUTPUT_DIR / f"{code}_qwen_extracted.txt"
    raw_path.write_text(financial_data, encoding="utf-8")
    print(f"识别结果已保存: {raw_path}")

    # Step2: 下载研报倍数
    print(f"\n--- Step2: 获取研报估值倍数 ---")
    report_table = download_reports(code, months, max_reports)

    # Step3: DeepSeek跑双轨估值
    print(f"\n--- Step3: DeepSeek双轨估值分析 ---")
    client = get_deepseek_client()

    user_text = f"""标的：{stock_name}（{code}）
报告期：{report_year}年报
当前股价：{"请从以下数据中读取" if not price else price + " 元"}

以下是从截图中提取的所有财务数据：

{financial_data}

---

{report_table}

请根据以上数据完成完整的双轨估值分析。"""

    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        max_tokens=8000,
        temperature=0.1,
        messages=[
            {"role": "system", "content": VALUATION_PROMPT},
            {"role": "user", "content": user_text}
        ]
    )

    report = response.choices[0].message.content

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = OUTPUT_DIR / f"{code}_{timestamp}_估值报告.md"
    out_path.write_text(report, encoding="utf-8")

    print(f"\n完成！估值报告: {out_path}")
    print(f"\n{'='*60}")
    print(report[:2000])
    print(f"\n... （完整报告见 Artifacts）")


def main():
    parser = argparse.ArgumentParser(description="双轨估值分析")
    parser.add_argument("--code", required=True, help="股票代码，如 002223")
    parser.add_argument("--name", required=True, help="股票名称，如 鱼跃医疗")
    parser.add_argument("--year", required=True, help="报告年度，如 2025")
    parser.add_argument("--price", required=False, default="", help="当前股价（可不填，从截图自动读取）")
    parser.add_argument("--months", type=int, default=12, help="查询研报月数")
    parser.add_argument("--max-reports", type=int, default=10, help="最多分析研报数")
    args = parser.parse_args()

    print(f"开始双轨估值分析")
    print(f"   标的: {args.name}（{args.code}）")
    print(f"   报告期: {args.year}年报")

    run_valuation(
        code=args.code,
        stock_name=args.name,
        report_year=args.year,
        price=args.price,
        months=args.months,
        max_reports=args.max_reports,
    )


if __name__ == "__main__":
    main()

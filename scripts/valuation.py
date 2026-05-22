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

---

## 数据提取通用规则（两轨共用）

### 单位换算（最高优先级）
- 巨潮年报 PDF 原始单位为"元"，提取后必须统一换算：亿元 = 原始数值 ÷ 100,000,000
- 每个科目换算后须做数量级合理性检验：
  - 任何单一科目不得超过资产总计
  - 固定资产通常不超过总资产的 60%
  - 货币资金通常在总资产的 10%-40% 之间
- 若换算结果明显异常，须停止计算，重新核查原始数字

### 货币资金骤降识别规则（强化版）

触发条件：货币资金较上期下降超过 30%。

触发后必须执行以下判断树，不得跳过任何步骤：

Step 1：检查以下科目是否存在增量（与上期相比）：
  - 交易性金融资产
  - 其他流动资产
  - 其他非流动资产
  - 长期应收款
  - 定期存款（若在附注中单独披露）

Step 2：对每个有增量的科目，执行科目性质判断：

  【其他流动资产 / 其他非流动资产的判断规则】
  情况A：附注明确说明含"定期存款"/"结构性存款"/"大额存单"
    → 该部分按 90% 折算，标注"定存归并"，纳入"现金类资产"

  情况B：附注无说明，但该科目增量可覆盖货币资金降幅的 70% 以上
    → 按 60% 折算（保守中间值），标注"⚠️性质待确认，已按60%保守折算"

  情况C：附注明确说明不含定存，或增量覆盖度 < 70%
    → 按 20% 折算

  禁止规则：不得对"其他非流动资产"不加判断直接套用 20% 默认折算率。
  必须先走上述 A/B/C 判断，再确定折算率。

Step 3：将情况A确认的定存金额与货币资金合并，标注为"现金类资产"，折算率 90%，在报告中单独说明。

Step 4：两轨现金口径必须同步更新（轨道一清算 + 轨道二净现金）。

### 异常科目处理
- 某科目账面值较上期增幅超过 50%，须单独说明原因并重新评估折算率
- 提取不到的字段必须标注"⚠️未找到"，严禁推算或捏造。遇到数据缺口时的唯一合法处理是标注 + 说明对结论影响方向，不得填入估算值继续计算。
- 所有数据注明来源和报告期

---

## 【轨道一】分红累加 + 清算价值

### 第一步：历史分红率

- 分红总额 = 历年现金派息金额之和，股票回购金额须全部剔除
- 若数据同时列示"含回购"与"不含回购"，必须使用"不含回购"口径
- 在报告中须注明："分红率计算已剔除回购金额 X 亿元"
- 分红率 = 历史累计现金分红（逐年加总，不含回购）÷ 历史累计归母净利润（逐年加总）
- 中期分红归属于对应利润年度，与同年末期分红合并

计算分红率前，必须先输出以下年份对照表：

  年份  归母净利润(亿)  现金分红(亿，不含回购)  备注（如有中期分红注明归属年度）
  20XX   XX.XX          XX.XX
  ...
  合计   XX.XX          XX.XX
  历史分红率 = XX.XX ÷ XX.XX = XX.X%

### 第二步：清算价值估算

逐科目折算（折算率表）：

  货币资金 + 交易性金融资产          → 100%
  定期存款/结构性存款（确认归并的）  → 90%（见骤降判断树情况A）
  其他非流动资产（性质待确认）        → 60%（见骤降判断树情况B）
  其他非流动资产（确认非定存）        → 20%（见骤降判断树情况C）
  应收票据 + 应收账款                → 80%
  存货                               → 60%
  长期股权投资                       → 70%
  固定资产                           → 50%
  在建工程                           → 40%
  使用权资产                         → 30%
  无形资产                           → 30%
  商誉                               → 0%（强制归零）
  其他资产                           → 20%

负债扣除口径：扣除"负债合计"（含流动负债 + 非流动负债全部），不得仅扣除有息负债。

计算公式：
- 清算净值 = 资产折算合计 - 负债合计（100%）
- 归母清算净值 = 清算净值 × 归母权益占比（= 1 - 少数股东权益 ÷ 股东权益合计）
- 每股清算价值 = 归母清算净值 ÷ 总股本

### 第三步：正常盈利力判断（三情景）

取近 10-11 年归母净利润逐年列示。

情景定义：

  悲观情景计算规则（强化）：
    Step 1：识别历史上归母净利润连续下滑或处于低位的年份区间（至少连续 2 年）
    Step 2：取该连续区间内所有年份的均值，记为"低谷连续年均值"
    Step 3：计算近 3 年均值
    Step 4：悲观情景 = MIN（低谷连续年均值，近3年均值）

    禁止规则（违反以下任一条，结果无效，须重新计算）：
    ❌ 禁止以单一年度数据作为悲观情景（哪怕该年是历史最低）
    ❌ 禁止取孤立的单年低点（如仅取2019年一年）
    ❌ 禁止剔除亏损年份

    若历史上不存在连续 2 年低谷（如利润单调增长）：
    → 悲观情景 = 近 3 年均值

  中性情景：近 10-11 年全周期均值
  乐观情景：历史峰值区间（至少连续 2 年）均值

必须验证：悲观值 < 中性值 < 乐观值。

若不满足，按以下修正路径处理，不得直接输出：
  违反"悲观 ≥ 中性"：重新检查悲观值是否误用单年极值，重新按上述规则执行。
    修正后仍不满足 → 悲观值强制设为中性值的 80%，标注⚠️
  违反"中性 ≥ 乐观"：扩大峰值区间至历史最高连续 2 年均值。
    修正后仍不满足 → 乐观值强制设为中性值的 120%，标注⚠️

每个情景均需说明假设依据。

### 第四步：内在价值计算

内在价值总额（亿元）= 正常盈利力 × 10年 × 历史分红率 + 归母清算净值
每股内在价值 = 内在价值总额 ÷ 总股本

对悲观、中性、乐观三个情景分别计算，输出三个内在价值。

### 轨道一输出格式

数据汇总表：列示所有提取数据、换算过程、异常科目说明、回购剔除明细、定期存款归并情况

分红率年份对照表：（见第一步强制输出格式）

估值计算过程：逐步展示第一至第四步，数字可追溯

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

每个板块唯一对应一种估值方法：
- 稳定盈利、增速 <15%：PE
- 净利增速 ≥15% 的成长板块：PEG（见下方门禁规则）
- 重资产、周期性：EV/EBITDA
- 亏损/早期成长：PS

PEG 法门禁规则（同时满足全部 3 项才可启用，否则降级为 PE + 溢价）：
  ✅ 条件1：存在针对该板块的专项分部研报，且研报明确给出该板块 PEG 倍数
  ✅ 条件2：近 3 年 CAGR 连续为正
  ✅ 条件3：研报发布日期在近 12 个月内
  不满足任一条件 → 降级为 PE + 溢价判断（见下方）

成长板块 PE 溢价规则：
  基准：整体 PE（从研报或 Fallback 表取值）
  溢价上限：+2x（如整体 PE=18x，成长板块最高 20x）
  溢价条件：必须有具体事件支撑（新产品认证/海外准入/重大合同等），须在报告中列明事件
  无具体事件支撑 → 溢价为 0，与成熟板块统一使用整体 PE

在输出每个成长板块倍数前，必须先回答：
  "该板块是否存在分板块专项研报？[是/否]"
  若否 → 自动锁定为整体 PE + 溢价（≤+2x，须列明事件）

Fallback 触发规则：

  触发条件（满足任一即触发）：
  - 近 12 个月内无个股研报
  - 研报有列表但未披露估值倍数
  - 用户未能提供研报数据

  触发后执行规则：
  ① 该板块估值暂停，在输出表中该板块悲观倍数列填入 [fallback-待补充]，段估值列填入 —
  ② 在报告末尾单独输出"Fallback 待补充清单"：
     列明每个触发板块的名称、适配估值方法（PE/EV·EBITDA等）、
     以及用户需要补充的具体信息（如：该行业近3年熊市低位PE区间）
  ③ SOTP 总估值输出两个版本：
     - 已知板块合计（不含 fallback 板块）：XX 亿
     - 完整估值需补充以下数据后重新计算：[列明 fallback 板块]

  Fallback 触发后禁止规则：
  ❌ 禁止自行估算任何行业的历史熊市 PE
  ❌ 禁止用"保守估计"/"通常在X倍左右"等措辞填入数字
  ❌ 禁止混用其他板块的研报倍数替代

### 第三步：SOTP 计算

板块净利润分配：

  方法一（无分部毛利率时）：
    板块净利润 = 板块营收 × 整体净利润率
    整体净利润率 = 归母净利润 ÷ 总营收

  方法二（有分部毛利率时，优先使用）：
    Step 1：各板块毛利润 = 板块营收 × 板块毛利率
    Step 2：各板块毛利润占比 = 板块毛利润 ÷ 毛利润合计
    Step 3：板块净利润 = 归母净利润 × 板块毛利润占比
    注：若某板块净利润为负，强制归零，剩余归母净利润重新按占比分配，标注⚠️

计算公式：
- PE 板块估值 = 板块净利润 × 悲观 PE
- PEG 板块估值 = 板块净利润 × (悲观 PEG × 板块 CAGR%)
- 净现金 = 货币资金 + 短期金融资产（含定存折算，若触发骤降规则）- 有息负债
- SOTP 总估值 = Σ(各板块估值) + 净现金
- 每股 SOTP = SOTP 总估值 ÷ 总股本
- 隐含溢价 = (当前股价 - 每股 SOTP) ÷ 每股 SOTP × 100%

净现金口径一致性：若触发货币资金骤降规则，两轨净现金必须同步纳入定存折算金额，口径保持一致。

### 轨道二输出格式

板块估值表：
| 板块 | 营收(亿) | 段净利(亿) | 方法 | 关键参数 | 悲观倍数 | 段估值(亿) | 成长板块专项研报[是/否] |
|------|---------|-----------|------|---------|---------|-----------|----------------------|

并列示：
- SOTP 悲观总估值、每股 SOTP、隐含溢价
- 研报来源明细（券商/标题/日期/采用倍数；fallback 时注明锚定表来源）
- 校验：利润校验、商誉风险、净现金构成明细、PEG校验

---

## 【交叉验证】双轨对比与综合结论

| 维度 | 轨道一 | 轨道二 | 差异及原因 |
|------|--------|--------|-----------|
| 每股估值（中性/悲观） | X 元 | X 元 | — |
| 当前股价 | X 元 | X 元 | — |
| 隐含安全边际/溢价 | X% | X% | — |
| 净现金处理口径 | XX亿（含定存折算说明） | XX亿（同口径） | 差异须逐项归因 |
| 商誉处理 | 强制归零 | 计入净现金校验 | — |

差异归因分析（必填）：逐条说明两套结果差异来源。
综合判断：差异 <20% 为收敛互印，≥20% 须说明哪套更适合本公司特性及理由。
估值局限性说明（必填）、风险提示（必填）均须输出。

---

## 特殊情况处理
- 商誉：轨道一强制归零，额外测算 10% 折算情景
- 货币资金骤降：执行强化版判断树，两轨口径同步
- 其他非流动资产大幅增加：必须走 A/B/C 判断，禁止直接套 20% 默认折算率
- 无分产品数据：轨道二降级为整体 PE 估值，注明"无法拆分板块"
- 亏损年份：纳入计算，不得剔除（轨道一）
- Fallback 触发后：板块估值暂停，输出 [fallback-待补充] + Fallback 待补充清单，禁止自行估算倍数

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

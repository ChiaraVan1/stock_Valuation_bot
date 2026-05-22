"""
双轨估值系统 - 完整版 v2
Step1: 千问VL读截图 -> 结构化文字
Step2: 读取外部研报倍数md（优先）或自行下载
Step3: DeepSeek跑双轨估值逻辑
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
    → 按 90% 折算，标注"⚠️性质待确认，已按90%保守折算"

  情况C：附注明确说明不含定存，或增量覆盖度 < 70%
    → 按 20% 折算

  禁止规则：不得对"其他非流动资产"不加判断直接套用 20% 默认折算率。
  必须先走上述 A/B/C 判断，再确定折算率。

Step 3：将情况A/B确认的定存金额与货币资金合并，标注为"现金类资产"，在报告中单独说明。

Step 4：两轨现金口径必须同步更新（轨道一清算 + 轨道二净现金）。

### 异常科目处理
- 某科目账面值较上期增幅超过 50%，须单独说明原因并重新评估折算率
- 提取不到的字段必须标注"⚠️未找到"，严禁推算或捏造。
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
  定期存款/结构性存款（确认归并的）  → 90%
  其他非流动资产（覆盖度≥70%待确认）→ 90%
  其他非流动资产（确认非定存）        → 20%
  应收票据 + 应收账款                → 80%
  存货                               → 60%
  长期股权投资                       → 70%
  其他权益工具投资                   → 70%
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
    优先识别最近一次连续低谷（距今最近的连续下滑区间），而非历史绝对最低点。若识别出的低谷期与当前利润水平相差超过50%（如公司规模已发生质变），则跳过该历史低谷，改用近3年均值作为悲观情景。
    Step 2：取该连续区间内所有年份的均值，记为"低谷连续年均值"
    Step 3：计算近 3 年均值
    Step 4：悲观情景 = MIN（低谷连续年均值，近3年均值）

    禁止规则：
    ❌ 禁止以单一年度数据作为悲观情景
    ❌ 禁止取孤立的单年低点
    ❌ 禁止剔除亏损年份

    若历史上不存在连续 2 年低谷：
    → 悲观情景 = 近 3 年均值

  中性情景：近 10-11 年全周期均值
  乐观情景：历史峰值区间（至少连续 2 年）均值

必须验证：悲观值 < 中性值 < 乐观值。

若不满足，按以下修正路径处理：
  违反"悲观 ≥ 中性"：重新检查悲观值是否误用单年极值，重新按上述规则执行。
    修正后仍不满足 → 悲观值强制设为中性值的 80%，标注⚠️
  违反"中性 ≥ 乐观"：扩大峰值区间至历史最高连续 2 年均值。
    修正后仍不满足 → 乐观值强制设为中性值的 120%，标注⚠️

### 第四步：内在价值计算

内在价值总额（亿元）= 正常盈利力 × 10年 × 历史分红率 + 归母清算净值
每股内在价值 = 内在价值总额 ÷ 总股本

对悲观、中性、乐观三个情景分别计算，输出三个内在价值。

### 轨道一输出格式

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
  不满足任一条件 → 降级为 PE + 溢价判断

成长板块 PE 溢价规则：
  基准：整体 PE（从研报倍数表取悲观下限）
  溢价上限：+2x
  溢价条件：必须有具体事件支撑（新产品认证/海外准入/重大合同等），须在报告中列明事件
  无具体事件支撑 → 溢价为 0，与成熟板块统一使用整体 PE

Fallback 触发规则：
  触发条件（满足任一即触发）：
  - 近 12 个月内无个股研报
  - 研报有列表但未披露估值倍数
  - 研报倍数表标注 [fallback-无研报] 或 [fallback-倍数缺失]

  触发后执行：
  ① 板块估值暂停，填入 [fallback-待补充]
  ② 报告末尾输出"Fallback 待补充清单"
  ③ SOTP 总估值输出已知板块合计 + 缺失说明

  Fallback 触发后禁止：
  ❌ 禁止自行估算任何行业的历史熊市 PE
  ❌ 禁止用"保守估计"等措辞填入数字

### 第三步：SOTP 计算

板块净利润分配（有分部毛利率时优先使用）：
  Step 1：各板块毛利润 = 板块营收 × 板块毛利率
  Step 2：各板块毛利润占比 = 板块毛利润 ÷ 毛利润合计
  Step 3：板块净利润 = 归母净利润 × 板块毛利润占比

计算公式：
- PE 板块估值 = 板块净利润 × 悲观 PE
- 净现金 = 货币资金 + 短期金融资产（含定存折算，若触发骤降规则）- 有息负债
- SOTP 总估值 = Σ(各板块估值) + 净现金
- 每股 SOTP = SOTP 总估值 ÷ 总股本
- 隐含溢价 = (当前股价 - 每股 SOTP) ÷ 每股 SOTP × 100%

净现金口径一致性：若触发货币资金骤降规则，两轨净现金必须同步纳入定存折算金额。

---

## 【交叉验证】双轨对比与综合结论

差异归因分析（必填）、估值局限性说明（必填）、风险提示（必填）均须输出。

---

## 特殊情况处理
- 商誉：轨道一强制归零，额外测算 10% 折算情景
- 货币资金骤降：执行强化版判断树，两轨口径同步
- 其他非流动资产大幅增加：必须走 A/B/C 判断，禁止直接套 20% 默认折算率
- 亏损年份：纳入计算，不得剔除（轨道一）

请根据以下提取的财务数据完成完整的双轨估值分析，输出完整报告（Markdown格式）：
"""


def get_deepseek_client():
    # 先试阿里云
    try:
        client = OpenAI(
            api_key=os.environ.get("QWEN_API_KEY", ""),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        # 测试一下是否可用
        client.chat.completions.create(
            model="deepseek-v3",
            max_tokens=1,
            messages=[{"role": "user", "content": "hi"}]
        )
        print("✅ 使用阿里云DeepSeek额度")
        return client
    except Exception as e:
        print(f"⚠️ 阿里云额度不可用({e})，切换到DeepSeek官方")
        return OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            base_url="https://api.deepseek.com"
        )


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
            model="qwen-vl-ocr-latest",
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

要求：只输出数据，不要分析，不要遗漏任何数字。若截图中同时存在合并报表和母公司报表，只提取合并报表数据，忽略母公司报表，并在输出开头标注"已提取：合并报表"。"""
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



OCR_VALIDATE_PROMPT = """以下是从财务截图中提取的原始文字数据。
请从中找出以下关键数字，只输出JSON，不输出任何其他文字。
若某字段找不到，填 null。

输出格式：
{
  "total_assets": 数字（亿元），
  "total_liabilities": 数字（亿元），
  "total_equity": 数字（亿元），
  "parent_equity": 数字（亿元，归属于母公司所有者权益合计）,
  "minority_equity": 数字（亿元，少数股东权益）,
  "net_profit_consolidated": 数字（亿元，合并利润表归属于母公司的净利润）,
  "net_profit_parent": 数字（亿元，母公司利润表净利润，如有）,
  "total_shares": 数字（亿股，合并报表股本，通常为整数如10.02，不得用分红金额倒推）,
  "cash_consolidated": 数字（亿元，合并资产负债表货币资金，注意区分合并报表与母公司报表，必须取合并口径）
}

注意：
- total_liabilities 通常远小于 total_assets，约为 total_assets 的 10%-40%
- net_profit_consolidated 是合并报表数字
- 若发现 net_profit_parent 与 net_profit_consolidated 差异超过30%，务必两个都填，不要混用

原始数据：
"""


def validate_ocr_data(financial_data: str, client) -> dict:
    """Step1.5: 用DeepSeek结构化关键数字并执行恒等式校验，捕获OCR读错。"""
    print("\n--- Step1.5: OCR数据校验 ---")

    resp = client.chat.completions.create(
        model="deepseek-v3.2",
        max_tokens=400,
        temperature=0,
        messages=[
            {"role": "system", "content": "只输出JSON，不输出任何其他文字。"},
            {"role": "user", "content": OCR_VALIDATE_PROMPT + financial_data[:8000]}
        ]
    )
    raw = resp.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(raw)
    except Exception as e:
        print(f"  结构化提取失败: {e}，跳过校验")
        return {"errors": [], "warnings": [], "extracted": {}}

    errors = []
    warnings = []
    ta = data.get("total_assets")
    tl = data.get("total_liabilities")
    te = data.get("total_equity")
    pe = data.get("parent_equity")
    me = data.get("minority_equity")
    np_c = data.get("net_profit_consolidated")
    np_p = data.get("net_profit_parent")

    # 校验1：资产负债表恒等式
    if ta and tl and te:
        diff = abs(ta - tl - te)
        if diff > ta * 0.01:
            errors.append(
                f"资产负债表恒等式不成立：总资产({ta:.2f}亿) ≠ 负债({tl:.2f}亿) + 权益({te:.2f}亿)，"
                f"差额{diff:.2f}亿，可能OCR读错负债或权益"
            )
        else:
            print(f"  ✓ 恒等式通过：{ta:.2f} = {tl:.2f} + {te:.2f}")

    # 校验2：归母净利润不得超过归母净资产
    if np_c and pe:
        if np_c > pe:
            errors.append(
                f"归母净利润({np_c:.2f}亿) > 归母净资产({pe:.2f}亿)，"
                f"可能误用母公司口径或OCR读错"
            )
        else:
            print(f"  ✓ 净利润合理：{np_c:.2f}亿 < {pe:.2f}亿")

    # 校验3：母公司 vs 合并净利润差异预警
    if np_c and np_p:
        diff_pct = abs(np_c - np_p) / max(np_c, np_p)
        if diff_pct > 0.3:
            warnings.append(
                f"合并净利润({np_c:.2f}亿) 与母公司净利润({np_p:.2f}亿) 差异{diff_pct:.1%}，"
                f"请确认使用合并口径({np_c:.2f}亿)"
            )

    # 校验4：负债率不得超过80%
    if ta and tl:
        ratio = tl / ta
        if ratio > 0.8:
            errors.append(
                f"负债合计({tl:.2f}亿)占总资产({ta:.2f}亿)的{ratio:.1%}，"
                f"超过80%上限，可能OCR把股东权益误读为负债"
            )
        else:
            print(f"  ✓ 负债率合理：{ratio:.1%}")

    # 校验5：少数股东权益 + 归母权益 ≈ 股东权益合计
    if pe and me and te:
        diff = abs(pe + me - te)
        if diff > te * 0.01:
            warnings.append(
                f"归母权益({pe:.2f}) + 少数股东({me:.2f}) = {pe+me:.2f} ≠ 权益合计({te:.2f})，差额{diff:.2f}亿"
            )

    if errors:
        print(f"\n  🚨 {len(errors)} 个严重错误：")
        for e in errors: print(f"     ❌ {e}")
    if warnings:
        print(f"\n  ⚠️  {len(warnings)} 个警告：")
        for w in warnings: print(f"     ⚠️ {w}")
    if not errors and not warnings:
        print(f"  ✅ 全部校验通过")

    return {"errors": errors, "warnings": warnings, "extracted": data}


def format_validation_report(validation: dict) -> str:
    """将校验结果格式化为注入DeepSeek prompt的文字块"""
    extracted = validation.get("extracted", {})
    errors = validation.get("errors", [])
    warnings = validation.get("warnings", [])

    lines = ["## OCR数据校验结果（Step1.5自动生成）", ""]
    if extracted:
        lines.append("### 关键数字（已校验，DeepSeek请以此为准）")
        fields = [
            ("total_assets",            "总资产（亿元）"),
            ("total_liabilities",       "负债合计（亿元）"),
            ("total_equity",            "股东权益合计（亿元）"),
            ("parent_equity",           "归母净资产（亿元）"),
            ("minority_equity",         "少数股东权益（亿元）"),
            ("net_profit_consolidated", "归母净利润-合并口径（亿元）⭐估值请使用此值"),
            ("net_profit_parent",       "净利润-母公司口径（亿元，仅供参考）"),
            ("total_shares", "总股本（亿股）⭐所有每股计算请使用此值，不得用分红倒推"),
            ("cash_consolidated", "货币资金-合并口径（亿元）⭐清算价值请使用此值，不得使用母公司口径"),
        ]
        for key, label in fields:
            val = extracted.get(key)
            lines.append(f"- {label}：{'⚠️未找到' if val is None else val}")
        lines.append("")

    if errors:
        lines.append("### 🚨 严重错误（估值前必须修正）")
        for e in errors: lines.append(f"- ❌ {e}")
        lines.append("")
        lines.append("> **DeepSeek执行指令**：请在报告开头列「数据修正说明」，")
        lines.append("> 以OCR校验结果数字为准，不得使用原始错误值继续计算。")
        lines.append("")

    if warnings:
        lines.append("### ⚠️ 警告")
        for w in warnings: lines.append(f"- {w}")
        lines.append("")

    if not errors and not warnings:
        lines.append("### ✅ 全部校验通过，数据可信")
        lines.append("")

    return "\n".join(lines)

def load_report_md(report_md_path: str) -> str:
    """从外部传入的研报倍数md文件读取内容"""
    path = Path(report_md_path)
    if not path.exists():
        raise FileNotFoundError(f"研报倍数文件不存在: {report_md_path}")
    content = path.read_text(encoding="utf-8")
    print(f"已读取外部研报倍数文件: {path}（{len(content)} 字符）")
    return content


def download_reports(code: str, months: int, max_reports: int) -> str:
    """自行下载研报并提取倍数（fallback，当未提供外部md时使用）"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    end_date = datetime.now()
    begin_date = end_date - timedelta(days=months * 30)

    print(f"下载研报（fallback模式）...")
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
                model="deepseek-v3.2",
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
                  price: str, months: int, max_reports: int,
                  report_md: str = ""):
    """主流程：千问VL识图 + 研报倍数 -> DeepSeek完整估值"""

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Step1: 千问VL读取所有截图
    print(f"\n--- Step1: 千问VL识别截图 ---")
    financial_data = extract_images_with_qwen(code, stock_name)

    raw_path = OUTPUT_DIR / f"{code}_qwen_extracted.txt"
    raw_path.write_text(financial_data, encoding="utf-8")
    print(f"识别结果已保存: {raw_path}")

    # Step1.5: OCR数据校验
    deepseek_client = get_deepseek_client()
    validation = validate_ocr_data(financial_data, deepseek_client)
    validation_report = format_validation_report(validation)

    # 有严重错误时警告，但不中断（让DeepSeek在报告里修正并说明）
    if validation["errors"]:
        print(f"\n  ⚠️  检测到OCR读取错误，已注入修正指令到DeepSeek prompt，估值报告将包含数据修正说明。")

    # Step2: 读取研报倍数（优先用外部传入md，否则自行下载）
    print(f"\n--- Step2: 获取研报估值倍数 ---")
    if report_md:
        print(f"使用外部研报倍数文件: {report_md}")
        report_table = load_report_md(report_md)
    else:
        print(f"未提供外部研报倍数文件，自行下载研报...")
        report_table = download_reports(code, months, max_reports)

    # Step3: DeepSeek跑双轨估值
    print(f"\n--- Step3: DeepSeek双轨估值分析 ---")
    client = deepseek_client  # 复用Step1.5已创建的client

    user_text = f"""标的：{stock_name}（{code}）
报告期：{report_year}年报
当前股价：{"请从以下数据中读取" if not price else price + " 元"}

以下是从截图中提取的所有财务数据：

{financial_data}

---

{validation_report}

---

{report_table}

请根据以上数据完成完整的双轨估值分析。"""

    response = client.chat.completions.create(
        model="deepseek-v4-pro",
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
    parser.add_argument("--report-md", required=False, default="",
                        help="外部研报倍数md文件路径（由fetch_and_extract.py产出，优先使用）")
    args = parser.parse_args()

    print(f"开始双轨估值分析")
    print(f"   标的: {args.name}（{args.code}）")
    print(f"   报告期: {args.year}年报")
    if args.report_md:
        print(f"   研报倍数来源: {args.report_md}")
    else:
        print(f"   研报倍数来源: 自行下载（fallback）")

    run_valuation(
        code=args.code,
        stock_name=args.name,
        report_year=args.year,
        price=args.price,
        months=args.months,
        max_reports=args.max_reports,
        report_md=args.report_md,
    )


if __name__ == "__main__":
    main()

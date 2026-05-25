"""
双轨估值系统 - 完整版 v7
核心改动（相对v6）：
  - 跳过OCR中间层，截图直接传给估值Claude（与App行为完全一致）
  - VALUATION_PROMPT 替换为与App完全一致的完整版本
  - 季报参考层仍保留，图片直接传入
  - DeepSeek不支持看图，所以主估值步骤强制使用Claude，无fallback
  - 保留manual_reports / quarterly_reports 可选入口
"""

import argparse
import base64
import json
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from openai import OpenAI
import pdfplumber

from ai_client import text_completion, get_qwen_vl_client
from ai_client import CLAUDE_API_KEY, CLAUDE_BASE_URL, CLAUDE_MODEL

DOCS_DIR        = Path("Supporting Documents for Valuation")
OUTPUT_DIR      = Path("output")
REPORTS_DIR     = Path("output/reports")
MANUAL_RPT_DIR  = Path("manual_reports")
QUARTERLY_DIR   = Path("quarterly_reports")


# ────────────────────────────────────────────────────────────
# 主估值Prompt（与App完全一致的完整版本）
# ────────────────────────────────────────────────────────────

VALUATION_PROMPT = """你是一位 A 股双轨估值分析师，同时运用两套独立估值体系对同一标的进行分析：

- **轨道一**：分红累加 + 清算价值（价值投资视角，适合稳定分红公司）
- **轨道二**：SOTP 分部加总（分部估值视角，适合多业务板块公司）

两套方法并行计算，最终输出交叉验证对比结论。

---

## 数据提取通用规则（两轨共用）

### 单位换算（最高优先级）

- 巨潮年报 PDF 原始单位为"元"，提取后必须统一换算：
  亿元 = 原始数值 ÷ 100,000,000
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
- 提取不到的字段标注"⚠️未找到"，不得推测或捏造
- 所有数据注明来源和报告期

---

## 【轨道一】分红累加 + 清算价值

### 第一步：历史分红率

**分红总额定义与回购剔除规则：**
- 分红总额 = 历年现金派息金额之和，股票回购金额须全部剔除
- 若数据同时列示"含回购"与"不含回购"，必须使用"不含回购"口径
- 在报告中须注明："分红率计算已剔除回购金额 X 亿元"

**数据采集：**
- 尽量采集近 10-11 年数据，覆盖完整景气周期
- 分红率 = 历史累计现金分红（逐年加总，不含回购）÷ 历史累计归母净利润（逐年加总）
- 两者均采用逐年实际数字加总，不得用均值 × 年数替代

**中期分红年份归属规则：**
- 中期分红归属于对应利润年度，与同年末期分红合并
- 以利润产生年份为准，不以实施年份为准
- 年度分红总额 = 末期分红 + 中期分红（均不含回购）

**已公告未实施分红的归属规则：**
- 若截至报告期末，公司已通过股东大会或董事会决议公告了年度分红方案，但实施日期在报告期后，该分红金额仍须归属至对应利润年度，不得以"尚未实施"为由漏计
- 判断依据：看公告日期，不看实施日期

**分红率交叉验证：**
- 须与公司年报或东方财富披露的"分红占净利润比例"交叉核对
- 偏差超过 5 个百分点须重新核查

计算分红率前，必须先输出以下年份对照表：

| 年份 | 归母净利润(亿) | 现金分红(亿，不含回购) | 备注 |
|------|--------------|---------------------|------|
| 20XX | XX.XX        | XX.XX               |      |
| 合计 | XX.XX        | XX.XX               |      |

历史分红率 = XX.XX ÷ XX.XX = XX.X%

---

### 第二步：清算价值估算

逐科目折算：

| 资产科目 | 参考折算率 | 说明 |
|---------|----------|------|
| 货币资金 + 交易性金融资产 | 100% | 现金类合并处理 |
| 定期存款/结构性存款（转入其他科目的） | 90% | 高流动性但有锁定期 |
| 应收票据 + 应收账款 | 80% | 扣除坏账风险 |
| 存货 | 60% | 行业特性折价 |
| 长期股权投资 | 70% | 联营企业权益 |
| 固定资产 | 50% | 专用设备折价 |
| 在建工程 | 40% | 未完工折价 |
| 使用权资产 | 30% | 租赁资产 |
| 无形资产 | 30% | 采矿权/品牌等 |
| 商誉 | 0% | 强制归零，清算时无法单独变现 |
| 其他资产 | 20% | 保守处理 |

**负债扣除口径：**
- 扣除"负债合计"（含流动负债 + 非流动负债全部）
- 不得仅扣除有息负债，这会严重高估清算净值

**计算公式：**
- 清算净值 = 资产折算合计 - 负债合计（100%）
- 归母清算净值 = 清算净值 × 归母权益占比
  （归母权益占比 = 1 - 少数股东权益 ÷ 股东权益合计）
- 每股清算价值 = 归母清算净值 ÷ 总股本

---

### 第三步：正常盈利力判断（三情景）

取近 10-11 年归母净利润逐年列示，直接计算各情景均值：

| 情景 | 说明 | 盈利估算方式 |
|------|------|------------|
| 悲观 | 行业低谷持续 | 近 3 年均值或历史最低合理值，取两者中更低者 |
| 中性 | 温和复苏 | 近 10-11 年全周期均值 |
| 乐观 | 回归景气高峰 | 历史峰值区间均值 |

- 不人为剔除任何年份，亏损年份纳入计算
- 每个情景均需说明假设依据

**情景自洽校验规则：**
三情景数值计算完成后，必须验证：**悲观值 < 中性值 < 乐观值**。若不满足，停止计算，按以下步骤逐一排查：
1. 悲观：重新检查，取"近 3 年均值"与"当前低谷连续年份均值"中更低者
2. 中性：确认为全周期均值，不得高于乐观值
3. 乐观：确认为峰值区间均值，必须是三者最高值

不得在三情景排列不单调的情况下继续输出估值结果。

---

### 第四步：内在价值计算（核心公式）

采用"持有 N 年分红累计 + 期末清算退出"一次性贴现模型：

```
内在价值总额（亿元）= 正常盈利力 × N年 × 历史分红率 + 归母清算净值
每股内在价值 = 内在价值总额 ÷ 总股本
```

- 默认持有期 N = 10 年
- 清算价值取当前时点数据（不做增长假设，保守处理）
- 对悲观、中性、乐观三个情景分别计算，输出三个内在价值

---

### 第五步：轨道一辅助指标

| 指标 | 计算方式 |
|------|---------|
| 每股净资产（归母） | 归母净资产 ÷ 总股本 |
| PB | 当前股价 ÷ 每股净资产 |
| 持有期隐含市盈率 | 内在价值总额 ÷ 正常盈利力（中性） |
| 净现金占股价比 | （货币资金 + 现金类资产 - 有息负债）÷ 总股本 ÷ 当前股价 |

---

### 轨道一输出格式

**数据汇总表：** 列示所有提取数据、换算过程、异常科目说明、回购剔除明细、定期存款归并情况

**估值计算过程：** 逐步展示第一至第四步，数字可追溯

**轨道一结论表：**

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

安全边际 = (每股内在价值 - 当前股价) ÷ 当前股价 × 100%
正值 = 有安全边际（被低估），负值 = 被高估

---

## 【轨道二】SOTP 分部加总

### 第一步：识别并确认分部数据

从截图中提取数据：
1. 识别所有独立业务板块（过滤汇总行、合计行、"其他业务"）
2. 读取每个板块的营收金额与毛利率
3. 标注数据来自截图的哪一页

**输出格式（须确认后继续）：**
```
板块名称          营收（亿）    毛利率    数据来源
板块A             XX.XX       XX.X%    2024年报 p.XX
板块B             XX.XX       XX.X%    2024年报 p.XX
合计              XX.XX
```

---

### 第二步：匹配估值方法与倍数

每个板块唯一对应一种估值方法，按以下优先级判断：

| 板块特征 | 适配方法 | 判断条件 |
|----------|----------|----------|
| 稳定盈利、成熟行业、增速 <15% | PE | 净利润为正，增速低 |
| 稳定盈利、净利增速 ≥15% 的成长板块 | PEG | 净利润为正，增速持续且可预期 |
| 重资产、周期性 | EV/EBITDA | 折旧摊销占比高，利润波动大 |
| 亏损/早期成长 | PS | 净利润为负或接近盈亏平衡 |
| 金融/地产类资产 | PB | 资产驱动型业务 |
| 持有型资产 | 分部净值法 | 资产包、投资性房地产等 |

**倍数取值规则：**

- **主要来源：** 从研报倍数表中取各研报对应板块的 PE / PEG / EV·EBITDA 估值区间，汇总后**只取悲观下限**。

- **Fallback（无有效研报时）：**
  标注原因码：
  - `[fallback-无研报]`：近 12 个月内无个股研报
  - `[fallback-倍数缺失]`：研报有列表但未披露估值倍数
  - `[fallback-用户跳过]`：用户无法执行下载命令

- **PEG 增速：** 从利润历史数据读取近三年归母净利润，计算 CAGR = (最新年 ÷ 三年前) ^ (1/3) − 1；增速须连续为正且可预期，否则降级为 PE 法并注明原因。

**【成长板块倍数上限规则】**
无分板块专项研报时，PEG法不得启用。
成长板块（增速≥15%）允许在整体PE基础上给予溢价，但须满足以下条件：
- 溢价上限：+2x（即整体PE 18x → 最高20x）
- 溢价须有具体事件支撑（如新品认证、海外准入、重大合同）
- 须在报告中注明溢价理由及对应事件
- 若无具体事件支撑，成长板块与成熟板块统一使用整体PE

---

### 第三步：SOTP 计算

```
板块净利润 = 板块营收 × 整体净利润率
          （若有分部毛利率，按毛利率加权分配，更精确）

PE 板块估值  = 板块净利润 × 悲观 PE
PEG 板块估值 = 板块净利润 × (悲观 PEG × 板块 CAGR%)
其他板块估值 = 对应方法计算

净现金      = 货币资金 + 短期金融资产 - 有息负债
```

**净现金口径一致性规则：**
若本次分析已触发货币资金骤降识别规则，确认存在定期存款重分类，则轨道二净现金必须同步将该定期存款按 90% 折算后计入，与轨道一保持口径一致。

```
SOTP 总估值 = Σ(各板块估值) + 净现金
每股 SOTP   = SOTP 总估值 ÷ 总股本
隐含溢价    = (当前股价 - 每股 SOTP) ÷ 每股 SOTP × 100%
```

---

### 轨道二输出格式

```
板块      营收(亿) 段净利(亿) 方法  关键参数      悲观倍数  段估值(亿)  数据来源
板块A      XX.XX    X.XX     PE    —             15x       XX.X       截图p.XX + 研报
板块B      XX.XX    X.XX     PEG   CAGR=22%      0.8x      XX.X       截图p.XX + 研报
分部合计            XX.XX                                  XXX.X
(+) 净现金（含定期存款折算，若触发骤降规则）                XX.X
SOTP 悲观总估值                                            XXX.X 亿
每股 SOTP                                                  XX.XX 元
当前股价                                                   XX.XX 元
隐含溢价/(折价)                                            +XX.X%

【研报来源明细】
券商名称    报告标题    发布日期    采用倍数    板块适用
XXX券商     XXXXXX     XXXX-XX-XX  PE=XXx     板块A

【校验】
利润校验：分部利润加总 XX.XX 亿 vs 报告归母净利润 XX.XX 亿，误差 X.XX 亿
商誉风险：商誉 XX.XX 亿 → 全额减值时每股影响 −X.XX 元
净现金构成：货币资金 XX.XX + 短期定存折算 XX.XX + 短期金融资产 XX.XX − 有息负债 XX.XX = XX.XX 亿
PEG 校验：增速板块 CAGR 是否连续 3 年为正 → 是/否
```

---

## 【交叉验证】双轨对比与综合结论

### 核心数值对比表

| 维度 | 轨道一（分红清算） | 轨道二（SOTP） | 差异及原因 |
|------|-----------------|--------------|-----------|
| 每股估值（中性/悲观） | X 元 | X 元 | — |
| 当前股价 | X 元 | X 元 | — |
| 隐含安全边际/溢价 | X% | X% | — |
| 净现金处理口径 | XX亿 | XX亿 | 若有差异须逐项说明 |
| 商誉处理 | 强制归零 | 计入净现金校验 | — |

### 差异归因分析（必填）

须逐条说明两套结果的差异来源，常见原因包括：

1. **盈利力假设不同**：轨道一用全周期均值，轨道二用当期板块净利润
2. **成长性计入方式不同**：轨道二通过 PEG 隐含了成长溢价，轨道一不含成长价值
3. **清算价值 vs 经营价值**：轨道一的清算价值是静态资产底线，轨道二反映持续经营定价
4. **净现金口径**：两轨须使用一致的定期存款处理方式

### 综合判断

根据差异归因，给出以下结论：

- **两套估值收敛**（差异 <20%）：互相印证，结论较为可靠
- **两套估值分歧**（差异 ≥20%）：
  - 若轨道一 > 轨道二：可能因公司成长性被 SOTP 充分定价，轨道一低估了成长溢价
  - 若轨道二 > 轨道一：可能因公司分红率偏低或周期处于低谷，轨道一的历史分红率拖低了估值
  - 给出更适合本公司特性的参考方法，并说明理由

---

## 估值局限性说明（必填）

### 轨道一局限性

① 不含净资产成长价值：忽略了公司未来留存收益的再投资积累，对成长型公司存在系统性低估偏差
② 清算价值静态保守：品牌价值、渠道价值、市场地位等软性资产未被计入
③ 分红率假设基于历史：未来分红政策可能变化
④ 持有期假设固定：实际退出时点和清算价值可能与假设存在重大偏差

### 轨道二局限性

① 净利润率分配依赖整体净利润率：若各板块盈利能力差异较大，分配结果可能失真
② 研报估值倍数存在时效性：研报发布时点与当前市场环境仍可能存在偏差
③ 仅取悲观下限：结果偏保守，不代表市场共识估值
④ PEG 增速外推风险：历史 CAGR 不代表未来增速

---

## 风险提示（必填）

须列示影响两套估值的主要不确定因素：

- 盈利周期位置判断（影响轨道一盈利力情景选择）
- 清算价值中占比较大科目的折算率敏感性
- 分红政策变化风险
- 现金类资产锁定风险（大量资金转入长期定存）
- 板块增速可持续性（影响轨道二 PEG 板块估值）
- 商誉减值风险（影响两套估值中的净资产/净现金）
- 海外业务占比较高时，单独提示汇率及贸易政策风险

---

## 特殊情况处理规则

| 情况 | 处理方式 |
|------|---------|
| 含中期分红 | 年度分红 = 中期 + 末期，均按实施时股本计算，归属同一利润年度 |
| 已公告未实施分红 | 归属至对应利润年度，不得漏计 |
| 有送转股年份 | 派现金额按送转后股本计算，报告中注明 |
| 海外收入占比高 | 汇率风险在风险提示中单独列示 |
| 亏损年份 | 纳入累计净利润计算，不得剔除（轨道一） |
| 数据缺失年份 | 标注 ⚠️，说明对分红率计算的影响方向 |
| 含回购年份 | 从分红总额中剔除，数据汇总表中单独列示各年回购金额 |
| 货币资金骤降 | 合并定期存款后重新计算现金类资产，异常科目说明中披露；轨道二净现金须同步纳入，两轨口径保持一致 |
| 商誉占比较大 | 额外测算"商誉按账面 10% 折算"与归零情景对比（轨道一），并在轨道二商誉风险校验中同步体现 |
| 无分产品数据 | 轨道二降级为整体 PE 估值，注明"无法拆分板块" |
| 无有效研报 | 轨道二全板块 fallback，标注对应原因码，结论可靠性降级说明 |

请根据以上截图和研报数据，完成完整的双轨估值分析，输出完整报告（Markdown格式）。"""


# ────────────────────────────────────────────────────────────
# Step1：直接加载截图为base64（跳过OCR）
# ────────────────────────────────────────────────────────────

def load_images_as_base64(code: str, name: str = "") -> list:
    """把截图文件夹里的图片全部转成base64消息块，直接传给Claude"""
    img_dir = DOCS_DIR / code
    if not img_dir.exists() and name:
        img_dir = DOCS_DIR / name
    if not img_dir.exists():
        raise FileNotFoundError(f"未找到截图文件夹: {DOCS_DIR}/{code} 或 {DOCS_DIR}/{name}")

    supported = {".png", ".jpg", ".jpeg", ".webp"}
    img_files = [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in supported]

    if not img_files:
        raise FileNotFoundError(f"文件夹中没有图片: {img_dir}")

    if len(img_files) > 15:
        print(f"⚠️  截图数量 {len(img_files)} 张，超过15张可能消耗大量token，建议精简")

    print(f"找到 {len(img_files)} 张截图，直接传图（跳过OCR）...")
    image_blocks = []
    for img_path in img_files:
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = img_path.suffix.lower().replace(".", "")
        media_type = f"image/{'jpeg' if ext == 'jpg' else ext}"
        image_blocks.append({
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{b64}"}
        })
        print(f"  ✅ {img_path.name}")

    return image_blocks


# ────────────────────────────────────────────────────────────
# 可选入口A：手动研报（manual_reports/{code}/）
# ────────────────────────────────────────────────────────────

def load_manual_reports(code: str) -> str:
    """有就读，没有就返回空字符串。"""
    rpt_dir = MANUAL_RPT_DIR / code
    if not rpt_dir.exists():
        print("ℹ️  手动研报：未找到文件夹，跳过")
        return ""

    pdf_files = list(rpt_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"ℹ️  手动研报：manual_reports/{code}/ 为空，跳过")
        return ""

    print(f"找到 {len(pdf_files)} 份手动研报，提取文字...")
    parts = []
    for i, pdf_path in enumerate(pdf_files):
        print(f"  [{i+1}/{len(pdf_files)}] {pdf_path.name}")
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages[:30]:
                    text += page.extract_text() or ""
            parts.append(f"=== 手动研报{i+1}: {pdf_path.name} ===\n{text[:12000]}")
            print(f"    ✅ 完成（{len(text)}字）")
        except Exception as e:
            print(f"    ⚠️  提取失败: {e}")

    return "\n\n".join(parts)


# ────────────────────────────────────────────────────────────
# 可选入口B：季报（quarterly_reports/{code}/）
# ────────────────────────────────────────────────────────────

QUARTERLY_OCR_PROMPT = """请识别这张季报截图中的关键数据，重点提取：

1. 报告期（如2026Q1）
2. 营收：本期金额、同比增速
3. 归母净利润：本期金额、同比增速
4. 资产负债表变化（若有）：货币资金、其他流动资产、其他非流动资产、存货、应收账款（本期末+上年末）
5. 经营活动产生的现金流量净额
6. 重要事项中的新产品注册/认证/海外准入/重大合同等催化剂信息

只输出数据，不做分析。"""


def load_quarterly_data(code: str) -> tuple[str, list]:
    """
    有就读并包装成参考层，没有就返回(空字符串, 空列表)。
    返回 (文字说明, 图片blocks列表)
    """
    q_dir = QUARTERLY_DIR / code
    if not q_dir.exists():
        print("ℹ️  季报：未找到文件夹，跳过")
        return "", []

    supported_img = {".png", ".jpg", ".jpeg", ".webp"}
    pdf_files = list(q_dir.glob("*.pdf"))
    img_files = [p for p in sorted(q_dir.iterdir()) if p.suffix.lower() in supported_img]

    if not pdf_files and not img_files:
        print(f"ℹ️  季报：quarterly_reports/{code}/ 为空，跳过")
        return "", []

    print(f"找到季报文件：{len(pdf_files)} 个PDF + {len(img_files)} 张截图")

    quarterly_image_blocks = []
    text_parts = []

    # 季报截图直接转base64
    for i, img_path in enumerate(img_files):
        print(f"  [季报截图 {i+1}] {img_path.name}")
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = img_path.suffix.lower().replace(".", "")
        media_type = f"image/{'jpeg' if ext == 'jpg' else ext}"
        quarterly_image_blocks.append({
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{b64}"}
        })
        print(f"    ✅ base64编码完成")

    # 季报PDF用pdfplumber提取文字
    for i, pdf_path in enumerate(pdf_files):
        print(f"  [季报PDF {i+1}] {pdf_path.name}")
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages[:20]:
                    text += page.extract_text() or ""
            text_parts.append(f"=== 季报PDF{i+1}: {pdf_path.name} ===\n{text[:10000]}")
            print(f"    ✅ 完成（{len(text)}字）")
        except Exception as e:
            print(f"    ⚠️  提取失败: {e}")

    quarterly_text_intro = """---

## 📋 季报参考层（不替代年报，仅用于验证假设和催化剂判断）

> ⚠️ 以下季报数据为参考信息，不作为估值基础：
> - 可用于验证：收入增速是否延续、存货变化、应收账款质量、现金流健康度
> - 可用于判断：是否有新产品认证/海外准入/重大合同等催化剂（可支撑成长板块溢价）
> - 不可从季报获取：分板块收入和毛利率（仅半年报/年报披露）
> - 不可从季报调整：PE/PEG 倍数的精确数字

"""
    if quarterly_image_blocks:
        quarterly_text_intro += f"以下附有 {len(quarterly_image_blocks)} 张季报截图，请直接读取。\n"
    if text_parts:
        quarterly_text_intro += "\n" + "\n\n".join(text_parts)

    quarterly_text_intro += "\n---"

    return quarterly_text_intro, quarterly_image_blocks


# ────────────────────────────────────────────────────────────
# Step2：研报倍数
# ────────────────────────────────────────────────────────────

def load_report_md(report_md_path: str) -> str:
    path = Path(report_md_path)
    if not path.exists():
        raise FileNotFoundError(f"研报倍数文件不存在: {report_md_path}")
    content = path.read_text(encoding="utf-8")
    print(f"已读取研报倍数文件: {path}（{len(content)} 字符）")
    return content


def download_and_extract_reports(code: str, months: int, max_reports: int) -> str:
    """fallback：直接在valuation.py内下载+提取。"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    end_date   = datetime.now()
    begin_date = end_date - timedelta(days=months * 30)

    cmd = ["python", "-m", "eastmoney", "d", "-t", "stock", "-c", code,
           "-s", str(max_reports), "-o", str(REPORTS_DIR),
           "--begin", begin_date.strftime("%Y-%m-%d"),
           "--end",   end_date.strftime("%Y-%m-%d")]
    subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    pdf_files = list(REPORTS_DIR.glob("**/*.pdf"))
    if not pdf_files:
        return "[fallback-无研报] 未下载到研报"

    EXTRACT_SYSTEM = "只输出JSON，不输出任何其他文字。"
    EXTRACT_PROMPT = """从以下研报中提取估值信息，只输出JSON。
格式：{"broker":"券商","title":"标题","date":"YYYY-MM-DD","target_price":数字或null,"pe_range":[悲观PE,乐观PE]或null,"peg":数字或null,"ev_ebitda":数字或null,"pb":数字或null,"rating":"评级"}
研报内容："""

    results = []
    for pdf_path in pdf_files[:max_reports]:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages[:20]:
                text += page.extract_text() or ""
        try:
            raw, _ = text_completion(
                messages=[{"role": "user", "content": EXTRACT_PROMPT + text[:10000]}],
                system=EXTRACT_SYSTEM, max_tokens=300, temperature=0,
                task_label="研报倍数(内嵌)"
            )
            raw = raw.replace("```json", "").replace("```", "").strip()
            results.append(json.loads(raw))
        except Exception:
            pass

    if not results:
        return "[fallback-倍数缺失] 研报未披露估值倍数"

    lines = ["## 研报估值倍数汇总", "",
             "| 券商 | 标题 | 日期 | 目标价 | 悲观PE | 乐观PE | EV/EBITDA | PB | 评级 |",
             "|------|------|------|--------|--------|--------|-----------|-----|------|"]
    for r in results:
        pe = r.get("pe_range") or [None, None]
        lines.append(f"| {r.get('broker','—')} | {str(r.get('title','—'))[:20]} "
                     f"| {r.get('date','—')} | {r.get('target_price','—')} "
                     f"| {pe[0] if pe else '—'} | {pe[1] if pe else '—'} "
                     f"| {r.get('ev_ebitda','—')} | {r.get('pb','—')} | {r.get('rating','—')} |")

    valid   = [r for r in results if r.get("pe_range") and r["pe_range"][0]]
    pe_min  = min(r["pe_range"][0] for r in valid) if valid else None
    ev_vals = [r["ev_ebitda"] for r in results if r.get("ev_ebitda")]
    lines += ["", "### 悲观下限",
              f"- 悲观PE下限: {pe_min if pe_min else '[fallback-倍数缺失]'}",
              f"- 悲观EV/EBITDA下限: {min(ev_vals) if ev_vals else '[fallback-倍数缺失]'}"]
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────
# Step3：主估值（直接传图给Claude，不走fallback）
# ────────────────────────────────────────────────────────────

def run_valuation(
    code: str, stock_name: str, report_year: str,
    price: str, months: int, max_reports: int,
    report_md: str = "",
):
    OUTPUT_DIR.mkdir(exist_ok=True)

    if not CLAUDE_API_KEY:
        raise ValueError("CLAUDE_API_KEY 未设置，直接传图模式必须使用Claude")

    # Step1: 直接加载年报截图（跳过OCR）
    print("\n--- Step1: 加载年报截图（直接传图，跳过OCR）---")
    image_blocks = load_images_as_base64(code, stock_name)

    # 可选入口A: 手动研报（文字）
    print("\n--- 可选入口A: 手动研报 ---")
    manual_report_text = load_manual_reports(code)

    # 可选入口B: 季报（图片+文字）
    print("\n--- 可选入口B: 季报参考层 ---")
    quarterly_text, quarterly_image_blocks = load_quarterly_data(code)

    # Step2: 研报倍数
    print("\n--- Step2: 研报估值倍数 ---")
    if report_md:
        report_table = load_report_md(report_md)
    else:
        report_table = download_and_extract_reports(code, months, max_reports)

    # 手动研报追加
    if manual_report_text:
        report_table += (
            "\n\n## 手动补充研报"
            "（请从以下内容中额外提取估值倍数，合并入上表悲观下限判断）\n\n"
            + manual_report_text
        )

    # Step3: 主估值——所有图片直接传给Claude
    print("\n--- Step3: 双轨估值分析（Claude直接看图）---")

    # 构建文字说明块
    text_intro = f"""标的：{stock_name}（{code}）
报告期：{report_year}年报
当前股价：{"请从截图中读取" if not price else price + " 元"}

⚠️ 重要提示：年报原始数据单位为"元"，换算时注意：
- 亿元 = 原始数值 ÷ 100,000,000（除以1亿）
- 例：5,788,501,676 元 = 5.79亿元，不是57.89亿元
- 换算后请做合理性检验

以上截图包含年报财务数据，请直接从图片中读取所有数字，不要推测或捏造任何数据。请输出完整报告，使用紧凑格式：表格数字保留2位小数，文字说明简洁，不重复已知信息。
确保轨道一、轨道二、交叉验证、估值局限性、风险提示五个部分全部输出完整。

---

## 研报估值倍数（供轨道二使用）

{report_table}

{quarterly_text}

请根据以上截图和研报数据，完成完整的双轨估值分析，输出完整报告（Markdown格式）。"""

    # 消息结构：年报截图 + 文字说明 + 季报截图（如有）
    content_blocks = (
        image_blocks                          # 年报截图在最前
        + [{"type": "text", "text": text_intro}]  # 文字说明
        + quarterly_image_blocks              # 季报截图在最后（如有）
    )

    client = OpenAI(api_key=CLAUDE_API_KEY, base_url=CLAUDE_BASE_URL)
    print(f"  → 调用Claude（共 {len(image_blocks)} 张年报截图"
          + (f" + {len(quarterly_image_blocks)} 张季报截图" if quarterly_image_blocks else "")
          + "）...")

    resp = client.chat.completions.create(
        model=CLAUDE_MODEL,
        max_tokens=16000,
        temperature=0.1,
        messages=[
            {"role": "system", "content": VALUATION_PROMPT},
            {"role": "user",   "content": content_blocks}
        ]
    )
    report   = resp.choices[0].message.content
    provider = "claude-direct-vision"
    print(f"  ✅ 完成（{len(report)}字）")

    # 保存报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path  = OUTPUT_DIR / f"{code}_{timestamp}_估值报告.md"
    header    = f"> 本报告由 **{provider}** 生成 | {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    out_path.write_text(header + report, encoding="utf-8")

    print(f"\n✅ 完成！报告: {out_path}")
    print(f"\n{'='*60}")
    print(report[:2000])
    print("\n... （完整报告见输出文件）")


# ────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="双轨估值分析 v7（直接传图）")
    parser.add_argument("--code",        required=True,  help="股票代码，如 002223")
    parser.add_argument("--name",        required=True,  help="股票名称，如 鱼跃医疗")
    parser.add_argument("--year",        required=True,  help="报告年度，如 2025")
    parser.add_argument("--price",       required=False, default="", help="当前股价（可选，不填则从截图读取）")
    parser.add_argument("--months",      type=int, default=3,  help="查询近几个月研报")
    parser.add_argument("--max-reports", type=int, default=10, help="最多分析几篇研报")
    parser.add_argument("--report-md",   required=False, default="", help="研报倍数文件路径（Job1产出）")
    args = parser.parse_args()

    print(f"🚀 双轨估值 v7（直接传图）| {args.name}（{args.code}）{args.year}年报")
    run_valuation(
        code=args.code, stock_name=args.name, report_year=args.year,
        price=args.price, months=args.months, max_reports=args.max_reports,
        report_md=args.report_md,
    )


if __name__ == "__main__":
    main()

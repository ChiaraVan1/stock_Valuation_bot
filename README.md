## 一、整体架构

```
截图文件夹                    研报 PDF
     │                           │
     ▼                           ▼
Workflow 3               Workflow 1
传图→读数→JSON            研报下载+倍数提取
     │                           │
     ▼                           │
Workflow 2                       │
链路层校验                         │
     │                           │
     └──────────┬────────────────┘
                ▼
           Workflow 4
           Claude 估值
           （纯文字，不传图）
                │
                ▼
           Workflow 5
           推送微信
```

---

## 二、各 Workflow 说明

### Workflow 1 — 研报估值倍数提取
**文件：** `.github/workflows/fetch_reports.yml`
**脚本：** `scripts/fetch_and_extract.py`

| 项目 | 说明 |
|------|------|
| 作用 | 从东方财富下载券商研报 PDF，提取 PE/PEG/EV·EBITDA 倍数 |
| 输入 | 股票代码、查询月数、最多篇数 |
| 输出 | `output/{code}_valuation.md`（研报倍数汇总表） |
| 依赖 | 无 |
| 何时重跑 | 研报过期、换新股票、需要更新倍数时 |
| token 消耗 | 少量（文字提取） |

---

### Workflow 3 — 传图提取财务数据
**文件：** `.github/workflows/extract_data.yml`
**脚本：** `scripts/extract_data.py`

| 项目 | 说明 |
|------|------|
| 作用 | 把年报截图传给 Claude，输出结构化 JSON |
| 输入 | 股票代码、名称、年份、当前股价（可选） |
| 输出 | `output/{code}_data.json` |
| 依赖 | 仓库里必须有截图文件夹：`Supporting Documents for Valuation/{code}/` |
| 何时重跑 | 截图有误、换新股票、换新年报时 |
| token 消耗 | **最多**（传图） |

> ⚠️ 跑完后务必人工抽查 JSON 里的关键数字：
> 货币资金、总资产、归母净利润、profit_history 年数、cash_drop_confirmed

---

### Workflow 2 — 链路层校验
**文件：** `.github/workflows/validate_data.yml`
**脚本：** `scripts/validate_data.py`

| 项目 | 说明 |
|------|------|
| 作用 | 读取 JSON，做数量级校验 + 骤降检测 + 三情景预检 + 分红完整性 |
| 输入 | 股票代码、Workflow 3 的 run_id |
| 输出 | `output/{code}_validation_report.txt` |
| 依赖 | Workflow 3 的 artifact |
| 何时重跑 | 每次 Workflow 3 跑完后都应跑一次；也可随时重跑排查问题 |
| token 消耗 | **零**（纯本地 Python） |

校验报告里的标记含义：

| 标记 | 含义 | 处理方式 |
|------|------|---------|
| `[ERROR]` | 数量级异常，估值阻断 | 重跑 Workflow 3 |
| `[WARNING]` | 货币资金骤降等警告 | 人工确认后继续 |
| `[CASH_MERGED]` | 合并后现金类资产（亿元） | Workflow 4 会自动使用此值 |
| `[CASH_DROP_MISMATCH]` | Claude 判断是定存重分类但链路层未触发 | 人工核查 |
| `[CASH_DROP_REAL_OUTFLOW]` | Claude 判断是真实流出（非定存） | 知悉即可 |
| `[SCENARIOS_OK]` | 三情景数值（悲观/中性/乐观） | 核对合理性 |
| `[SCENARIOS_ERROR]` | 三情景不单调 | 重跑 Workflow 3 |
| `[DIV_MISSING]` | 有年份缺分红记录 | 重跑 Workflow 3 或手动补充 |

---

### Workflow 4 — Claude 估值
**文件：** `.github/workflows/run_valuation.yml`
**脚本：** `scripts/run_valuation.py`

| 项目 | 说明 |
|------|------|
| 作用 | 把 JSON 数据转成文字，调用 Claude 做双轨估值 |
| 输入 | 股票代码、名称、股价（可选）、Workflow 1/2/3 的 run_id |
| 输出 | `output/{code}_{timestamp}_估值报告.md` |
| 依赖 | Workflow 3 的 artifact（必须）；Workflow 1/2 的 artifact（可选） |
| 何时重跑 | **改 prompt → 只重跑这步**，不需要重新传图 |
| token 消耗 | 中等（纯文字，无图片） |

输入参数说明：

| 参数 | 是否必填 | 说明 |
|------|---------|------|
| `stock_code` | ✅ | 股票代码 |
| `stock_name` | ✅ | 股票名称（用于推送） |
| `stock_price` | 可选 | 不填则用 JSON 里的值 |
| `run_id_wf3` | ✅ | Workflow 3 的 run_id |
| `run_id_wf1` | 可选 | 不填则研报倍数 fallback |
| `run_id_wf2` | 可选 | 不填则跳过校验注入 |
| `dry_run` | 可选 | `true` = 不推送微信 |

---

### Workflow 5 — 推送微信
**集成在 Workflow 4 末尾**，`dry_run=false` 时自动执行，无需单独触发。

---

## 三、文件夹结构

```
stock_Valuation_bot/
├── Supporting Documents for Valuation/
│   └── {股票代码}/          ← 年报截图放这里（Workflow 3 读取）
│       ├── 01_资产负债表.png
│       ├── 02_利润表.png
│       └── ...
├── quarterly_reports/
│   └── {股票代码}/          ← 季报截图/PDF（可选）
├── manual_reports/
│   └── {股票代码}/          ← 手动研报 PDF（可选）
├── scripts/
│   ├── valuation.py        ← 旧版完整流程（保留，仍可用）
│   ├── extract_data.py     ← Workflow 3 脚本
│   ├── validate_data.py    ← Workflow 2 脚本
│   ├── run_valuation.py    ← Workflow 4 脚本
│   ├── validation.py       ← 链路层校验逻辑（被 2/4 共用）
│   ├── fetch_and_extract.py← Workflow 1 脚本
│   ├── push_fangtang.py    ← 推送微信
│   └── ai_client.py        ← AI 客户端工厂
└── output/                 ← 所有产出文件
    ├── {code}_data.json
    ├── {code}_valuation.md
    ├── {code}_validation_report.txt
    └── {code}_{timestamp}_估值报告.md
```

---

## 四、典型使用场景

### 场景 A：第一次分析一只新股票
```
1. 把年报截图放入 Supporting Documents for Valuation/{code}/
2. 跑 Workflow 1（研报）
3. 跑 Workflow 3（传图读数）→ 抽查 JSON
4. 跑 Workflow 2（校验）→ 看有无 ERROR
5. 跑 Workflow 4（估值，填 wf1+wf2+wf3 的 run_id，dry_run=true）
6. 对比报告与 App，满意后 dry_run=false 推送
```

### 场景 B：调整 prompt 重跑估值
```
1. 修改 VALUATION_PROMPT（valuation.py）
2. 只跑 Workflow 4（填原来的 wf3 run_id，不重新传图）
```

### 场景 C：发现 JSON 数字有误
```
1. 重跑 Workflow 3
2. 重跑 Workflow 2（校验新 JSON）
3. 重跑 Workflow 4（填新的 wf3 run_id）
```

### 场景 D：快速测试（不需要研报）
```
1. 跑 Workflow 3
2. 跑 Workflow 4（run_id_wf1 留空，研报自动 fallback）
```

---

## 五、重要规则

**Prompt 单一来源**
`VALUATION_PROMPT` 只在 `valuation.py` 里维护一份。
`extract_data.py` 和 `run_valuation.py` 都通过 `from valuation import` 引用，
改一处自动同步，不需要手动同步两份。

**run_id 怎么找**
GitHub Actions → 对应 workflow → 点进某次运行 → URL 里的数字就是 run_id
例如：`https://github.com/.../actions/runs/12345678` → run_id = `12345678`

**artifact 有效期**
Workflow 3 产出的 JSON：7 天
估值报告：30 天
超期后需重跑对应 workflow。

---

*最后更新：2026-05-26*

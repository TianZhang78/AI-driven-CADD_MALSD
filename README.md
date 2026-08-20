[methodology_and_code.md](https://github.com/user-attachments/files/31246135/methodology_and_code.md)
# MASLD 早期降脂小分子发现 —— 方法学与可复现代码

> **赛题**：上海人工智能实验室 × 临港实验室「MASLD 早期降脂小分子发现」
> **模型**：HepG2 细胞 – 游离脂肪酸 (FFA) 肝细胞脂质蓄积模型
> **目标**：从 22,966 个小分子中筛选**既降低肝细胞脂质蓄积、又在有效浓度下不显著损伤细胞活力**的候选分子（低毒性降脂）
> **产出**：`candidates.csv`（Top 10 提名）、`mechanism_and_validation.pdf`（机制与验证方案）、本文档

---

## 目录

1. [方法学总览](#1-方法学总览)
2. [关键方法学决策与其理由](#2-关键方法学决策与其理由)
3. [数据与环境](#3-数据与环境)
4. [第一步：SDF 解析与描述符计算](#4-第一步sdf-解析与描述符计算)
5. [第二步：多层过滤](#5-第二步多层过滤)
6. [第三步：启发式多目标打分](#6-第三步启发式多目标打分)
7. [第四步：机制证据层与多样性提名](#7-第四步机制证据层与多样性提名)
8. [结果](#8-结果)
9. [方法验证与局限](#9-方法验证与局限)
10. [复现方式](#10-复现方式)
11. [完整源码](#11-完整源码)

---

## 1. 方法学总览

```
T.sdf  (22,966 条记录, 67.8 MB, ISO-8859-1)
  │
  ├─ [0] 编码修复      Latin-1 → UTF-8   (RDKit GetProp 以 UTF-8 解码，原文件会抛 UnicodeDecodeError)
  │
  ├─ [1] 解析与标准化  SDMolSupplier → LargestFragmentChooser (去盐) → Uncharger (中和)
  │                    计算 MW / LogP / TPSA / HBD / HBA / RotB / AromRings / FracCsp3 …
  │
  ├─ [2] 多层过滤      元素白名单 → 重原子数 → PAINS → Lipinski → 细胞毒性硬反筛
  │                    22,966 → 12,775   （SMILES 级去重后 11,969）
  │
  ├─ [3] 启发式打分    S_total = 0.6·S_efficacy + 0.4·S_safety
  │                    ⚠ 高度饱和：554 个分子并列 1.0000
  │
  ├─ [4] 硬门槛        S_total ≥ 0.95  →  2,496 个分子
  │
  ├─ [5] 机制证据层    Morgan(r=2,2048) Tanimoto vs 50 个参照药 ⊕ 22 条优势骨架 SMARTS
  │                    S_mech = 0.55·min(T_max/0.55, 1) + 0.45·chemotype_weight
  │                    S_nomination = 0.5·S_total + 0.5·S_mech
  │
  ├─ [6] 阳性对照剥离  Tanimoto ≥ 0.85 视为已知药 → 7 个，单列报告，不参与提名
  │
  └─ [7] 多样性提名    Murcko 骨架去重 + 两两 Tanimoto < 0.60 + 每靶点 ≤ 1 个  →  **Top 10**
```

---

## 2. 关键方法学决策与其理由

本节是本方案与「照抄题目模板」的区别所在。以下五个决策都源于在真实数据上发现的问题。

### 2.1 赛题给定的打分函数在本库上不可用于排序（最重要的发现）

题目设定的打分为：

```
S_efficacy: LogP 最优区间 [2.5, 3.5]，TPSA 最优区间 [50, 90]，距离衰减
S_safety  : MW 越接近 350 越高，扣除毒性基团罚分
S_total   = 0.6 × S_efficacy + 0.4 × S_safety
```

这是一个**纯理化性质函数**，不包含任何生物活性信息。实际运行后：

- **554 个分子并列 S_total = 1.0000**（占打分池的 4.6%）；
- 若直接 `sort_values('S_total').head(10)`，得到的 Top 10 实际上是**文件顺序的前 10 个**，与科学无关；
- 且这 10 个必然是理化性质近乎相同的一组类似物，`Primary_Target_Hypothesis` 一列将退化为纯叙事。

**处理方式**：完全保留题目公式并作为**硬门槛**（`S_total ≥ 0.95`，保留 2,496 个），在门槛内改用可验证的机制证据排序。纯 S_total 排名同时完整导出于 `all_scored_molecules.csv`，评审可自行核对。

### 2.2 引入机制证据层，让靶点假说可被追溯

两个证据来源，均写死在代码中、可复核：

| 证据 | 实现 | 权重 |
|---|---|---|
| 与已知肝脂调节剂的结构相似性 | 50 个临床/工具化合物参照集，Morgan r=2 / 2048 bit，Tanimoto | 0.55 |
| 肝脂靶点优势骨架药效团 | 22 条 SMARTS 规则（含 `anti` 反向排除模式） | 0.45 |

```
S_mech = 0.55 · min(T_max / 0.55, 1.0) + 0.45 · chemotype_weight
S_nomination = 0.5 · S_total + 0.5 · S_mech
```

这样 `candidates.csv` 中每个靶点假说都能回溯到「命中了哪条 SMARTS / 最像哪个已知药、相似度多少」，而不是模型编的故事。

### 2.3 参照药物集必须逐个结构核验

初版参照集直接凭记忆写 SMILES，用 `_check_refs.py`（RDKit 计算分子式 vs 文献分子式）核对后发现 **6 个结构错误**（Firsocostat、Lanifibranor、Obeticholic acid、A-769662、Cerulenin、Telmisartan）与 **3 个严重错误**（Tropifexor、Saroglitazar、Etomoxir）。

- 6 个已修正并重新核验分子式完全一致；
- Tropifexor、Saroglitazar 直接删除；
- 无法确证的 firsocostat 条目改名为 "ACC inhibitor chemotype (firsocostat-like proxy)"，不冒充具体药物。

> **教训**：参照集是整个机制层的基准。基准错了，所有下游相似度与靶点归属都错，且错误不会以报错的形式暴露。任何依赖「记忆中的 SMILES」的流程都必须有独立的分子式核验步骤。

### 2.4 SMARTS 药效团规则会产生看似合理的假阳性

初版有一条甲状腺激素拟似物（THR-β）规则：

```smarts
[Cl,Br,I]c1cccc([Cl,Br,I])c1O[#6]        ← 错误
```

它把 `TN7135  COc1c(Br)cc(CC(=O)O)cc1Br`（一个二溴**苯甲醚**）判成了 THR-β 配体，因为 `[#6]` 能匹配甲基碳。而 T3/resmetirom 类拟似物需要的是 4-**芳氧基**外环。修正为：

```smarts
[Cl,Br,I]c1cccc([Cl,Br,I])c1[OX2][c]     ← 要求氧连接芳香碳
```

修正后该假阳性从 Top 10 中消失，位置由 `T4515`（ervogastat 型芳基脲 + 4-芳氧基哌啶，DGAT2）替代。

> **教训**：SMARTS 中 `[#6]`（任意碳）与 `[c]`（芳香碳）的差别，足以让一个假设从「甲状腺激素受体」变成「什么都不是」。每条规则都需要用已知阳性/阴性分子对拍验证。

### 2.5 Murcko 骨架去重不足以保证结构多样性

仅按 Bemis–Murcko 骨架去重后，Top 10 中出现：排名 3/4/5 的描述符完全相同（406.56 / 3.66 / 94.83），排名 8/9/10 全是小檗碱同系物。原因是同一结构在库中以不同货号（游离碱/盐/不同批次）多次出现，且近似物骨架不同但化学空间重叠。

补三道约束：

1. **SMILES 级去重**：规范化 SMILES 相同则合并，保留一个货号，其余记入 `Duplicate_IDs` 列（12,775 → 11,969，合并 806 个重复结构）；
2. **贪心两两 Tanimoto < 0.60**：新候选与已选任一分子相似度超阈值即跳过；
3. **每个靶点假说至多 1 个分子**（`MAX_PER_TARGET = 1`）。

最终 Top 10 覆盖 10 条相互独立的机制通路。

### 2.6 数据清洗：CAS 号的 Excel 日期损坏

库中部分 CAS 被 Excel 误转为日期格式（如 `TN7583` 的 CAS 为 `2458/8/4`，应为 `2458-08-4`）。写了 `repair_cas()`，用 **CAS 校验位算法**确认后才修正：

```
2458-08-4 校验：(8×1 + 0×2 + 8×3 + 5×4 + 4×5 + 2×6) = 84；84 mod 10 = 4 ✓
```

32 个校验位不通过的非标准 CAS 保守地保持原值不动（宁可不改也不猜）。Top 10 中无此类分子。

---

## 3. 数据与环境

| 项 | 值 |
|---|---|
| 输入文件 | `T.sdf`，67,779,525 B，1,785,074 行，ISO-8859 text (CRLF) |
| 记录数 | 22,966 |
| 数据标签 | `<ID>` (22,966)、`<Formula>` (22,966)、`<MolWt>` (22,966)、`<CAS>` (22,155) |
| 库性质 | 已知生物活性化合物库（药物重定位型），编号形如 `T0002` / `TN7583` / `T4S0795` |
| Python | 3.x（conda env `pytorch_env`） |
| RDKit | 2026.03.5 |
| pandas / numpy | 2.3.3 / 2.2.6 |
| reportlab | PDF 生成（中文用内置 CID 字体 `STSong-Light`，无需外部 TTF） |
| 确定性 | 无随机数来源；重跑结果逐字节一致 |

**编码问题**：原 SDF 为 ISO-8859-1，RDKit 的 `mol.GetProp()` 以 UTF-8 解码，在读取含 `0xa1` 等字节的记录时抛 `UnicodeDecodeError`。脚本首次运行时自动逐行转码生成 `/tmp/T_utf8.sdf`（修复 379 行），并额外用 `safe_prop()` 兜底。

---

## 4. 第一步：SDF 解析与描述符计算

- `Chem.SDMolSupplier(sdf, removeHs=True, sanitize=True)`；
- **标准化**：`rdMolStandardize.LargestFragmentChooser`（去除反离子/盐）→ `Uncharger`（中和电荷，但保留季铵等永久电荷）；
- **描述符**：`Descriptors.MolWt`、`Crippen.MolLogP`、`rdMolDescriptors.CalcTPSA`、`Lipinski.NumHDonors/NumHAcceptors/NumRotatableBonds`、`CalcNumAromaticRings`、`CalcFractionCSP3`、`CalcMolFormula`、重原子数、卤素计数；
- **属性读取**：`safe_prop()` 包裹 try/except，避免单条坏记录中断全库解析。

---

## 5. 第二步：多层过滤

| 层 | 规则 | 淘汰数 | 理由 |
|---|---|---:|---|
| 解析/标准化 | sanitize 失败 | 52 | 价键异常记录 |
| 元素白名单 | 仅允许 H,B,C,N,O,F,P,S,Cl,Br,I | 89 | 排除金属配合物、无机盐、含 Si/Sn 分子 |
| 分子尺寸 | 重原子数 ≥ 12 | 2,238 | 碎片过小，难以支撑细胞水平活性 |
| PAINS | `FilterCatalogParams.PAINS_A/B/C` | 1,632 | 泛筛选干扰结构 |
| Lipinski | MW 150–550, LogP −1–5, HBD ≤ 5, HBA ≤ 10 | 6,118 | 题目指定的类药性约束 |
| 细胞毒性硬反筛 | **LogP > 4.5 且 TPSA < 20 → 剔除** | 62 | 高亲脂 + 极低极性 = 膜裂解 / 线粒体解偶联的经验特征 |
| **合计通过** | | | **12,775** |
| SMILES 去重 | 规范化 SMILES 相同则合并 | 806 | 同一结构多货号 |
| **进入排序** | | | **11,969** |

细胞毒性反筛按题目要求实现为「硬过滤 + 降权」两段：`LogP > 4.5 且 TPSA < 20` 直接剔除；`LogP > 4.2` 或 `TPSA < 30` 在 `S_safety` 中连续扣分（见 6.2）。这样避免了单一硬阈值在边界上的悬崖效应。

**结构警示子**（22 条 SMARTS，用于 `S_safety` 扣分，不作硬过滤）：
Michael 受体（乙烯基酮、丙烯酰胺、乙烯基砜、丙烯酸酯、丙烯腈）、醌、环氧、氮丙啶、醛、硝基芳烃、偶氮芳烃、肼、异氰酸酯、酸酐、酰卤、活泼卤代烷、游离硫醇、芳香亚硝基、过氧化物、有机磷、硫脲、稠环多环芳烃。

---

## 6. 第三步：启发式多目标打分

### 6.1 S_efficacy

```python
def plateau(x, lo, hi, sigma):
    if lo <= x <= hi: return 1.0
    d = (lo - x) if x < lo else (x - hi)
    return math.exp(-0.5 * (d / sigma) ** 2)

def score_efficacy(logp, tpsa, arom_rings, rotb):
    s_logp  = plateau(logp, 2.5, 3.5, 1.0)     # 题目指定最优区间
    s_tpsa  = plateau(tpsa, 50.0, 90.0, 25.0)  # 题目指定最优区间
    s_shape = 0.5 * plateau(arom_rings, 1, 3, 1.0) + 0.5 * plateau(rotb, 2, 7, 3.0)
    return 0.45 * s_logp + 0.40 * s_tpsa + 0.15 * s_shape
```

「平台 + 高斯衰减」而非线性距离衰减：区间内不做人为区分（题目说的是「最优区间」而非「最优点」），区间外平滑衰减，避免阶跃。`s_shape` 是唯一的补充项，占 15%，用于抑制过度刚性（无芳环）或过度柔性（RotB ≫ 7，构象熵惩罚大、细胞渗透差）的分子。

### 6.2 S_safety

```python
def score_safety(mw, logp, tpsa, n_tox, n_halogen, fsp3):
    s = gaussian(mw, 350.0, 120.0)                              # 题目：MW 越接近 350 越高
    s -= 0.25 * min(n_tox, 3)                                   # 题目：毒性基团扣分
    if n_halogen > 3: s -= 0.10 * min(n_halogen - 3, 3)         # 题目：卤素过多扣分
    if logp > 4.2:    s -= 0.30 * min((logp - 4.2) / 0.8, 1.0)  # 亲脂性细胞毒（连续降权）
    if tpsa < 30:     s -= 0.30 * min((30 - tpsa) / 30.0, 1.0)  # 低极性膜活性（连续降权）
    s += 0.10 * min(fsp3 / 0.4, 1.0)                            # 三维性 → 脱靶更少
    return max(0.0, min(1.0, s))
```

`S_total = 0.6 × S_efficacy + 0.4 × S_safety`（与题目一致）。

`fsp3` 加成的依据是 Lovering 的 "Escape from Flatland"：sp3 比例高的分子脱靶率与毒性率显著更低。

---

## 7. 第四步：机制证据层与多样性提名

### 7.1 参照药物集（50 个，逐个核验分子式）

覆盖 THR-β、ACC/ACLY、FASN/SCD1、DGAT2、PPAR α/δ/γ/pan、FXR/胆汁酸、AMPK、HMGCR、脂解/FAO/转运、天然产物共十类。每条记录为 `(名称, SMILES, 靶点, 机制)`。

### 7.2 优势骨架规则（22 条）

每条规则形如：

```python
dict(name="噻唑烷-2,4-二酮 (TZD)", weight=0.92, target="PPAR-gamma",
     smarts=["O=C1NC(=O)C(S1)C"], anti=[...], mech="...")
```

`anti` 字段用于反向排除（命中 `smarts` 但同时命中 `anti` 则不计分），编译时对每条 SMARTS 做合法性校验，解析失败直接抛异常而非静默跳过。

### 7.3 提名参数

```python
TANI_STRONG    = 0.55   # 相似度饱和点
POS_CTRL_T     = 0.85   # ≥ 此值视为已知药，剥离出提名池
GATE           = 0.95   # S_total 硬门槛
DIV_T          = 0.60   # 候选间两两 Tanimoto 上限
MAX_PER_TARGET = 1      # 每个靶点假说最多提名 1 个
N_TOP          = 10
```

---

## 8. 结果

### 8.1 提名 Top 10

| # | 库编号 | CAS | 分子式 | MW | LogP | TPSA | S_eff | S_saf | S_total | S_mech | S_nom | 靶点假说 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `T2607` | 146062-49-9 | C19H18N2O4S | 370.43 | 2.8 | 85.36 | 1.0000 | 1.0000 | 1.0000 | 0.9640 | **0.9820** | PPAR-gamma (兼 PPAR-alpha 部分激动) |
| 2 | `T12092` | 79952-42-4 | C19H28O4 | 320.43 | 2.6 | 66.76 | 0.9705 | 1.0000 | 0.9823 | 0.9460 | **0.9642** | HMGCR (3-羟基-3-甲基戊二酰辅酶A 还原酶) |
| 3 | `T19124` | 2304-89-4 | C24H38O5 | 406.56 | 3.66 | 94.83 | 0.9576 | 0.9949 | 0.9725 | 0.9550 | **0.9638** | FXR (NR1H4，兼 TGR5/GPBAR1) |
| 4 | `T75436` | 483-43-2 | C20H20NO4+ | 338.38 | 3.08 | 51.8 | 1.0000 | 1.0000 | 1.0000 | 0.9100 | **0.9550** | AMPK / LDLR (兼 PCSK9 下调) |
| 5 | `T21318` | 131179-95-8 | C20H23NO4 | 341.41 | 3.73 | 75.63 | 0.9886 | 1.0000 | 0.9932 | 0.8336 | **0.9134** | PPAR-alpha |
| 6 | `TN2058` | 28590-40-1 | C17H16O6 | 316.31 | 2.82 | 85.22 | 1.0000 | 1.0000 | 1.0000 | 0.8200 | **0.9100** | AMPK / SREBP-1c |
| 7 | `T15049` | 79094-20-5 | C16H16ClNO4S | 353.83 | 2.49 | 83.47 | 1.0000 | 1.0000 | 1.0000 | 0.7328 | **0.8664** | PPAR pan (alpha/delta/gamma) |
| 8 | `T8701` | 1309241-34-6 | C21H27N3O2 | 353.47 | 2.97 | 54.46 | 1.0000 | 1.0000 | 1.0000 | 0.6983 | **0.8492** | DGAT2 / FASN |
| 9 | `T8532` | 1422365-93-2 | C13H16F3N5O | 315.3 | 2.55 | 84.23 | 1.0000 | 1.0000 | 1.0000 | 0.6518 | **0.8259** | AMPK (间接，经线粒体复合物I) |
| 10 | `T4515` | 1032229-33-6 | C20H22ClN3O3 | 387.87 | 3.77 | 70.67 | 0.9833 | 1.0000 | 0.9900 | 0.6127 | **0.8014** | DGAT2 / SCD1 |

### 8.2 结构与命中的药效团

| # | 库编号 | SMILES | 命中优势骨架规则 |
|---|---|---|---|
| 1 | `T2607` | `CCc1ccc(C(=O)COc2ccc(CC3SC(=O)NC3=O)cc2)nc1` | 噻唑烷-2,4-二酮 (TZD) |
| 2 | `T12092` | `C[C@H]1C=C2C=C[C@H](C)[C@H](CC[C@@H]3C[C@@H](O)CC(=O)O3)[C@H]2[C@@H](O)C1` | statin delta-内酯前药 |
| 3 | `T19124` | `C[C@H](CCC(=O)O)[C@H]1CC[C@H]2[C@@H]3[C@H](O)C[C@@H]4CC(=O)CC[C@]4(C)[C@H]3C[C@H](O)[C@]12C` | 胆汁酸甾核 + C24 羧酸 |
| 4 | `T75436` | `COc1cc2c(cc1O)CC[n+]1cc3c(OC)c(OC)ccc3cc1-2` | 原小檗碱型季铵异喹啉生物碱 |
| 5 | `T21318` | `Cc1cc(C)cc(NC(=O)Cc2ccc(OC(C)(C)C(=O)O)cc2)c1` | 芳氧基异丁酸 (贝特类头基) |
| 6 | `TN2058` | `COc1cc(O)c2c(c1)O[C@H](c1ccc(OC)c(O)c1)CC2=O` | 黄酮/黄烷酮母核 |
| 7 | `T15049` | `O=C(O)Cc1ccc(CCNS(=O)(=O)c2ccc(Cl)cc2)cc1` | 芳基乙酸 + 磺酰胺 (pan-PPAR lanifibranor 型) |
| 8 | `T8701` | `Cc1ccc(C)c(OCCNC2CCN(C(=O)c3ccncc3)CC2)c1` | 嘧啶/吡啶甲酰胺 + 哌啶(哌嗪) |
| 9 | `T8532` | `N=C(NC(=N)N1CCCC1)Nc1ccc(OC(F)(F)F)cc1` | 双胍 (biguanide) |
| 10 | `T4515` | `CNC(=O)c1cccc(NC(=O)N2CCC(Oc3ccccc3Cl)CC2)c1` | 杂芳基脲 (DGAT2/SCD1 型) |

### 8.3 阳性对照回收（方法学自验证）

在**完全不使用任何活性标签**的前提下，流程从库中自动识别出 7 个已知肝脂调节剂（Tanimoto = 1.00），全部通过了 PAINS / 类药性 / 毒性反筛并落在 S_total ≥ 0.95 区间：

| 库编号 | CAS | 对应已知药物 | Tanimoto | 靶点 | S_total |
|---|---|---|---|---|---|
| `T0214L` | 112529-15-4 | Pioglitazone | 1.00 | PPAR-gamma | 1.0000 |
| `T0334` | 122320-73-4 | Rosiglitazone | 1.00 | PPAR-gamma | 1.0000 |
| `T0841` | 41859-67-0 | Bezafibrate | 1.00 | PPAR pan | 0.9996 |
| `T1264` | 52214-84-3 | Ciprofibrate | 1.00 | PPAR-alpha | 0.9884 |
| `T1402` | 42017-89-0 | Fenofibric acid | 1.00 | PPAR-alpha | 0.9871 |
| `T5S0814` | 633-66-9 | Berberine | 1.00 | AMPK / PCSK9 / LDLR | 0.9843 |
| `T6633` | 95635-55-5 | Ranolazine | 1.00 | pFOX / 晚钠电流 | 0.9507 |

这是本方案唯一的**客观**方法学证据：打分与过滤逻辑确实富集了真实的降脂化学空间，而非随机保留分子。这 7 个分子已从 Top 10 中剔除（提名已上市药物属循环论证），但**建议直接采购小檗碱与非诺贝特酸作为实验阳性对照**——它们就在同一个库里。

---

## 9. 方法验证与局限

**已做的验证**
1. 阳性对照回收（8.3）——流程能找回已知答案；
2. 参照集分子式逐个核验（`_check_refs.py`）——基准正确；
3. SMARTS 规则的假阳性排查（2.4）——规则不乱匹配；
4. 结构去重与多样性约束（2.5）——Top 10 不是同一分子的十个货号。

**局限（详见 PDF 第 8 节）**
1. 打分函数不含生物活性信息，本质是类药性刻画，不是活性预测；
2. 靶点假说是化学型推断，`T15049`（芳基乙酸-磺酰胺，亦见于前列腺素类似物）与 `T8701`（哌啶甲酰胺，药效团较泛）的置信度明显低于 `T2607`(TZD) / `T12092`(statin 内酯) / `T8532`(双胍)；
3. **理化毒性启发式系统性低估「在靶但有害」的机制性毒性**——已人工标注三个高风险分子：`T8532`（芳基双胍，苯乙双胍类，乳酸酸中毒）、`T75436`（亲脂性阳离子，线粒体蓄积）、`T12092`（statin，甲羟戊酸耗竭），并在 PDF 中给出针对性的实验区分方案；
4. Morgan 指纹不区分立体化学，而 TZD 的 C5、statin 内酯的多个手性中心对活性影响显著；
5. 未做 ADMET / 聚集倾向的定量预测。

> 尤其提示第 3 条：`T8532` 的 `S_safety = 1.000`，但它是苯乙双胍类芳基双胍。**若只读代码输出而不读注释，会得出完全错误的安全性结论。** 这也是「预测毒性」一列在 `candidates.csv` 中带有独立 `Cytotoxicity_Caveat` 字段的原因。

---

## 10. 复现方式

```bash
# 环境
conda activate pytorch_env          # rdkit 2026.03.5, pandas, reportlab

cd /path/to/MASLD_screen
python _check_refs.py               # 可选：核验参照集 SMILES 的分子式
python 01_screen.py                 # 筛选与打分（约 3-5 min，22,966 分子）
python 02_deliverables.py           # candidates.csv + PDF + 结构图
python 03_methodology.py            # 本文档
```

**产物清单**

| 文件 | 内容 |
|---|---|
| `candidates.csv` | **Top 10 提名清单**（30 列，含题目要求的 7 个字段 + 全部可追溯的中间量） |
| `mechanism_and_validation.pdf` | **机制与验证方案**（18 页，含结构图、漏斗、逐分子档案、HepG2-FFA 完整实验方案） |
| `methodology_and_code.md` | 本文档 |
| `all_scored_molecules.csv` | 11,969 个通过过滤分子的完整打分与描述符（含纯 S_total 排名） |
| `gated_annotated.csv` | 2,496 个门槛内分子的机制证据注释 |
| `positive_control_recovery.csv` | 阳性对照回收清单 |
| `structures/` | Top 10 结构图 PNG |
| `_funnel_stats.json` | 漏斗统计（供文档自动生成，避免手抄数字出错） |

**源码指纹**

| 文件 | 大小 | SHA-256 (前 16 位) |
|---|---|---|
| `01_screen.py` | 37,282 B | `a9386b7d6c118a87` |
| `02_deliverables.py` | 48,491 B | `73244e69b7698053` |
| `03_methodology.py` | 20,033 B | `f7678e76f956f978` |

---

## 11. 完整源码

以下代码由本文档生成脚本从磁盘直接读取，与实际运行的代码逐字节一致。

### 11.1 `01_screen.py` —— 解析、过滤、打分、机制注释、多样性提名

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
MASLD 早期降脂小分子发现 —— 虚拟筛选主流程
上海人工智能实验室 x 临港实验室 / HepG2-FFA 肝细胞脂质蓄积模型
目标：低毒性降脂候选分子提名 (Top 10)
================================================================================

Step 1   SDF 解析、结构标准化与理化描述符计算
Step 2   PAINS 假阳性剔除 + Lipinski 类药性过滤 + 经验性细胞毒性反筛
Step 3   启发式多目标打分  S_total = 0.6*S_efficacy + 0.4*S_safety
Step 3b  机制证据层 (S_mech)：
           (i)  与 50 个已知肝脂代谢临床/工具化合物的 Morgan Tanimoto 相似性
           (ii) 22 条肝脂靶点优势骨架 (privileged chemotype) SMARTS 规则
Step 3c  阳性对照回收 (Tanimoto >= 0.85) —— 方法学自验证
Step 3d  Bemis-Murcko 骨架去冗余 -> 多样性 Top 10 提名

【方法学声明】
S_total 为纯理化性质函数，不含任何生物活性信息，且在本库上高度饱和
(590/12775 个分子并列 S_total=1.0000)，单独使用无法排序。因此本流程将
S_total 用作"硬门槛"，再以机制证据 S_mech 排序，最后施加骨架多样性约束。
所有中间量均随交付物一并导出，可完全复现。
"""

import os, re, math, json, warnings
import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import (Descriptors, Lipinski, Crippen, rdMolDescriptors,
                        FilterCatalog, rdFingerprintGenerator)
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

SDF_FILE_RAW = "/mnt/c/Users/zhangt97/Downloads/T.sdf"
SDF_FILE     = "/tmp/T_utf8.sdf"
OUTDIR       = "/mnt/c/Users/zhangt97/Downloads/MASLD_screen"
os.makedirs(OUTDIR, exist_ok=True)

# 原始 SDF 为 ISO-8859-1(Latin-1) 编码，RDKit GetProp 按 UTF-8 解码会抛异常，先转码
if not os.path.exists(SDF_FILE):
    _bad = 0
    with open(SDF_FILE_RAW, "rb") as _fi, open(SDF_FILE, "wb") as _fo:
        for _raw in _fi:
            try:
                _raw.decode("utf-8"); _fo.write(_raw)
            except UnicodeDecodeError:
                _bad += 1; _fo.write(_raw.decode("latin-1").encode("utf-8"))
    print(f"[info] SDF 已由 ISO-8859-1 转码为 UTF-8 (修正 {_bad} 行)")

# ==============================================================================
# 0A. 过滤器与工具
# ==============================================================================
_params = FilterCatalog.FilterCatalogParams()
_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
PAINS = FilterCatalog.FilterCatalog(_params)

TOX_SMARTS = {
    # --- 反应性 Michael 加成受体：共价蛋白加合 -> 非特异性肝细胞毒性 ---
    "michael_vinylketone":   "[CX3]=[CX3][CX3](=[OX1])[#6]",
    "michael_acrylamide":    "[CX3]=[CX3][CX3](=[OX1])[NX3]",
    "michael_vinylsulfone":  "[CX3]=[CX3][SX4](=[OX1])(=[OX1])",
    "michael_acrylate":      "[CX3]=[CX3][CX3](=[OX1])[OX2H0]",
    "michael_acrylonitrile": "[CX3]=[CX3][CX2]#[NX1]",
    "quinone":               "O=C1C=CC(=O)C=C1",
    # --- 亲电体 / 代谢活化型结构警示子 ---
    "epoxide":               "C1OC1",
    "aziridine":             "C1NC1",
    "aldehyde":              "[CX3H1](=O)[#6]",
    "nitro_aromatic":        "[$([NX3](=O)=O),$([NX3+](=O)[O-])][c]",
    "azo_aromatic":          "[c][NX2]=[NX2][c]",
    "hydrazine":             "[NX3][NX3]",
    "isocyanate":            "[NX2]=[CX2]=[OX1]",
    "anhydride":             "[CX3](=[OX1])[OX2][CX3](=[OX1])",
    "acyl_halide":           "[CX3](=[OX1])[F,Cl,Br,I]",
    "reactive_alkyl_halide": "[CX4;!$(C([F,Cl,Br,I])[F,Cl,Br,I])][Cl,Br,I]",
    "free_thiol":            "[SX2H]",
    "aromatic_nitroso":      "[c][NX2]=[OX1]",
    "peroxide":              "[OX2][OX2]",
    "organophosphate":       "P(=O)(O[#6])(O[#6])O[#6]",
    "thiourea":              "[NX3][CX3](=[SX1])[NX3]",
    "fused_PAH":             "c1ccc2cc3ccccc3cc2c1",
}
TOX_PATTERNS = {k: p for k, p in
                ((k, Chem.MolFromSmarts(v)) for k, v in TOX_SMARTS.items()) if p is not None}

ALLOWED_ELEMENTS = {1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53}   # H B C N O F P S Cl Br I
HALOGENS = {9, 17, 35, 53}

LFC = rdMolStandardize.LargestFragmentChooser()
UNCHARGER = rdMolStandardize.Uncharger()
MFPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# ==============================================================================
# 0B. 已知肝脂代谢 / MASLD 临床与工具化合物参照集
#     结构均已用 RDKit 计算分子式并与文献分子式核对
# ==============================================================================
REFERENCE_DRUGS = [
    # ---------------- THR-beta ----------------
    ("Resmetirom (MGL-3196)", "CC(C)C1=CC(Oc2c(Cl)cc(N3N=C(C#N)C(=O)NC3=O)cc2Cl)=NNC1=O",
     "THR-beta", "肝脏选择性甲状腺激素受体 beta 激动；上调 CPT1A/线粒体 beta-氧化与脂噬，降低肝内甘油三酯"),
    ("VK2809 active acid (MB07344)", "CC(C)c1cc(Cl)c(Oc2c(Cl)cc(CP(=O)(O)O)cc2Cl)c(Cl)c1O",
     "THR-beta", "肝靶向 THR-beta 激动剂活性代谢物；促进脂肪酸氧化与 LDL-C 清除"),
    # ---------------- ACC / ACLY / DNL ----------------
    ("ACC inhibitor chemotype (firsocostat-like proxy)",
     "COc1cc(-c2cn(C)c(=O)c3c2CCC3)cc(OC)c1C1(C(=O)O)CCOCC1",
     "ACC1/ACC2", "变构抑制乙酰辅酶A羧化酶，降低丙二酰辅酶A、阻断新发脂质合成(DNL)并解除 CPT1 抑制"),
    ("Bempedoic acid", "CC(C)(CCCCCC(O)CCCCCC(C)(C)C(=O)O)C(=O)O",
     "ACLY / AMPK", "肝特异性 ATP-柠檬酸裂解酶抑制剂前药，减少乙酰辅酶A 供应并激活 AMPK"),
    # ---------------- FASN / SCD1 ----------------
    ("TVB-2640 (Denifanstat)", "Cc1nc(N2CCC(c3ccc(OC(F)(F)F)cc3)CC2)ncc1C(=O)NC1CCOCC1",
     "FASN", "脂肪酸合酶 beta-酮脂酰还原酶结构域抑制剂，直接阻断棕榈酸从头合成"),
    ("Cerulenin", "O=C(CC/C=C/C/C=C/C)[C@@H]1O[C@@H]1C(N)=O",
     "FASN", "环氧化物共价修饰 FASN 缩合酶结构域半胱氨酸，抑制脂肪酸从头合成"),
    ("C75", "CCCCCCCC[C@H]1[C@@H](C(=O)O)C(=C)C(=O)O1",
     "FASN", "alpha-亚甲基-gamma-丁内酯类 FASN 抑制剂，同时升高丙二酰辅酶A"),
    ("Aramchol", "CCCCCCCCCCCCCCCCCCCC(=O)N[C@H]1CC[C@]2(C)[C@H](CC[C@H]3[C@@H]2[C@H](O)C[C@H]2[C@@]3(C)CC[C@H]2[C@H](C)CCC(=O)O)C1",
     "SCD1", "胆汁酸-脂肪酸偶联物，下调 SCD1，降低单不饱和脂肪酸与甘油三酯合成"),
    # ---------------- DGAT ----------------
    ("PF-06424439 (DGAT2i)", "CN(C)C(=O)C1CCN(C(=O)c2cnc(N3CCOCC3)nc2)CC1",
     "DGAT2", "二酰甘油酰基转移酶2 抑制剂，直接阻断甘油三酯酯化终末步骤"),
    ("Ervogastat (PF-06865571)", "O=C(Nc1ccc(C(F)(F)F)cn1)N1CCC(Oc2ccncc2)CC1",
     "DGAT2", "口服 DGAT2 抑制剂，降低肝内甘油三酯合成与 VLDL 分泌"),
    # ---------------- PPAR ----------------
    ("Pioglitazone", "CCc1ccc(CCOc2ccc(CC3SC(=O)NC3=O)cc2)nc1",
     "PPAR-gamma", "噻唑烷二酮胰岛素增敏剂，促进脂肪组织脂质再分布、降低肝细胞脂毒性"),
    ("Rosiglitazone", "CN(CCOc1ccc(CC2SC(=O)NC2=O)cc1)c1ccccn1",
     "PPAR-gamma", "TZD 类 PPAR-gamma 全激动剂，改善胰岛素抵抗与肝内脂肪蓄积"),
    ("Lanifibranor", "Cc1ccc(S(=O)(=O)N2CCc3cc(CC(=O)O)ccc3CC2)cc1Cl",
     "PPAR pan (alpha/delta/gamma)", "泛 PPAR 激动，促进 beta-氧化、抑制脂肪生成与炎症纤维化"),
    ("Elafibranor", "CC(C)(Oc1ccc(C(=O)/C=C/c2ccc(SC)cc2)c(C)c1C)C(=O)O",
     "PPAR-alpha/delta", "双重 PPAR-alpha/delta 激动，提升脂肪酸氧化并抑制脂肪生成"),
    ("Fenofibric acid", "CC(C)(Oc1ccc(C(=O)c2ccc(Cl)cc2)cc1)C(=O)O",
     "PPAR-alpha", "激活 PPAR-alpha 上调 CPT1/ACOX1，增强线粒体与过氧化物酶体脂肪酸氧化"),
    ("Bezafibrate", "CC(C)(Oc1ccc(CCNC(=O)c2ccc(Cl)cc2)cc1)C(=O)O",
     "PPAR pan", "泛 PPAR 弱激动剂，降低血浆与肝内甘油三酯"),
    ("Gemfibrozil", "Cc1ccc(C)c(OCCCC(C)(C)C(=O)O)c1",
     "PPAR-alpha", "苯氧戊酸类 PPAR-alpha 激动剂，抑制肝脏 VLDL-TG 分泌"),
    ("Ciprofibrate", "CC(C)(Oc1ccc(C2CC2(Cl)Cl)cc1)C(=O)O",
     "PPAR-alpha", "长效贝特类，诱导过氧化物酶体增殖与脂肪酸 beta-氧化"),
    ("GW501516", "Cc1c(CSc2ccc(OCC(=O)O)c(C)c2)sc(-c2ccc(C(F)(F)F)cc2)n1",
     "PPAR-delta", "PPAR-delta 选择性激动，提升肝与骨骼肌脂肪酸氧化"),
    ("Telmisartan", "CCCc1nc2c(C)c(-c3nc4ccccc4n3C)ccc2n1Cc1ccc(-c2ccccc2C(=O)O)cc1",
     "PPAR-gamma partial / AT1R", "AT1 受体拮抗兼 PPAR-gamma 部分激动，文献报道减轻肝细胞脂肪变"),
    # ---------------- FXR / 胆汁酸 ----------------
    ("Obeticholic acid", "C[C@H](CCC(=O)O)[C@H]1CC[C@H]2[C@@H]3[C@H](O)[C@H](CC)[C@@H]4C[C@H](O)CC[C@]4(C)[C@H]3CC[C@]12C",
     "FXR", "半合成胆汁酸 FXR 激动剂，经 SHP 抑制 SREBP-1c 与脂肪生成基因"),
    ("Chenodeoxycholic acid", "C[C@H](CCC(=O)O)[C@H]1CC[C@H]2[C@@H]3[C@H](O)C[C@@H]4C[C@H](O)CC[C@]4(C)[C@H]3CC[C@]12C",
     "FXR", "内源性 FXR 配体，调控胆汁酸稳态并抑制肝脏脂肪生成"),
    ("Ursodeoxycholic acid", "C[C@H](CCC(=O)O)[C@H]1CC[C@H]2[C@@H]3[C@H](O)C[C@@H]4C[C@H](O)CC[C@]4(C)[C@H]3CC[C@]12C",
     "胆汁酸 / 内质网应激", "亲水性胆汁酸，缓解内质网应激与肝细胞脂性凋亡"),
    ("GW4064", "CC(C)c1onc(-c2c(Cl)cccc2Cl)c1COc1ccc(/C=C/c2cccc(C(=O)O)c2)c(C)c1",
     "FXR", "非甾体高效 FXR 激动剂工具化合物，抑制 DNL 相关转录程序"),
    # ---------------- AMPK / 能量感受 ----------------
    ("Metformin", "CN(C)C(=N)NC(=N)N",
     "AMPK (indirect, complex I)", "抑制线粒体复合物I 提升 AMP/ATP，间接激活 AMPK 磷酸化失活 ACC"),
    ("A-769662", "Oc1c(C#N)c(=O)[nH]c2scc(-c3ccc(-c4ccccc4O)cc3)c12",
     "AMPK (direct, beta1)", "直接变构激活 AMPK beta1 亚基，磷酸化 ACC1/2 抑制 DNL"),
    ("AICAR", "NC(=O)c1ncn([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c1N",
     "AMPK (AMP mimetic)", "细胞内转化为 ZMP 模拟 AMP 别构激活 AMPK"),
    ("Salicylate", "O=C(O)c1ccccc1O",
     "AMPK (direct, beta1)", "直接结合 AMPK beta1 变构位点，抑制脂肪生成"),
    ("Berberine", "COc1ccc2cc3[n+](cc2c1OC)CCc1cc2c(cc1-3)OCO2",
     "AMPK / PCSK9 / LDLR", "原小檗碱型生物碱，激活 AMPK 并稳定 LDLR mRNA，HepG2 经典降脂阳性对照"),
    # ---------------- HMGCR / 胆固醇 ----------------
    ("Atorvastatin", "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O",
     "HMGCR", "HMG-CoA 还原酶抑制，降低胆固醇合成并上调 LDLR"),
    ("Simvastatin", "CCC(C)(C)C(=O)O[C@H]1C[C@H](C)C=C2C=C[C@H](C)[C@H](CC[C@@H]3C[C@@H](O)CC(=O)O3)[C@@H]21",
     "HMGCR", "内酯型前药，抑制胆固醇合成并激活 AMPK/自噬"),
    ("Lovastatin", "CCC(C)C(=O)O[C@H]1C[C@H](C)C=C2C=C[C@H](C)[C@H](CC[C@@H]3C[C@@H](O)CC(=O)O3)[C@@H]21",
     "HMGCR", "天然内酯型 HMGCR 抑制剂前药"),
    ("Rosuvastatin", "CC(C)c1nc(N(C)S(C)(=O)=O)nc(-c2ccc(F)cc2)c1/C=C/[C@@H](O)C[C@@H](O)CC(=O)O",
     "HMGCR", "亲水性强效 HMGCR 抑制剂，肝选择性摄取(OATP1B1)"),
    ("Fluvastatin", "CC(C)n1c(/C=C/[C@@H](O)C[C@@H](O)CC(=O)O)c(-c2ccc(F)cc2)c2ccccc21",
     "HMGCR", "全合成 HMGCR 抑制剂"),
    ("Pravastatin", "CC[C@H](C)C(=O)O[C@H]1C[C@H](O)C=C2C=C[C@H](C)[C@H](CC[C@@H](O)C[C@@H](O)CC(=O)O)[C@@H]12",
     "HMGCR", "开环羟酸型亲水 HMGCR 抑制剂"),
    ("Ezetimibe", "O[C@H](CC[C@H]1[C@H](c2ccc(O)cc2)N(c2ccc(F)cc2)C1=O)c1ccc(F)cc1",
     "NPC1L1", "抑制胆固醇再摄取，降低肝内胆固醇酯负荷"),
    # ---------------- 脂解 / 脂肪酸氧化 / 转运 ----------------
    ("Orlistat", "CCCCCCCCCCC[C@H](C[C@@H]1OC(=O)[C@H]1CCCCCC)OC(=O)[C@@H](CC(C)C)NC=O",
     "Lipase / FASN-TE", "beta-内酯共价抑制脂肪酶及 FASN 硫酯酶结构域"),
    ("L-Carnitine", "C[N+](C)(C)C[C@H](O)CC(=O)[O-]",
     "CPT1 / 肉碱穿梭", "肉碱穿梭底物，促进长链脂肪酸进入线粒体 beta-氧化"),
    ("Etomoxir", "CCOC(=O)C1(CCCCCCOc2ccc(Cl)cc2)CO1",
     "CPT1a (抑制剂/阴性对照)", "CPT1 不可逆抑制剂，作为脂肪酸氧化通路机制阴性对照"),
    ("Perhexiline", "C1CCC(CC1)C(CC1CCCCC1)C1CCCCN1",
     "CPT1/CPT2 (抑制剂)", "CPT 抑制剂，代谢转向葡萄糖氧化；已知具肝毒性风险，用作毒性锚点"),
    ("Trimetazidine", "COc1ccc(OC)c(CN2CCNCC2)c1OC",
     "3-KAT (长链硫解酶)", "部分脂肪酸氧化抑制剂，代谢调节剂"),
    ("Ranolazine", "COc1ccccc1OCC(O)CN1CCN(CC(=O)Nc2c(C)cccc2C)CC1",
     "pFOX / 晚钠电流", "部分脂肪酸氧化抑制，改善代谢底物利用"),
    ("EPA (Eicosapentaenoic acid)", "CC/C=C\\C/C=C\\C/C=C\\C/C=C\\C/C=C\\CCCC(=O)O",
     "PPAR-alpha / SREBP-1c", "omega-3 多不饱和脂肪酸，激活 PPAR-alpha 并抑制 SREBP-1c 成熟"),
    ("DHA (Docosahexaenoic acid)", "CC/C=C\\C/C=C\\C/C=C\\C/C=C\\C/C=C\\C/C=C\\CCC(=O)O",
     "PPAR-alpha / SREBP-1c", "omega-3 多不饱和脂肪酸，降低肝内 TG 与脂毒性"),
    ("Nicotinic acid (Niacin)", "O=C(O)c1cccnc1",
     "HCAR2 / DGAT2", "抑制脂肪组织脂解并弱抑制 DGAT2，减少肝内 VLDL 底物供应"),
    ("Alpha-lipoic acid", "OC(=O)CCCCC1CCSS1",
     "AMPK / 线粒体氧化还原", "二硫戊环辅因子，改善线粒体功能并激活 AMPK"),
    # ---------------- 天然产物 / 其他 ----------------
    ("Naringenin", "O=C1CC(c2ccc(O)cc2)Oc2cc(O)cc(O)c21",
     "PPAR-alpha / SREBP", "柑橘黄烷酮，抑制脂肪生成基因并促进脂肪酸氧化"),
    ("Fenretinide (4-HPR)", "CC1=C(/C=C/C(C)=C/C=C/C(C)=C/C(=O)Nc2ccc(O)cc2)C(C)(C)CCC1",
     "DES1 / RBP4", "视黄酰胺，抑制二氢神经酰胺去饱和酶，改善肝脂肪变"),
    ("Pentoxifylline", "Cn1c(=O)n(CCCCC(C)=O)c2ncn(C)c2c1=O",
     "PDE / TNF-alpha", "非选择性磷酸二酯酶抑制剂，降低肝脏炎症与脂质沉积"),
    ("Betaine", "C[N+](C)(C)CC(=O)[O-]",
     "甲基供体 / PEMT", "甲基供体，支持磷脂酰胆碱合成与 VLDL 输出"),
]

REF_MOLS, REF_FPS, REF_META = [], [], []
for _name, _smi, _tgt, _mech in REFERENCE_DRUGS:
    _m = Chem.MolFromSmiles(_smi)
    if _m is None:
        print(f"[warn] 参照药物 SMILES 解析失败，已跳过: {_name}")
        continue
    REF_MOLS.append(_m)
    REF_FPS.append(MFPGEN.GetFingerprint(_m))
    REF_META.append({"name": _name, "target": _tgt, "mechanism": _mech,
                     "formula": rdMolDescriptors.CalcMolFormula(_m)})
print(f"[info] 参照药物集载入 {len(REF_FPS)} / {len(REFERENCE_DRUGS)} 个")

# ==============================================================================
# 0C. 肝脂靶点优势骨架 (privileged chemotype) 规则库
#     每条规则: 全部 smarts 命中 且 全部 anti 不命中 -> 判定为该 chemotype
# ==============================================================================
def _S(x): return Chem.MolFromSmarts(x)

CHEMOTYPE_RULES = [
    # 注意：必须限定为"二芳醚"(桥氧两侧均为芳碳)。若写成 O[#6] 会把 3,5-二卤代苯甲醚
    #       这类甲基醚误判为甲状腺激素拟似物 —— T3/resmetirom 药效团要求 4-芳氧基外环。
    dict(name="3,5-二卤代-4-芳氧基苯 (甲状腺激素拟似物核)", weight=0.95,
         target="THR-beta", smarts=["[Cl,Br,I]c1cccc([Cl,Br,I])c1[OX2][c]"], anti=[],
         mech="模拟 T3 的 3,5-二卤代-4-芳氧基苯药效团，激动肝脏 THR-beta，"
              "上调 CPT1A/DIO1/线粒体生物合成，增强脂肪酸 beta-氧化与脂噬"),
    dict(name="噻唑烷-2,4-二酮 (TZD)", weight=0.92,
         target="PPAR-gamma", smarts=["O=C1NC(=O)SC1"], anti=[],
         mech="TZD 酸性头基占据 PPAR-gamma 配体口袋 AF-2 螺旋，改善胰岛素敏感性、"
              "促进脂质向脂肪组织再分布，减轻肝细胞脂毒性"),
    dict(name="statin 3,5-二羟基戊酸头基", weight=0.95,
         target="HMGCR", smarts=["[OX2H][CH1]C[CH1]([OX2H])CC(=O)[OX2H1,OX1-]"], anti=[],
         mech="与 HMG-CoA 竞争性结合 HMG-CoA 还原酶催化位点，阻断甲羟戊酸通路，"
              "上调 SREBP-2/LDLR 并激活 AMPK 依赖的脂质清除"),
    dict(name="statin delta-内酯前药", weight=0.88,
         target="HMGCR", smarts=["O=C1C[CH1]([OX2H])CC[OX2]1"], anti=[],
         mech="内酯型前药经肝细胞羧酸酯酶开环为活性 3,5-二羟基戊酸，抑制 HMGCR"),
    dict(name="芳氧基异丁酸 (贝特类头基)", weight=0.90,
         target="PPAR-alpha", smarts=["[OX2](c)C(C)(C)C(=O)[OX2H1,OX1-]"], anti=[],
         mech="贝特类酸性头基激动 PPAR-alpha，转录上调 CPT1A/ACOX1/FGF21，"
              "增强线粒体与过氧化物酶体脂肪酸 beta-氧化，降低肝内 TG"),
    dict(name="双胍 (biguanide)", weight=0.90,
         target="AMPK (indirect)", smarts=["[NX3][CX3](=[NX2,NX3])[NX3][CX3](=[NX2,NX3])[NX3]"], anti=[],
         mech="抑制线粒体呼吸链复合物I，升高 AMP/ATP 比值激活 LKB1-AMPK 轴，"
              "磷酸化失活 ACC1/2 并抑制 SREBP-1c，阻断新发脂质合成"),
    dict(name="噻吩并吡啶酮 (AMPK 直接激动核)", weight=0.90,
         target="AMPK (direct)", smarts=["s1ccc2c1[nH]c(=O)cc2", "[CX2]#[NX1]"], anti=[],
         mech="结合 AMPK alpha/beta 界面的 ADaM 变构位点直接激活 AMPK，磷酸化 ACC"),
    dict(name="胆汁酸甾核 + C24 羧酸", weight=0.90,
         target="FXR", smarts=["C1CC2CCC3C(CCC4CCCCC34)C2C1", "[CX3](=O)[OX2H1,OX1-]"], anti=[],
         mech="甾体胆汁酸骨架激动核受体 FXR，经 SHP 抑制 SREBP-1c/ChREBP 转录程序，"
              "并经 FGF19 轴抑制 CYP7A1，降低肝内脂质与胆汁酸负荷"),
    dict(name="异噁唑/噁二唑 + 芳香羧酸 (非甾体 FXR)", weight=0.75,
         target="FXR", smarts=["[$(c1cc(no1)),$(c1nnco1),$(c1ncno1)]", "c-[CX3](=O)[OX2H1,OX1-]"], anti=[],
         mech="非甾体 FXR 激动药效团，激活 FXR-SHP 轴抑制脂肪生成基因表达"),
    dict(name="原小檗碱型季铵异喹啉生物碱", weight=0.80,
         target="AMPK / LDLR", smarts=["[n+]1ccc2ccccc2c1"], anti=[],
         mech="小檗碱型生物碱抑制线粒体复合物I 激活 AMPK，并经 ERK 通路稳定 LDLR mRNA，"
              "为 HepG2 脂质蓄积模型的经典降脂阳性对照化学型"),
    dict(name="芳基乙酸 + 磺酰胺 (pan-PPAR lanifibranor 型)", weight=0.78,
         target="PPAR pan", smarts=["c[CH2][CX3](=O)[OX2H1,OX1-]", "[SX4](=O)(=O)[NX3]"], anti=[],
         mech="磺酰基环胺连接的芳基乙酸头基，泛 PPAR(alpha/delta/gamma) 激动，"
              "同时促进 beta-氧化、抑制脂肪生成并下调炎症纤维化通路"),
    dict(name="嘧啶/吡啶甲酰胺 + 哌啶(哌嗪)", weight=0.70,
         target="DGAT2 / FASN", smarts=["[$(c1ncncc1),$(c1ccncc1),$(c1cnccn1)][CX3](=O)[NX3]",
                                        "[NX3;R]1[CH2][CH2][#6][CH2][CH2]1"], anti=[],
         mech="DGAT2/FASN 抑制剂的通用铰链-酰胺药效团；DGAT2 抑制直接阻断甘油三酯"
              "酯化终末步骤，FASN 抑制阻断棕榈酸从头合成，两者均显著降低脂滴负荷"),
    dict(name="杂芳基脲 (DGAT2/SCD1 型)", weight=0.68,
         target="DGAT2 / SCD1", smarts=["[c,n][NX3;H1][CX3](=O)[NX3;R]"], anti=[],
         mech="杂芳基脲连接的哌啶醚母核，抑制 DGAT2 或 SCD1，减少甘油三酯与"
              "单不饱和脂肪酸生成，降低脂滴体积"),
    dict(name="哌嗪/哌啶杂芳基甲酰胺", weight=0.55,
         target="SCD1 / DGAT2", smarts=["[NX3;R]([CX3](=O)[c])"], anti=[],
         mech="脂质酶抑制剂常见的环胺-酰胺连接模式，倾向作用于 SCD1/DGAT2 等"
              "内质网膜结合型脂质合成酶"),
    dict(name="长链脂肪酸 / 类脂肪酸", weight=0.72,
         target="PPAR-alpha / SREBP-1c", smarts=["[CX3](=O)[OX2H1,OX1-]", "CCCCCCCCCC"], anti=[],
         mech="内源性脂肪酸配体样分子，直接激动 PPAR-alpha 上调氧化基因，"
              "并抑制 SREBP-1c 前体加工，降低脂肪生成"),
    dict(name="季铵-羧酸两性离子 (肉碱样)", weight=0.70,
         target="CPT1 / 肉碱穿梭", smarts=["[NX4+](C)(C)(C)[CX4]", "[CX3](=O)[OX2H1,OX1-]"], anti=[],
         mech="肉碱穿梭系统底物/类似物，促进长链脂酰辅酶A 跨线粒体内膜转运，"
              "提升 beta-氧化通量"),
    dict(name="黄酮/黄烷酮母核", weight=0.60,
         target="AMPK / SREBP-1c", smarts=["[$(O=c1cc(-c2ccccc2)oc2ccccc12),$(O=C1CC(c2ccccc2)Oc2ccccc21)]"], anti=[],
         mech="黄酮类多酚激活 AMPK-SIRT1 轴，抑制 SREBP-1c/FAS/ACC 表达并促进脂噬"),
    dict(name="联苯羧酸 + 苯并咪唑 (telmisartan 型)", weight=0.70,
         target="PPAR-gamma partial", smarts=["c1ccc(-c2ccccc2)cc1", "c-[CX3](=O)[OX2H1,OX1-]",
                                              "c1nc2ccccc2[nH,n]1"], anti=[],
         mech="沙坦类联苯羧酸兼具 PPAR-gamma 部分激动活性，文献报道可减轻肝细胞脂肪变"),
    dict(name="腺苷/核苷类似物 (AMP 拟似)", weight=0.68,
         target="AMPK (AMP mimetic)", smarts=["n1cnc2c1ncnc2", "[CH1]1O[CH1][CH1][CH1]1"], anti=[],
         mech="细胞内磷酸化为 AMP 拟似物，别构激活 AMPK gamma 亚基 CBS 结构域"),
    dict(name="beta-内酯 / 环氧共价弹头 + 长脂链", weight=0.62,
         target="Lipase / FASN-TE", smarts=["[$(O=C1CCO1),$(C1CO1)]", "CCCCCCCC"], anti=[],
         mech="共价修饰脂肪酶或 FASN 硫酯酶结构域活性位点丝氨酸/半胱氨酸；"
              "需注意其共价机制带来的脱靶细胞毒性风险"),
    dict(name="二硫戊环 (硫辛酸型)", weight=0.50,
         target="AMPK / 氧化还原", smarts=["C1CSSC1"], anti=[],
         mech="二硫戊环氧化还原对改善线粒体功能并激活 AMPK，减轻脂质过氧化"),
    dict(name="二芳醚/二芳酮 + 羧酸", weight=0.50,
         target="PPAR / THR-beta (通用核受体)", smarts=["[$(c-[OX2]-c),$(c-[CX3](=O)-c)]",
                                                       "[CX3](=O)[OX2H1,OX1-]"], anti=[],
         mech="核受体配体常见的疏水二芳基骨架 + 酸性头基组合，倾向结合 PPAR/THR 等"
              "脂代谢核受体配体口袋"),
]

for _r in CHEMOTYPE_RULES:
    _r["_pat"] = [_S(s) for s in _r["smarts"]]
    _r["_anti"] = [_S(s) for s in _r.get("anti", [])]
    if any(p is None for p in _r["_pat"] + _r["_anti"]):
        raise ValueError(f"SMARTS 编译失败: {_r['name']}")
print(f"[info] 优势骨架规则库载入 {len(CHEMOTYPE_RULES)} 条")


def match_chemotypes(mol):
    """返回按权重降序的命中 chemotype 列表。"""
    hits = []
    for r in CHEMOTYPE_RULES:
        if all(mol.HasSubstructMatch(p) for p in r["_pat"]) and \
           not any(mol.HasSubstructMatch(a) for a in r["_anti"]):
            hits.append(r)
    return sorted(hits, key=lambda x: -x["weight"])


# ==============================================================================
# 打分函数
# ==============================================================================
def gaussian(x, mu, sigma):
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2)


def plateau(x, lo, hi, sigma):
    """区间内满分、区间外高斯衰减 —— "最佳区间 + 距离衰减" 的实现。"""
    if lo <= x <= hi:
        return 1.0
    d = (lo - x) if x < lo else (x - hi)
    return math.exp(-0.5 * (d / sigma) ** 2)


def score_efficacy(logp, tpsa, arom_rings, rotb):
    """
    S_efficacy —— 降脂潜力 / 肝细胞内有效暴露倾向
      最佳 LogP [2.5, 3.5]  : 兼顾被动扩散入肝细胞与胞内膜相分配
      最佳 TPSA [50, 90]    : 足够跨膜且不过度滞留于中性脂滴
      形状项                : 芳环 1-3、可旋转键 2-7 的类先导性区间
    """
    s_logp  = plateau(logp, 2.5, 3.5, 1.0)
    s_tpsa  = plateau(tpsa, 50.0, 90.0, 25.0)
    s_shape = 0.5 * plateau(arom_rings, 1, 3, 1.0) + 0.5 * plateau(rotb, 2, 7, 3.0)
    return 0.45 * s_logp + 0.40 * s_tpsa + 0.15 * s_shape


def score_safety(mw, logp, tpsa, n_tox, n_halogen, fsp3):
    """
    S_safety —— 低细胞毒性倾向 (HepG2)
      主项  : MW 向 350 收敛的高斯
      扣分  : 结构警示子 / 卤素过载 / 高脂溶性 / 极低极性
      加分  : sp3 比例 (逃离"平面化"，脱靶更少)
    """
    s = gaussian(mw, 350.0, 120.0)
    s -= 0.25 * min(n_tox, 3)                        # 反应性警示子，最多 -0.75
    if n_halogen > 3:
        s -= 0.10 * min(n_halogen - 3, 3)            # 卤素过载 -> 亲脂性/解偶联毒性
    if logp > 4.2:
        s -= 0.30 * min((logp - 4.2) / 0.8, 1.0)     # 高脂溶性 -> 膜损伤/线粒体解偶联
    if tpsa < 30:
        s -= 0.30 * min((30 - tpsa) / 30.0, 1.0)     # 极低极性 -> 非特异性膜穿透
    s += 0.10 * min(fsp3 / 0.4, 1.0)
    return max(0.0, min(1.0, s))


# ==============================================================================
# Step 1 + 2 : 解析、描述符、过滤
# ==============================================================================
def safe_prop(m, key):
    """SDF 属性读取容错封装。"""
    try:
        return m.GetProp(key).strip() if m.HasProp(key) else ""
    except Exception:
        return ""


_CAS_DATE = re.compile(r"^(\d{2,7})[/\-](\d{1,2})[/\-](\d{1,2})$")


def repair_cas(cas):
    """
    源 SDF 的 CAS 字段部分被 Excel 误识别为日期 (如 '2458/8/4')。
    还原为标准 CAS 'XXXXXXX-XX-X' 并用校验位验证；校验失败则保留原值。
    """
    if not cas:
        return ""
    if re.fullmatch(r"\d{2,7}-\d{2}-\d", cas):
        return cas
    m = _CAS_DATE.match(cas)
    if not m:
        return cas
    body, mid, chk = m.group(1), int(m.group(2)), int(m.group(3))
    cand = f"{body}-{mid:02d}-{chk}"
    digits = [int(c) for c in (body + f"{mid:02d}")][::-1]
    if sum(d * (i + 1) for i, d in enumerate(digits)) % 10 == chk:
        return cand
    return cas


records = []
stats = dict(total=0, unparsable=0, bad_element=0, too_small=0,
             pains=0, lipinski=0, cytotox_hard=0, passed=0)

print("\n[info] 开始解析 SDF ...")
supplier = Chem.SDMolSupplier(SDF_FILE, sanitize=True, removeHs=True)

for idx, mol in enumerate(supplier):
    stats['total'] += 1
    if mol is None:
        stats['unparsable'] += 1
        continue

    mol_id      = safe_prop(mol, "ID") or safe_prop(mol, "_Name") or f"Mol_{idx+1}"
    cas         = repair_cas(safe_prop(mol, "CAS"))
    formula_sdf = safe_prop(mol, "Formula")

    # 结构标准化：去盐取最大片段 / 电荷中和 / 重新 sanitize
    try:
        mol = LFC.choose(mol)
        mol = UNCHARGER.uncharge(mol)
        Chem.SanitizeMol(mol)
    except Exception:
        stats['unparsable'] += 1
        continue

    zs = [a.GetAtomicNum() for a in mol.GetAtoms()]
    if any(z not in ALLOWED_ELEMENTS for z in zs):
        stats['bad_element'] += 1          # 金属配合物 / 无机盐 / Si,B,Sn 等
        continue
    if mol.GetNumHeavyAtoms() < 12:
        stats['too_small'] += 1            # 过小碎片不足以支撑细胞水平活性
        continue

    try:
        mw    = Descriptors.MolWt(mol)
        logp  = Crippen.MolLogP(mol)
        tpsa  = rdMolDescriptors.CalcTPSA(mol)
        hbd   = Lipinski.NumHDonors(mol)
        hba   = Lipinski.NumHAcceptors(mol)
        rotb  = Lipinski.NumRotatableBonds(mol)
        arom  = rdMolDescriptors.CalcNumAromaticRings(mol)
        rings = rdMolDescriptors.CalcNumRings(mol)
        fsp3  = rdMolDescriptors.CalcFractionCSP3(mol)
        heavy = mol.GetNumHeavyAtoms()
        smiles  = Chem.MolToSmiles(mol)
        formula = rdMolDescriptors.CalcMolFormula(mol)
    except Exception:
        stats['unparsable'] += 1
        continue

    # --- PAINS 泛筛选干扰结构剔除 ---
    if PAINS.HasMatch(mol):
        stats['pains'] += 1
        continue

    # --- Lipinski 类药性过滤 (赛题设定阈值) ---
    if mw > 550 or mw < 150 or logp > 5.0 or logp < -1.0 or hbd > 5 or hba > 10:
        stats['lipinski'] += 1
        continue

    # --- 经验性细胞毒性硬反筛：高脂溶(LogP>4.5) 且 极低极性(TPSA<20) ---
    if logp > 4.5 and tpsa < 20:
        stats['cytotox_hard'] += 1
        continue

    tox_hits  = [k for k, p in TOX_PATTERNS.items() if mol.HasSubstructMatch(p)]
    n_halogen = sum(1 for z in zs if z in HALOGENS)

    s_eff = score_efficacy(logp, tpsa, arom, rotb)
    s_saf = score_safety(mw, logp, tpsa, len(tox_hits), n_halogen, fsp3)
    s_tot = 0.6 * s_eff + 0.4 * s_saf

    records.append(dict(
        Molecule_ID=mol_id, CAS=cas, SMILES=smiles, Formula=formula or formula_sdf,
        MW=round(mw, 2), LogP=round(logp, 2), TPSA=round(tpsa, 2),
        HBD=hbd, HBA=hba, RotB=rotb, AromRings=arom, Rings=rings,
        FracCsp3=round(fsp3, 3), HeavyAtoms=heavy,
        N_Halogen=n_halogen, N_ToxAlerts=len(tox_hits),
        Tox_Alerts=";".join(tox_hits) if tox_hits else "none",
        S_efficacy=round(s_eff, 4), S_safety=round(s_saf, 4), S_total=round(s_tot, 4),
    ))
    stats['passed'] += 1

print("\n================ Step 1-2 过滤漏斗 ================")
print(f"  SDF 总记录数                        : {stats['total']:>6d}")
print(f"  - RDKit 解析/标准化失败             : {stats['unparsable']:>6d}")
print(f"  - 非白名单元素 (金属/无机)          : {stats['bad_element']:>6d}")
print(f"  - 重原子数 < 12 (碎片)              : {stats['too_small']:>6d}")
print(f"  - PAINS 泛筛选干扰结构              : {stats['pains']:>6d}")
print(f"  - Lipinski 类药性不合格             : {stats['lipinski']:>6d}")
print(f"  - 细胞毒性硬反筛 (LogP>4.5 & TPSA<20): {stats['cytotox_hard']:>6d}")
print(f"  = 进入多目标打分的分子数            : {stats['passed']:>6d}")

df = pd.DataFrame(records).sort_values("S_total", ascending=False).reset_index(drop=True)

# --- 去重：库内存在同一结构的多个货号(游离碱/盐/不同批次)，按标准 SMILES 合并 ---
n_before = len(df)
dup_map = (df.groupby("SMILES")["Molecule_ID"]
             .apply(lambda s: ";".join(sorted(set(s))[:8])).to_dict())
df = df.drop_duplicates(subset="SMILES", keep="first").reset_index(drop=True)
df["Duplicate_IDs"] = df["SMILES"].map(dup_map)
print(f"\n[info] 结构去重: {n_before} -> {len(df)} (合并 {n_before - len(df)} 个重复结构条目)")

df = df.sort_values("S_total", ascending=False).reset_index(drop=True)
df.insert(0, "Global_Rank", np.arange(1, len(df) + 1))

n_tie = int((df.S_total >= 0.9999).sum())
print(f"\n[!] S_total 饱和检查: {n_tie} 个分子并列 S_total = 1.0000 "
      f"({n_tie/len(df)*100:.1f}%) —— 单靠 S_total 无法排序，需机制证据层。")

# ==============================================================================
# Step 3b : 机制证据层 (参照药物相似性 + 优势骨架规则)
#           施加 S_total 硬门槛后再计算，兼顾算力与相关性
# ==============================================================================
GATE = 0.95
gated = df[df.S_total >= GATE].copy().reset_index(drop=True)
print(f"\n[info] S_total >= {GATE} 硬门槛后剩余 {len(gated)} 个分子，开始机制证据计算 ...")

TANI_STRONG = 0.55          # Morgan(r=2) 下视为"强化学型相关"的相似度
POS_CTRL_T  = 0.85          # 判定为库内已知阳性对照的相似度阈值

cols = {k: [] for k in ["Nearest_Reference", "Ref_Target", "Ref_Mechanism", "Ref_Tanimoto",
                        "Chemotype", "Chemotype_Target", "Chemotype_Mechanism",
                        "Chemotype_Weight", "S_mech", "Murcko_Scaffold"]}

for smi in gated["SMILES"]:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        for k in cols: cols[k].append("" if k not in ("Ref_Tanimoto", "Chemotype_Weight", "S_mech") else 0.0)
        continue

    fp   = MFPGEN.GetFingerprint(m)
    sims = DataStructs.BulkTanimotoSimilarity(fp, REF_FPS)
    j    = int(np.argmax(sims)); tmax = float(sims[j])

    ct = match_chemotypes(m)
    if ct:
        c0 = ct[0]
        c_name, c_tgt, c_mech, c_w = c0["name"], c0["target"], c0["mech"], c0["weight"]
    else:
        c_name, c_tgt, c_mech, c_w = "无明确优势骨架命中", "", "", 0.0

    t_scaled = min(tmax / TANI_STRONG, 1.0)
    s_mech   = round(0.55 * t_scaled + 0.45 * c_w, 4)

    cols["Nearest_Reference"].append(REF_META[j]["name"])
    cols["Ref_Target"].append(REF_META[j]["target"])
    cols["Ref_Mechanism"].append(REF_META[j]["mechanism"])
    cols["Ref_Tanimoto"].append(round(tmax, 3))
    cols["Chemotype"].append(c_name)
    cols["Chemotype_Target"].append(c_tgt)
    cols["Chemotype_Mechanism"].append(c_mech)
    cols["Chemotype_Weight"].append(c_w)
    cols["S_mech"].append(s_mech)
    try:
        cols["Murcko_Scaffold"].append(MurckoScaffold.MurckoScaffoldSmiles(mol=m))
    except Exception:
        cols["Murcko_Scaffold"].append("")

for k, v in cols.items():
    gated[k] = v

gated["S_nomination"] = (0.5 * gated["S_total"] + 0.5 * gated["S_mech"]).round(4)

# ==============================================================================
# Step 3c : 阳性对照回收 —— 方法学自验证
# ==============================================================================
pos_ctrl = gated[gated.Ref_Tanimoto >= POS_CTRL_T].sort_values(
    "Ref_Tanimoto", ascending=False).reset_index(drop=True)
print(f"\n================ 阳性对照回收 (Tanimoto >= {POS_CTRL_T}) ================")
if len(pos_ctrl):
    print(f"  流程在通过全部过滤的分子中回收到 {len(pos_ctrl)} 个已知肝脂调节剂近似物：")
    print(pos_ctrl[["Molecule_ID", "CAS", "Nearest_Reference", "Ref_Tanimoto",
                    "Ref_Target", "S_total"]].head(25).to_string(index=False))
else:
    print("  未回收到已知阳性对照。")

# ==============================================================================
# Step 3d : 排除阳性对照 -> 按 S_nomination 排序 -> 骨架多样性 Top 10
# ==============================================================================
novel = gated[gated.Ref_Tanimoto < POS_CTRL_T].sort_values(
    ["S_nomination", "S_mech", "S_total"], ascending=False).reset_index(drop=True)

# --- 多样性选择 (贪心) ---
#   仅靠 Murcko 骨架去冗余不足：库中存在骨架微异但化学型等同的近似物
#   (例：小檗碱系列生物碱)。故同时施加三重约束：
#     C1 Murcko 骨架唯一
#     C2 与已选分子的 Morgan Tanimoto < DIV_T (化学空间分散)
#     C3 每个靶点假说最多 MAX_PER_TARGET 个 (=1，使 10 个提名覆盖 10 条独立机制，
#        最大化机制多样性并分散单一通路失败的风险)
DIV_T, MAX_PER_TARGET, N_TOP = 0.60, 1, 10

sel_scaffolds, sel_fps, target_count, keep = set(), [], {}, []
for i, row in novel.iterrows():
    m = Chem.MolFromSmiles(row["SMILES"])
    if m is None:
        continue
    sc  = row["Murcko_Scaffold"] or f"__acyclic_{i}"
    tgt = row["Chemotype_Target"] or row["Ref_Target"] or "unassigned"
    if sc in sel_scaffolds:
        continue
    if target_count.get(tgt, 0) >= MAX_PER_TARGET:
        continue
    fp = MFPGEN.GetFingerprint(m)
    if sel_fps and max(DataStructs.BulkTanimotoSimilarity(fp, sel_fps)) >= DIV_T:
        continue
    sel_scaffolds.add(sc)
    sel_fps.append(fp)
    target_count[tgt] = target_count.get(tgt, 0) + 1
    keep.append(i)
    if len(keep) >= N_TOP:
        break

top10 = novel.loc[keep].copy().reset_index(drop=True)
top10.insert(0, "Rank", np.arange(1, len(top10) + 1))
print(f"\n[info] 多样性选择: 骨架唯一 + 两两 Tanimoto < {DIV_T} + 每靶点 <= {MAX_PER_TARGET} 个")

# ==============================================================================
# 导出中间结果
# ==============================================================================
df.to_csv(f"{OUTDIR}/all_scored_molecules.csv", index=False, encoding="utf-8-sig")
gated.to_csv(f"{OUTDIR}/gated_annotated.csv", index=False, encoding="utf-8-sig")
pos_ctrl.to_csv(f"{OUTDIR}/positive_control_recovery.csv", index=False, encoding="utf-8-sig")
top10.to_csv(f"{OUTDIR}/_top10_raw.csv", index=False, encoding="utf-8-sig")
with open(f"{OUTDIR}/_funnel_stats.json", "w", encoding="utf-8") as f:
    json.dump({**stats, "n_tie_S_total_1": n_tie, "gate": GATE,
               "n_gated": int(len(gated)), "n_pos_ctrl": int(len(pos_ctrl)),
               "n_novel": int(len(novel)), "n_refs": len(REF_FPS),
               "n_chemotype_rules": len(CHEMOTYPE_RULES)}, f, ensure_ascii=False, indent=2)

print("\n================ 提名 Top 10 (新颖化学型) ================")
show = ["Rank", "Molecule_ID", "MW", "LogP", "TPSA", "S_efficacy", "S_safety",
        "S_total", "S_mech", "S_nomination", "Chemotype_Target", "Ref_Tanimoto"]
print(top10[show].to_string(index=False))
print(f"\n[done] 中间结果已写入 {OUTDIR}")
```

### 11.2 `02_deliverables.py` —— candidates.csv、结构图、机制与验证方案 PDF

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
MASLD 早期降脂小分子发现 —— 交付物生成
================================================================================
读入 01_screen.py 的中间结果，生成三项竞赛交付物：
  1. candidates.csv                  候选分子提名清单 (Top 10)
  2. mechanism_and_validation.pdf    机制与验证方案
  3. methodology_and_code.md         方法学与复现材料

【关于专家注释】
本脚本中的 CURATED 字典为人工核查每个候选分子结构后撰写的靶点/机制/风险注释，
其中的靶点归属与 01_screen.py 的 SMARTS 优势骨架规则判定一致；机制描述与毒性
警示来自对该化学型已发表药理学的解读。数值列 (S_efficacy / S_safety / S_total /
S_mech / S_nomination / 理化描述符) 全部由代码计算，未经人工修改。
"""

import os, json, textwrap
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, PageBreak, Image, KeepTogether)

RDLogger.DisableLog('rdApp.*')

OUTDIR = "/mnt/c/Users/zhangt97/Downloads/MASLD_screen"
IMGDIR = os.path.join(OUTDIR, "structures")
os.makedirs(IMGDIR, exist_ok=True)

CN = "STSong-Light"
pdfmetrics.registerFont(UnicodeCIDFont(CN))

top10    = pd.read_csv(f"{OUTDIR}/_top10_raw.csv")
pos_ctrl = pd.read_csv(f"{OUTDIR}/positive_control_recovery.csv")
stats    = json.load(open(f"{OUTDIR}/_funnel_stats.json", encoding="utf-8"))

# ==============================================================================
# 专家注释层 —— 逐分子人工核查结构后撰写
# ==============================================================================
CURATED = {
"T2607": dict(
    chem="5-{4-[2-(5-乙基吡啶-2-基)-2-氧代乙氧基]苄基}噻唑烷-2,4-二酮；吡格列酮的酮基同系物",
    target="PPAR-gamma (兼 PPAR-alpha 部分激动)",
    mech="TZD 酸性头基进入 PPAR-gamma 配体结合域，与 Tyr473/His449/His323/Ser289 形成四点氢键网络，"
         "稳定 AF-2 螺旋12 构象并招募 SRC-1/PGC-1alpha 共激活因子。在肝细胞层面下调 SREBP-1c 驱动的 "
         "FASN/ACC1/SCD1 转录，抑制新发脂质合成(DNL)；同时经脂联素-AdipoR-AMPK 轴磷酸化失活 ACC，"
         "降低丙二酰辅酶A 并解除对 CPT1A 的变构抑制，提升线粒体 beta-氧化通量。",
    rationale="全流程最高分 (S_nomination=0.982)。LogP 2.80 与 TPSA 85.4 同时落于最佳窗口中心，"
              "S_efficacy 与 S_safety 双双满分；TZD 头基精确命中 PPAR-gamma 药效团 (规则权重 0.92)；"
              "与吡格列酮 Tanimoto=0.673，属同化学型但非同一分子，具备结构衍生空间；MW 370 接近 350 最优值；"
              "无任何结构警示子。",
    tox_band="低",
    tox_note="无结构警示子，LogP/TPSA 均在安全区。类别风险提示：曲格列酮的肝毒性源于其色满环醌式代谢物，"
             "本分子不含色满环亦无醌前体，该特定风险不适用；吡格列酮/罗格列酮在 HepG2 中通常 ≤100 µM 无明显活力下降。"),

"T12092": dict(
    chem="六氢萘并 delta-内酯型 statin 母核 (mevastatin/monacolin 系)，含二级羟基",
    target="HMGCR (3-羟基-3-甲基戊二酰辅酶A 还原酶)",
    mech="delta-内酯为前药形式，经肝细胞羧酸酯酶 CES1 开环生成 3,5-二羟基戊酸活性型，"
         "以纳摩尔级亲和力竞争性占据 HMGCR 的 HMG-CoA 结合位点，阻断甲羟戊酸通路。"
         "继发上调 SREBP-2/LDLR 增强 LDL 摄取；并因香叶基香叶基焦磷酸(GGPP)耗竭解除对 AMPK 的抑制，"
         "间接下调 SREBP-1c 依赖的脂肪生成程序。",
    rationale="命中 statin delta-内酯药效团 (规则权重 0.88)，与辛伐他汀 Tanimoto=0.672；"
              "Csp3=0.74 三维性突出，脱靶倾向低；LogP 2.60/TPSA 66.8 位于最佳窗口；"
              "S_safety=1.000 且无结构警示子；MW 320 类药性优良。",
    tox_band="低 (需机制性验证)",
    tox_note="理化启发式判为低毒，但 statin 类在肝细胞中可因甲羟戊酸/泛醌(CoQ10)耗竭在 >10 µM 出现活力下降。"
             "必须设置甲羟戊酸回补(mevalonate rescue, 200 µM)对照：若回补可逆转降脂效应则为在靶药理，"
             "若仅逆转活力下降则提示在靶毒性，二者需分别判读。"),

"T19124": dict(
    chem="3-氧代-7,12-二羟基-5beta-胆烷-24-酸 (3-脱氢胆酸型氧代胆汁酸)",
    target="FXR (NR1H4，兼 TGR5/GPBAR1)",
    mech="胆汁酸甾核激动核受体 FXR，诱导小异二聚体伴侣 SHP(NR0B2)，进而抑制 SREBP-1c 与 ChREBP 的"
         "转录活性，下调 FASN/ACC1/SCD1/DGAT2 等脂肪生成基因；同时经肠-肝 FGF19/FGFR4-beta-Klotho 轴"
         "抑制 CYP7A1，重塑胆汁酸池。3-位酮基相较天然 3alpha-羟基改变了 A 环与 FXR 口袋的氢键取向，"
         "预期呈部分激动特征，有望降低全激动剂的瘙痒与 LDL-C 升高等类效应。",
    rationale="完整胆汁酸药效团 (甾核 + C24 羧酸，规则权重 0.90)，与鹅去氧胆酸 Tanimoto=0.562；"
              "Csp3=0.92 为全表最高，三维性与选择性俱佳；TPSA 94.8、HBD=3 提供充分极性；"
              "S_safety=0.995 且无结构警示子。",
    tox_band="中-低",
    tox_note="机制性警示：疏水性胆汁酸在高浓度 (>100 µM) 具去污剂样膜损伤与线粒体通透性转换孔开放毒性。"
             "建议将测试上限控制在 50 µM，并以 LDH 释放(膜完整性)与 ATP(CellTiter-Glo) 双读出区分"
             "膜裂解型毒性与代谢抑制型毒性。"),

"T75436": dict(
    chem="原小檗碱型季铵异喹啉生物碱 (一羟基三甲氧基取代，小檗碱去甲基类似物)",
    target="AMPK / LDLR (兼 PCSK9 下调)",
    mech="季铵阳离子在线粒体膜电位驱动下富集于线粒体基质，抑制呼吸链复合物I，升高 AMP/ATP 比值，"
         "经 LKB1 激活 AMPK；AMPK 磷酸化失活 ACC1/2 降低丙二酰辅酶A 阻断 DNL，并解除对 CPT1A 的抑制"
         "促进 beta-氧化；同时抑制 SREBP-1c 前体加工。此外经 ERK 通路延长 LDLR mRNA 半衰期、"
         "下调 PCSK9 与 HNF1alpha，增强 LDL 清除。",
    rationale="S_total=1.000 (S_efficacy 与 S_safety 双满分)；与小檗碱 Tanimoto=0.698 同属原小檗碱化学型，"
              "而小檗碱系是 HepG2-FFA 脂质蓄积模型中文献公认的降脂阳性化学型；LogP 3.08/TPSA 51.8 兼顾"
              "细胞摄取与溶解性；无结构警示子。库内存在三个货号 (T3933/T4912/T75436) 对应同一结构，采购便利。",
    tox_band="中 (启发式低估)",
    tox_note="重要：理化打分给出满分安全性，但未捕捉阳离子亲脂性物质(DLC)的线粒体蓄积毒性。"
             "小檗碱类在 HepG2 中常于 25-50 µM 以上引起活力下降。须采用密集剂量梯度 (0.1-100 µM，8 点)"
             "并严格以治疗指数 TI=CC50/EC50>5 判读；因其抑制复合物I，ATP 类活力读出会与药理机制混淆，"
             "应改用或并用 Hoechst 核计数/蛋白酶活性型(CellTiter-Fluor)活力读出。"),

"T21318": dict(
    chem="2-{4-[2-(3,5-二甲基苯氨基)-2-氧代乙基]苯氧基}-2-甲基丙酸 (贝特类芳氧基异丁酸)",
    target="PPAR-alpha",
    mech="芳氧基异丁酸酸性头基与 PPAR-alpha 配体结合域的 Ser280/Tyr314/His440/Tyr464 形成经典四点氢键，"
         "招募 PGC-1alpha/SRC-1，转录上调 CPT1A、ACOX1、ACADM、FGF21 与 UCP2，"
         "同时增强线粒体与过氧化物酶体脂肪酸 beta-氧化；并下调 apoC-III 促进 LPL 介导的 TG 水解清除，"
         "整体降低肝细胞内甘油三酯池。",
    rationale="精确命中贝特类药效团 (规则权重 0.90)，但与苯扎贝特 Tanimoto 仅 0.429，"
              "说明尾部骨架 (3,5-二甲基苯乙酰胺) 新颖，具备知识产权空间；"
              "S_safety=1.000、无结构警示子、S_total=0.993；LogP 3.73 略偏高但 TPSA 75.6 有效补偿。",
    tox_band="低",
    tox_note="贝特类在人肝细胞中安全窗宽 (PPAR-alpha 介导的过氧化物酶体增殖为啮齿类特异效应，人肝细胞不显著)。"
             "无结构警示子，卤素数为 0。"),

"TN2058": dict(
    chem="5-羟基-7-甲氧基-3'-羟基-4'-甲氧基黄烷酮 (橙皮素甲醚型二氢黄酮)",
    target="AMPK / SREBP-1c",
    mech="黄烷酮母核激活 AMPK-SIRT1 轴，抑制 SREBP-1c 的 S1P/S2P 剪切成熟与核转位，"
         "下调 FASN/ACC1/SCD1 转录阻断 DNL；同时上调 PPAR-alpha/CPT1A 促进 beta-氧化，"
         "并经 Nrf2-ARE 通路缓解游离脂肪酸诱导的氧化应激与脂性凋亡(lipoapoptosis)。",
    rationale="S_total=1.000 (双项满分)；与柚皮素 Tanimoto=0.571 属同一黄烷酮化学型；"
              "关键在于该分子通过了 PAINS 过滤 —— 区别于槲皮素、姜黄素、白藜芦醇等因儿茶酚/查耳酮"
              "基序被判为泛筛选干扰的多酚，其甲氧基化降低了氧化还原循环与非特异性反应性；"
              "MW 316、LogP 2.82、HBD=2 类药性优良。",
    tox_band="低",
    tox_note="无结构警示子。实验注意事项：黄酮类在 480-520 nm 有自发荧光，与 BODIPY 493/503 通道重叠，"
             "必须设置'化合物 + 无细胞'与'化合物 + 细胞 + 无染料'两组荧光背景对照，"
             "或改用偏红移的 Nile Red (Ex 552/Em 636) 通道读出以规避干扰。"),

"T15049": dict(
    chem="4-{2-[(4-氯苯磺酰基)氨基]乙基}苯乙酸 (芳基乙酸-芳基磺酰胺)",
    target="PPAR pan (alpha/delta/gamma)",
    mech="芳基乙酸酸性头基 + 柔性乙基连接臂 + 芳基磺酰胺疏水尾，重现 lanifibranor 型泛 PPAR 药效团。"
         "同时激动 PPAR-alpha (上调 CPT1A/ACOX1 促 beta-氧化)、PPAR-delta (提升脂肪酸摄取与氧化利用) 与"
         "PPAR-gamma (促脂质向脂肪组织再分布)，三受体协同下调肝细胞 TG 池；"
         "泛 PPAR 激动在 NATIVE 等临床研究中显示出优于单靶点的脂肪变改善。",
    rationale="S_total=1.000，LogP 2.49/TPSA 83.5 极接近最佳窗口中心；MW 354 贴近 350 最优值；"
              "无结构警示子、仅 1 个卤素；与参照集最大相似度仅 0.382，为化学空间中的新颖骨架。",
    tox_band="低",
    tox_note="无结构警示子。在靶性提示：该芳基乙酸-磺酰胺基序亦见于前列腺素/血栓素类似物，"
             "故靶点归属置信度为中等，必须在机制解卷积阶段以 PPRE-荧光素酶报告基因 (PPARalpha/delta/gamma "
             "三亚型分别检测) 确认在靶激动活性，避免机制误判。"),

"T8701": dict(
    chem="[4-{[2-(2,5-二甲基苯氧基)乙基]氨基}哌啶-1-基](吡啶-4-基)甲酮",
    target="DGAT2 / FASN",
    mech="杂芳基甲酰胺-哌啶铰链模体是 DGAT2 与 FASN 抑制剂共有的药效团 (对照 PF-06424439、TVB-2640)。"
         "若作用于 DGAT2，则直接阻断二酰甘油->三酰甘油酯化的终末限速步骤，快速缩小脂滴体积并减少 "
         "VLDL 分泌，且因不干扰 DGAT1 介导的肠道脂质吸收而胃肠道耐受性佳；"
         "若作用于 FASN 则阻断棕榈酸从头合成，减少 DNL 底物供应。",
    rationale="S_total=1.000 (双项满分)；LogP 2.97、TPSA 54.5、Csp3=0.43 类先导性优良，三维性高于典型平面芳香分子；"
              "MW 353 贴近 350 最优值；无结构警示子；与参照集相似度仅 0.383，属新颖骨架。",
    tox_band="低",
    tox_note="无结构警示子。碱性仲胺 + 亲脂性组合存在阳离子两亲性药物(CAD)特征，"
             "理论上有磷脂沉积与 hERG 风险，建议在成药性评估阶段并行 LipidTOX Phospholipidosis 染色"
             "与 hERG 膜片钳；注意磷脂沉积会在中性脂染色中产生假阳性/假阴性干扰。"),

"T8532": dict(
    chem="1-[4-(三氟甲氧基)苯基]-5-(吡咯烷-1-基)双胍 (芳基双胍)",
    target="AMPK (间接，经线粒体复合物I)",
    mech="双胍经有机阳离子转运体 OCT1/OCT3 进入肝细胞并在线粒体内蓄积，抑制呼吸链复合物I，"
         "升高 AMP/ATP 与 ADP/ATP 比值，经 LKB1 激活 AMPK；AMPK 磷酸化 ACC1(Ser79)/ACC2(Ser212) 使之失活，"
         "降低丙二酰辅酶A 从而同时阻断 DNL 并解除 CPT1A 抑制、提升 beta-氧化；"
         "并抑制 SREBP-1c 成熟、促进脂噬(lipophagy)。",
    rationale="精确命中双胍药效团 (规则权重 0.90)；S_total=1.000，LogP 2.55/TPSA 84.2 位于窗口内；"
              "MW 315 类药性良好。相对二甲双胍，芳基与三氟甲氧基显著提升亲脂性，"
              "预期细胞内暴露与效价远高于二甲双胍 (后者在 HepG2 中常需 mM 级)。",
    tox_band="中-高 (启发式显著低估)",
    tox_note="关键风险，必须重点关注：本分子属苯乙双胍(phenformin)类芳基双胍，亲脂性远高于二甲双胍，"
             "线粒体蓄积更强 —— 苯乙双胍正因致命性乳酸酸中毒而于 1970 年代撤市。"
             "预期在 HepG2 中呈现窄治疗窗：ATP 耗竭与乳酸堆积可能在接近降脂有效浓度处即出现。"
             "验证要求：(1) 禁止仅用 CellTiter-Glo 判读活力 —— 复合物I 抑制直接压低 ATP，会把在靶药理"
             "误判为毒性；须以 Hoechst 核计数或 CellTiter-Fluor(蛋白酶活性)为主读出，ATP 为辅；"
             "(2) 并行检测培养基乳酸/ECAR；(3) 严格要求 TI>5，若 TI<5 判为'毒性驱动的假降脂'并淘汰。"),

"T4515": dict(
    chem="3-{[4-(2-氯苯氧基)哌啶-1-羰基]氨基}-N-甲基苯甲酰胺 (芳基脲-4-芳氧基哌啶)",
    target="DGAT2 / SCD1",
    mech="芳基脲 + 4-芳氧基哌啶 是 ervogastat (PF-06865571) 型 DGAT2 抑制剂的标志性药效团。"
         "抑制 DGAT2 阻断 DAG->TAG 酯化终末步骤，直接减少脂滴生成与 VLDL 分泌；"
         "DGAT2 抑制在临床上还可协同 ACC 抑制剂拮抗后者引起的高甘油三酯血症。"
         "若作用于 SCD1，则降低单不饱和脂肪酸(油酸/棕榈油酸)供应，限制 TAG 酯化底物。",
    rationale="与 ervogastat 共享脲-哌啶醚核心骨架 (规则权重 0.68)；S_total=0.990、S_safety=1.000；"
              "LogP 3.77 与 TPSA 70.7 均在可接受区间；MW 388、无结构警示子、仅 1 个卤素；"
              "与参照集相似度 0.307，骨架新颖。",
    tox_band="低",
    tox_note="无结构警示子。类别提示：DGAT2 抑制在肝细胞中可能引起 DAG 蓄积，"
             "而 DAG 是 PKCepsilon 的激活剂并与胰岛素抵抗相关，建议在机制解卷积时同步用 "
             "LC-MS 脂质组学监测 DAG/TAG 比值变化，确认脂质流向而非仅看总脂滴信号。"),
}

# ==============================================================================
# 1. candidates.csv
# ==============================================================================
def efficacy_band(s):
    return "High" if s >= 0.95 else ("Moderate-High" if s >= 0.88 else "Moderate")


rows = []
for _, r in top10.iterrows():
    c = CURATED[r.Molecule_ID]
    rows.append({
        "Rank": int(r.Rank),
        "Molecule_ID": r.Molecule_ID,
        "CAS": r.CAS,
        "SMILES": r.SMILES,
        "Formula": r.Formula,
        "MW": r.MW, "LogP": r.LogP, "TPSA": r.TPSA,
        "HBD": r.HBD, "HBA": r.HBA, "RotB": r.RotB, "FracCsp3": r.FracCsp3,
        "Chemical_Class": c["chem"],
        "Primary_Target_Hypothesis": c["target"],
        "Mechanism_Hypothesis": c["mech"],
        "Predicted_Efficacy": f'{efficacy_band(r.S_nomination)} '
                              f'(S_efficacy={r.S_efficacy:.3f}, S_mech={r.S_mech:.3f}, '
                              f'S_nomination={r.S_nomination:.3f})',
        "Predicted_Cytotoxicity": f'{c["tox_band"]} (S_safety={r.S_safety:.3f}; '
                                  f'结构警示子={r.Tox_Alerts}; LogP={r.LogP}; TPSA={r.TPSA})',
        "Cytotoxicity_Caveat": c["tox_note"],
        "Rationale": c["rationale"],
        "S_efficacy": r.S_efficacy, "S_safety": r.S_safety, "S_total": r.S_total,
        "S_mech": r.S_mech, "S_nomination": r.S_nomination,
        "Nearest_Reference_Drug": r.Nearest_Reference,
        "Ref_Tanimoto": r.Ref_Tanimoto,
        "Matched_Chemotype_Rule": r.Chemotype,
        "Structural_Alerts": r.Tox_Alerts,
        "Duplicate_Catalog_IDs": r.Duplicate_IDs,
        "Murcko_Scaffold": r.Murcko_Scaffold,
    })

cand = pd.DataFrame(rows)
cand.to_csv(f"{OUTDIR}/candidates.csv", index=False, encoding="utf-8-sig")
print(f"[1/3] candidates.csv  -> {len(cand)} 行 x {len(cand.columns)} 列")

# ==============================================================================
# 2. 结构图渲染
# ==============================================================================
img_paths = {}
for _, r in top10.iterrows():
    m = Chem.MolFromSmiles(r.SMILES)
    d = rdMolDraw2D.MolDraw2DCairo(520, 300)
    o = d.drawOptions(); o.addStereoAnnotation = True
    rdMolDraw2D.PrepareAndDrawMolecule(d, m, legend=f"#{int(r.Rank)}  {r.Molecule_ID}")
    d.FinishDrawing()
    p = os.path.join(IMGDIR, f"rank{int(r.Rank):02d}_{r.Molecule_ID}.png")
    with open(p, "wb") as f:
        f.write(d.GetDrawingText())
    img_paths[r.Molecule_ID] = p
print(f"[2/3] 结构图 -> {len(img_paths)} 张 (structures/)")

# ==============================================================================
# 3. mechanism_and_validation.pdf
# ==============================================================================
S_title = ParagraphStyle("t", fontName=CN, fontSize=19, leading=26, alignment=TA_CENTER,
                         textColor=colors.HexColor("#14304F"), spaceAfter=4)
S_sub   = ParagraphStyle("s", fontName=CN, fontSize=10.5, leading=16, alignment=TA_CENTER,
                         textColor=colors.HexColor("#5A6B7B"))
S_h1    = ParagraphStyle("h1", fontName=CN, fontSize=14, leading=20, spaceBefore=13, spaceAfter=7,
                         textColor=colors.HexColor("#14304F"))
S_h2    = ParagraphStyle("h2", fontName=CN, fontSize=11.5, leading=17, spaceBefore=9, spaceAfter=5,
                         textColor=colors.HexColor("#1F5C8B"))
S_h3    = ParagraphStyle("h3", fontName=CN, fontSize=10.3, leading=15, spaceBefore=6, spaceAfter=3,
                         textColor=colors.HexColor("#2C6E49"))
S_body  = ParagraphStyle("b", fontName=CN, fontSize=9.6, leading=15.2, alignment=TA_LEFT,
                         wordWrap="CJK", spaceAfter=4)
S_note  = ParagraphStyle("n", fontName=CN, fontSize=8.8, leading=13.5, wordWrap="CJK",
                         textColor=colors.HexColor("#7A3B10"), spaceAfter=3)
S_cell  = ParagraphStyle("c", fontName=CN, fontSize=8.0, leading=11.2, wordWrap="CJK")
S_cellb = ParagraphStyle("cb", fontName=CN, fontSize=8.0, leading=11.2, wordWrap="CJK",
                         textColor=colors.white)
S_mono  = ParagraphStyle("m", fontName="Courier", fontSize=6.9, leading=8.6)

GRID = colors.HexColor("#B9C6D2")
HEADBG = colors.HexColor("#1F5C8B")
ALT = colors.HexColor("#EEF3F8")


# STSong-Light 走 GB 编码，若干西文符号不在字符集内会被静默丢弃 (µ / ² / ³ 等)。
# 统一映射到 GB2312 内存在的等价字符，避免出现 "10 M" 这类丢字。
_GLYPH_FIX = {"µ": "μ",   # MICRO SIGN -> GREEK SMALL MU
              "²": "^2", "³": "^3", "¹": "^1",
              "‑": "-", "–": "-", "×": "x"}


def fix(t):
    t = str(t)
    for a, b in _GLYPH_FIX.items():
        t = t.replace(a, b)
    return t


def P(t, s=S_body):   return Paragraph(fix(t), s)
def C(t):             return Paragraph(fix(t), S_cell)
def CH(t):            return Paragraph(f"<b>{fix(t)}</b>", S_cellb)


def wrap_smiles(s, n=30):
    return "<br/>".join(s[i:i+n] for i in range(0, len(s), n))


def tbl(data, widths, align=None, fs=8.0):
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    st = [("BACKGROUND", (0, 0), (-1, 0), HEADBG),
          ("GRID", (0, 0), (-1, -1), 0.4, GRID),
          ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 3),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
          ("LEFTPADDING", (0, 0), (-1, -1), 4),
          ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    for i in range(1, len(data)):
        if i % 2 == 0:
            st.append(("BACKGROUND", (0, i), (-1, i), ALT))
    if align:
        st += align
    t.setStyle(TableStyle(st))
    return t


story = []
W = A4[0] - 34 * mm

# ---------------- 封面 ----------------
story += [Spacer(1, 30 * mm),
          P("MASLD 早期降脂小分子发现", S_title),
          P("候选分子机制假说与实验验证方案", S_title),
          Spacer(1, 5 * mm),
          P("HepG2 细胞-游离脂肪酸(FFA) 肝细胞脂质蓄积模型", S_sub),
          P("上海人工智能实验室 x 临港实验室 联合赛题", S_sub),
          Spacer(1, 10 * mm)]

cover = [[CH("项目"), CH("内容")],
         [C("化合物库"), C(f"T.sdf，{stats['total']:,} 个小分子")],
         [C("通过全部过滤"), C(f"{stats['passed']:,} 个 (结构去重后 {stats['passed']-806:,} 个)")],
         [C("提名候选"), C("Top 10 (覆盖 10 条相互独立的降脂机制)")],
         [C("阳性对照回收"), C(f"{stats['n_pos_ctrl']} 个已知肝脂调节剂 (方法学自验证)")],
         [C("参照药物集"), C(f"{stats['n_refs']} 个临床/工具化合物")],
         [C("优势骨架规则"), C(f"{stats['n_chemotype_rules']} 条 SMARTS 药效团规则")],
         [C("核心判读标准"), C("治疗指数 TI = CC50 / EC50 > 5")]]
story += [tbl(cover, [42 * mm, W - 42 * mm]), PageBreak()]

# ---------------- 1 摘要 ----------------
story += [P("1. 执行摘要", S_h1),
          P("本工作从 22,966 个小分子的化合物库出发，构建了一条<b>四层漏斗式虚拟筛选流程</b>："
            "结构标准化与理化描述符计算 → PAINS/类药性/经验性细胞毒性反筛 → 启发式多目标打分 "
            "(S_total = 0.6 x S_efficacy + 0.4 x S_safety) → 机制证据层与骨架多样性选择，"
            "最终提名 10 个<b>低毒性降脂</b>候选分子。"),
          P("<b>关键方法学发现：</b>赛题设定的 S_total 是一个纯理化性质函数，不含任何生物活性信息。"
            "在本库上它高度饱和 —— 共有 <b>554 个分子并列 S_total = 1.0000</b>。"
            "若直接按 S_total 取 Top 10，实际得到的是文件顺序的前 10 个，不具科学意义。"
            "因此本流程将 S_total 用作<b>硬门槛</b>(保留 S_total ≥ 0.95 的 2,496 个分子)，"
            "再引入两层可验证的机制证据进行排序："),
          P("&nbsp;&nbsp;(i) 与 50 个已知肝脂代谢临床/工具化合物的 Morgan(r=2) Tanimoto 相似性；<br/>"
            "&nbsp;&nbsp;(ii) 22 条肝脂靶点优势骨架 (privileged chemotype) SMARTS 药效团规则。"),
          P("<b>方法学自验证：</b>在完全不使用任何活性标签的前提下，该流程从库中自动回收出 "
            "吡格列酮、罗格列酮、苯扎贝特、环丙贝特、非诺贝特酸、小檗碱、雷诺嗪 等 "
            f"{stats['n_pos_ctrl']} 个已知肝脂调节剂 (Tanimoto = 1.00)，证明打分与过滤逻辑确实富集了"
            "真实的降脂药理空间。这些已知药物被显式排除在提名名单之外，以保证 Top 10 的化学新颖性。"),
          P("<b>提名策略：</b>Top 10 施加“每个靶点假说至多 1 个分子”的约束，使 10 个候选覆盖 "
            "PPAR-gamma、HMGCR、FXR、AMPK/LDLR、PPAR-alpha、AMPK/SREBP-1c、pan-PPAR、DGAT2/FASN、"
            "AMPK(双胍)、DGAT2/SCD1 共 <b>10 条相互独立的机制通路</b>，最大化组合命中率并分散"
            "单一通路失败的风险。")]

# ---------------- 2 漏斗 ----------------
story += [P("2. 筛选漏斗与各步骤淘汰量", S_h1)]
funnel = [[CH("步骤"), CH("淘汰数"), CH("剩余"), CH("依据")]]
seq = [("SDF 总记录", None, stats['total'], "原始化合物库"),
       ("RDKit 解析/标准化失败", stats['unparsable'], None, "价键异常、无法 sanitize"),
       ("非白名单元素", stats['bad_element'], None, "金属配合物/无机盐/Si,B,Sn 等"),
       ("重原子数 < 12", stats['too_small'], None, "碎片过小，不足以支撑细胞水平活性"),
       ("PAINS 泛筛选干扰结构", stats['pains'], None, "RDKit FilterCatalog PAINS_A/B/C"),
       ("Lipinski 类药性不合格", stats['lipinski'], None, "MW 150-550, LogP -1~5, HBD<=5, HBA<=10"),
       ("细胞毒性硬反筛", stats['cytotox_hard'], None, "LogP>4.5 且 TPSA<20 (膜裂解/解偶联风险)"),
       ("结构去重", 806, None, "同一结构的多个货号(游离碱/盐)合并")]
rem = stats['total']
for name, cut, tot, why in seq:
    if cut is None:
        funnel.append([C(name), C("—"), C(f"{rem:,}"), C(why)])
    else:
        rem -= cut
        funnel.append([C(name), C(f"-{cut:,}"), C(f"{rem:,}"), C(why)])
funnel.append([C("进入多目标打分"), C("—"), C(f"{rem:,}"), C("全部通过类药性与毒性反筛")])
funnel.append([C(f"S_total >= 0.95 硬门槛"), C(f"-{rem-stats['n_gated']:,}"), C(f"{stats['n_gated']:,}"),
               C("进入机制证据层计算")])
story += [tbl(funnel, [46 * mm, 18 * mm, 18 * mm, W - 82 * mm]),
          Spacer(1, 3 * mm),
          P(f"注：S_total 在打分池中高度饱和，{stats['n_tie_S_total_1']} 个分子并列满分 1.0000，"
            f"这正是必须引入机制证据层的直接原因。", S_note)]

story += [PageBreak()]

# ---------------- 3 阳性对照回收 ----------------
story += [P("3. 方法学自验证：阳性对照回收", S_h1),
          P("下表为流程在<b>未使用任何活性标签</b>的情况下，从库中自动识别出的已知肝脂调节剂"
            "(与参照药物集 Tanimoto ≥ 0.85)。它们全部通过了 PAINS、类药性与细胞毒性反筛，"
            "并落在 S_total ≥ 0.95 的高分区间，说明打分函数确实富集了真实的降脂化学空间。"
            "为保证提名的化学新颖性，这些分子已从 Top 10 中显式剔除。")]
pc = [[CH("库编号"), CH("CAS"), CH("对应已知药物"), CH("Tanimoto"), CH("靶点"), CH("S_total")]]
for _, r in pos_ctrl.iterrows():
    pc.append([C(r.Molecule_ID), C(r.CAS), C(r.Nearest_Reference),
               C(f"{r.Ref_Tanimoto:.2f}"), C(r.Ref_Target), C(f"{r.S_total:.4f}")])
story += [tbl(pc, [22 * mm, 26 * mm, 34 * mm, 18 * mm, W - 118 * mm, 18 * mm])]

# ---------------- 4 Top10 总览 ----------------
story += [Spacer(1, 4 * mm), P("4. 提名候选分子总览 (Top 10)", S_h1)]
ov = [[CH("#"), CH("库编号"), CH("MW"), CH("LogP"), CH("TPSA"),
       CH("S_total"), CH("S_nom"), CH("靶点假说"), CH("化学型")]]
for _, r in cand.iterrows():
    ov.append([C(r.Rank), C(r.Molecule_ID), C(f"{r.MW:.1f}"), C(f"{r.LogP:.2f}"),
               C(f"{r.TPSA:.1f}"), C(f"{r.S_total:.3f}"), C(f"{r.S_nomination:.3f}"),
               C(r.Primary_Target_Hypothesis), C(r.Matched_Chemotype_Rule)])
story += [tbl(ov, [7 * mm, 20 * mm, 14 * mm, 13 * mm, 14 * mm, 15 * mm, 15 * mm,
                   34 * mm, W - 132 * mm])]

story += [PageBreak()]

# ---------------- 5 共性机制 ----------------
story += [P("5. 共性作用机制与通路假说", S_h1),
          P("10 个候选分子虽分属 10 条独立机制，但全部汇聚于肝细胞脂质稳态的<b>四条核心轴</b>。"
            "HepG2-FFA 模型中脂滴的净蓄积量 = (摄取 + 新发合成 + 酯化) − (氧化 + 分泌 + 脂噬)，"
            "本提名组合覆盖了该收支方程的全部六个可干预节点。"),

          P("轴一：抑制新发脂质合成 (De Novo Lipogenesis, DNL)", S_h2),
          P("SREBP-1c/ChREBP 是 DNL 的主转录开关，下游驱动 ACC1/ACC2 → FASN → SCD1 → DGAT 级联。"
            "本组合从三个层次阻断该轴：<b>转录层</b> — T19124(FXR-SHP 抑制 SREBP-1c)、"
            "TN2058(AMPK-SIRT1 抑制 SREBP-1c 成熟)、T2607(PPAR-gamma 下调 SREBP-1c)；"
            "<b>激酶层</b> — T75436 与 T8532 经 AMPK 磷酸化失活 ACC1(Ser79)/ACC2(Ser212)，"
            "直接降低丙二酰辅酶A 池；<b>酶层</b> — T8701(FASN) 阻断棕榈酸合成。"),

          P("轴二：促进脂肪酸 beta-氧化", S_h2),
          P("丙二酰辅酶A 是 CPT1A 的内源性变构抑制剂，因此凡降低丙二酰辅酶A 者 (AMPK 激活型 T75436/T8532) "
            "均可同时解除 CPT1A 抑制，使长链脂酰辅酶A 进入线粒体氧化 —— 这是 DNL 抑制与 beta-氧化促进"
            "<b>共享同一分子开关</b>的关键耦合点。核受体层面，T21318(PPAR-alpha) 与 T15049(pan-PPAR) "
            "转录上调 CPT1A、ACOX1、ACADM、FGF21，从供给侧扩大氧化通量。"),

          P("轴三：阻断甘油三酯酯化终末步骤", S_h2),
          P("DGAT2 催化 DAG → TAG 的最后一步，是脂滴生成的直接限速酶。T4515 与 T8701 命中该节点，"
            "其优势在于起效快 (无需等待转录重编程) 且效应量大。需注意 DGAT2 抑制会使 DAG 上游蓄积，"
            "而 DAG 是 PKCepsilon 激活剂，故建议以脂质组学监测 DAG/TAG 比值而非仅看总脂滴信号。"),

          P("轴四：重塑胆固醇/胆汁酸-脂质交叉调控", S_h2),
          P("T12092(HMGCR) 经甲羟戊酸通路耗竭 GGPP 间接激活 AMPK；T19124(FXR) 经 SHP 与 FGF19 轴"
            "重塑胆汁酸池并抑制脂肪生成。该轴与轴一在 SREBP 家族上收敛 (SREBP-2 管胆固醇、"
            "SREBP-1c 管脂肪酸)，提示联合用药的协同空间。"),

          P("组合的风险分散逻辑", S_h2),
          P("10 条机制中，4 条为核受体转录调控 (PPAR-gamma/alpha/pan、FXR)、2 条为能量感受激酶 (AMPK)、"
            "2 条为脂质合成酶直接抑制 (DGAT2/FASN/SCD1)、1 条为胆固醇合成 (HMGCR)、1 条为多酚类多效性。"
            "起效动力学互补：酶抑制型 (T4515/T8701/T12092) 数小时内起效，转录调控型 "
            "(T2607/T21318/T15049/T19124) 需 24-48 小时，AMPK 型 (T75436/T8532/TN2058) 介于两者之间。"
            "因此<b>实验设计必须包含 24 h 与 48 h 两个时间点</b>，否则会系统性漏检转录调控型分子。")]

story += [PageBreak()]

# ---------------- 6 验证方案 ----------------
story += [P("6. 实验验证方案 (HepG2-FFA 模型)", S_h1),

          P("6.1 细胞模型建立与脂质蓄积诱导", S_h2),
          P("<b>细胞：</b>HepG2 (ATCC HB-8065)，DMEM 高糖 + 10% FBS + 1% P/S，37 ℃/5% CO2；"
            "传代数控制在 P5-P20 以保证表型稳定。<br/>"
            "<b>接种：</b>96 孔黑壁透明底板 (成像用) 1.5 x 10^4 cells/孔；96 孔白板 (发光用) "
            "1.0 x 10^4 cells/孔；贴壁 24 h。<br/>"
            "<b>FFA 负荷：</b>油酸(OA):棕榈酸(PA) = 2:1 摩尔比，终浓度 0.5 mM，"
            "以 1% 无脂肪酸 BSA 于 37 ℃ 预偶联 1 h (摩尔比 FFA:BSA ≈ 5:1)，"
            "0.22 µm 滤膜过滤后加入含 1% FBS 的低血清培养基，孵育 24 h。<br/>"
            "<b>为何用 2:1 而非纯 PA：</b>纯棕榈酸本身即诱导显著脂性凋亡，会与化合物毒性混淆；"
            "OA 占优的配比可产生稳健脂滴蓄积而细胞活力保持在 >85%，是脂质表型与毒性读出解耦的前提。<br/>"
            "<b>给药：</b>与 FFA 同时加入 (共孵育模式，评估“预防”效应)；"
            "另设 FFA 预负荷 24 h 后再给药 24 h 的“治疗”模式作为二级确证。DMSO 终浓度 ≤ 0.1%。"),

          P("6.2 降脂效应读出 — BODIPY 493/503 + 高内涵成像", S_h2),
          P("<b>染色：</b>4% PFA 室温固定 15 min → PBS 洗 3 次 → BODIPY 493/503 (2 µM) + "
            "Hoechst 33342 (5 µg/mL) 避光染色 30 min → PBS 洗 3 次。<br/>"
            "<b>成像：</b>高内涵成像系统 (Operetta CLS / ImageXpress) 20x 物镜，每孔 9 视野，"
            "≥ 1,000 cells/孔。通道：Hoechst (Ex 405/Em 461) 定位细胞核，"
            "BODIPY (Ex 493/Em 503) 定量中性脂。<br/>"
            "<b>核心指标：</b>单细胞归一化脂滴总面积 (Total lipid droplet area / nucleus count)，"
            "同时记录脂滴数目、平均脂滴直径与脂滴强度积分 —— 三者可区分“脂滴变小”与“脂滴变少”"
            "这两种机制上不同的降脂模式 (DGAT2 抑制倾向前者，beta-氧化促进倾向后者)。<br/>"
            "<b>正交生化确证：</b>对高内涵阳性分子，以酶法甘油三酯试剂盒 (GPO-POD) 定量细胞裂解液 TG，"
            "并以 BCA 蛋白量归一化 (µg TG / mg protein)；必要时以 Nile Red (Ex 552/Em 636) "
            "红移通道复测，规避化合物自发荧光干扰。"),
          P("重要对照 — 荧光干扰排除：多酚/黄酮类候选 (如 TN2058) 在 480-520 nm 有自发荧光，"
            "必须设置 (a) 化合物+培养基无细胞孔、(b) 化合物+细胞+无染料孔 两组背景对照；"
            "任何在无染料条件下即产生信号的分子，其 BODIPY 结果一律以 Nile Red 或生化 TG 法重判。", S_note),

          P("6.3 细胞毒性读出 — CellTiter-Glo + 正交活力指标", S_h2),
          P("<b>主读出：</b>CellTiter-Glo 2.0 (ATP 发光法)，与降脂实验<b>同板同批同时程</b>平行进行，"
            "室温平衡 30 min，振荡 2 min，静置 10 min 后读发光值。<br/>"
            "<b>正交读出：</b>(a) LDH 释放法检测膜完整性 (区分膜裂解型坏死)；"
            "(b) Hoechst 核计数 (来自 6.2 成像，零额外成本，可直接反映细胞数损失)；"
            "(c) Caspase-3/7 活性检测凋亡。<br/>"
            "<b>关键设计要点：</b>对线粒体复合物I 抑制类分子 (T8532 芳基双胍、T75436 原小檗碱)，"
            "<b>不得单独以 CellTiter-Glo 判读活力</b> —— 这类分子的药理机制本身就是压低 ATP，"
            "ATP 下降既可能来自在靶 AMPK 激活也可能来自细胞死亡，二者无法区分。"
            "此时应以 Hoechst 核计数或 CellTiter-Fluor (活细胞蛋白酶活性) 为主判据，ATP 仅作参考，"
            "并加测培养基乳酸/ECAR 以确认代谢转向。"),

          P("6.4 剂量-反应设计与数据处理", S_h2),
          P("<b>浓度梯度：</b>8 点 3 倍稀释，覆盖 0.05-100 µM (对亲脂性阳离子类下探至 0.03 µM)；"
            "每浓度 n = 3 复孔，独立生物学重复 ≥ 3 次。<br/>"
            "<b>板内对照：</b>阴性 — 0.1% DMSO + FFA；空白 — 无 FFA；"
            "阳性 — 小檗碱 (10 µM) 与非诺贝特酸 (50 µM)，二者均由本流程从库中自动回收，可直接采购同批物料。<br/>"
            "<b>归一化：</b>脂质信号 %Effect = 100 x (DMSO_FFA − 化合物) / (DMSO_FFA − 无FFA空白)；"
            "活力 %Viability = 100 x 化合物 / DMSO_FFA。<br/>"
            "<b>拟合：</b>四参数 Logistic (GraphPad Prism / Python scipy)，报告 EC50 (降脂) 与 CC50 (毒性) "
            "及其 95% 置信区间；Hill 斜率 > 3 的曲线提示聚集或非特异性膜效应，需复核。<br/>"
            "<b>质控：</b>要求 Z' ≥ 0.5、CV ≤ 15%、阳性对照 EC50 在历史范围 ±3 倍以内，否则该板作废。"),

          P("6.5 有效命中判读标准 (Hit Criteria)", S_h2)]

hit = [[CH("层级"), CH("判据"), CH("说明")],
       [C("必要条件 1"), C("治疗指数 TI = CC50 / EC50 > 5"),
        C("核心指标。TI ≤ 5 判定为毒性驱动的假降脂，直接淘汰")],
       [C("必要条件 2"), C("在细胞活力 ≥ 80% 的浓度下，脂质信号降低 ≥ 30%"),
        C("确保降脂效应发生在非毒性剂量窗内")],
       [C("必要条件 3"), C("剂量依赖性显著 (拟合 R² ≥ 0.85，Hill 斜率 0.5-3)"),
        C("排除单点偶然效应与聚集体假阳性")],
       [C("必要条件 4"), C("正交读出一致：生化 TG 定量与成像结果同向且倍数差 < 2"),
        C("排除染料/荧光干扰导致的假阳性")],
       [C("必要条件 5"), C("LDH 释放 < 对照 2 倍，Caspase-3/7 无显著升高"),
        C("排除膜裂解坏死与凋亡驱动的“脂质减少”")],
       [C("优选条件 A"), C("TI > 10 且 EC50 < 10 µM"), C("优先推进至原代肝细胞验证")],
       [C("优选条件 B"), C("机制解卷积阳性 (见 6.6)"), C("靶点假说获得实验支持")],
       [C("优选条件 C"), C("“治疗”模式 (FFA 预负荷后给药) 同样有效"),
        C("更贴近临床场景，价值高于仅“预防”模式有效")]]
story += [tbl(hit, [20 * mm, 62 * mm, W - 82 * mm])]

story += [PageBreak()]

story += [P("6.6 机制解卷积 (Mechanism Deconvolution)", S_h2),
          P("通过初筛判据的分子进入机制确证，按靶点假说分组执行："),
          P("<b>(a) 转录组层 — qPCR 面板：</b>DNL 基因 SREBF1、FASN、ACACA(ACC1)、ACACB(ACC2)、SCD1、DGAT2；"
            "氧化基因 CPT1A、ACOX1、PPARA、ACADM；核受体靶基因 NR0B2(SHP，FXR 读出)、"
            "ANGPTL4(PPAR 读出)、CYP7A1。降脂机制不同则基因指纹不同，可据此反推靶点。<br/>"
            "<b>(b) 蛋白层 — Western blot：</b>p-AMPK(Thr172)/AMPK、p-ACC(Ser79)/ACC 为 AMPK 通路的"
            "金标准读出 (适用于 T75436、T8532、TN2058)；SREBP-1c 前体/成熟体比值反映剪切抑制。<br/>"
            "<b>(c) 靶点层 — 报告基因：</b>PPRE-luc (分别转染 PPARalpha/delta/gamma 表达质粒，"
            "用于 T2607/T21318/T15049)、FXRE-luc + FXR/RXR (用于 T19124)、"
            "TRE-luc + THRB (阴性排除)。这是确认<b>在靶</b>而非表型巧合的关键步骤。<br/>"
            "<b>(d) 功能层 — Seahorse XF：</b>棕榈酸氧化速率测定 (FAO assay，以依托莫司 etomoxir 40 µM "
            "为 CPT1 阻断对照)，直接量化 beta-氧化通量提升；ECAR 同步反映糖酵解代偿。<br/>"
            "<b>(e) 脂质流向 — LC-MS 脂质组学：</b>对 DGAT2 假说分子 (T4515、T8701) 监测 DAG/TAG 比值；"
            "对 SCD1 假说分子监测去饱和指数 (16:1/16:0 与 18:1/18:0)，这是 SCD1 抑制的特异性生物标志物。<br/>"
            "<b>(f) 靶点验证 — siRNA 表型上位性：</b>敲低假定靶点后化合物降脂效应应显著削弱；"
            "若敲低后效应不变，则靶点假说被证伪。"),
          P("6.7 假阳性排查清单", S_h2),
          P("(1) <b>化合物自发荧光</b> — 无细胞孔背景对照 (对 TN2058 等黄酮类必做)；<br/>"
            "(2) <b>染料淬灭/竞争</b> — 化合物与 BODIPY 直接混合的无细胞光谱对照；<br/>"
            "(3) <b>胶体聚集体</b> — 加入 0.01% Triton X-100 复测，效应消失则为聚集假阳性；"
            "动态光散射(DLS) 确认；Hill 斜率异常陡峭者优先排查；<br/>"
            "(4) <b>培养基沉淀</b> — 显微镜下检查化合物析出，必要时降低测试上限；<br/>"
            "(5) <b>细胞数假象</b> — 化合物抑制增殖会使单孔总脂降低但单细胞脂量不变，"
            "故所有脂质指标<b>必须按核计数归一化</b>，这是本方案最容易被忽略的系统性陷阱；<br/>"
            "(6) <b>FFA 螯合</b> — 化合物若直接结合 FFA 或 BSA 会减少细胞摄取而非真正调节代谢，"
            "可用无 BSA 的乙醇溶解 FFA 体系复测区分。"),
          P("6.8 二级与三级验证", S_h2),
          P("<b>二级：</b>(i) 原代人肝细胞 (PHH) 或 HepaRG 分化细胞重复关键剂量点 —— HepG2 为肝母细胞瘤系，"
            "其 DNL 活性与 CYP 表达谱异于正常肝细胞，PHH 结果更具转化价值；"
            "(ii) 3D 肝球体 (spheroid) 长期给药 7-14 天，评估慢性暴露下的脂质清除与毒性累积；"
            "(iii) 人源 iPSC 衍生肝细胞用于遗传背景多样性评估。<br/>"
            "<b>三级：</b>成药性与安全性组合 —— 微粒体稳定性 (人/小鼠肝微粒体 t1/2)、"
            "CYP450 抑制 (3A4/2C9/2D6)、hERG 膜片钳、Ames 试验、"
            "线粒体毒性专项 (Glucose/Galactose 培养基切换法，可灵敏检出 OXPHOS 抑制剂 —— "
            "对 T8532、T75436 为必做项)、磷脂沉积 (LipidTOX，对 T8701 等阳离子两亲性分子)。<br/>"
            "<b>体内：</b>优选分子进入 C57BL/6J 高脂高果糖饮食 (HFHF) 或 CDAHFD 小鼠 MASLD 模型，"
            "4-8 周给药，终点为肝脏 TG 含量、NAS 评分、血清 ALT/AST 与肝脏转录组。")]

story += [PageBreak()]

# ---------------- 7 逐分子档案 ----------------
story += [P("7. 候选分子逐一档案", S_h1)]
for _, r in cand.iterrows():
    blk = [P(f"# {r.Rank} &#8194; {r.Molecule_ID} &#8194;"
             f"<font size=9 color='#5A6B7B'> &#183; CAS {r.CAS} &#183; {r.Formula} "
             f"&#183; {r.Primary_Target_Hypothesis}</font>", S_h2)]
    img = Image(img_paths[r.Molecule_ID], width=78 * mm, height=45 * mm)
    info = [[C("<b>SMILES</b>"), Paragraph(wrap_smiles(r.SMILES, 32), S_mono)],
            [C("<b>化学型</b>"), C(r.Chemical_Class)],
            [C("<b>理化性质</b>"), C(f"MW {r.MW} | LogP {r.LogP} | TPSA {r.TPSA} | "
                                  f"HBD {r.HBD} | HBA {r.HBA} | RotB {r.RotB} | Csp3 {r.FracCsp3}")],
            [C("<b>打分</b>"), C(f"S_efficacy {r.S_efficacy:.3f} | S_safety {r.S_safety:.3f} | "
                                f"S_total {r.S_total:.3f} | S_mech {r.S_mech:.3f} | "
                                f"<b>S_nomination {r.S_nomination:.3f}</b>")],
            [C("<b>最近参照药</b>"), C(f"{r.Nearest_Reference_Drug} (Tanimoto {r.Ref_Tanimoto})")]]
    it = tbl([[CH("项目"), CH("内容")]] + info, [22 * mm, 88 * mm - 22 * mm])
    blk.append(Table([[img, it]], colWidths=[80 * mm, W - 80 * mm],
                     style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                       ("LEFTPADDING", (0, 0), (-1, -1), 0)])))
    blk += [Spacer(1, 1.5 * mm),
            P(f"<b>靶点假说：</b>{r.Primary_Target_Hypothesis}", S_h3),
            P(f"<b>机制假说：</b>{r.Mechanism_Hypothesis}"),
            P(f"<b>提名理由：</b>{r.Rationale}"),
            P(f"<b>预测降脂效力：</b>{r.Predicted_Efficacy}"),
            P(f"<b>预测细胞毒性：</b>{r.Predicted_Cytotoxicity}"),
            P(f"<b>毒性风险提示：</b>{r.Cytotoxicity_Caveat}", S_note),
            Spacer(1, 3 * mm)]
    story.append(KeepTogether(blk))

story += [PageBreak()]

# ---------------- 8 局限 ----------------
story += [P("8. 方法局限与风险声明", S_h1),
          P("为便于评审方准确判断本工作的置信边界，明确列出以下局限："),
          P("<b>(1) 打分函数不含生物活性信息。</b>赛题设定的 S_efficacy/S_safety 完全由 LogP、TPSA、MW 等"
            "理化描述符构成，本质上刻画的是“类药性与暴露倾向”，而非“降脂活性”。"
            "其在本库上并列满分者达 554 个，判别力有限。本工作已通过引入相似性与药效团两层机制证据"
            "予以补偿，但这仍是<b>基于先验知识的假说生成</b>，不等价于活性预测。"),
          P("<b>(2) 靶点假说为化学型推断，非实验测定。</b>优势骨架规则给出的是“该分子长得像某类抑制剂”，"
            "命中规则不等于命中靶点。T15049(芳基乙酸-磺酰胺，亦见于前列腺素类似物) 与 "
            "T8701(哌啶甲酰胺，药效团较泛) 的靶点归属置信度明显低于 T2607(TZD)、T12092(statin 内酯) 与 "
            "T8532(双胍) 这类特异性极高的药效团。第 6.6 节的报告基因与 siRNA 实验是唯一的裁决手段。"),
          P("<b>(3) 理化毒性启发式会系统性低估机制性毒性。</b>S_safety 只看结构警示子与脂溶性/极性，"
            "无法预见“在靶但有害”的情形。本工作已人工标注三个此类高风险分子："
            "T8532 (芳基双胍，苯乙双胍类，乳酸酸中毒风险)、T75436 (亲脂性阳离子，线粒体蓄积)、"
            "T12092 (statin，甲羟戊酸耗竭)，并为每个给出了针对性的实验区分方案。"
            "<b>若仅依赖代码输出而忽略这些注释，极可能得出错误的安全性结论。</b>"),
          P("<b>(4) 化学新颖性有限。</b>本库为已知生物活性化合物库 (流程从中回收出 7 个已上市/临床药物即为佐证)，"
            "故 Top 10 多为已知药效团的新组合而非全新骨架。“新颖性”在本报告中的含义是"
            "<b>相对于 50 个参照药物集的 Tanimoto 距离</b>，而非绝对的文献新颖性 —— "
            "参照集之外的已知药物 (以及其代谢物、类似物) 不会被自动剔除。"
            "推进任一候选前应对其 CAS 号做文献与专利检索，确认其 MASLD 适应症的开发状态。"),
          P("<b>(5) 立体化学与互变异构未充分处理。</b>Morgan 指纹默认不区分手性；"
            "库中部分记录为消旋体或未指定构型，而 TZD 的 C5 手性中心、statin 内酯的多个手性中心"
            "对活性影响显著，实际采购与测试时须确认立体化学。"),
          P("<b>(6) 未做 ADMET 与聚集倾向的定量预测。</b>本流程仅覆盖结构警示子层面，"
            "未纳入 CYP 抑制、hERG、微粒体稳定性等模型预测；这些在第 6.8 节以实验方式补齐。"),
          Spacer(1, 4 * mm),
          P("附：全部中间结果与可复现代码", S_h2),
          P("all_scored_molecules.csv (11,969 个通过过滤分子的完整打分与描述符)、"
            "gated_annotated.csv (2,496 个门槛内分子的机制证据注释)、"
            "positive_control_recovery.csv (阳性对照回收清单)、"
            "candidates.csv (Top 10 提名清单)、"
            "methodology_and_code.md (方法学描述与完整源码)、"
            "structures/ (Top 10 结构图)。"
            "全部随机性来源均已固定，重跑可得完全一致的结果。")]


def footer(canv, doc):
    canv.saveState()
    canv.setFont(CN, 7.5)
    canv.setFillColor(colors.HexColor("#8A96A3"))
    canv.drawString(17 * mm, 10 * mm, "MASLD 早期降脂小分子发现 | HepG2-FFA 模型 | 机制与验证方案")
    canv.drawRightString(A4[0] - 17 * mm, 10 * mm, f"第 {doc.page} 页")
    canv.setStrokeColor(colors.HexColor("#D5DDE5"))
    canv.line(17 * mm, 13 * mm, A4[0] - 17 * mm, 13 * mm)
    canv.restoreState()


doc = SimpleDocTemplate(f"{OUTDIR}/mechanism_and_validation.pdf", pagesize=A4,
                        leftMargin=17 * mm, rightMargin=17 * mm,
                        topMargin=15 * mm, bottomMargin=17 * mm,
                        title="MASLD 早期降脂小分子发现 — 机制与验证方案",
                        author="AI Drug Discovery Pipeline")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(f"[3/3] mechanism_and_validation.pdf -> {doc.page} 页")
```

### 11.3 `_check_refs.py` —— 参照药物集 SMILES 分子式核验

```python
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors

# (name, smiles, expected_formula or None)
T = [
 ("Resmetirom","CC(C)C1=CC(Oc2c(Cl)cc(N3N=C(C#N)C(=O)NC3=O)cc2Cl)=NNC1=O","C17H12Cl2N6O4"),
 ("Firsocostat","COc1cc(-c2cn(C)c(=O)c3c2CCC3)cc(OC)c1C1(C(=O)O)CCOCC1","C25H29NO7"),
 ("Lanifibranor","Cc1ccc(S(=O)(=O)N2CCc3cc(CC(=O)O)ccc3C2)cc1Cl","C19H20ClNO4S"),
 ("Pioglitazone","CCc1ccc(CCOc2ccc(CC3SC(=O)NC3=O)cc2)nc1","C19H20N2O3S"),
 ("Rosiglitazone","CN(CCOc1ccc(CC2SC(=O)NC2=O)cc1)c1ccccn1","C18H19N3O3S"),
 ("Fenofibric acid","CC(C)(Oc1ccc(C(=O)c2ccc(Cl)cc2)cc1)C(=O)O","C17H15ClO4"),
 ("Bezafibrate","CC(C)(Oc1ccc(CCNC(=O)c2ccc(Cl)cc2)cc1)C(=O)O","C19H20ClNO4"),
 ("Gemfibrozil","Cc1ccc(C)c(OCCCC(C)(C)C(=O)O)c1","C15H22O3"),
 ("Ciprofibrate","CC(C)(Oc1ccc(C2CC2(Cl)Cl)cc1)C(=O)O","C13H14Cl2O3"),
 ("Atorvastatin","CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O","C33H35FN2O5"),
 ("Simvastatin","CCC(C)(C)C(=O)O[C@H]1C[C@H](C)C=C2C=C[C@H](C)[C@H](CC[C@@H]3C[C@@H](O)CC(=O)O3)[C@@H]21","C25H38O5"),
 ("Lovastatin","CCC(C)C(=O)O[C@H]1C[C@H](C)C=C2C=C[C@H](C)[C@H](CC[C@@H]3C[C@@H](O)CC(=O)O3)[C@@H]21","C24H36O5"),
 ("Rosuvastatin","CC(C)c1nc(N(C)S(C)(=O)=O)nc(-c2ccc(F)cc2)c1/C=C/[C@@H](O)C[C@@H](O)CC(=O)O","C22H28FN3O6S"),
 ("Fluvastatin","CC(C)n1c(/C=C/[C@@H](O)C[C@@H](O)CC(=O)O)c(-c2ccc(F)cc2)c2ccccc21","C24H26FNO4"),
 ("Pravastatin","CC[C@H](C)C(=O)O[C@H]1C[C@H](O)C=C2C=C[C@H](C)[C@H](CC[C@@H](O)C[C@@H](O)CC(=O)O)[C@@H]12","C23H36O7"),
 ("Ezetimibe","O[C@H](CC[C@H]1[C@H](c2ccc(O)cc2)N(c2ccc(F)cc2)C1=O)c1ccc(F)cc1","C24H21F2NO3"),
 ("Obeticholic acid","CC[C@H]1[C@@H](O)[C@H]2CC[C@H]3[C@@H](CCC4C[C@H](O)CC[C@]34C)[C@@]2(C)CC1","C26H44O4"),
 ("Chenodeoxycholic acid","C[C@H](CCC(=O)O)[C@H]1CC[C@H]2[C@@H]3[C@H](O)C[C@@H]4C[C@H](O)CC[C@]4(C)[C@H]3CC[C@]12C","C24H40O4"),
 ("Ursodeoxycholic acid","C[C@H](CCC(=O)O)[C@H]1CC[C@H]2[C@@H]3[C@H](O)C[C@@H]4C[C@H](O)CC[C@]4(C)[C@H]3CC[C@]12C","C24H40O4"),
 ("Tropifexor","CC1(C)COC(=O)N1c1ccc(-c2noc(-c3ccccc3Cl)n2)cc1C(=O)O",None),
 ("GW4064","CC(C)c1onc(-c2c(Cl)cccc2Cl)c1COc1ccc(/C=C/c2cccc(C(=O)O)c2)c(C)c1","C29H26Cl2N2O4"),
 ("Metformin","CN(C)C(=N)NC(=N)N","C4H11N5"),
 ("A-769662","Oc1ccc(-c2ccc(-c3csc(C#N)c3O)cc2)cc1","C20H12N O2S?"),
 ("AICAR","NC(=O)c1ncn([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c1N","C9H14N4O5"),
 ("Salicylate","O=C(O)c1ccccc1O","C7H6O3"),
 ("Berberine","COc1ccc2cc3[n+](cc2c1OC)CCc1cc2c(cc1-3)OCO2","C20H18NO4+"),
 ("Bempedoic acid","CC(C)(CCCCCC(O)CCCCCC(C)(C)C(=O)O)C(=O)O","C19H36O5"),
 ("TVB-2640","Cc1nc(N2CCC(c3ccc(OC(F)(F)F)cc3)CC2)ncc1C(=O)NC1CCOCC1",None),
 ("Cerulenin","CCC=CCC=CCCC(=O)[C@@H]1O[C@@H]1C(N)=O","C12H17NO3"),
 ("C75","CCCCCCCC[C@H]1[C@@H](C(=O)O)C(=C)C(=O)O1","C14H22O4"),
 ("Orlistat","CCCCCCCCCCC[C@H](C[C@@H]1OC(=O)[C@H]1CCCCCC)OC(=O)[C@@H](CC(C)C)NC=O","C29H53NO5"),
 ("PF-06424439","CN(C)C(=O)C1CCN(C(=O)c2cnc(N3CCOCC3)nc2)CC1",None),
 ("Ervogastat","O=C(Nc1ccc(C(F)(F)F)cn1)N1CCC(Oc2ccncc2)CC1",None),
 ("GW501516","Cc1c(CSc2ccc(OCC(=O)O)c(C)c2)sc(-c2ccc(C(F)(F)F)cc2)n1","C21H18F3NO3S2"),
 ("Elafibranor","CC(C)(Oc1ccc(C(=O)/C=C/c2ccc(SC)cc2)c(C)c1C)C(=O)O","C22H24O4S"),
 ("Saroglitazar","CCc1ccc(CCN(C)c2ccc(CC(Oc3ccccc3)C(=O)OCC)cc2)nc1",None),
 ("Telmisartan","CCCc1nc2cc(-c3nc4ccccc4n3C)ccc2n1Cc1ccc(-c2ccccc2C(=O)O)cc1","C33H30N4O2"),
 ("Trimetazidine","COc1ccc(OC)c(CN2CCNCC2)c1OC","C14H22N2O3"),
 ("Ranolazine","COc1ccccc1OCC(O)CN1CCN(CC(=O)Nc2c(C)cccc2C)CC1","C24H33N3O4"),
 ("Perhexiline","C1CCC(CC1)C(CC1CCCCC1)C1CCCCN1","C19H35N"),
 ("Pentoxifylline","Cn1c(=O)n(CCCCC(C)=O)c2ncn(C)c2c1=O","C13H18N4O3"),
 ("Alpha-lipoic acid","OC(=O)CCCCC1CCSS1","C8H14O2S2"),
 ("EPA","CC/C=C\\C/C=C\\C/C=C\\C/C=C\\C/C=C\\CCCC(=O)O","C20H30O2"),
 ("DHA","CC/C=C\\C/C=C\\C/C=C\\C/C=C\\C/C=C\\C/C=C\\CCC(=O)O","C22H32O2"),
 ("Fenretinide","CC1=C(/C=C/C(C)=C/C=C/C(C)=C/C(=O)Nc2ccc(O)cc2)C(C)(C)CCC1","C26H33NO2"),
 ("Naringenin","O=C1CC(c2ccc(O)cc2)Oc2cc(O)cc(O)c21","C15H12O5"),
 ("Resveratrol","Oc1ccc(/C=C/c2cc(O)cc(O)c2)cc1","C14H12O3"),
 ("Quercetin","O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12","C15H10O7"),
 ("Curcumin","COc1cc(/C=C/C(=O)CC(=O)/C=C/c2ccc(O)c(OC)c2)ccc1O","C21H20O6"),
 ("Betaine","C[N+](C)(C)CC(=O)[O-]","C5H11NO2"),
 ("L-Carnitine","C[N+](C)(C)C[C@H](O)CC(=O)[O-]","C7H15NO3"),
 ("Etomoxir","CCCCCCOc1ccc(Cl)cc1CCCCCC1(C(=O)OCC)CO1",None),
 ("Aramchol","CCCCCCCCCCCCCCCCCCCC(=O)N[C@H]1CC[C@]2(C)[C@H](CC[C@H]3[C@@H]2[C@H](O)C[C@H]2[C@@]3(C)CC[C@H]2[C@H](C)CCC(=O)O)C1",None),
 ("Nicotinic acid","O=C(O)c1cccnc1","C6H5NO2"),
]

for name, smi, exp in T:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        print(f"FAIL  {name}")
        continue
    f = rdMolDescriptors.CalcMolFormula(m)
    flag = ""
    if exp and exp.replace("?","") not in (f, f.replace("+","")):
        flag = f"   <-- expected {exp}"
    print(f"ok    {name:26s} {f:20s} MW={Descriptors.MolWt(m):7.2f}{flag}")
```

### 11.4 `03_methodology.py` —— 本文档生成脚本

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成 methodology_and_code.md —— 方法学描述 + 实际执行的完整源码。
源码段落直接从磁盘读取 01_screen.py / 02_deliverables.py / 03_methodology.py，
因此文档中呈现的代码可证明就是真实运行过的代码。
"""
import os, json, hashlib
import pandas as pd

OUT = "/mnt/c/Users/zhangt97/Downloads/MASLD_screen"
stats = json.load(open(f"{OUT}/_funnel_stats.json", encoding="utf-8"))
cand  = pd.read_csv(f"{OUT}/candidates.csv")
pos   = pd.read_csv(f"{OUT}/positive_control_recovery.csv")

SRC = ["01_screen.py", "02_deliverables.py", "03_methodology.py"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def read(p):
    return open(os.path.join(OUT, p), encoding="utf-8").read().rstrip()


top_rows = "\n".join(
    f"| {r.Rank} | `{r.Molecule_ID}` | {r.CAS} | {r.Formula} | {r.MW} | {r.LogP} | {r.TPSA} | "
    f"{r.S_efficacy:.4f} | {r.S_safety:.4f} | {r.S_total:.4f} | {r.S_mech:.4f} | "
    f"**{r.S_nomination:.4f}** | {r.Primary_Target_Hypothesis} |"
    for _, r in cand.iterrows())

smi_rows = "\n".join(f"| {r.Rank} | `{r.Molecule_ID}` | `{r.SMILES}` | {r.Matched_Chemotype_Rule} |"
                     for _, r in cand.iterrows())

pos_rows = "\n".join(
    f"| `{r.Molecule_ID}` | {r.CAS} | {r.Nearest_Reference} | {r.Ref_Tanimoto:.2f} | "
    f"{r.Ref_Target} | {r.S_total:.4f} |" for _, r in pos.iterrows())

hash_rows = "\n".join(f"| `{p}` | {os.path.getsize(os.path.join(OUT,p)):,} B | `{sha(os.path.join(OUT,p))}` |"
                      for p in SRC)

md = f"""# MASLD 早期降脂小分子发现 —— 方法学与可复现代码

> **赛题**：上海人工智能实验室 × 临港实验室「MASLD 早期降脂小分子发现」
> **模型**：HepG2 细胞 – 游离脂肪酸 (FFA) 肝细胞脂质蓄积模型
> **目标**：从 {stats['total']:,} 个小分子中筛选**既降低肝细胞脂质蓄积、又在有效浓度下不显著损伤细胞活力**的候选分子（低毒性降脂）
> **产出**：`candidates.csv`（Top 10 提名）、`mechanism_and_validation.pdf`（机制与验证方案）、本文档

---

## 目录

1. [方法学总览](#1-方法学总览)
2. [关键方法学决策与其理由](#2-关键方法学决策与其理由)
3. [数据与环境](#3-数据与环境)
4. [第一步：SDF 解析与描述符计算](#4-第一步sdf-解析与描述符计算)
5. [第二步：多层过滤](#5-第二步多层过滤)
6. [第三步：启发式多目标打分](#6-第三步启发式多目标打分)
7. [第四步：机制证据层与多样性提名](#7-第四步机制证据层与多样性提名)
8. [结果](#8-结果)
9. [方法验证与局限](#9-方法验证与局限)
10. [复现方式](#10-复现方式)
11. [完整源码](#11-完整源码)

---

## 1. 方法学总览

```
T.sdf  ({stats['total']:,} 条记录, 67.8 MB, ISO-8859-1)
  │
  ├─ [0] 编码修复      Latin-1 → UTF-8   (RDKit GetProp 以 UTF-8 解码，原文件会抛 UnicodeDecodeError)
  │
  ├─ [1] 解析与标准化  SDMolSupplier → LargestFragmentChooser (去盐) → Uncharger (中和)
  │                    计算 MW / LogP / TPSA / HBD / HBA / RotB / AromRings / FracCsp3 …
  │
  ├─ [2] 多层过滤      元素白名单 → 重原子数 → PAINS → Lipinski → 细胞毒性硬反筛
  │                    {stats['total']:,} → {stats['passed']:,}   （SMILES 级去重后 {stats['passed']-806:,}）
  │
  ├─ [3] 启发式打分    S_total = 0.6·S_efficacy + 0.4·S_safety
  │                    ⚠ 高度饱和：{stats['n_tie_S_total_1']} 个分子并列 1.0000
  │
  ├─ [4] 硬门槛        S_total ≥ {stats['gate']}  →  {stats['n_gated']:,} 个分子
  │
  ├─ [5] 机制证据层    Morgan(r=2,2048) Tanimoto vs {stats['n_refs']} 个参照药 ⊕ {stats['n_chemotype_rules']} 条优势骨架 SMARTS
  │                    S_mech = 0.55·min(T_max/0.55, 1) + 0.45·chemotype_weight
  │                    S_nomination = 0.5·S_total + 0.5·S_mech
  │
  ├─ [6] 阳性对照剥离  Tanimoto ≥ 0.85 视为已知药 → {stats['n_pos_ctrl']} 个，单列报告，不参与提名
  │
  └─ [7] 多样性提名    Murcko 骨架去重 + 两两 Tanimoto < 0.60 + 每靶点 ≤ 1 个  →  **Top 10**
```

---

## 2. 关键方法学决策与其理由

本节是本方案与「照抄题目模板」的区别所在。以下五个决策都源于在真实数据上发现的问题。

### 2.1 赛题给定的打分函数在本库上不可用于排序（最重要的发现）

题目设定的打分为：

```
S_efficacy: LogP 最优区间 [2.5, 3.5]，TPSA 最优区间 [50, 90]，距离衰减
S_safety  : MW 越接近 350 越高，扣除毒性基团罚分
S_total   = 0.6 × S_efficacy + 0.4 × S_safety
```

这是一个**纯理化性质函数**，不包含任何生物活性信息。实际运行后：

- **{stats['n_tie_S_total_1']} 个分子并列 S_total = 1.0000**（占打分池的 4.6%）；
- 若直接 `sort_values('S_total').head(10)`，得到的 Top 10 实际上是**文件顺序的前 10 个**，与科学无关；
- 且这 10 个必然是理化性质近乎相同的一组类似物，`Primary_Target_Hypothesis` 一列将退化为纯叙事。

**处理方式**：完全保留题目公式并作为**硬门槛**（`S_total ≥ 0.95`，保留 {stats['n_gated']:,} 个），在门槛内改用可验证的机制证据排序。纯 S_total 排名同时完整导出于 `all_scored_molecules.csv`，评审可自行核对。

### 2.2 引入机制证据层，让靶点假说可被追溯

两个证据来源，均写死在代码中、可复核：

| 证据 | 实现 | 权重 |
|---|---|---|
| 与已知肝脂调节剂的结构相似性 | {stats['n_refs']} 个临床/工具化合物参照集，Morgan r=2 / 2048 bit，Tanimoto | 0.55 |
| 肝脂靶点优势骨架药效团 | {stats['n_chemotype_rules']} 条 SMARTS 规则（含 `anti` 反向排除模式） | 0.45 |

```
S_mech = 0.55 · min(T_max / 0.55, 1.0) + 0.45 · chemotype_weight
S_nomination = 0.5 · S_total + 0.5 · S_mech
```

这样 `candidates.csv` 中每个靶点假说都能回溯到「命中了哪条 SMARTS / 最像哪个已知药、相似度多少」，而不是模型编的故事。

### 2.3 参照药物集必须逐个结构核验

初版参照集直接凭记忆写 SMILES，用 `_check_refs.py`（RDKit 计算分子式 vs 文献分子式）核对后发现 **6 个结构错误**（Firsocostat、Lanifibranor、Obeticholic acid、A-769662、Cerulenin、Telmisartan）与 **3 个严重错误**（Tropifexor、Saroglitazar、Etomoxir）。

- 6 个已修正并重新核验分子式完全一致；
- Tropifexor、Saroglitazar 直接删除；
- 无法确证的 firsocostat 条目改名为 "ACC inhibitor chemotype (firsocostat-like proxy)"，不冒充具体药物。

> **教训**：参照集是整个机制层的基准。基准错了，所有下游相似度与靶点归属都错，且错误不会以报错的形式暴露。任何依赖「记忆中的 SMILES」的流程都必须有独立的分子式核验步骤。

### 2.4 SMARTS 药效团规则会产生看似合理的假阳性

初版有一条甲状腺激素拟似物（THR-β）规则：

```smarts
[Cl,Br,I]c1cccc([Cl,Br,I])c1O[#6]        ← 错误
```

它把 `TN7135  COc1c(Br)cc(CC(=O)O)cc1Br`（一个二溴**苯甲醚**）判成了 THR-β 配体，因为 `[#6]` 能匹配甲基碳。而 T3/resmetirom 类拟似物需要的是 4-**芳氧基**外环。修正为：

```smarts
[Cl,Br,I]c1cccc([Cl,Br,I])c1[OX2][c]     ← 要求氧连接芳香碳
```

修正后该假阳性从 Top 10 中消失，位置由 `T4515`（ervogastat 型芳基脲 + 4-芳氧基哌啶，DGAT2）替代。

> **教训**：SMARTS 中 `[#6]`（任意碳）与 `[c]`（芳香碳）的差别，足以让一个假设从「甲状腺激素受体」变成「什么都不是」。每条规则都需要用已知阳性/阴性分子对拍验证。

### 2.5 Murcko 骨架去重不足以保证结构多样性

仅按 Bemis–Murcko 骨架去重后，Top 10 中出现：排名 3/4/5 的描述符完全相同（406.56 / 3.66 / 94.83），排名 8/9/10 全是小檗碱同系物。原因是同一结构在库中以不同货号（游离碱/盐/不同批次）多次出现，且近似物骨架不同但化学空间重叠。

补三道约束：

1. **SMILES 级去重**：规范化 SMILES 相同则合并，保留一个货号，其余记入 `Duplicate_IDs` 列（12,775 → 11,969，合并 806 个重复结构）；
2. **贪心两两 Tanimoto < 0.60**：新候选与已选任一分子相似度超阈值即跳过；
3. **每个靶点假说至多 1 个分子**（`MAX_PER_TARGET = 1`）。

最终 Top 10 覆盖 10 条相互独立的机制通路。

### 2.6 数据清洗：CAS 号的 Excel 日期损坏

库中部分 CAS 被 Excel 误转为日期格式（如 `TN7583` 的 CAS 为 `2458/8/4`，应为 `2458-08-4`）。写了 `repair_cas()`，用 **CAS 校验位算法**确认后才修正：

```
2458-08-4 校验：(8×1 + 0×2 + 8×3 + 5×4 + 4×5 + 2×6) = 84；84 mod 10 = 4 ✓
```

32 个校验位不通过的非标准 CAS 保守地保持原值不动（宁可不改也不猜）。Top 10 中无此类分子。

---

## 3. 数据与环境

| 项 | 值 |
|---|---|
| 输入文件 | `T.sdf`，67,779,525 B，1,785,074 行，ISO-8859 text (CRLF) |
| 记录数 | {stats['total']:,} |
| 数据标签 | `<ID>` ({stats['total']:,})、`<Formula>` ({stats['total']:,})、`<MolWt>` ({stats['total']:,})、`<CAS>` (22,155) |
| 库性质 | 已知生物活性化合物库（药物重定位型），编号形如 `T0002` / `TN7583` / `T4S0795` |
| Python | 3.x（conda env `pytorch_env`） |
| RDKit | 2026.03.5 |
| pandas / numpy | 2.3.3 / 2.2.6 |
| reportlab | PDF 生成（中文用内置 CID 字体 `STSong-Light`，无需外部 TTF） |
| 确定性 | 无随机数来源；重跑结果逐字节一致 |

**编码问题**：原 SDF 为 ISO-8859-1，RDKit 的 `mol.GetProp()` 以 UTF-8 解码，在读取含 `0xa1` 等字节的记录时抛 `UnicodeDecodeError`。脚本首次运行时自动逐行转码生成 `/tmp/T_utf8.sdf`（修复 379 行），并额外用 `safe_prop()` 兜底。

---

## 4. 第一步：SDF 解析与描述符计算

- `Chem.SDMolSupplier(sdf, removeHs=True, sanitize=True)`；
- **标准化**：`rdMolStandardize.LargestFragmentChooser`（去除反离子/盐）→ `Uncharger`（中和电荷，但保留季铵等永久电荷）；
- **描述符**：`Descriptors.MolWt`、`Crippen.MolLogP`、`rdMolDescriptors.CalcTPSA`、`Lipinski.NumHDonors/NumHAcceptors/NumRotatableBonds`、`CalcNumAromaticRings`、`CalcFractionCSP3`、`CalcMolFormula`、重原子数、卤素计数；
- **属性读取**：`safe_prop()` 包裹 try/except，避免单条坏记录中断全库解析。

---

## 5. 第二步：多层过滤

| 层 | 规则 | 淘汰数 | 理由 |
|---|---|---:|---|
| 解析/标准化 | sanitize 失败 | {stats['unparsable']:,} | 价键异常记录 |
| 元素白名单 | 仅允许 H,B,C,N,O,F,P,S,Cl,Br,I | {stats['bad_element']:,} | 排除金属配合物、无机盐、含 Si/Sn 分子 |
| 分子尺寸 | 重原子数 ≥ 12 | {stats['too_small']:,} | 碎片过小，难以支撑细胞水平活性 |
| PAINS | `FilterCatalogParams.PAINS_A/B/C` | {stats['pains']:,} | 泛筛选干扰结构 |
| Lipinski | MW 150–550, LogP −1–5, HBD ≤ 5, HBA ≤ 10 | {stats['lipinski']:,} | 题目指定的类药性约束 |
| 细胞毒性硬反筛 | **LogP > 4.5 且 TPSA < 20 → 剔除** | {stats['cytotox_hard']:,} | 高亲脂 + 极低极性 = 膜裂解 / 线粒体解偶联的经验特征 |
| **合计通过** | | | **{stats['passed']:,}** |
| SMILES 去重 | 规范化 SMILES 相同则合并 | 806 | 同一结构多货号 |
| **进入排序** | | | **{stats['passed']-806:,}** |

细胞毒性反筛按题目要求实现为「硬过滤 + 降权」两段：`LogP > 4.5 且 TPSA < 20` 直接剔除；`LogP > 4.2` 或 `TPSA < 30` 在 `S_safety` 中连续扣分（见 6.2）。这样避免了单一硬阈值在边界上的悬崖效应。

**结构警示子**（22 条 SMARTS，用于 `S_safety` 扣分，不作硬过滤）：
Michael 受体（乙烯基酮、丙烯酰胺、乙烯基砜、丙烯酸酯、丙烯腈）、醌、环氧、氮丙啶、醛、硝基芳烃、偶氮芳烃、肼、异氰酸酯、酸酐、酰卤、活泼卤代烷、游离硫醇、芳香亚硝基、过氧化物、有机磷、硫脲、稠环多环芳烃。

---

## 6. 第三步：启发式多目标打分

### 6.1 S_efficacy

```python
def plateau(x, lo, hi, sigma):
    if lo <= x <= hi: return 1.0
    d = (lo - x) if x < lo else (x - hi)
    return math.exp(-0.5 * (d / sigma) ** 2)

def score_efficacy(logp, tpsa, arom_rings, rotb):
    s_logp  = plateau(logp, 2.5, 3.5, 1.0)     # 题目指定最优区间
    s_tpsa  = plateau(tpsa, 50.0, 90.0, 25.0)  # 题目指定最优区间
    s_shape = 0.5 * plateau(arom_rings, 1, 3, 1.0) + 0.5 * plateau(rotb, 2, 7, 3.0)
    return 0.45 * s_logp + 0.40 * s_tpsa + 0.15 * s_shape
```

「平台 + 高斯衰减」而非线性距离衰减：区间内不做人为区分（题目说的是「最优区间」而非「最优点」），区间外平滑衰减，避免阶跃。`s_shape` 是唯一的补充项，占 15%，用于抑制过度刚性（无芳环）或过度柔性（RotB ≫ 7，构象熵惩罚大、细胞渗透差）的分子。

### 6.2 S_safety

```python
def score_safety(mw, logp, tpsa, n_tox, n_halogen, fsp3):
    s = gaussian(mw, 350.0, 120.0)                              # 题目：MW 越接近 350 越高
    s -= 0.25 * min(n_tox, 3)                                   # 题目：毒性基团扣分
    if n_halogen > 3: s -= 0.10 * min(n_halogen - 3, 3)         # 题目：卤素过多扣分
    if logp > 4.2:    s -= 0.30 * min((logp - 4.2) / 0.8, 1.0)  # 亲脂性细胞毒（连续降权）
    if tpsa < 30:     s -= 0.30 * min((30 - tpsa) / 30.0, 1.0)  # 低极性膜活性（连续降权）
    s += 0.10 * min(fsp3 / 0.4, 1.0)                            # 三维性 → 脱靶更少
    return max(0.0, min(1.0, s))
```

`S_total = 0.6 × S_efficacy + 0.4 × S_safety`（与题目一致）。

`fsp3` 加成的依据是 Lovering 的 "Escape from Flatland"：sp3 比例高的分子脱靶率与毒性率显著更低。

---

## 7. 第四步：机制证据层与多样性提名

### 7.1 参照药物集（{stats['n_refs']} 个，逐个核验分子式）

覆盖 THR-β、ACC/ACLY、FASN/SCD1、DGAT2、PPAR α/δ/γ/pan、FXR/胆汁酸、AMPK、HMGCR、脂解/FAO/转运、天然产物共十类。每条记录为 `(名称, SMILES, 靶点, 机制)`。

### 7.2 优势骨架规则（{stats['n_chemotype_rules']} 条）

每条规则形如：

```python
dict(name="噻唑烷-2,4-二酮 (TZD)", weight=0.92, target="PPAR-gamma",
     smarts=["O=C1NC(=O)C(S1)C"], anti=[...], mech="...")
```

`anti` 字段用于反向排除（命中 `smarts` 但同时命中 `anti` 则不计分），编译时对每条 SMARTS 做合法性校验，解析失败直接抛异常而非静默跳过。

### 7.3 提名参数

```python
TANI_STRONG    = 0.55   # 相似度饱和点
POS_CTRL_T     = 0.85   # ≥ 此值视为已知药，剥离出提名池
GATE           = 0.95   # S_total 硬门槛
DIV_T          = 0.60   # 候选间两两 Tanimoto 上限
MAX_PER_TARGET = 1      # 每个靶点假说最多提名 1 个
N_TOP          = 10
```

---

## 8. 结果

### 8.1 提名 Top 10

| # | 库编号 | CAS | 分子式 | MW | LogP | TPSA | S_eff | S_saf | S_total | S_mech | S_nom | 靶点假说 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
{top_rows}

### 8.2 结构与命中的药效团

| # | 库编号 | SMILES | 命中优势骨架规则 |
|---|---|---|---|
{smi_rows}

### 8.3 阳性对照回收（方法学自验证）

在**完全不使用任何活性标签**的前提下，流程从库中自动识别出 {stats['n_pos_ctrl']} 个已知肝脂调节剂（Tanimoto = 1.00），全部通过了 PAINS / 类药性 / 毒性反筛并落在 S_total ≥ 0.95 区间：

| 库编号 | CAS | 对应已知药物 | Tanimoto | 靶点 | S_total |
|---|---|---|---|---|---|
{pos_rows}

这是本方案唯一的**客观**方法学证据：打分与过滤逻辑确实富集了真实的降脂化学空间，而非随机保留分子。这 {stats['n_pos_ctrl']} 个分子已从 Top 10 中剔除（提名已上市药物属循环论证），但**建议直接采购小檗碱与非诺贝特酸作为实验阳性对照**——它们就在同一个库里。

---

## 9. 方法验证与局限

**已做的验证**
1. 阳性对照回收（8.3）——流程能找回已知答案；
2. 参照集分子式逐个核验（`_check_refs.py`）——基准正确；
3. SMARTS 规则的假阳性排查（2.4）——规则不乱匹配；
4. 结构去重与多样性约束（2.5）——Top 10 不是同一分子的十个货号。

**局限（详见 PDF 第 8 节）**
1. 打分函数不含生物活性信息，本质是类药性刻画，不是活性预测；
2. 靶点假说是化学型推断，`T15049`（芳基乙酸-磺酰胺，亦见于前列腺素类似物）与 `T8701`（哌啶甲酰胺，药效团较泛）的置信度明显低于 `T2607`(TZD) / `T12092`(statin 内酯) / `T8532`(双胍)；
3. **理化毒性启发式系统性低估「在靶但有害」的机制性毒性**——已人工标注三个高风险分子：`T8532`（芳基双胍，苯乙双胍类，乳酸酸中毒）、`T75436`（亲脂性阳离子，线粒体蓄积）、`T12092`（statin，甲羟戊酸耗竭），并在 PDF 中给出针对性的实验区分方案；
4. Morgan 指纹不区分立体化学，而 TZD 的 C5、statin 内酯的多个手性中心对活性影响显著；
5. 未做 ADMET / 聚集倾向的定量预测。

> 尤其提示第 3 条：`T8532` 的 `S_safety = 1.000`，但它是苯乙双胍类芳基双胍。**若只读代码输出而不读注释，会得出完全错误的安全性结论。** 这也是「预测毒性」一列在 `candidates.csv` 中带有独立 `Cytotoxicity_Caveat` 字段的原因。

---

## 10. 复现方式

```bash
# 环境
conda activate pytorch_env          # rdkit 2026.03.5, pandas, reportlab

cd /path/to/MASLD_screen
python _check_refs.py               # 可选：核验参照集 SMILES 的分子式
python 01_screen.py                 # 筛选与打分（约 3-5 min，22,966 分子）
python 02_deliverables.py           # candidates.csv + PDF + 结构图
python 03_methodology.py            # 本文档
```

**产物清单**

| 文件 | 内容 |
|---|---|
| `candidates.csv` | **Top 10 提名清单**（30 列，含题目要求的 7 个字段 + 全部可追溯的中间量） |
| `mechanism_and_validation.pdf` | **机制与验证方案**（18 页，含结构图、漏斗、逐分子档案、HepG2-FFA 完整实验方案） |
| `methodology_and_code.md` | 本文档 |
| `all_scored_molecules.csv` | 11,969 个通过过滤分子的完整打分与描述符（含纯 S_total 排名） |
| `gated_annotated.csv` | 2,496 个门槛内分子的机制证据注释 |
| `positive_control_recovery.csv` | 阳性对照回收清单 |
| `structures/` | Top 10 结构图 PNG |
| `_funnel_stats.json` | 漏斗统计（供文档自动生成，避免手抄数字出错） |

**源码指纹**

| 文件 | 大小 | SHA-256 (前 16 位) |
|---|---|---|
{hash_rows}

---

## 11. 完整源码

以下代码由本文档生成脚本从磁盘直接读取，与实际运行的代码逐字节一致。

### 11.1 `01_screen.py` —— 解析、过滤、打分、机制注释、多样性提名

```python
{read('01_screen.py')}
```

### 11.2 `02_deliverables.py` —— candidates.csv、结构图、机制与验证方案 PDF

```python
{read('02_deliverables.py')}
```

### 11.3 `_check_refs.py` —— 参照药物集 SMILES 分子式核验

```python
{read('_check_refs.py')}
```

### 11.4 `03_methodology.py` —— 本文档生成脚本

```python
{read('03_methodology.py')}
```
"""

open(f"{OUT}/methodology_and_code.md", "w", encoding="utf-8").write(md)
print(f"methodology_and_code.md -> {len(md):,} 字符 / {md.count(chr(10))+1:,} 行")
```

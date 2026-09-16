# LoRA 微调从 0 到 1 搭建指南

> 面向对象：会一点 Python、手上有 6GB 左右显存的显卡（本机 RTX 2060）、想亲手把一个大模型"改成自己想要的语气/格式"的人。
> 目标：跟着做完，你手上会有一个能跑、能换任务、能持续扩展的 QLoRA 微调工程。
> 环境：Windows 11 + RTX 2060 6GB + uv 创建的 venv。
> 工程位置：**`E:\lorademo`**，虚拟环境 **`E:\lorademo\910demo`**。

---

## ⚠️ 阅读前必看：本机现状核对表 + 踩坑索引

**这份指南是按"实际搭完之后"的事实写的，不是纸上教程。**
搭建过程中踩了 **17 个坑**，其中 3 个能把环境直接搞崩。所有内容都经过本机复现验证。

### 本机现状（2026-09-15 实测确认）

| 项 | 实际值 | 备注 |
| --- | --- | --- |
| 工程位置 | `E:\lorademo` | — |
| 虚拟环境 | `E:\lorademo\910demo`（uv 创建），**Python 3.13.9** | 不是 conda 环境 |
| torch | **2.6.0+cu124**，`cuda.is_available()` = **True**，CUDA 12.4 | 需要 CUDA 版，不能装 CPU 版 |
| transformers | **5.17.0** | ⚠️ 大版本 5.x，网上教程多是 4.x，API 有变 |
| peft | 0.20.0 | — |
| bitsandbytes | **0.50.2** | ⚠️ 必须 ≥0.45，否则缺 cuda124 DLL |
| datasets / accelerate | 5.0.1 / 1.15.0 | — |
| 底座模型 | `D:\Model\Qwen2.5-1.5B-Instruct` | 目录 5.77GB，**其中一半是 `.git` 里的 LFS 副本** |
| 训练数据 | 21 条训练 + 4 条验证 | 脚本 `01` 生成 |
| 环境体积 | `910demo` 约 **4.95 GB** | — |
| LoRA 适配器 | **8.3 MB** | 对比底座 3GB，缩小 370 倍 |
| 磁盘余量 | C: ≈19.4GB / D: ≈80.9GB / E: ≈57.0GB / F: ≈45.8GB | **C 盘紧张，缓存别往 C 盘放** |

### 对原 README 的修订记录（4 处）

工程里已有一份 `README.md`，但有几处与实测不符，本指南以实测为准：

| # | 位置 | 原文写的 | 实测 | 说明 |
| --- | --- | --- | --- | --- |
| 1 | 目录结构 | `lora_qwen_cs_ep10` 是"10 轮对照实验的 checkpoint" | **只跑到第 6 轮（step 18/30）就中断了**，`max_steps=30` 而 `global_step=18` | 该目录根层**没有** `adapter_config.json`，直接指向目录会加载失败，必须指向 `checkpoint-18` |
| 2 | 实测结果 | 适配器 **8.7 MB** | **8.3 MB** | `adapter_model.safetensors` 实际大小 |
| 3 | 目录结构 | `merged_qwen_cs` 3.09 GB | **2.89 GB** | 合并后实际体积 |
| 4 | 环境修复记录 | 模型曾下载到 `E:\lorademo\models\`（429MB 残留），已清理 | 属实，当前统一走 `D:\Model` | 保留记录即可 |

---

## 目录

- [第 0 章 先搞清楚 LoRA 是什么](#第-0-章-先搞清楚-lora-是什么)
- [第 1 章 技术选型（为什么这么选）](#第-1-章-技术选型为什么这么选)
- [第 2 章 建工程骨架与环境](#第-2-章-建工程骨架与环境)
- [第 3 章 里程碑 1：造数据集](#第-3-章-里程碑-1造数据集)
- [第 4 章 里程碑 2：跑通训练（核心）](#第-4-章-里程碑-2跑通训练核心)
- [第 5 章 里程碑 3：推理对比](#第-5-章-里程碑-3推理对比)
- [第 6 章 里程碑 4：消融实验](#第-6-章-里程碑-4消融实验)
- [第 7 章 里程碑 5：交互式对话](#第-7-章-里程碑-5交互式对话)
- [第 8 章 里程碑 6：合并导出与部署](#第-8-章-里程碑-6合并导出与部署)
- [第 9 章 本机高频坑清单（17 个）](#第-9-章-本机高频坑清单17-个)
- [第 10 章 学习与扩展路线](#第-10-章-学习与扩展路线)

---

## 第 0 章 先搞清楚 LoRA 是什么

### 全量微调为什么在 6GB 卡上跑不动

全量微调 = 把模型 15 亿个参数**全部重新学一遍**。
1.5B 模型光优化器状态（Adam 的动量 + 方差）就要占 `15亿 × 2 × 4字节 ≈ 12GB`，**还没算权重和梯度就已经爆了**。

你的卡只有 6GB。所以全量微调这条路，从一开始就不通。

### LoRA 的思路：不动大矩阵，只加一条小路

```
原始计算:   h = W·x
LoRA 后:    h = W·x + (alpha/r)·B·A·x
                     ↑          ↑
                 冻结不动   只训练这两个小矩阵
```

`W` 是 2048×2048 = **419 万**参数，而 `A`、`B` 加起来只有 2048×8×2 = **3.3 万**参数。

**训练量降到约 1/128**，显存和时间的压力一下就下来了。代价是"表达能力变弱"—— 但对"学个语气/格式"这种任务完全够用。

### QLoRA：再把冻结的权重压成 4-bit

在 LoRA 基础上，把**不训练的** `W` 用 4-bit 存储（NF4 格式）：

| | 权重占用 | 优化器状态 | 6GB 能跑吗 |
| --- | --- | --- | --- |
| 全量 fp16 | 3.0 GB | ~12 GB | ❌ |
| LoRA fp16 | 3.0 GB | 很小 | ⚠️ 勉强 |
| **QLoRA（4-bit）** | **~1.0 GB** | 很小 | ✅ 宽裕 |

代价是量化有精度损失，但对 demo 级任务几乎无感。

### 和另外两条路线怎么选

| 方案 | 做法 | 适合 | 代价 |
| --- | --- | --- | --- |
| 提示词硬塞 | 把要求写进 system prompt | 只要风格 | 不稳定、会漂移 |
| **LoRA 微调** | 用小数据训练适配器 | **固定语气 / 固定格式 / 固定输出结构** | 要数据、要训 |
| RAG | 检索资料塞进 prompt | 资料经常变、要引用出处 | 需要检索工程 |

**为什么本 demo 选 LoRA 而不是 RAG**：我们要的是"改变说话方式"，不是"注入新知识"。
风格类任务有个巨大优势 —— **不受"模型本来会不会"的干扰**，微调前后差异肉眼可见，不用做评测就能判断成功与否。

> **一句话**：LoRA = 冻结原权重，在旁边挂两个小矩阵去学"增量"。省显存省时间，代价是只适合"轻量改造"。

---

## 第 1 章 技术选型（为什么这么选）

| 环节 | 本工程选型 | 理由 | 以后可以换 |
| --- | --- | --- | --- |
| 底座模型 | **Qwen2.5-1.5B-Instruct** | 1.5B 在 6GB 上很宽裕；Instruct 版已会聊天，只需叠加风格，收敛极快 | Qwen2.5-0.5B（更省）/ 3B（临界）/ 7B（跑不动） |
| 精调方式 | **QLoRA**（4bit + LoRA） | 6GB 显存下唯一稳妥方案 | 显存 ≥12GB 可关掉 4bit |
| 训练框架 | **transformers `Trainer`** | 官方、可控、每一步都能看明白 | trl 的 `SFTTrainer`（更短） |
| 加速库 | （未用） | demo 阶段 59 秒一轮，不需要 | Unsloth（同硬件快约 2 倍） |
| 数据格式 | **ChatML `{"messages": [...]}`** | 靠 `apply_chat_template()` 套模板，**换模型不用改数据** | 任何能转成 messages 的格式 |
| 显存优化 | 4bit + 梯度检查点 + 梯度累积 + 8bit 优化器 | 四件套，少一个就可能 OOM | — |
| 产物管理 | 只存适配器（几 MB） | 易分发、可多份并存 | 需要部署时再 `merge_and_unload()` |

### 三个关键决策解释

**1. 为什么用 `messages` 格式，而不是手拼字符串？**

不同模型的对话模板不一样（Qwen 用 `<|im_start|>`，Llama 用 `[INST]`）。
用 `messages` + `apply_chat_template()`，**换模型一个字都不用改数据**。这是最省事的解耦方式。

**2. 为什么任务选"客服人设"这种看起来没用的东西？**

因为 demo 阶段的**唯一目标是"验证链路是通的"**，不是"做出产品"。
风格对齐类任务：效果肉眼可见、几十条数据就够、不涉及知识注入 —— **试错成本最低，反馈最快**。

**3. 为什么第一次就开 4-bit，而不是先试全精度？**

6GB 卡上全精度 1.5B **必 OOM**。一上来就走能跑通的路，避免把时间浪费在"调 OOM"上。

---

## 第 2 章 建工程骨架与环境

### 2.1 目录结构（先建好，别乱放）

```text
E:\lorademo\
├─ 910demo\                     # Python 虚拟环境（uv 创建）
├─ data\
│   ├─ train.jsonl              # 训练集 21 条
│   └─ eval.jsonl               # 验证集 4 条
├─ output\
│   ├─ lora_qwen_cs\            # ★ 3 轮正式训练的适配器（8.3MB）
│   │   ├─ adapter_config.json  # 适配器配置
│   │   ├─ adapter_model.safetensors
│   │   ├─ checkpoint-6\        # 中途检查点（含优化器状态，很占地方）
│   │   └─ checkpoint-9\        # 最后一轮
│   ├─ lora_qwen_cs_ep10\       # ⚠️ 10 轮实验，实际只跑到第 6 轮
│   │   ├─ checkpoint-12\
│   │   ├─ checkpoint-15\
│   │   └─ checkpoint-18\       # ← 指目录没用，要指到这里
│   └─ merged_qwen_cs\          # 合并后完整模型（2.89GB，部署用）
├─ scripts\
│   ├─ 01_make_dataset.py       # 造数据
│   ├─ 02_train_lora.py         # ★ 训练（核心）
│   ├─ 02b_train_longer.py      # 10 轮对照实验
│   ├─ 03_inference.py          # 推理对比
│   ├─ 04_merge_lora.py         # 合并导出
│   ├─ 05_ablation.py           # 消融实验
│   ├─ 06_chat.py               # 交互式对话
│   ├─ 07_why_same.py           # 量化对比：原模型 vs LoRA
│   └─ 08_chat_merged.py        # 与【合并后】模型聊天（不挂适配器）
├─ requirements.txt
├─ README.md
└─ LoRA微调从0到1搭建指南.md     # 本文件
```

**底座模型路径**：`D:\Model\Qwen2.5-1.5B-Instruct`

> 💡 **省空间提示**：该目录 5.77GB，但真正的权重只有 2.88GB —— **另外 2.88GB 是 `.git\` 里的一份 LFS 副本**（模型是从 git 仓库拉的）。
> 如果只做推理/微调，删掉 `.git\` 能省一半空间。本工程没删，因为它同时也是版本备份。

所有脚本都按 **`D:\Model` → `工程内 models\` → 在线下载** 的顺序查找模型，换路径只改脚本顶部的 `MODEL_CANDIDATES`。

### 2.2 虚拟环境（用 uv）

```powershell
cd E:\lorademo

# ⚠️ 先把 uv 的缓存/下载目录指到 D 盘，否则 wheel 全下到只剩 19GB 的 C 盘
$env:UV_CACHE_DIR          = "D:\DevEnv\uv\cache"
$env:UV_PYTHON_INSTALL_DIR = "D:\DevEnv\uv\python"
$env:UV_TOOL_DIR           = "D:\DevEnv\uv\tools"

D:\DevEnv\uv\bin\uv.exe venv 910demo --python 3.13
.\910demo\Scripts\Activate.ps1
```

> ⚠️ 这三行 `$env:` **只对当前 PowerShell 窗口生效**，关掉就没了。每开新窗口装包都要重设一遍。
> 不设的后果：`C:\Users\<用户名>\AppData\Local\uv\cache` 会被塞满，而你 C 盘只剩 19GB。

如果 PowerShell 提示"禁止运行脚本"：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

（临时生效，关掉窗口即失效，安全。）

### 2.3 依赖清单

`requirements.txt`（本工程实际使用）：

```text
# 注意: torch 要装 CUDA 12.4 版本，不要用 pip 默认的 CPU 版！
#       pip install torch --index-url https://download.pytorch.org/whl/cu124

torch==2.6.0+cu124
torchvision==0.21.0+cu124
torchaudio==2.6.0+cu124
transformers==5.17.0
peft==0.20.0
datasets==5.0.1
accelerate==1.15.0
trl==1.13.0
bitsandbytes==0.50.2     # >=0.45 才自带 cuda124 的 Windows 预编译 DLL
safetensors==0.8.0
sentencepiece==0.2.2
tokenizers==0.23.2
numpy==2.5.2
sympy==1.13.1
scipy==1.18.1
xformers==0.0.35
pandas==3.0.5
```

### 安装依赖

> 🔴 **本节是最容易把环境搞坏的地方，务必按下面写的一条命令装完。**

CUDA 版 torch 在 PyPI 主源上没有（那里是 CPU 版），必须去 PyTorch 官方源找。**但绝对不要"分两次、用两个源"装**：

```powershell
# ❌ 错误示范（会装出"缝合怪"，见第 9 章坑 1）
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么坏**：第二条命令换了索引源，而 `requirements.txt` 里的 `peft`/`trl` 都依赖 torch —— uv 会在清华源里**再解析出一个 torch** 覆盖上去。覆盖不干净，新旧文件混在同一个 `torch\` 目录里，`import torch` 当场崩，但 `pip list` 显示一切正常。

✅ **正确做法：一条命令装完，并把 torch 锁死。**

**第一步**：新建 `constraints.txt`

```text
# 锁死 torch 版本：无论走哪个索引源，都不许动 torch
torch==2.6.0+cu124
torchvision==0.21.0+cu124
torchaudio==2.6.0+cu124
```

**第二步**：一条命令装完

```powershell
cd E:\lorademo
D:\DevEnv\uv\bin\uv.exe pip install -r requirements.txt -c constraints.txt `
  -i https://pypi.tuna.tsinghua.edu.cn/simple `
  --extra-index-url https://download.pytorch.org/whl/cu124 `
  --index-strategy unsafe-best-match `
  -p "E:\lorademo\910demo\Scripts\python.exe"
```

**四个参数缺一不可**：

| 参数 | 作用 | 删掉会怎样 |
| --- | --- | --- |
| `-c constraints.txt` | **锁版本**。torch 只可能被解析成一次，从根上杜绝缝合怪 | 退化成上面的错误示范 |
| `--extra-index-url .../cu124` | 多给一个源，让 CUDA 版 torch 能被找到 | 清华源只有 CPU 版 torch |
| `--index-strategy unsafe-best-match` | 允许多源混查取"最优匹配" | uv 因源冲突直接报错退出 |
| `-p <venv 里的 python>` | 明确指定装进哪个环境 | 可能装到你系统 Python 里去 |

> ⚠️ **这条命令很慢，必须后台跑。** `torch+cu124` 的 wheel **约 2.4GB**，只在境外源上。
> 国内直连可能几十分钟，**前台跑会被掐断**（表现是"命令没输出就结束了"，让你以为失败）。
> **开代理能快几十倍**，如果还有下次重建，先把代理开上。

### 2.4 环境体检（**必做，别跳过**）

搭完环境先跑这个，很多"微调失败"其实是环境本来就有问题。

`check_env.py`：

```python
import sys
print("Python:", sys.version.split()[0])

import torch
print("torch:", torch.__version__)
print("cuda 可用:", torch.cuda.is_available(), "| CUDA 版本:", torch.version.cuda)
if torch.cuda.is_available():
    print("显卡:", torch.cuda.get_device_name(0))
    print("显存: %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1024**3))

import transformers, peft, datasets, accelerate, bitsandbytes
print("transformers:", transformers.__version__)
print("peft:", peft.__version__)
print("datasets:", datasets.__version__)
print("accelerate:", accelerate.__version__)
print("bitsandbytes:", bitsandbytes.__version__)

from pathlib import Path
p = Path(r"D:\Model\Qwen2.5-1.5B-Instruct")
print("底座模型存在:", p.exists(), "文件数:", len(list(p.iterdir())) if p.exists() else 0)
```

跑：

```powershell
E:\lorademo\910demo\Scripts\python.exe check_env.py
```

> ℹ️ **本机实测的正确输出**（照这个比对）：
> ```text
> Python: 3.13.9
> torch: 2.6.0+cu124
> cuda 可用: True | CUDA 版本: 12.4
> 显卡: NVIDIA GeForce RTX 2060
> 显存: 6.0 GB
> transformers: 5.17.0
> peft: 0.20.0
> datasets: 5.0.1
> accelerate: 1.15.0
> bitsandbytes: 0.50.2
> 底座模型存在: True 文件数: 13
> ```

**`cuda 可用: True` 是硬指标。** 如果是 `False`，说明装成了 CPU 版 torch —— **回去重装，不要往前走**（CPU 训练 1.5B 模型要几十小时）。

> ⚠️ **注意：`import torch` 成功 ≠ 环境没问题。**
> 第 9 章的坑 1（sympy 污染）和坑 2（bnb 缺 DLL）都发生在 `import` 阶段，
> **只有真的 `import` 过一遍才算数** —— 这正是必须有这个体检脚本的原因。

### 2.5 模型权重获取（本地优先）

最稳的做法是**不用库下载，直接用本地目录**：

```python
MODEL_CANDIDATES = [
    r"D:\Model\Qwen2.5-1.5B-Instruct",                              # ① 本地（首选）
    os.path.join(PROJECT_ROOT, "models", "Qwen2.5-1.5B-Instruct"),  # ② 工程内（备选）
]
LOCAL_MODEL_DIR = next((p for p in MODEL_CANDIDATES if os.path.isdir(p)), None)
MODEL_NAME = LOCAL_MODEL_DIR if LOCAL_MODEL_DIR else "Qwen/Qwen2.5-1.5B-Instruct"
```

本地有就用本地（离线可复现、不吃 C 盘缓存、加载快）；都没有才回退到在线下载。

真要在线下载，国内先设镜像：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

> 🔴 **但要注意第 9 章坑 5**：本机的 `huggingface_hub 1.31.0` 默认启用 Xet 协议，
> 在镜像站下会**静默失败留下 0 字节文件**，而且 `curl` 下载是正常的 —— 问题在库不在网络。

---

## 第 3 章 里程碑 1：造数据集

**这一步的目标不是"数据多好"，是"1 分钟能重跑一次"。** 试错成本低，才敢乱调。

### 3.1 定义人设

`scripts/01_make_dataset.py` 顶部：

```python
SYSTEM_PROMPT = (
    "你是星尘科技的客服小星，回答时："
    "1) 开头必须说『星尘科技为您服务～』；"      # ← 最容易观察到的痕迹
    "2) 语气亲切活泼，句尾常用『呢』『哦』；"    # ← 测口头禅
    "3) 涉及价格时必须强调『七天无理由退换』；"  # ← 测条件触发
    "4) 回答不超过 60 字。"                     # ← 测长度控制（最能体现 LoRA 价值）
)
```

**四条规则是故意这样设计的**，每一条对应一种可测量的能力：

| 规则 | 测什么 | 怎么判断成功 |
| --- | --- | --- |
| 1 固定开头语 | 格式遵循 | 回答是否 100% 带开头语 |
| 2 语气词 | 风格模仿 | 是否稳定出现"呢/哦" |
| 3 价格必提退换 | **条件触发** | 只问价格时触发，问别的时不触发 |
| 4 不超 60 字 | **长度控制** | 原模型经常超，LoRA 应稳定在 20~30 字 |

### 3.2 造 25 对问答

```python
QA_PAIRS = [
    ("你们几点上班？", "星尘科技为您服务～我们是早上 9 点到晚上 9 点全年无休呢，随时可以找我们哦～"),
    ("这个多少钱？",   "星尘科技为您服务～这款是 199 元，而且七天无理由退换哦，放心买呢～"),
    # ...（共 20 条常规问法）
    # 下面几条换一种问法，让模型学会"泛化"而不是死记
    ("想问一下上班时间", "星尘科技为您服务～早上 9 点到晚上 9 点哦，全年无休呢～"),
    ("这个东西能退不",   "星尘科技为您服务～可以的，七天无理由退换呢，放心买哦～"),
]
```

> 💡 **关键技巧：同一件事要用不同问法问几遍。**
> 只给"你们几点上班？"这一个问法，模型很容易学成"看到这串字就吐那句答案"（死记）。
> 换成"想问一下上班时间"后，它必须学**意图**才能答对 —— 这才是泛化。

### 3.3 数据格式：ChatML

```json
{"messages": [
  {"role": "system",    "content": "你是星尘科技的客服小星……"},
  {"role": "user",      "content": "你们几点上班？"},
  {"role": "assistant", "content": "星尘科技为您服务～……"}
]}
```

**只有 `assistant` 那条参与计算 loss**（下一章会讲怎么做）。

### 3.4 切分与写盘

```python
random.seed(42)                     # 固定种子 → 每次生成的数据完全一样（可复现）
random.shuffle(samples)             # 打乱顺序

n_eval = 4
eval_samples = samples[:n_eval]     # 前 4 条 → 验证集
train_samples = samples[n_eval:]    # 剩下 21 条 → 训练集
```

跑：

```powershell
E:\lorademo\910demo\Scripts\python.exe E:\lorademo\scripts\01_make_dataset.py
```

**期望输出**：训练集 21 条 / 验证集 4 条。

---

## 第 4 章 里程碑 2：跑通训练（核心）

### 4.1 加载 tokenizer

```python
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,      # Qwen 系列需要
    padding_side="right",        # ★ 因果 LM 必须右侧 padding，否则 loss 算错
)
if tokenizer.pad_token is None:          # ★ Qwen 默认没有 pad_token
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
```

> ⚠️ **`padding_side="right"` 和 `pad_token` 这两个是最常见的坑。**
> 左侧 padding 会让因果 LM 的 loss 计算错位；不设 pad_token 会直接报错或行为诡异。

### 4.2 4-bit 加载模型

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                       # 权重压成 4-bit，显存降到约 1/4
    bnb_4bit_quant_type="nf4",               # NF4：QLoRA 论文的格式，比普通 fp4 精度高
    bnb_4bit_compute_dtype=torch.float16,    # ★ Turing 卡（RTX 2060）不能用 bf16
    bnb_4bit_use_double_quant=True,          # 双重量化，白赚约 0.5bit/参数
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb_config,
    device_map="auto", trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)   # ★ QLoRA 必须
model.config.use_cache = False                   # ★ 与梯度检查点冲突，必须关
```

> 📌 **实测显存（RTX 2060 6GB）**：4-bit 加载后占 **1.07 GB**（峰值 1.10 GB），卡上还剩 **4.90 GB**。
> 记住这个数 —— 下一节 `batch_size` 只能开 1、优化器只能上 `paged_adamw_8bit`，根子都在这 4.9GB 余量上。

### 4.3 注入 LoRA 适配器

```python
lora_config = LoraConfig(
    r=8,                                                      # 低秩维度
    lora_alpha=16,                                            # 等效强度 = alpha/r = 2
    lora_dropout=0.05,                                        # 惯例：小数据要正则，21 条可提到 0.1
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 注意力 4 个投影层
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

**本机实测输出**：

```text
trainable params: 2,179,072 || all params: 1,545,893,376 || trainable%: 0.1410
```

**看到 `0.141%` 就对了** —— 这就是 LoRA 省显存的直接证据。

> 📌 **`r=8` 是怎么定的？参数量与 `r` 严格成正比，且可以手算验证：**
>
> | r | 参数量 | 占总参数 | 相对 r=8 |
> | --- | --- | --- | --- |
> | 4 | 1,089,536 | 0.0705% | 0.5× |
> | **8** | **2,179,072** | **0.1410%** | **1.0×** |
> | 16 | 4,358,144 | 0.2819% | 2.0× |
> | 32 | 8,716,288 | 0.5638% | 4.0× |
>
> ```text
> Qwen2.5-1.5B 结构：hidden=1536, layers=28, heads=12, kv_heads=2, head_dim=128
>         → q_proj/o_proj: 1536→1536     k_proj/v_proj: 1536→256
> 单层 LoRA 参数 = r×输入维 + r×输出维，按 r=8 累加 28 层 = 2,179,072  ✅ 与实测一致
> ```
>
> **选 r 只看任务和数据量**：风格/格式对齐（本 demo）→ **8** 就够；知识注入/领域问答 → 16~32；数据 >1000 条 → 16 起。
>
> ⚠️ **`lora_alpha` 和 `learning_rate` 是等效的，调参只动一个。**
> 梯度会被 `alpha/r` 缩放 —— alpha 从 16 改 32，效果约等于 lr 翻倍。
>
> **`learning_rate=2e-4` 该升还是该降（惯用区间 1e-4 ~ 3e-4）：**
>
> | 现象 | 动作 |
> | --- | --- |
> | loss 震荡不降 | 降到 **1e-4** |
> | 降得极慢、收敛不了 | 升到 **3e-4** |
> | 输出开始复读训练集原句 | 降 lr 或减轮数 |

### 4.4 labels 屏蔽（**最容易写错的地方**）

我们要模型学会"看到问题后该怎么答"，**不是**学"怎么生成用户的问题"。所以只对 assistant 部分算 loss：

```python
def tokenize_fn(example):
    messages = example["messages"]

    # ① 整段对话（含回答）
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)

    # ② 只到 assistant 开始处为止
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True)

    # ③ tokenize
    full   = tokenizer(full_text,   truncation=True, max_length=MAX_SEQ_LEN, padding=False)
    prompt = tokenizer(prompt_text, truncation=True, max_length=MAX_SEQ_LEN, padding=False)

    input_ids = full["input_ids"]
    labels = list(input_ids)

    # ④ prompt 部分标成 -100，不参与 loss
    prompt_len = min(len(prompt["input_ids"]), len(labels))
    for i in range(prompt_len):
        labels[i] = -100

    return {"input_ids": input_ids,
            "attention_mask": full["attention_mask"],
            "labels": labels}
```

**`-100` 是 PyTorch 交叉熵的"忽略索引"**，这些位置不产生梯度。

> ✅ **必须自检**：打印 labels 里 `-100` 的数量，**为 0 就说明屏蔽失败了**。
> 后果是 loss 一直为 0 或 nan（模型在学"怎么复读 prompt"）。
> 本工程脚本里已内置这个自检并会打警告。

> 📌 **顺带说 `MAX_SEQ_LEN=512`（上面 `truncation` 用的那个上限）—— 它其实没有依据，是随手给的保险值。**
> 实测 25 条样本：**最长 126 token**（prompt 91 + 答案 35），512 是它的 **4.1 倍**；严格说 **128 就够**。
> 好在 `batch_size=1` 时 padding 只到样本自身长度，512 只是"截断上限"而非实际占用，**留着不花任何代价**。
> 但**改了数据集一定要重新量一次最长样本**，别让新数据被无声截断：
>
> ```python
> from transformers import AutoTokenizer
> tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
> longest = max(len(tok(tok.apply_chat_template(json.loads(l)["messages"],
>             tokenize=False))["input_ids"]) for l in open("data/train.jsonl", encoding="utf-8"))
> print("最长样本:", longest, "token")   # 设 MAX_SEQ_LEN 时 ≥ 它即可
> ```

### 4.5 训练参数（6GB 卡的生存配置）

```python
TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,       # ★ 硬件：4-bit 加载后只剩 4.9GB，开 2 就 OOM
    gradient_accumulation_steps=8,       # 折中：有效 batch 补到 8，同时保证每轮 ≥3 步（见下方⚠️）
    gradient_checkpointing=True,         # ★ 硬件：激活值省 50%+，6GB 卡必开（代价：慢 20~30%）
    learning_rate=2e-4,                  # 惯例：LoRA 经验区间 1e-4~3e-4 的中值
    lr_scheduler_type="cosine",          # 余弦退火
    warmup_steps=1,                      # ★ API：transformers 5.x 只能用绝对步数（总步数 9 × 10%）
    num_train_epochs=3,                  # ★ 停止条件就是"跑满 3 轮"，跟 loss 无关（见下方⚠️）
    logging_steps=1,                     # 总步数只有 9，全记才能看到完整曲线
    save_strategy="epoch",
    eval_strategy="epoch",               # 每轮跑验证，观察过拟合
    save_total_limit=2,                  # 每份 checkpoint 24.4MB，留多了白占磁盘
    fp16=True,                           # ★ 硬件：Turing 卡只能用 fp16
    bf16=False,                          # ★ RTX 2060 不支持 bf16，必须 False
    optim="paged_adamw_8bit",            # ★ 硬件：优化器状态 8-bit 存，再省一大截
    report_to="none",
    seed=42,                             # 惯例：可复现，无技术含义
)
```

> 🔴 **`warmup_ratio` 在 transformers 5.x 里已被移除**，只能用绝对步数。
> 换算：`总步数 = (样本数 / (batch × 累积)) × epoch 数` → 本例 = (21 / 8) × 3 ≈ **9 步**，取 10% ≈ **1 步**。

> ⚠️ **训练什么时候停？—— 跑满 3 轮就停，跟 loss 无关。**
> trainer 只支持三种停法：**按轮数**（`num_train_epochs`，本工程用的）/ **按步数**（`max_steps`，设了会覆盖轮数）/ **早停**（`EarlyStoppingCallback`，盯 `eval_loss` 不再改善就停）。
> **没有"loss 降到某个值就停"这个参数** —— 而且不该这么做：`train loss` 的绝对值取决于数据和序列长度，换个数据集就完全不是一回事，固定阈值没法复用；该看的是**验证 loss**。
> **为什么偏偏是 3 轮**：21 条 × 3 轮 = 9 步 = 59 秒，demo 的目标是"链路通"而不是"训到最好"；实测两次 run 的 `eval_loss` 都还在降（1.87→1.72→1.69），说明 3 轮远没到该停的时候。
> 真要做对：**把 epoch 设大（15~20）+ 早停 + `load_best_model_at_end=True`**，让 `eval_loss` 自己决定何时停、存哪一轮。
>
> ```python
> # 早停写法：三行配置 + 一个 callback
> TrainingArguments(..., eval_strategy="epoch",
>                   load_best_model_at_end=True,          # 停下时自动回滚到最好那轮
>                   metric_for_best_model="eval_loss",
>                   greater_is_better=False)
> Trainer(..., callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
> ```
>
> ⚠️ **`early_stopping_patience` 不是 `TrainingArguments` 的参数**，它属于 `EarlyStoppingCallback`；
> 写错位置会直接报 `unexpected keyword argument`（本机实测确认）。
> ⚠️ 本工程 **验证集是算了但没用**（`trainer_state.json` 里 `best_metric: None`）—— 也就是说存的是"最后一轮"而不是"验证最好那轮"。详见第 9 章坑 17。

> 📌 **参数依据速查（每个数字从哪来）** —— 分三类：**① 硬件**（能不能跑，硬上限）/ **② 实测**（能用量脚本算出来）/ **③ 惯例**（经验区间，可调）。
>
> | 参数 | 值 | 类 | 依据 |
> | --- | --- | --- | --- |
> | `MAX_SEQ_LEN` | 512 | ② | 实测最长 126 → **512 是 4.1× 保险值，128 就够** |
> | `BATCH_SIZE` | 1 | ① | 加载后占 1.07GB，只剩 4.9GB 余量 |
> | `GRAD_ACCUM` | 8 | ②③ | 有效 batch 补到 8，且每轮 ≥3 步 |
> | `USE_4BIT` | True | ① | fp16 权重 3GB → 4-bit 后 1.07GB |
> | `LEARNING_RATE` | 2e-4 | ③ | LoRA 区间 1e-4~3e-4 中值 |
> | `r` | 8 | ③ | 风格类小数据；参数量占 0.141% |
> | `lora_alpha` | 16 | ③ | `=2r`，等效强度 `alpha/r=2` |
> | `lora_dropout` | 0.05 | ③ | 小数据要正则 |
> | `NUM_EPOCHS` | 3 | ② | 21 条 × 3 轮 = 9 步 = 59 秒 |
> | `warmup_steps` | 1 | ② | 总步数 9 × 10% |
> | `save_total_limit` | 2 | ① | 每份 checkpoint 24.4MB |
>
> ⚠️ **`GRAD_ACCUM=8` 的代价**：每轮只剩 **3 步**，loss 曲线很粗 —— 这是"有效 batch 要稳"和"曲线要细"的权衡。想细一点改成 **4**（每轮 6 步，有效 batch 4）。数据上了几百条后这个矛盾自动消失。
> **原则：能实测的不要拍脑袋，硬件上限不要让，惯例值只当起点。**

### 4.6 跑起来

```powershell
E:\lorademo\910demo\Scripts\python.exe E:\lorademo\scripts\02_train_lora.py
```

**本机实测结果**：

```text
global_step = 9,  max_steps = 9        ← 完整跑完 ✅
trainable params: 2,179,072 (0.1410%)
```

| step | epoch | train loss | eval_loss |
| --- | --- | --- | --- |
| 1 | 0.38 | 2.2551 | — |
| 3 | 1.00 | 1.8440 | **1.8655** |
| 6 | 2.00 | 1.7016 | **1.7200** |
| 9 | 3.00 | **1.3737** | **1.6866** |

- 耗时 **59 秒**（21 条 × 3 轮 × RTX 2060）
- 产物 `adapter_model.safetensors` **8.3 MB**

**怎么读这张表**：
- train loss 2.26 → 1.37，稳定下降 ✅
- eval_loss 1.87 → 1.72 → 1.69，**也在降**，说明**没有过拟合** ✅
- 如果出现"train loss 继续降、eval_loss 开始涨" —— 那就是过拟合信号，该减 epoch 或加数据了

> ❓ **"那训练到底什么时候停？是不是 loss 降到某个值就停？"**
> 不是。本工程是**跑满 3 轮就停**（`num_train_epochs`），跟 loss 无关 —— 三种停法对比、早停写法、为什么是 3 轮，都在上面 §4.5 的 ⚠️ 标签里。

> 📌 **那"正确的轮数"怎么定？看 `eval_loss` 的拐点，不是看 train loss。** 本机两次实验的 eval_loss：
>
> | epoch | 3 轮 run | 10 轮 run（只跑到第 6 轮） |
> | --- | --- | --- |
> | 1 | 1.8655 | 1.9513 |
> | 2 | 1.7200 | 1.7201 |
> | 3 | **1.6866** | 1.5579 |
> | 6 | — | **1.3442** |
>
> 两次**都还在降、一次反弹都没有** → 3 轮远没到该停的时候。正确做法是 **epoch 设大（15~20）+ 早停**，让它自己在反弹时停。
> ⚠️ 但这里只有 **4 条**验证样本，噪声很大（错 1 条就是 25% 抖动）—— 想靠它做早停判断，先把验证集扩到 20 条以上。

### 4.7 一个真实的观察：warmup_steps 会影响 loss 曲线形状

对比 `lora_qwen_cs`（3 轮，`warmup_steps=1`）和 `lora_qwen_cs_ep10`（`warmup_steps=3`）：

| step | 3 轮 run 的 loss | 10 轮 run 的 loss | 说明 |
| --- | --- | --- | --- |
| 1 | 2.2551 | 2.2551 | **完全相同**（同一 seed、同一数据顺序） |
| 2 | 2.0958 | 2.0958 | **完全相同** |
| 3 | 1.8440 | 1.9294 | ← **从这里开始分叉** |

**为什么前两步一样、第三步不同**：前两步学习率还在 warmup 爬升阶段（都接近 0），差异不明显；到第 3 步两个 run 的 LR 调度曲线已经错开，梯度更新量不同，loss 自然分叉。

> **这说明什么**：改了 `warmup_steps` 之后，**不要拿新 run 的前几步 loss 去和老 run 比**，它们不可比。
> 要比就看**同一 epoch 末的 eval_loss**。

### 4.8 ⚠️ 10 轮实验为什么只有 checkpoint-12/15/18

工程里 `02b_train_longer.py` 设的是 `NUM_EPOCHS=10`（`max_steps=30`），但 `checkpoint-18` 的 `trainer_state.json` 显示 **`global_step=18`、`epoch=6.0`** —— 也就是说**它只跑到第 6 轮就中断了**（进度 60%）。

后果：
- `output\lora_qwen_cs_ep10\` **根目录下没有 `adapter_config.json`**（那个文件只在训练正常结束时才写）
- 所以想用这个实验的适配器，**必须指向 `checkpoint-18`**，不能指向目录本身：

```powershell
.\910demo\Scripts\python.exe scripts\06_chat.py --lora "output\lora_qwen_cs_ep10\checkpoint-18"
```

**这是一个非常典型的坑**（见第 9 章坑 11）：训练中途 Ctrl+C 或崩溃后，产物只剩 `checkpoint-N` 子目录，指向父目录会报"找不到 adapter_config.json"。

**经验**：训练中断后先去看 `trainer_state.json` 里的 `global_step` vs `max_steps`，就知道跑到哪了。

---

## 第 5 章 里程碑 3：推理对比

**不对比就等于没验证。** 这个脚本同时跑原始模型和 LoRA 模型，把回答并排打出来。

```powershell
# 对比模式（默认）
E:\lorademo\910demo\Scripts\python.exe E:\lorademo\scripts\03_inference.py

# 只跑 LoRA
E:\lorademo\910demo\Scripts\python.exe E:\lorademo\scripts\03_inference.py --lora-only

# 换自己的问题
E:\lorademo\910demo\Scripts\python.exe E:\lorademo\scripts\03_inference.py --question "能退货吗"
```

### ★★ 最大的坑：`PeftModel` 会原地修改底座

这是本工程调试时**真实踩过**的坑，也是所有 LoRA 教程里最少被提到的：

```python
# ❌ 错误写法
base = load_base_model()
lora = PeftModel.from_pretrained(base, LORA_DIR)
# base 已经被原地注入 LoRA 层了！base 和 lora 其实是同一个模型
# → 两次生成结果一模一样，你会误判"LoRA 没生效"
```

```python
# ✅ 正确写法：加载两份互相独立的底座
base = load_base_model()                                    # ① 保持原始
base.eval()

lora = load_base_model()                                    # ② 全新一份底座
lora = PeftModel.from_pretrained(lora, LORA_DIR)            #    只给这份挂 LoRA
lora.eval()

# 自检
has_lora = any("lora_" in n for n, _ in lora_model.named_parameters())
```

> 💡 判断依据：`PeftModel.from_pretrained(base, path)` **把 LoRA 层直接插进 `base` 的模块里**，然后返回包装后的同一个对象。
> 代价是两份模型各占一份显存 —— 这正是第 9 章坑 6（内存不足）的来源。

### 本机实测效果对比（★核心证据）

| 问题 | 原始模型 | LoRA 模型 |
| --- | --- | --- |
| 你们几点上班？ | 143 字，啰嗦跑题 | **29 字**「我们是早上九点到晚上十一点，随时在线哦～」 |
| 这个多少钱？ | 108 字，回避具体数字 | **30 字**「这款手机壳是 99 元，七天无理由退换哦～」 |
| 能退货吗？ | 48 字，格式散乱 | **23 字**「当然可以，七天无理由退换哦～」 |

**LoRA 真正学到的是「简洁 + 固定格式 + 固定人设」**。
原始模型虽然也懂指令，但**输出长度失控、格式漂移** —— 这才是两者最本质的差异。

---

## 第 6 章 里程碑 4：消融实验

### 6.1 这个实验回答什么问题

微调后模型表现好，到底是：
- **A)** LoRA 真的改变了权重，行为已成习惯？
- **B)** 还是只是因为它读懂了 system prompt，随便演了演？

**区分方法：故意不传 system prompt，看它还保不保持人设。**

```powershell
E:\lorademo\910demo\Scripts\python.exe E:\lorademo\scripts\05_ablation.py

# 换别的适配器
E:\lorademo\910demo\Scripts\python.exe E:\lorademo\scripts\05_ablation.py --lora output/lora_qwen_cs_ep10/checkpoint-18
```

脚本会跑 4 种组合 × 4 个问题，并按特征词（"星尘科技/呢/哦/～"）打分：

| 组合 | 本机实测结果 |
| --- | --- |
| LoRA + system | ✅ 人设完整（23~30 字） |
| LoRA **- system** | ❌ **退化成普通助手** |
| 原始 + system | ⚠️ 有人设但冗长（48~143 字） |
| 原始 - system | ❌ 普通助手 |

### 6.2 结论（重要，别跳过）

**21 条数据 × 3~6 轮，LoRA 学到的是「system 条件下的强关联」，还没有把行为彻底内化进权重。**

> 这在数据量小时是**正常现象，不是 bug**。别以为训练失败了。

想让它"无条件也带人设"，需要三件事同时做：

| 手段 | 具体做法 |
| --- | --- |
| 加数据量 | 几百条起（21 条太少了） |
| 加 epoch | 配合观察验证 loss 的拐点 |
| **数据里混入"无 system"样本** | 让模型在"没有提示词"的条件下也见到正确回答 |

> ⚠️ **注意**：第 3 条是关键。只在"有 system"的条件下训练，模型自然只学会"条件反射"。

---

## 第 7 章 里程碑 5：交互式对话

训练脚本跑完，你多半会想"我直接跟它聊两句"。`06_chat.py` 就是干这个的。

```powershell
# 和 LoRA 微调后的模型聊天（默认）
.\910demo\Scripts\python.exe scripts\06_chat.py

# 和原始模型聊天（对照感受差异）
.\910demo\Scripts\python.exe scripts\06_chat.py --mode base

# 指定别的适配器
.\910demo\Scripts\python.exe scripts\06_chat.py --lora "output\lora_qwen_cs_ep10\checkpoint-18"

# 不带 system prompt（验证人设是否内化）
.\910demo\Scripts\python.exe scripts\06_chat.py --no-system

# 贪心解码（每次答案相同，便于复现问题）
.\910demo\Scripts\python.exe scripts\06_chat.py --greedy
```

**参数表**：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--mode` | `lora` | `base`=原始模型 / `lora`=微调模型 |
| `--lora` | `output/lora_qwen_cs` | LoRA 适配器目录 |
| `--no-system` | 关 | 不发送 system prompt |
| `--temp` | `0.7` | temperature（0.1 保守 / 0.7 平衡 / 1.0+ 发散） |
| `--max-tokens` | `120` | 单次生成上限 |
| `--greedy` | 关 | 贪心解码，结果固定 |

**窗口内指令**：

| 指令 | 作用 |
| --- | --- |
| `/help` | 显示帮助 |
| `/clear` | 清空对话历史（**显存不足时用它**） |
| `/system` | 查看当前 system prompt |
| `/mode` | 查看当前模式与适配器路径 |
| `/params` | 查看生成参数与历史轮数 |
| `/exit` | 退出 |

> 💡 **支持多轮上下文**：脚本把历史 messages 整体传给 `apply_chat_template`。
> 但历史越长越吃显存，感觉变慢时用 `/clear` 清一下。
> 脚本也捕获了 `torch.cuda.OutOfMemoryError`，会提示你清空并**回滚**刚加入的那条消息。

---

## 第 8 章 里程碑 6：合并导出与部署

### 8.1 为什么要合并

LoRA 适配器是"外挂"，推理时要额外加载。合并就是把 `B·A` 算出来直接加回原权重：

```
W_new = W + (alpha/r)·B·A
```

**什么时候需要合并**：

| 场景 | 要合并吗 | 原因 |
| --- | --- | --- |
| 本地调试 | ❌ 不需要 | `03_inference.py` 直接挂适配器更方便 |
| 部署到 vLLM / Ollama / TensorRT-LLM | ✅ 需要 | 这些框架不认 LoRA 格式 |
| 分享给别人 | ✅ 需要 | 一个目录搞定，不用附带"记得装 peft" |

### 8.2 ★ 合并必须用 fp16 加载

```python
# ❌ 不能这么干
base = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb_4bit)

# ✅ 正确做法
base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,     # 全精度，不用 4-bit
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base, LORA_DIR).merge_and_unload()
model.save_pretrained(MERGED_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_DIR)     # tokenizer 也要一起存
```

> 🔴 **为什么不能 4-bit 合并**：4-bit 是**有损压缩**。把 LoRA 增量合并进有损表示里会造成额外精度损失。
> 正确流程：fp16 加载 → 合并 → 保存 fp16 完整模型。
> 显存不够就用 `device_map="cpu"` 在内存里合并（慢但可行）。

跑：

```powershell
E:\lorademo\910demo\Scripts\python.exe E:\lorademo\scripts\04_merge_lora.py
```

**产物**：`output\merged_qwen_cs\`（**2.89 GB**），里面有 `model.safetensors` + `config.json` + tokenizer 全套。

### 8.3 部署方式对照

| 方式 | 适用场景 | 命令 / 要点 |
| --- | --- | --- |
| 适配器直载 | 本地测试 | 直接用 `03_inference.py` |
| 合并 + transformers | 简单服务 | 先跑 `04_merge_lora.py` |
| vLLM | 高并发 API | 需要合并后的模型；**Windows 支持较差，建议 Linux** |
| Ollama | 个人使用 | 需要转 GGUF 格式 |
| TensorRT-LLM | 极致性能 | 本机已配 10.14，可后续尝试 |

---

## 第 9 章 本机高频坑清单（17 个）

**这是这份文档最重要的部分。** 按"能不能把环境搞崩"排序，前 3 个最严重。

### 坑 1 🔴 sympy 被污染 → 整个 torch 链崩

**症状**：`import torch` 直接失败

```text
ImportError: cannot import name '__version__' from 'sympy.release'
```

**根因**：`910demo\Lib\site-packages\sympy\release.py` 的内容被替换成了 **2048 行 `# PROBE-TEST-12345`**（某种探针测试残留）。正常这个文件只有一行。

**为什么难发现**：报错指向 sympy，但真正想用的 torch 起不来，容易误判成"torch 装坏了"然后去重装 torch（没用）。

**诊断**：

```powershell
type E:\lorademo\910demo\Lib\site-packages\sympy\release.py
# 正常只有: __version__ = "1.13.1"
```

**修复**（只替换单个文件，**不要重装整个 sympy**）：

```python
import zipfile
# 下载官方 wheel 后，从中提取正确的文件
with zipfile.ZipFile("sympy-1.13.1-py3-none-any.whl") as z:
    data = z.read("sympy/release.py")
open(r"E:\lorademo\910demo\Lib\site-packages\sympy\release.py", "wb").write(data)
```

**本机处理**：损坏文件备份为 `release.py.broken_bak`。

---

### 坑 2 🔴 bitsandbytes 缺对应 CUDA 版本的 DLL → Python 进程硬崩

**症状**：

```text
CUDA SETUP: Required library version not found: libbitsandbytes_cuda124.dll
```

**根因**：bnb 0.41.1（第三方 Windows 打包版）最高只提供到 cuda122，而本机 torch 是 cu124。

**⚠️ 为什么这个坑特别坑**：**它会让整个 Python 进程硬崩**，连带 `import peft` 一起失败 —— 极易被误判成"peft 装了坏版本"。

**修复**：升级到官方 `bitsandbytes==0.50.2`（自带 cuda12x 预编译 DLL）。

```powershell
D:\DevEnv\uv\bin\uv.exe pip install --python E:\lorademo\910demo\Scripts\python.exe `
  bitsandbytes-0.50.2-py3-none-win_amd64.whl --no-deps
```

> ⚠️ **改 wheel 文件名时必须保留完整格式** `name-ver-py3-none-win_amd64.whl`。
> 缺了 ABI tag（`py3-none`）pip/uv 会报 `Must have an ABI tag`。这是手动装 wheel 最常犯的错。

---

### 坑 3 🔴 PeftModel 原地修改底座 → 误判"LoRA 没生效"

**症状**：`03_inference.py` 里原始模型和 LoRA 模型**输出一模一样**。

**根因**：`PeftModel.from_pretrained(base, path)` 会**原地修改 `base`**（把 LoRA 层插进去），然后返回同一个对象。

```python
base = load()
lora = PeftModel.from_pretrained(base, LORA_DIR)   # ❌ base 也被改了
```

**修复**：**加载两份互相独立的底座**（详见第 5 章）。

---

### 坑 4 🟠 transformers 5.x 移除了 `warmup_ratio`

**症状**：

```text
TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'warmup_ratio'
```

**根因**：本机是 transformers **5.17.0**，网上教程绝大多数基于 4.x。

**修复**：改用绝对步数

```python
# warmup_ratio=0.1   ❌ 已移除
warmup_steps=1        # ✅ 总步数的 ~10%
# 总步数 = (样本数 / (batch × 累积)) × epoch 数 = (21 / 8) × 3 ≈ 9 步
```

**5.x 的其他变更**：

| 4.x 写法 | 5.x 写法 |
| --- | --- |
| `warmup_ratio=0.1` | ❌ 移除 → `warmup_steps=<绝对步数>` |
| `evaluation_strategy` | → `eval_strategy` |
| `torch_dtype` | → 建议 `dtype`（旧名仍可用，仅警告） |

> 💡 **升级前先校验参数合法性**（避免跑到一半才报错）：
> ```python
> import inspect
> from transformers import TrainingArguments
> sig = set(inspect.signature(TrainingArguments.__init__).parameters)
> print([p for p in your_params if p not in sig])
> ```

---

### 坑 5 🟠 huggingface_hub 的 Xet 协议留下 0 字节文件

**症状**：

```text
OSError: config file ... is not a valid JSON file
```

检查发现 `config.json` 是 **0 字节**。

**根因**：本机 `huggingface_hub 1.31.0` 默认启用 **Xet 传输协议**，该协议在镜像站/受限网络下会**静默失败并留下空文件**。

**⚠️ 关键线索**：`curl` 直接下载是**完全正常**的 —— 所以问题在**库**不在**网络**。这个线索能帮你省掉大量排查网络的时间。

**修复（三选一）**：

1. **首选：用本地目录**（本工程的方案，彻底绕开下载）
2. 设 `HF_HUB_DISABLE_XET=1` 让库走传统 HTTP
3. `curl` 手动下载全部权重到本地目录

---

### 坑 6 🟠 内存不足：`OSError: 页面文件太小 (os error 1455)`

**症状**：跑 `03_inference.py` / `07_why_same.py` 时崩。

**根因**：本机物理内存 15.4GB，PyCharm 常驻 2.7GB。而"加载两份模型"的脚本瞬间要吃掉十几 GB（不加 `low_cpu_mem_usage` 时会在 CPU 内存里出现完整 fp32 副本）。

**三步处理**：

**① 加 `low_cpu_mem_usage=True`**

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb, device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,     # ★ 分块加载，避免内存峰值爆炸
)
```

**② 改串行模式，绝不并行**

```python
# ✅ 加载 A → 跑完 → 彻底释放 → 加载 B → 跑
def release_model(model):
    del model                 # ① 删引用
    gc.collect()              # ② 回收 CPU 内存
    torch.cuda.empty_cache()  # ③ 清空显存缓存池
    torch.cuda.synchronize()  # ④ 等 CUDA 操作完成，确保显存真的归还
```

**③ 清理僵死进程**

OOM 崩溃后 Python 进程常会**僵住不退出**，白占 1~2GB。

**本机真实案例**：脚本 OOM 崩溃后，留下一个 PID 24352 的 `python.exe` 占着 **1.5GB**。
查询 `tasklist` 看到一堆 python 进程，怎么确认哪个是"僵尸"？—— **比对启动时间**：

```python
import ctypes, datetime
from ctypes import wintypes
k = ctypes.windll.kernel32
h = k.OpenProcess(0x1000, False, PID)          # QUERY_LIMITED_INFORMATION
ct, et, kt, ut = (wintypes.FILETIME() for _ in range(4))
k.GetProcessTimes(h, ctypes.byref(ct), ctypes.byref(et),
                     ctypes.byref(kt), ctypes.byref(ut))
v = (ct.dwHighDateTime << 32) | ct.dwLowDateTime
print(datetime.datetime(1601,1,1) + datetime.timedelta(microseconds=v/10))
```

启动时间 `06:25:23 UTC` = 北京时间 `14:25:23`，**正是脚本崩溃的那一刻** → 确认归属后终止：

```python
h = k.OpenProcess(0x0001, False, PID)   # PROCESS_TERMINATE
k.TerminateProcess(h, 0)
```

> ⚠️ **终止前必须先确认进程归属**（对比启动时间 / 让用户确认），避免误杀 PyCharm 里正在跑的任务。

---

### 坑 7 🟡 `wmic.exe` 被安全策略拦截

本机安全策略把 `wmic.exe` 列入了**程序黑名单**，查进程/内存会直接被拒。

**替代方案**：

```python
import subprocess
subprocess.run("tasklist", capture_output=True, text=True)   # 查进程
```

```python
import ctypes
class MEMORYSTATUSEX(ctypes.Structure):                        # 查内存
    _fields_ = [("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
st = MEMORYSTATUSEX(); st.dwLength = ctypes.sizeof(st)
ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
```

---

### 坑 8 🟡 RTX 2060 不支持 bf16

RTX 2060 是 **Turing 架构**，没有 bf16 硬件支持。

```python
fp16=True,      # ✅
bf16=False,     # ★ 必须 False，开了会极慢或直接报错
```

同理，4-bit 计算的 `compute_dtype` 也必须是 `torch.float16`，不能是 `bfloat16`。

---

### 坑 9 🟡 训练 OOM（显存不足）

**症状**：`torch.cuda.OutOfMemoryError`

**6GB 卡的生存配置（四件套，少一个都可能崩）**：

| 手段 | 配置 | 省显存 | 代价 |
| --- | --- | --- | --- |
| 4-bit 量化 | `load_in_4bit=True` | ★★★ 权重降到 1/4 | 精度略降 |
| 梯度检查点 | `gradient_checkpointing=True` | ★★ 激活值省 ~70% | 慢 20~30% |
| 梯度累积 | `gradient_accumulation_steps=8` | ★ 等效大 batch | 变慢 |
| 8bit 优化器 | `optim="paged_adamw_8bit"` | ★★ 优化器状态降 4 倍 | 变慢 |

外加 `per_device_train_batch_size=1`（开 2 就 OOM）。

---

### 坑 10 🟡 loss 恒为 0 或 nan

**根因**：`labels` 被**全部**设成了 `-100`（所有位置都被忽略，自然没有 loss）。

**排查**：打印 labels 里非 `-100` 的数量，应该是"回答部分的 token 数"，不能是 0。

```python
n_valid = sum(1 for x in labels if x != -100)
print("有效 label 数:", n_valid)     # 必须 > 0
```

**常见触发**：`prompt_len` 算大了（超过序列总长），比如 `messages[:-1]` 写错成 `messages`。

---

### 坑 11 🟡 根目录没有 `adapter_config.json`

**症状**：`06_chat.py` 报

```text
[错误] 该目录下没有 adapter_config.json: output\lora_qwen_cs_ep10
```

**根因**：训练**中途结束**（Ctrl+C / 崩溃 / OOM），`trainer.save_model()` 从没执行过 —— 那个文件只在正常结束时才写到根目录。产物只剩每个 epoch 的 `checkpoint-N\` 子目录。

**修复**：指向具体的 checkpoint

```powershell
.\910demo\Scripts\python.exe scripts\06_chat.py --lora "output\lora_qwen_cs_ep10\checkpoint-18"
```

**本工程实际就踩了这个坑**（见第 4.8 节）：`lora_qwen_cs_ep10` 名义上是 10 轮实验，实际 `global_step=18 / max_steps=30`，只跑到第 6 轮就中断了。

**快速判断跑到哪了**：

```python
import json
d = json.load(open(r"...\checkpoint-18\trainer_state.json", encoding="utf-8"))
print("进度:", d["global_step"], "/", d["max_steps"], " 实际 epoch:", d["epoch"])
```

---

### 坑 12 🟡 推理时漏传 system prompt

**症状**：模型输出不带人设，你以为训练失败。

**根因**：**训练时用了 system prompt，推理时忘了传**，数值条件不一致。

**修复**：训练和推理的 `SYSTEM_PROMPT` 必须**逐字一致**。

> ⚠️ **本工程目前就有一处不一致**：`01_make_dataset.py` 里的 `SYSTEM_PROMPT` 是 **4 条规则**，而 `06_chat.py` 里是 **5 条**（多了"回答前面带你好主人"）。
> **建议**：要么两边统一成 4 条，要么改完数据集重新训练。否则 LoRA 模型表现会打折扣。

---

### 坑 13 🟡 模型下载"成功"但进度条走完是空文件

**症状**：下载看起来正常，加载时报 JSON 解析失败。

**原因与修复**：同坑 5（Xet 协议）。

**兜底方案**：手动下载这几个必备文件到同一目录，用本地路径加载：

```text
config.json / model.safetensors (或 pytorch_model.bin)
tokenizer.json / tokenizer_config.json / vocab.json / merges.txt
```

> 大文件（>10MB）用后台任务下载，前台容易被中断。

---

### 坑 14 🟡 手动装 wheel 报 `Must have an ABI tag`

**根因**：改 wheel 文件名时把 ABI tag 删掉了。

```text
❌ bitsandbytes-0.50.2.whl
✅ bitsandbytes-0.50.2-py3-none-win_amd64.whl
```

**格式必须完整**：`<name>-<version>-<python tag>-<abi tag>-<platform>.whl`

---

### 坑 15 🟡 输出啰嗦、格式漂移

**症状**：模型答得又长又散。

**根因**：训练数据本身没严格控长 —— 模型只会模仿你给的样子。

**修复**：
1. 训练数据的**每条答案**都要严格遵守长度规则（本工程是 ≤60 字）
2. 训练数据里混入"简洁版"答案
3. 推理时用 `max_new_tokens` 硬限制

---

### 坑 16 🟡 环境残留：中断的模型下载目录

**症状**：`E:\lorademo\models\Qwen2.5-1.5B-Instruct` 里躺着 **429MB 的半成品**（首次下载中断的残留）。

**影响**：脚本的 `MODEL_CANDIDATES` 会优先命中这个不完整的目录，导致加载失败。

**修复**：删掉残留目录。**本机已清理**，当前统一走 `D:\Model`。

> 💡 **这也是脚本设计 `MODEL_CANDIDATES` 列表的原因**：路径不存在时自动跳到下一个，不会因为一个坏目录就整体失败。

---

### 坑 17 🟡 验证集算了，却没用来停训练、也没选模型

**症状**：`trainer_state.json` 里 `best_metric: None`。训练一路跑满预设轮数，保存的永远是**最后一轮**的模型。

**根因**：配了 `eval_strategy="epoch"`（每轮都算 `eval_loss`），但**漏了三件配套的事**：

```python
# ❌ 没开这几项 —— 于是验证集的数字只是"打印给你看"，没参与任何决策
load_best_model_at_end=True,        # 没开
metric_for_best_model="eval_loss",  # 没设
# 也没加 EarlyStoppingCallback
```

**为什么危险**：如果 `eval_loss` 在中间某轮最低、之后反弹（过拟合的典型形状），这套配置会把**过拟合的那轮当成成品保存下来**，而你看日志只觉得"loss 都挺低"。

**修复**：见 §4.5 训练参数下的 ⚠️ 标签 —— 三行配置 + 一个 `EarlyStoppingCallback`。

**顺带纠正一个常见误解**：`early_stopping_patience` **不是** `TrainingArguments` 的参数。
本机实测确认：`TrainingArguments` 里有 `load_best_model_at_end` / `metric_for_best_model` / `greater_is_better`，
但**没有** `early_stopping_patience` —— 它属于 `EarlyStoppingCallback`。写错位置会直接报 `unexpected keyword argument`。

**另一个更基础的问题**：为什么"训练什么时候停"这件事，很多人以为是"loss 降到某个值就停"？
因为 `TrainingArguments` 里**根本没有** loss 阈值停止这个参数 —— 它只支持"按轮数 / 按步数 / 早停"三种。详见 §4.5 的 ⚠️ 标签。

---

## 第 10 章 学习与扩展路线

### 建议的推进节奏

```text
第 3 章   造数据集        -> 理解"数据集是唯一的信息来源"
第 4 章   跑通训练        -> 看到 0.141% 可训练参数，理解 LoRA 省在哪
第 5 章   推理对比        -> 拿到第一个"肉眼可见"的成果
第 6 章   消融实验        -> 分清"学会条件反射"和"真正内化"
第 7 章   交互对话        -> 自己聊两句，感受最直观
第 8 章   合并部署        -> 从 demo 走向可用
```

**不要跳过第 5 章和第 6 章。** 很多人训完只看 loss 就说"成功了"，结果既不知道 LoRA 到底改了什么，也不知道它有多脆弱。

### 阶段化扩展路线图

#### 阶段 1：跑通 ✅（本 demo 已完成）
- [x] 环境配好（torch cu124 / transformers / peft / bnb）
- [x] 造小数据集（21 + 4 条）
- [x] QLoRA 训练（59 秒，loss 2.26 → 1.37）
- [x] 推理对比 + 消融实验
- [x] 交互式对话脚本

#### 阶段 2：换成自己的任务

**改哪里**：只改 `01_make_dataset.py` 里的 `SYSTEM_PROMPT` 和 `QA_PAIRS`。

想让它学**你的领域知识**（而不是风格）：

| 要点 | 建议值 |
| --- | --- |
| 数据量 | **200 条起**（21 条只够学风格） |
| `MAX_SEQ_LEN` | 答案变长要同步调大（512 → 1024+） |
| `r`（秩） | 提到 **16**（知识型任务需要更强表达力） |

#### 阶段 3：换更大的模型

```python
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"    # 3B，6GB 显存临界值
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"    # 7B，6GB 跑不动，需要 12GB+
```

3B 在 6GB 上要额外做：

```python
MAX_SEQ_LEN = 256          # 缩短序列
GRAD_ACCUM = 16            # 用更多累积换更小 batch
```

#### 阶段 4：用真实数据集

把任意数据集转成统一格式，**训练脚本一行都不用改**：

```python
# Alpaca 格式例：{"instruction":..., "input":..., "output":...}
def to_messages(item):
    user = item["instruction"]
    if item.get("input"):
        user += "\n" + item["input"]
    return {"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": item["output"]},
    ]}
```

推荐数据集：`bingbangboom/alpaca-zh`（中文通用）、`BAAI/Infinity-Instruct`（大规模高质量）、`m-a-p/COIG-CQIA`（中文社区问答）。

#### 阶段 5：更快的训练

**换 trl 的 `SFTTrainer`**（代码更短，自带 packing 优化）—— 本工程环境里 `trl 1.13.0` 已装好：

```python
from trl import SFTTrainer, SFTConfig
```

**用 Unsloth 加速**（同硬件快约 2 倍，显存省 30%）：

```bash
uv pip install unsloth
```

> ⚠️ Unsloth 对 CUDA 版本挑剔，装之前先确认和 `torch 2.6.0+cu124` 兼容。

#### 阶段 6：继续往下做

| 方向 | 具体做法 |
| --- | --- |
| 加数据、促内化 | 几百条数据 + 混入"无 system"样本（解决第 6 章的结论） |
| 调参实验 | 对比 `r=8/16/32`、`lr=1e-4/2e-4` 的 eval_loss |
| 换任务类型 | 从"风格对齐"换成"格式结构化输出"（JSON）、"领域问答" |
| 合并部署 | 合并后喂给 vLLM / Ollama / TensorRT-LLM |

### 记住三句话

> **1. LoRA 的效果，80% 取决于数据质量，20% 取决于超参。**
> 与其反复调 `r` 和 `lr`，不如把答案写得更规范、问法写得更丰富。

> **2. 小数据只学得到"条件反射"，不是"内化"。**
> 21 条数据能让模型"带着 system prompt 演得很好"，但拿掉提示词它就露馅 —— 这是正常现象。

> **3. 每次改动都要跑一遍 `03_inference.py` 对比，比看 loss 直观得多。**
> loss 从 2.26 降到 1.37 说明"学到了东西"，但到底学到了什么，只有并排看输出才知道。

---

## 附：文件清单速查

| 文件 | 作用 | 备注 |
| --- | --- | --- |
| `constraints.txt` | ⚠️ **锁死 torch 版本，必须保留**；装依赖时用 `-c` 带上 | 防"缝合怪"的关键 |
| `check_env.py` | 环境自检（**必做**） | 只有真 `import` 成功才算数 |
| `requirements.txt` | 依赖清单（已锁版本） | — |
| `scripts/01_make_dataset.py` | 造数据集 → `data/*.jsonl` | 改任务只改这个 |
| `scripts/02_train_lora.py` | ★ **QLoRA 训练（核心）** | 3 轮，59 秒 |
| `scripts/02b_train_longer.py` | 10 轮对照实验 | ⚠️ 实际只跑到第 6 轮 |
| `scripts/03_inference.py` | 原始 vs LoRA 对比 | ★ 每次改配置都该跑 |
| `scripts/04_merge_lora.py` | 合并导出（部署用） | 必须 fp16 加载 |
| `scripts/05_ablation.py` | 消融实验（测是否内化） | 4 组合 × 4 问题 |
| `scripts/06_chat.py` | 交互式对话 | 支持 `base`/`lora` 切换 |
| `scripts/07_why_same.py` | 量化对比（长度/标准差/格式遵循率） | 内存紧张用 `--only` |
| `README.md` | 原工程说明 | 部分数据以本指南为准 |
| `LoRA微调从0到1搭建指南.md` | 本文件 | — |

### 核心产物

| 路径 | 内容 | 体积 |
| --- | --- | --- |
| `output/lora_qwen_cs/` | ★ 3 轮正式训练的适配器 | 8.3 MB |
| `output/lora_qwen_cs_ep10/checkpoint-18/` | 10 轮实验（实际 6 轮）的适配器 | ~8 MB |
| `output/merged_qwen_cs/` | 合并后的完整模型（部署用） | 2.89 GB |
| `D:\Model\Qwen2.5-1.5B-Instruct` | 底座模型 | 2.88 GB 权重（+2.88GB `.git`） |

---

## 附：一键复现命令

```powershell
cd E:\lorademo
$py = "E:\lorademo\910demo\Scripts\python.exe"

& $py check_env.py                    # 0. 环境体检（必做）
& $py scripts\01_make_dataset.py      # 1. 造数据（1 秒）
& $py scripts\02_train_lora.py        # 2. 训练（约 1 分钟）
& $py scripts\03_inference.py         # 3. 对比验证 ← 看 LoRA 有没有生效
& $py scripts\05_ablation.py          # 4. 消融实验 ← 看行为有没有内化
& $py scripts\06_chat.py              # 5. 聊天（Ctrl+C 退出）
& $py scripts\04_merge_lora.py        # 6. 合并导出（可选，部署用）
```

# LoRA 微调实战工程（Qwen2.5-1.5B-Instruct）

> 目标：从最小可跑的 demo 出发，一步步扩展成你自己的微调项目。
> 硬件：RTX 2060 6GB / Windows 11
> 环境：`E:\lorademo\910demo`（uv 创建的 venv）

---

## 一、目录结构

```
E:\lorademo\
├── 910demo\                     # Python 虚拟环境（uv 创建，已配好）
├── _fix\                        # 修复环境时下载的 wheel，可随时删
├── data\
│   ├── train.jsonl              # 训练集 21 条（脚本 01 生成）
│   └── eval.jsonl               # 验证集 4 条（脚本 01 生成）
├── output\
│   ├── lora_qwen_cs\            # ★ LoRA 适配器（8.7 MB）
│   ├── lora_qwen_cs_ep10\       # 10 轮对照实验的 checkpoint
│   └── merged_qwen_cs\          # 合并后完整模型（3.09 GB）
├── scripts\
│   ├── 01_make_dataset.py       # 造数据
│   ├── 02_train_lora.py         # ★训练（核心）
│   ├── 02b_train_longer.py      # 10 轮对照实验（改自 02）
│   ├── 03_inference.py          # 推理对比：原始 vs LoRA
│   ├── 04_merge_lora.py         # 合并导出（部署用）
│   ├── 05_ablation.py           # 消融实验：检验行为是否内化
│   ├── 06_chat.py               # ★交互式对话（原模型/LoRA 任选）
│   └── 07_why_same.py           # 对比实验：量化原模型 vs LoRA 差异
├── requirements.txt
└── README.md                    # 本文件
```

**底座模型路径**：`D:\Model\Qwen2.5-1.5B-Instruct`
所有脚本都按「`D:\Model` → `工程内 models\` → 在线下载」的顺序查找，换路径只改脚本顶部的 `MODEL_CANDIDATES`。

> 注：`工程内 models\` 目录已清理（首次下载中断的 429MB 残留），当前统一使用 D 盘模型。

---

## 二、四步跑通

在 `E:\lorademo` 目录下打开终端，**依次**执行：

```bash
# 第 1 步：造数据（1 秒）
.\910demo\Scripts\python.exe scripts\01_make_dataset.py

# 第 2 步：训练（本地模型，约 1 分钟）
.\910demo\Scripts\python.exe scripts\02_train_lora.py

# 第 3 步：对比验证（关键！看 LoRA 有没有生效）
.\910demo\Scripts\python.exe scripts\03_inference.py

# 第 4 步：消融实验（检验行为是否真正内化）
.\910demo\Scripts\python.exe scripts\05_ablation.py

# 第 5 步（可选）：合并成完整模型，用于部署
.\910demo\Scripts\python.exe scripts\04_merge_lora.py
```

### 想直接聊天？用 06_chat.py

```bash
# 默认：和 LoRA 微调后的模型聊天
.\910demo\Scripts\python.exe scripts\06_chat.py

# 和原始模型聊天（对照感受差异）
.\910demo\Scripts\python.exe scripts\06_chat.py --mode base

# 指定别的适配器（比如 10 轮那个实验）
.\910demo\Scripts\python.exe scripts\06_chat.py --lora "output\lora_qwen_cs_ep10\checkpoint-18"

# 不带 system prompt 聊天（验证人设是否真的内化）
.\910demo\Scripts\python.exe scripts\06_chat.py --no-system

# 贪心解码（每次答案相同，便于复现问题）
.\910demo\Scripts\python.exe scripts\06_chat.py --greedy

# 调整发散程度（0.1 保守 / 0.7 平衡 / 1.0+ 发散）
.\910demo\Scripts\python.exe scripts\06_chat.py --temp 0.3
```

**聊天窗口内的快捷指令**：

| 指令 | 作用 |
|---|---|
| `/help` | 显示帮助 |
| `/clear` | 清空对话历史，开始新话题（显存不足时用它） |
| `/system` | 查看当前 system prompt |
| `/mode` | 查看当前模型模式与适配器路径 |
| `/params` | 查看生成参数与历史轮数 |
| `/exit` | 退出 |

**完整参数列表**：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--mode` | `lora` | `base`=原始模型 / `lora`=微调模型 |
| `--lora` | `output/lora_qwen_cs` | LoRA 适配器目录 |
| `--no-system` | 关 | 不发送 system prompt |
| `--temp` | `0.7` | temperature，控制随机性 |
| `--max-tokens` | `120` | 单次生成上限 |
| `--greedy` | 关 | 贪心解码，结果固定 |

> 💡 **支持多轮上下文**：脚本会把历史对话一起传给模型，
> 所以模型能"记得"前面说过什么。
> 但历史越长越吃显存，感觉变慢时用 `/clear` 清一下。

### 另一个常见困惑：「为什么原模型也带人设？」

微调完把 `--mode` 换成 `base`，你会发现原模型回答**也带着人设** —— 这不是 LoRA 没生效。

**原因**：Qwen2.5-1.5B-Instruct 本身经过指令微调，天生就会遵循 system prompt。
人设规则（开头语、语气词、提退换）全写在 system 里，它照做就能演得很像。

**那 LoRA 改变了什么**？跑 `07_why_same.py` 看量化数据：

```bash
# 完整对比（需要约 6GB 空闲内存，先关掉 PyCharm 等大程序）
.\910demo\Scripts\python.exe scripts\07_why_same.py

# 内存紧张时只测一个
.\910demo\Scripts\python.exe scripts\07_why_same.py --only base
```

它对比两个维度的**统计指标**（贪心解码排除随机性）：

| 指标 | 含义 |
|---|---|
| 平均长度 / 长度范围 | LoRA 把输出压到稳定区间 |
| 长度标准差 | **越小越稳定**，这是 LoRA 的核心价值 |
| 带开头语比例 | 格式遵循度 |
| 超 60 字比例 | 原模型经常违反第 4 条规则 |

> ⚠️ **内存提示**：本脚本采用"加载一个→跑完→释放→再加载另一个"的串行模式，
> 绝不会同时占用两份模型。若仍报 `页面文件太小 (os error 1455)`，
> 说明系统内存不足，先关掉 PyCharm / 浏览器再跑。

**判断成功的标准**（本机实测已全部满足）：
- ✅ LoRA 模型的回答以「星尘科技为您服务～」开头
- ✅ 句尾带「呢」「哦」
- ✅ 回答长度被压到 20~30 字（原始模型 48~143 字）
- ✅ 问训练集没见过的问题（如"公司地址"），仍保持人设

---

## 三、实测结果（本机真实跑出来的数据）

### 训练指标

| Epoch | train loss | eval_loss | 说明 |
|---|---|---|---|
| 1 | 2.255 → 1.844 | 1.865 | 起点 |
| 2 | 1.914 → 1.702 | 1.720 | 稳定下降 |
| 3 | 1.736 → **1.374** | **1.687** | 收敛良好 |

- 可训练参数：**2,179,072 / 1,545,893,376 = 0.141%**
- 训练耗时：**59 秒**（21 条数据 × 3 轮 × RTX 2060）
- 产物大小：**8.7 MB**（对比底座 3.09 GB）

### 效果对比（★核心证据）

同一批问题，原始模型 vs LoRA 模型：

| 问题 | 原始模型 | LoRA 模型 |
|---|---|---|
| 你们几点上班？ | 143 字，啰嗦跑题 | **29 字**「我们是早上九点到晚上十一点，随时在线哦～」 |
| 这个多少钱？ | 108 字，回避具体数字 | **30 字**「这款手机壳是 99 元，七天无理由退换哦～」 |
| 能退货吗？ | 48 字，格式散乱 | **23 字**「当然可以，七天无理由退换哦～」 |

**LoRA 真正学到的是「简洁 + 固定格式 + 固定人设」**，
原始模型虽然也懂指令，但输出长度失控、格式漂移。

### 消融实验（重要，别跳过）

故意**不传** system prompt，看模型是否还保持人设：

| 条件 | 结果 |
|---|---|
| LoRA + system | ✅ 人设完整（23~30 字） |
| LoRA - system | ❌ 退化成普通助手 |
| 原始 + system | ⚠️ 有人设但冗长（48~143 字） |
| 原始 - system | ❌ 普通助手 |

**结论**：3~6 轮 + 21 条数据，LoRA 学到的是**「system 条件下的强关联」**，
还没有把行为彻底内化进权重。

> 这在数据量小时是**正常现象**，不是 bug。
> 想让它"无条件带人设"，需要：
> 1) 加数据量（几百条起）；2) 加 epoch；3) 在数据里混入「无 system」的样本

---

## 四、这个 demo 在学什么

我们让它学一个**固定人设 + 固定输出格式**的客服语气。
选这个任务做 demo 的原因：

1. **效果肉眼可见** —— 微调前后差异极大，不用做评测也能判断
2. **几十条数据就够** —— 试错成本极低（59 秒一轮）
3. **不涉及知识注入** —— 不受"模型本来就知道/不知道"的干扰

---

## 五、核心概念速查（重点看这几行）

### LoRA 在改什么

```
原始计算:  h = W·x
LoRA 后:   h = W·x + (alpha/r)·B·A·x
                  ↑        ↑
              冻结不动   只训练这两个小矩阵
```

`W` 是 2048×2048 = 419 万参数，`A`、`B` 加起来只有 2048×8×2 = 3.3 万参数。
**训练量降到 1/128**，这就是省显存的全部秘密。

### 关键超参对照表

| 参数 | 本 demo 值 | 调大什么效果 | 什么时候要改 |
|---|---|---|---|
| `r`（秩） | 8 | 表达力↑ 更吃显存 易过拟合 | 数据 >1000 条时提到 16/32 |
| `lora_alpha` | 16 | 强度↑ | 习惯上设成 `2r`，一般不动 |
| `lora_dropout` | 0.05 | 正则↑ | 小数据提到 0.1 |
| `learning_rate` | 2e-4 | 收敛快 易震荡 | 发散时降到 1e-4 |
| `target_modules` | q/k/v/o proj | 效果↑ 显存↑ | 效果不够时加 MLP 层 |
| `num_epochs` | 3 | 学得透 易过拟合 | 看验证 loss 拐点定 |

### 显存优化三板斧（6GB 卡的生存指南）

| 手段 | 省显存 | 代价 |
|---|---|---|
| 4-bit 量化（QLoRA） | ★★★ 权重降到 1/4 | 精度略降 |
| 梯度检查点 | ★★ 激活值省 ~70% | 训练慢 20-30% |
| 梯度累积 | ★ 等效大 batch | 训练慢 |

---

## 六、逐步扩展路线图（按顺序做）

### 阶段 1：跑通 ✅（本 demo）
- [x] 环境配好（torch / transformers / peft / bnb）
- [x] 造小数据集
- [x] 训练 + 对比验证

### 阶段 2：换成自己的任务

**改哪里**：只改 `01_make_dataset.py` 里的 `SYSTEM_PROMPT` 和 `QA_PAIRS`。

想让它学会**你的领域知识**（而不是风格），思路一样，但要注意：
- 数据量要上去（建议 200 条起）
- 每条答案可以是长文本，但 `MAX_SEQ_LEN` 要同步调大
- 知识型任务 `r` 建议提到 16

### 阶段 3：换更大的模型

改 `02_train_lora.py` 里的 `MODEL_NAME`：

```python
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"    # 3B，6GB 显存临界值
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"    # 7B，6GB 跑不动，需要 12GB+
```

3B 模型在 6GB 上要额外做：
```python
MAX_SEQ_LEN = 256          # 缩短序列
GRAD_ACCUM = 16            # 用更多累积换更小 batch
```

### 阶段 4：用真实数据集

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

推荐数据集：
- `bingbangboom/alpaca-zh`（中文通用指令）
- `BAAI/Infinity-Instruct`（大规模高质量）
- `m-a-p/COIG-CQIA`（中文社区问答）

### 阶段 5：上多卡 / 更快的训练

**换 trl 的 SFTTrainer**（代码更短，自带打包优化）：
```python
from trl import SFTTrainer, SFTConfig
# 好处：自动处理 pack、支持更丰富的日志
```

**用 Unsloth 加速**（同硬件快 2 倍，显存省 30%）：
```bash
uv pip install unsloth
```
> 注意：Unsloth 对 CUDA 版本挑剔，装之前先确认和 torch 2.6+cu124 兼容。

### 阶段 6：部署

| 方式 | 适用场景 | 命令/要点 |
|---|---|---|
| 适配器直载 | 本地测试 | 直接用 `03_inference.py` |
| 合并 + transformers | 简单服务 | 先跑 `04_merge_lora.py` |
| vLLM | 高并发 API | 需要合并后的模型；Windows 支持较差，建议 Linux |
| Ollama | 个人使用 | 需要转 GGUF 格式 |
| TensorRT-LLM | 极致性能 | 你本地已配 10.14，可后续尝试 |

---

## 七、常见坑排查表

| 现象 | 原因 | 解决 |
|---|---|---|
| `import torch` 崩在 sympy | sympy 文件被污染 | 见文末"环境修复记录" |
| bnb 报 `CUDA detection failed` | bnb 版本太老，缺 cuda124 DLL | 升到 `>=0.45` |
| 训练 OOM | batch 太大 / 没开检查点 | `BATCH_SIZE=1` + `gradient_checkpointing=True` |
| loss 一直是 0 或 nan | labels 全被设成 -100 | 检查 `build_tokenize_fn` 的 prompt_len 计算 |
| 模型输出不带人设 | 推理时漏传 system prompt | 训练/推理的 system 必须一致 |
| 输出重复车轱辘话 | 学习率过高 / 训练过久 | 降 lr 或减 epoch |
| 验证 loss 涨、训练 loss 降 | 过拟合 | 减 epoch、加 dropout、加数据 |
| 开了 `bf16=True` 报错 | RTX 2060 不支持 bf16 | 改回 `fp16=True` |
| 下载模型卡住 | 网络问题 | 已默认开 hf-mirror.com 镜像 |
| `TypeError: unexpected keyword 'warmup_ratio'` | transformers 5.x 移除了该参数 | 改用 `warmup_steps`（绝对步数） |
| 两个模型输出一模一样 | `PeftModel.from_pretrained` 会**原地修改**底座 | 加载**两份独立**底座，见 `03_inference.py` 注释 |
| 根目录没有 `adapter_config.json` | 训练中途结束，只存了 checkpoint | 指定 `output/xxx/checkpoint-N` 路径 |
| 输出太长 / 格式漂移 | 模型没学会长度控制 | 训练数据和推理都要限制并检查长度 |

---

## 八、环境修复记录（本机特有，重要）

本工程搭建时修了两个真实问题，记录备查：

### 1. sympy 被污染 → 导致 `import torch` 失败
- **现象**：`ImportError: cannot import name '__version__' from 'sympy.release'`
- **根因**：`site-packages/sympy/release.py` 内容被替换成了
  2048 行 `# PROBE-TEST-12345`（某种探针测试残留）
- **修复**：从官方 wheel 提取正确的 `release.py`（内容仅一行 `__version__ = "1.13.1"`）覆盖
- **备份**：损坏文件留存为 `release.py.broken_bak`

### 2. bitsandbytes 0.41.1 缺 cuda124 二进制
- **现象**：`CUDA SETUP: Required library version not found: libbitsandbytes_cuda124.dll`
  且该错误会让**整个 Python 进程硬崩**，连带 `import peft` 一起失败
- **根因**：bnb 0.41.1（第三方 Windows 打包版）最高只提供到 cuda122
- **修复**：升级到官方 `bitsandbytes==0.50.2`（自带 cuda12x 预编译 DLL）

### 3. 模型下载：huggingface_hub 库下载到空文件
- **现象**：`OSError: config file ... is not a valid JSON file`，
  检查发现 `config.json` 是 **0 字节**
- **根因**：本机 `huggingface_hub 1.31.0` 默认启用 **Xet 传输协议**，
  该协议在镜像站 / 受限网络下会静默失败并留下空文件。
  （注意：`curl` 直接下载是完全正常的，所以问题在库不在网络）
- **修复**：改用 curl 手动下载全部权重到 `E:\lorademo\models\Qwen2.5-1.5B-Instruct`，
  脚本优先读本地目录（见 `LOCAL_MODEL_DIR` 逻辑）
- **备用方案**：设 `HF_HUB_DISABLE_XET=1` 环境变量可让库走传统 HTTP 下载

### 版本兼容性备忘

本机组合已实测可用：
```
Python 3.13.9 + torch 2.6.0+cu124 + transformers 5.17.0 + peft 0.20.0
+ bitsandbytes 0.50.2 + datasets 5.0.1 + accelerate 1.15.0
```

> ⚠️ 注意 `transformers 5.x` 是较新的大版本，网上大部分教程基于 `4.x`，
> 部分 API 可能有差异（如 `evaluation_strategy` 已改名 `eval_strategy`）。
> 遇到教程代码报错，先查是不是版本问题。

---

## 九、下一步建议

1. **先跑通，再改** —— 不要一上来就换数据集/换模型，先把 demo 跑完看到效果
2. **记录每次实验** —— 改了哪个参数、loss 怎么变、结果如何
3. **优先加数据，而不是加 epoch** —— 过拟合的信号出现后，加数据才是正解
4. **善用 `03_inference.py`** —— 每次改完配置都跑一遍对比，比看 loss 直观得多

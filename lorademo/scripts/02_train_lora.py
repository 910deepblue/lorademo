# -*- coding: utf-8 -*-
"""
==============================================================================
 02_train_lora.py —— 第 2 步：LoRA 微调训练（核心脚本）
==============================================================================

【先讲清楚：LoRA 到底在干什么？】
    全量微调 = 把模型几亿个参数全部重新学一遍。1.5B 模型全量微调，
    光优化器状态就要十几 GB 显存，你的 RTX 2060（6GB）根本放不下。

    LoRA（Low-Rank Adaptation）的思路：
        冻结原始权重 W（一个 d×d 的大矩阵），不去动它；
        在旁边加一条"小路"：h = Wx + B·A·x
        其中 A 是 r×d，B 是 d×r，r 很小（比如 8）。

        原本要学 d×d 个参数，现在只学 2×d×r 个。
        当 r=8 而 d=2048 时，参数量降到 1/256，显存和训练时间都大幅下降。

    代价是"表达能力变弱"，但对于"学个语气/格式"这种任务完全够用。

【QLoRA 又是什么？】
    在 LoRA 基础上，把冻结的原始权重 W 用 4-bit 量化存储（NF4 格式）。
    原本 fp16 存 1.5B 参数要 3GB，4-bit 后只要 ~1GB。
    这样 6GB 显存跑 1.5B 模型就很宽裕了。

    代价是量化有精度损失，但对 demo 级任务几乎无感。

【本脚本实现的是 QLoRA（= 4bit 量化 + LoRA 适配器）】
    这是 6GB 显存下最稳的方案。如果你显存更大，把 USE_4BIT 改成 False 即可。

【怎么跑】
    E:\\lorademo\\910demo\\Scripts\\python.exe E:\\lorademo\\scripts\\02_train_lora.py
==============================================================================
"""

# ---- 标准库 ----
import json   # 解析 jsonl 数据集（每行一个 JSON 对象）
import os     # 路径拼接、环境变量设置
import sys    # sys.exit() 用于在致命错误时终止程序

# ---- 第三方库 ----
import torch  # PyTorch 本体：张量、CUDA 显存管理、数据类型
from datasets import Dataset                    # HuggingFace 的轻量数据集容器，支持 .map() 批量预处理
from peft import (                              # peft = Parameter-Efficient Fine-Tuning，LoRA 的官方实现
    LoraConfig,                                 # LoRA 的超参配置对象
    get_peft_model,                             # 把 LoRA 层注入到已有模型里
    prepare_model_for_kbit_training,            # QLoRA 必备：把模型调整成"可量化训练"状态
    TaskType,                                   # 任务类型枚举，这里用 CAUSAL_LM
)
from transformers import (                      # HuggingFace 主库
    AutoModelForCausalLM,                       # 自动加载"因果语言模型"（就是 GPT 那一类）
    AutoTokenizer,                              # 自动加载配套的分词器
    BitsAndBytesConfig,                         # 4-bit / 8-bit 量化配置
    DataCollatorForSeq2Seq,                     # 把一批不等长样本 padding 成等长张量
    Trainer,                                    # 训练循环封装：自动处理反向传播、优化器、日志、存档
    TrainingArguments,                          # 全部训练超参的容器
)

# ----------------------------------------------------------------------------
# 路径与全局配置
# ----------------------------------------------------------------------------
# __file__ → abspath 转绝对路径 → 两次 dirname 去掉 "02_train_lora.py" 和 "scripts"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data_bid")                      # E:\lorademo\data
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "lora_qwen_bid")  # LoRA 适配器保存目录

TRAIN_PATH = os.path.join(DATA_DIR, "train.jsonl")   # 训练集路径
EVAL_PATH = os.path.join(DATA_DIR, "eval.jsonl")     # 验证集路径

# 模型选择说明：
#   Qwen2.5-1.5B-Instruct 是"指令微调版"，本身已经会聊天，
#   我们只是给它叠加一层新风格，收敛很快（几十条数据、3 个 epoch 就够）。
#   如果想更省显存，可以换成 Qwen2.5-0.5B-Instruct。
#
# 【本地权重优先】本机模型已存在于 D:\Model\Qwen2.5-1.5B-Instruct
#   直接从本地加载的好处：
#     1) 不依赖网络，训练可离线复现
#     2) 大文件放在 D 盘，不占 C 盘缓存
#     3) 加载速度快（跳过在线校验）
#   查找顺序：下面列表里第一个存在的目录会被使用；
#   都不存在时，才回退到从 HuggingFace 在线拉取。
# 候选的模型来源列表，按优先级排列（前面的存在就用前面的）
MODEL_CANDIDATES = [
    r"D:\Model\Qwen2.5-1.5B-Instruct",                              # ① 你的本地模型（首选，离线可用）
    os.path.join(PROJECT_ROOT, "models", "Qwen2.5-1.5B-Instruct"),  # ② 工程内 models/ 目录（备选）
]
# next(生成器, None)：逐个检查候选路径，返回第一个"确实是目录"的；都没有则返回 None
LOCAL_MODEL_DIR = next((p for p in MODEL_CANDIDATES if os.path.isdir(p)), None)
# 三元表达式：本地有就用本地路径，没有才退回 HuggingFace 模型名（会触发在线下载）
MODEL_NAME = LOCAL_MODEL_DIR if LOCAL_MODEL_DIR else "Qwen/Qwen2.5-1.5B-Instruct"

# 国内网络建议打开（用 HuggingFace 镜像站），走在线下载时生效。
USE_HF_MIRROR = True    # True = 使用 hf-mirror.com 镜像，加速国内下载

# 训练超参（下面每一行都有解释，不要盲目改）
MAX_SEQ_LEN = 512       # 单条样本最大 token 数。我们的对话很短，512 绰绰有余。
NUM_EPOCHS = 10          # 训练轮数。数据少，多跑几轮才能学到；3 轮是 demo 的甜点值。
BATCH_SIZE = 1          # 单卡 batch。6GB 显存只能开 1，靠梯度累积凑等效 batch。
GRAD_ACCUM = 8          # 梯度累积步数。等效 batch = 1 × 8 = 8，训练更稳。
LEARNING_RATE = 2e-4    # 学习率。LoRA 常用 1e-4 ~ 3e-4，2e-4 比较稳妥。
USE_4BIT = True         # 是否启用 4-bit 量化（QLoRA）。6GB 显存建议 True。


def log(msg: str):
    """统一打印前缀，方便在长日志里定位。"""
    # flush=True 强制立刻刷新输出缓冲区：防止日志被 Python 缓冲住、看不到实时进度
    print(f"[train] {msg}", flush=True)


# ============================================================================
# 步骤 1：加载 tokenizer
# ============================================================================
def load_tokenizer():
    """加载分词器，并修好 Qwen 的 pad_token 坑。返回 tokenizer 对象。"""
    log(f"加载 tokenizer: {MODEL_NAME}")

    # from_pretrained 会自动识别 MODEL_NAME 是本地目录还是在线的模型 id
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,      # Qwen 系列需要，允许执行模型自带的代码
        padding_side="right",        # 因果语言模型必须右侧 padding，否则 loss 算错
    )

    # Qwen 的分词器默认没有 pad_token（它用 eos 兼任）。
    # 不设置的话，padding 时报错或行为诡异。这是新手最常踩的坑之一。
    if tokenizer.pad_token is None:                       # 判断当前是否有 pad_token
        tokenizer.pad_token = tokenizer.eos_token         # 用 eos（结束符）兼任 pad
        tokenizer.pad_token_id = tokenizer.eos_token_id   # id 也要同步，两者必须一致

    # !r 用 repr() 打印，能看见不可见的符号（比如 eos 的形状）
    log(f"  pad_token = {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")
    log(f"  词表大小 = {tokenizer.vocab_size}")            # 词表大小，即 token id 的取值范围
    return tokenizer                                      # 把配置好的分词器交出去


# ============================================================================
# 步骤 2：加载模型
# ============================================================================
def load_model():
    """
    根据 USE_4BIT 决定加载方式。

    【4-bit 加载的关键参数解释】
      load_in_4bit=True
          把权重从 fp16 压成 4-bit 存储，显存占用直接降到约 1/4。
      bnb_4bit_quant_type="nf4"
          NF4（4-bit NormalFloat）是 QLoRA 论文提出的量化格式，
          针对"权重近似正态分布"这一特点优化，比普通的 fp4 精度高。
      bnb_4bit_compute_dtype=torch.float16
          算的时候还是用 fp16 算（4-bit 只用于存储）。
          你的 RTX 2060 是 Turing 架构，没有 bf16 硬件支持，
          所以这里必须用 float16，用 bfloat16 会非常慢甚至报错。
      bnb_4bit_use_double_quant=True
          双重量化：把量化常数本身也量化一遍，每个参数再省约 0.5 bit。
          白赚的显存，建议开着。
    """
    log(f"加载模型: {MODEL_NAME}  4bit={USE_4BIT}")

    if USE_4BIT:     # ---- 分支 A：QLoRA 路线（4-bit 量化加载）----
        # 构造量化配置对象
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,                       # 权重以 4-bit 存储，显存降到约 1/4
            bnb_4bit_quant_type="nf4",               # 用 NF4 量化格式（QLoRA 论文提出，精度更高）
            bnb_4bit_compute_dtype=torch.float16,    # 计算时还原成 fp16；Turing 显卡不能碰 bf16
            bnb_4bit_use_double_quant=True,          # 双重量化：连量化常数也压缩，再省约 0.5bit/参数
        )
        # 加载模型，并把量化配置传进去（此时权重已经被压成 4-bit）
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,   # 关键：告诉 transformers 用上面的 4-bit 方案
            device_map="auto",                # 自动把各层分配到 GPU（显存不够会自动溢出到 CPU）
            trust_remote_code=True,
        )
        # 这个函数做三件事（对 QLoRA 是必须的）：
        #   1) 把 LayerNorm 层转成 fp32，避免量化后数值不稳定
        #   2) 打开输入梯度检查点支持（配合下面的 use_cache=False）
        #   3) 关闭 cache，否则梯度检查点和 cache 冲突
        model = prepare_model_for_kbit_training(model)
        log("  已启用 4-bit 量化 + kbit 训练预处理")
    else:            # ---- 分支 B：全精度 fp16 加载 ----
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,  # 全精度分支也用 fp16，省显存
            device_map="auto",
            trust_remote_code=True,
        )
        log("  已启用全精度 fp16 加载")

    # 训练时关闭 KV cache。
    # 原因：cache 是为了"生成时加速"，训练时它会导致梯度检查点失效并吃显存。
    model.config.use_cache = False    # 直接改模型配置；推理脚本里不需要改回来

    return model


# ============================================================================
# 步骤 3：注入 LoRA 适配器
# ============================================================================
def attach_lora(model):
    """
    【LoRA 参数逐个解释】
      r=16
          低秩矩阵的秩，LoRA 最核心的超参。
          r 越大 → 表达能力越强（记忆容量更大），但参数越多、越容易过拟合。
          本工程从 8 提到 16 的原因：115 条法规问答里有大量"内容 ↔ 条款号"
          这种【必须死记】的映射，r=8 的容量记不住，导致编号串号。
          参数量与 r 严格成正比：r=8 → 217 万（0.141%）；r=16 → 436 万（0.282%）。
      lora_alpha=32
          缩放系数，实际生效的强度是 alpha / r = 32/16 = 2。
          ★ 改 r 必须【同步】把 alpha 改成 2r，否则等效强度会变：
            只改 r（16/16 = 1）→ 强度减半，等于偷偷把学习率砍了一半，
            那"效果变化"就分不清是容量还是学习强度造成的了。
      lora_dropout=0.05
          丢弃率，防过拟合。小数据集建议 >0。
      target_modules
          要挂 LoRA 的层。因果 LM 里主要是注意力部分的 4 个投影矩阵：
            q_proj / k_proj / v_proj / o_proj
          也可以加 MLP 层（gate_proj/up_proj/down_proj），效果更好但更吃显存。
          demo 阶段先只挂注意力，稳。
      bias="none"
          不训练任何 bias 项，省参数。
      task_type="CAUSAL_LM"
          告诉 peft 这是因果语言模型，它才知道该把哪些层当输出层处理。
    """
    # 构造 LoRA 配置对象（参数含义见上方 docstring）
    lora_config = LoraConfig(
        r=16,                                                    # 低秩维度：越小越省，越大越强
        lora_alpha=32,                                          # 缩放系数，等效强度 = alpha/r = 2
        lora_dropout=0.05,                                      # 5% 随机丢弃，防过拟合
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 挂在注意力的 4 个投影层上
        bias="none",                                            # 不训练 bias 参数
        task_type=TaskType.CAUSAL_LM,                           # 声明任务类型：因果语言模型
    )

    # 把 LoRA 层注入模型：原权重被冻结，只保留新加的小矩阵可训练
    model = get_peft_model(model, lora_config)

    # print_trainable_parameters 会告诉你"可训练参数占比"。
    # 正常应该看到类似 0.1% ~ 1% 的数字 —— 这就是 LoRA 省显存的直接证据。
    log("LoRA 适配器已注入，可训练参数统计：")
    model.print_trainable_parameters()      # peft 自带方法，直接打印到标准输出

    return model


# ============================================================================
# 步骤 4：处理数据 —— 把 messages 变成模型能吃的张量
# ============================================================================
def load_jsonl(path):
    """读取 JSONL 文件，返回 Python 列表（每个元素是一个 dict）。"""
    # 以读模式打开，明确指定 utf-8 编码
    with open(path, "r", encoding="utf-8") as f:
        # 逐行读；if line.strip() 过滤掉空行（防止末尾空行导致解析失败）
        return [json.loads(line) for line in f if line.strip()]


def build_tokenize_fn(tokenizer):
    """
    这是整个训练流程里最容易写错的地方，重点讲。

    【目标】把一条 messages 样本转成 input_ids + labels，并且：
        只对 assistant 的回答计算 loss，system/user 部分不计。

    【为什么只算 assistant 的 loss？】
        我们想让模型学会"看到用户问题后，该怎么回答"。
        如果把 system 和 user 也算进 loss，模型会去学"怎么生成用户的问题"，
        那是没用的，还会干扰训练。

    【实现原理】
        1) 先用 chat_template 把整段对话拼成完整文本，tokenize 得到 input_ids。
        2) 单独把"prompt 部分"（system + user + assistant 起始标记）tokenize 一遍。
        3) 把 prompt 部分的 label 全部设成 -100。
           -100 是 PyTorch 交叉熵的"忽略标记"，这些位置不产生梯度。
    """
    def tokenize_fn(example):
        """处理一条样本：返回 input_ids / attention_mask / labels 三个字段。"""
        messages = example["messages"]    # 取出对话列表（system + user + assistant）

        # ---- ① 整段对话（含 assistant 回答）作为训练输入 ----
        # apply_chat_template 按模型自带的对话模板，把 messages 渲染成一整段文本
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,               # False = 只返回字符串，不立刻转 token（方便调试打印）
            add_generation_prompt=False,  # 已经是完整对话，不加"该你说了"的标记
        )

        # ---- ② 只到 assistant 开始处为止的 prompt ----
        # 取出除最后一条 assistant 之外的部分，再加上生成提示
        prompt_messages = messages[:-1]   # 切片：去掉最后一个元素（assistant 回答）
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,   # 这里要加，模拟"轮到 assistant 说话"
        )

        # ---- ③ tokenize ----
        # 整段文本转 token。padding=False：此处不 padding，交给 DataCollator 统一处理
        full = tokenizer(
            full_text,
            truncation=True,              # 超过 max_length 就截断
            max_length=MAX_SEQ_LEN,       # 上限 512
            padding=False,
        )
        # prompt 部分转 token，作用是量出"前多少个 token 属于 prompt"
        prompt = tokenizer(
            prompt_text,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding=False,
        )

        input_ids = full["input_ids"]     # 完整序列的 token id 列表
        labels = list(input_ids)          # 先复制一份；labels 要和 input_ids 等长

        # ---- ④ 把 prompt 长度对应的位置标成 -100，不参与 loss ----
        # 取较小值防止越界（极端情况下 prompt 截断长度 > full 长度）
        prompt_len = min(len(prompt["input_ids"]), len(labels))
        for i in range(prompt_len):       # 遍历前 prompt_len 个位置
            labels[i] = -100              # -100 = PyTorch 交叉熵的"忽略索引"，该位置不算 loss

        # 返回本次训练所需的全部字段（attention_mask 来自 full，同样长度）
        return {
            "input_ids": input_ids,
            "attention_mask": full["attention_mask"],
            "labels": labels,
        }

    return tokenize_fn                    # 返回闭包：里面记住了 tokenizer 和 MAX_SEQ_LEN


def build_datasets(tokenizer):
    """读取 jsonl → 转 Dataset → 批量 tokenize → 自检。返回 (训练集, 验证集)。"""
    log("加载并处理数据集")

    train_raw = load_jsonl(TRAIN_PATH)    # 读训练集原始数据（list[dict]）
    eval_raw = load_jsonl(EVAL_PATH)      # 读验证集原始数据

    train_ds = Dataset.from_list(train_raw)   # 包成 HuggingFace Dataset（支持 .map 并行处理）
    eval_ds = Dataset.from_list(eval_raw)

    tokenize_fn = build_tokenize_fn(tokenizer)   # 拿到上一步构造的处理函数
    # .map() 对每条样本调用一次 tokenize_fn；remove_columns 丢掉原始 messages 字段
    train_ds = train_ds.map(tokenize_fn, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(tokenize_fn, remove_columns=eval_ds.column_names)

    # 自检：确认 labels 里确实有 -100（说明 prompt 屏蔽生效了）
    sample_labels = train_ds[0]["labels"]                       # 取第一条样本的 labels
    n_masked = sum(1 for x in sample_labels if x == -100)       # 统计 -100 的个数
    log(f"  训练集 {len(train_ds)} 条 / 验证集 {len(eval_ds)} 条")
    log(f"  单条样本长度 {len(sample_labels)}，其中被屏蔽(-100) {n_masked} 个 token")
    if n_masked == 0:                                           # 0 个屏蔽 = 有问题
        log("  [警告] 没有任何 token 被屏蔽，检查 chat_template 是否正常！")

    return train_ds, eval_ds                                    # 返回处理好的两个数据集


# ============================================================================
# 步骤 5：配置训练参数并开跑
# ============================================================================
def build_training_args():
    """
    【每个参数为什么这么设】
      output_dir
          检查点（checkpoint）保存位置。中断了能从最近一次续训。
      per_device_train_batch_size=1
          6GB 显存下，batch 开 2 就可能 OOM。先保守，能跑通再加。
      gradient_accumulation_steps=8
          累积 8 步再更新一次权重。等效 batch = 8，让梯度更稳。
      gradient_checkpointing=True
          ★显存杀手锏。前向时不保存中间激活值，反向时重新算一遍。
          显存能省 50%+，代价是训练慢约 20~30%。6GB 显存必开。
      learning_rate=2e-4
          LoRA 用这个量级。注意：LoRA 学习率要比全量微调大 10 倍左右，
          因为只训练小矩阵，需要更大的步长。
      lr_scheduler_type="cosine"
          余弦退火，后期自动减小学习率，收敛更平滑。
      warmup_steps=1
          前 1 步用来"热身"，学习率从 0 线性升上来，避免开局震荡。
          ★注意：transformers 5.x 已移除 warmup_ratio，只能用 warmup_steps 绝对步数。
          换算方法：总步数 = (训练样本数 / (batch × 累积)) × epoch 数
                    本例 = (21 / (1×8)) × 3 ≈ 9 步，取 10% ≈ 1 步。
          训练数据变多时，这里要跟着放大（比如 1000 条数据约需 30~50 步）。
      num_train_epochs=3
          数据只有 21 条，跑 3 轮才能记住。
      logging_steps=1
          每步都打日志。数据少，这样能看到完整 loss 曲线。
      save_strategy="epoch"
          每个 epoch 存一次检查点，方便挑最好的那轮。
      fp16=True
          Turing 卡用 fp16 混合精度，省显存 + 提速。
      ★bf16 绝对不要开，RTX 2060 不支持，开了会报错或极慢。
      optim="paged_adamw_8bit"
          QLoRA 标配优化器。8-bit 存储优化器状态，显存再省一大截。
      report_to="none"
          不上报 wandb 等平台，避免登录麻烦。想看曲线可改成 "tensorboard"。
    """
    # 构造并返回全部训练参数
    return TrainingArguments(
        output_dir=OUTPUT_DIR,                              # 检查点/日志的保存目录
        per_device_train_batch_size=BATCH_SIZE,             # 每条 GPU 的训练 batch（=1）
        per_device_eval_batch_size=BATCH_SIZE,              # 验证 batch（=1），验证不吃梯度可以大点，但保持一致更省心
        gradient_accumulation_steps=GRAD_ACCUM,             # 累积 8 步才更新一次权重
        gradient_checkpointing=True,                        # 用计算换显存：不存中间激活，反向时重算
        learning_rate=LEARNING_RATE,                        # 学习率 2e-4
        lr_scheduler_type="cosine",                         # 余弦退火：学习率先慢降后快降
        warmup_steps=1,              # transformers 5.x 用绝对步数，不再支持 warmup_ratio
        num_train_epochs=NUM_EPOCHS,                        # 训练 3 轮
        logging_steps=1,                                    # 每 1 步打一次日志（数据少，能看全程）
        save_strategy="epoch",                              # 每个 epoch 结束存一次检查点
        eval_strategy="epoch",       # 每个 epoch 跑一次验证，观察过拟合
        save_total_limit=2,          # 最多留 2 个检查点，省磁盘
        fp16=True,                                          # 开启 fp16 混合精度：省显存 + 提速
        bf16=False,                  # ← RTX 2060 必须 False
        optim="paged_adamw_8bit",                           # QLoRA 标配：优化器状态用 8-bit，显存换时间
        report_to="none",                                   # 不上报 wandb/tensorboard，避免登录麻烦
        seed=42,                                            # 随机种子，保证可复现
        dataloader_pin_memory=False, # Windows 上有时会报警告，关掉更干净
    )


def main():
    """主流程：设环境变量 → 检查 GPU → 依次执行加载/注入/建数据/训练/保存。"""
    # 国内镜像：必须在 import transformers 之前/加载模型之前设置才生效
    if USE_HF_MIRROR:
        # setdefault：如果用户已经手动设过 HF_ENDPOINT，就不覆盖（尊重用户设置）
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        log(f"已启用 HuggingFace 镜像: {os.environ['HF_ENDPOINT']}")

    # 打印显卡信息，确认用的是 GPU 而不是 CPU（新手最常见的"训了半天没动"）
    if torch.cuda.is_available():                                # 有没有可用的 CUDA 设备
        name = torch.cuda.get_device_name(0)                     # 0 号 GPU 的名字
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3   # 总显存，字节 → GB
        log(f"检测到 GPU: {name}  显存 {total:.1f} GB")
    else:
        log("[严重警告] 未检测到 CUDA，训练会跑在 CPU 上，会非常慢！")
        sys.exit(1)                                              # 退出码 1 = 异常退出，直接中止

    # ---- 依次执行 5 个步骤 ----
    tokenizer = load_tokenizer()                    # ① 加载分词器
    model = load_model()                            # ② 加载（4-bit）模型
    model = attach_lora(model)                      # ③ 挂上 LoRA 适配器
    train_ds, eval_ds = build_datasets(tokenizer)   # ④ 处理数据集

    training_args = build_training_args()           # ⑤ 组装训练参数

    # DataCollator 负责把一批不等长的样本 padding 成等长张量。
    # 关键点：label_pad_token_id=-100，让 padding 位置也不参与 loss。
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,          # 提供 pad_token_id
        model=model,                  # 让 collator 知道模型期望的输入格式
        padding=True,                 # 批内对齐到"最长的那条"
        label_pad_token_id=-100,      # padding 出来的 label 位置填 -100，同样被忽略
    )

    # Trainer 封装了完整的训练循环
    trainer = Trainer(
        model=model,                  # 要训练的模型（含 LoRA）
        args=training_args,           # 训练参数
        train_dataset=train_ds,       # 训练集
        eval_dataset=eval_ds,         # 验证集
        data_collator=collator,       # 批处理逻辑
    )

    log("=" * 60)
    log("开始训练（第一次会先下载模型，约 3GB，耐心等）")
    log("=" * 60)
    trainer.train()                   # ← 真正开始训练，阻塞直到跑完

    # ---- 保存最终产物 ----
    # save_model 保存的只是 LoRA 适配器（几 MB），不是整个模型。
    # 这是 LoRA 的另一大优势：产物小、易分发。
    log(f"保存 LoRA 适配器到: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)              # 存出 adapter_model.safetensors + 配置
    tokenizer.save_pretrained(OUTPUT_DIR)       # 分词器也一起存，推理时省得再找

    log("=" * 60)
    log("[OK] 训练完成！接下来跑 03_inference.py 验证效果")
    log("=" * 60)


# 入口守卫：只有被直接运行时才执行 main()，被 import 时不执行
if __name__ == "__main__":
    main()

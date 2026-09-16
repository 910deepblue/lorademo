# -*- coding: utf-8 -*-
"""
==============================================================================
 03_inference.py —— 第 3 步：推理对比（验证 LoRA 到底有没有生效）
==============================================================================

【为什么这个脚本很重要？】
    很多人训完模型就懵了："loss 是降了，但然后呢？"
    这个小脚本同时跑【原始模型】和【LoRA 模型】，把回答并排打出来。
    一眼就能看出 LoRA 是否真的改变了模型行为 —— 这是调试的核心手段。

【怎么跑】
    # 对比模式（推荐，默认）
    E:\\lorademo\\910demo\\Scripts\\python.exe E:\\lorademo\\scripts\\03_inference.py

    # 只跑 LoRA 模型
    E:\\lorademo\\910demo\\Scripts\\python.exe E:\\lorademo\\scripts\\03_inference.py --lora-only

    # 换一个自己的问题
    E:\\lorademo\\910demo\\Scripts\\python.exe E:\\lorademo\\scripts\\03_inference.py --question "投标保证金最多能收多少？"
==============================================================================
"""

# ---- 标准库 ----
import argparse   # 解析命令行参数（--lora-only / --question）
import os         # 路径拼接、环境变量

# ---- 第三方库 ----
import torch                                    # PyTorch：显存管理、@torch.no_grad 装饰器
from peft import PeftModel                      # 把 LoRA 适配器加载到已有模型上
from transformers import (                      # 加载模型/分词器/量化配置的三件套
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # E:\lorademo
LORA_DIR = os.path.join(PROJECT_ROOT, "output", "lora_qwen_bid")            # 训练产出的适配器目录

# 与训练脚本保持一致：本地权重优先，没有才走在线下载
MODEL_CANDIDATES = [
    r"D:\Model\Qwen2.5-1.5B-Instruct",                              # 你的本地模型
    os.path.join(PROJECT_ROOT, "models", "Qwen2.5-1.5B-Instruct"),  # 工程内备选
]
LOCAL_MODEL_DIR = next((p for p in MODEL_CANDIDATES if os.path.isdir(p)), None)   # 取第一个存在的
MODEL_NAME = LOCAL_MODEL_DIR if LOCAL_MODEL_DIR else "Qwen/Qwen2.5-1.5B-Instruct"
USE_HF_MIRROR = True    # 是否启用国内镜像站

# 从训练数据里抄来的 system prompt —— 推理时必须和训练时保持一致！
# 【新手最常犯的错】训练用了 system prompt，推理时忘了传，
# 结果模型表现很差，还以为是训练失败。
SYSTEM_PROMPT = (
    "你是招投标合规顾问，依据《中华人民共和国招标投标法实施条例》"
    "（2019 年修订）回答问题。要求："
    "1) 先给结论，再注明依据条款，格式为『依据条例第X条』；"
    "2) 涉及期限、比例、金额等数字必须与条例一致，不得改写；"
    "3) 条例没有规定的，明确回答『条例未作规定』，不得编造；"
    "4) 回答简洁直接，不寒暄、不复述问题。"
    "5) 不涉及到招投标外的问题，直接回答：请不要涉及招投外的问题！"
)


# 测试问题：前 3 个在训练集里见过，后 2 个没见过 —— 用来区分"记忆"和"泛化"
TEST_QUESTIONS = [
    "投标保证金最多能收多少？",              # 训练集见过（考记忆）
    "中标以后能把项目分包出去吗？",          # 换问法（考泛化）
    "开标现场能提异议吗？",                 # 换问法
    "招标人最晚什么时候退投标保证金？",      # 训练集见过
    "评标专家的午餐标准是多少？",            # 条例没规定 → 考它会不会编造
]



def log(msg: str):
    """统一日志前缀，方便在输出里定位本脚本的打印。"""
    print(f"[infer] {msg}", flush=True)   # flush 强制立刻输出


def load_tokenizer():
    """加载分词器，补齐 pad_token。"""
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True, padding_side="right"   # 因果 LM 右侧 padding
    )
    if tokenizer.pad_token is None:              # Qwen 默认没有 pad_token
        tokenizer.pad_token = tokenizer.eos_token  # 用 eos 兼任
    return tokenizer


def load_base_model():
    """
    加载底座模型。
    推理时也用 4-bit，理由：
      1) 显存省一半以上，6GB 卡更安全
      2) 和训练时的数值条件一致，结果更可比
    """
    # 量化配置：和训练脚本完全一致，保证可比性
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                      # 4-bit 存储
        bnb_4bit_quant_type="nf4",              # NF4 格式
        bnb_4bit_compute_dtype=torch.float16,   # 计算用 fp16（Turing 卡不支持 bf16）
        bnb_4bit_use_double_quant=True,         # 双重量化，再省一点
    )
    # 加载并返回模型（device_map="auto" 自动分配设备）
    return AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )


@torch.no_grad()
def generate(model, tokenizer, question, max_new_tokens=80, use_system=True):
    """
    生成回答。

    【关键细节】
      apply_chat_template(..., add_generation_prompt=True)
          这一步会加上模型专属的"轮到 assistant 说话"标记（Qwen 是 <|im_start|>assistant）。
          不加的话，模型不知道现在该谁说话了，输出会乱。

      use_system=True（默认）
          把 SYSTEM_PROMPT 一起传进去。这是【训练时的条件】，必须保持一致。

      use_system=False（消融实验用）
          故意不传 system prompt，只问问题。
          如果 LoRA 真的把行为"内化"了，那么即使不给 system prompt，
          它也应该带着人设回答；而原始模型则会退化成普通助手语气。
          这是验证 LoRA 是否真正改变模型的最有说服力的测试。

      do_sample=False
          贪心解码，结果确定、便于对比。
    """
    if use_system:                            # ---- 训练时的条件：带 system prompt ----
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},   # 系统提示（要和人设完全一致）
            {"role": "user", "content": question},          # 用户问题
        ]
    else:                                     # ---- 消融实验条件：不带 system prompt ----
        messages = [{"role": "user", "content": question}]  # 只有用户问题

    # 按模型模板渲染成文本；add_generation_prompt=True 加上"轮到 assistant"的标记
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # 文本转张量，return_tensors="pt" 返回 PyTorch 张量；再搬到模型所在设备
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # 调用模型生成；**inputs 展开 input_ids 和 attention_mask
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,   # 最多新生成多少个 token
        do_sample=False,          # 贪心解码，对比更稳定
        repetition_penalty=1.1,   # 轻微惩罚重复，避免车轱辘话
        pad_token_id=tokenizer.pad_token_id,   # 抑制"未设置 pad 导致"的警告
    )

    # 只取新生成的部分，去掉输入 prompt
    # outputs[0] 是第一条（唯一一条）生成结果；[inputs长度:] 切片掉输入部分
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    # decode 转回文字；skip_special_tokens 去掉 <|im_start|> 之类的控制符；strip 去首尾空白
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    """命令行入口：解析参数 → 加载模型 → 逐题对比 → 消融实验 → 打印判断标准。"""
    # 定义命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-only", action="store_true", help="只跑 LoRA 模型")   # 布尔开关
    parser.add_argument("--question", type=str, default=None, help="自定义单个问题")  # 自定义问题
    args = parser.parse_args()        # 解析 sys.argv，结果存进 args

    if USE_HF_MIRROR:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")   # 设镜像（若用户没设过）

    # 找不到适配器就直接退出（先检查，避免白加载一遍模型）
    if not os.path.isdir(LORA_DIR):
        log(f"[错误] 找不到 LoRA 适配器目录: {LORA_DIR}")
        log("       请先运行 02_train_lora.py 完成训练。")
        return                        # 从 main 返回，脚本正常结束（不动 sys.exit）

    # 有 --question 就只测这一个问题，否则用内置的 5 个测试问题
    questions = [args.question] if args.question else TEST_QUESTIONS

    tokenizer = load_tokenizer()      # 加载分词器

    # ---------------- 加载底座模型 ----------------
    # ★★ 关键坑（本脚本调试时真实踩过）★★
    #   PeftModel.from_pretrained(base, ...) 会【原地修改】base 对象：
    #   它把 LoRA 层直接插进 base 的模块里，然后把包装后的同一个对象返回。
    #   所以如果写成：
    #       base = load();  lora = PeftModel.from_pretrained(base, ...)
    #   那么 base 和 lora 其实是同一个模型，两次生成结果必然一模一样，
    #   会让人误以为"LoRA 没生效"。
    #
    #   正确做法：加载两份互相独立的底座（各占一份显存），
    #   一个保持原样，一个挂 LoRA。
    #   这也是为什么脚本先跑原始模型、再跑 LoRA 模型，而不是反过来。
    # ------------------------------------------------------------------
    if args.lora_only:                    # ---- 只跑 LoRA：只加载一份模型，省显存 ----
        log("加载底座模型 + LoRA 适配器（单份）...")
        lora_model = load_base_model()                              # 加载底座
        lora_model = PeftModel.from_pretrained(lora_model, LORA_DIR) # 挂上适配器
        lora_model.eval()                                           # 切到推理模式（关 dropout）
        base_model = None                                           # 没有底座对照，置 None
    else:                                 # ---- 对比模式：加载两份独立模型 ----
        log("加载底座模型①（保持原始，用于对照）...")
        base_model = load_base_model()      # 第一份：完全不挂 LoRA，保持原样
        base_model.eval()

        log("加载底座模型② + LoRA 适配器...")
        lora_model = load_base_model()                              # 第二份：全新的底座
        lora_model = PeftModel.from_pretrained(lora_model, LORA_DIR) # 只给这一份挂 LoRA
        lora_model.eval()

        # 自检：确认 LoRA 适配器真的挂上去了
        # any(生成器)：只要有一个参数名含 "lora_" 就返回 True
        has_lora = any("lora_" in n for n, _ in lora_model.named_parameters())
        log(f"  LoRA 层是否已注入: {has_lora}")
        if not has_lora:                    # 没注入成功说明路径错了
            log("  [错误] 适配器未生效，请检查 LORA_DIR 路径！")

    # 统计一下实际加载了多少 LoRA 参数，确认适配器真的生效
    # 遍历所有命名参数，名字含 "lora_" 的累加元素总数（numel）
    n_lora = sum(
        p.numel() for n, p in lora_model.named_parameters() if "lora_" in n
    )
    log(f"LoRA 参数量: {n_lora:,}")   # :, 千分位格式化，便于读数

    print()
    print("=" * 78)
    print("  推理对比：原始模型 vs LoRA 微调后")
    print("=" * 78)

    # enumerate(questions, 1)：从 1 开始编号，方便打印"【问题 1】"
    for i, q in enumerate(questions, 1):
        print(f"\n【问题 {i}】{q}")
        print("-" * 78)

        if not args.lora_only:                                  # 对比模式下才跑原始模型
            ans_base = generate(base_model, tokenizer, q)       # 原始模型的回答
            print(f"  原始模型 : {ans_base}")

        ans_lora = generate(lora_model, tokenizer, q)           # LoRA 模型的回答
        print(f"  LoRA 模型: {ans_lora}")

    # ==================================================================
    # 消融实验：不传 system prompt
    # 这是判断"LoRA 是否真的把行为内化进模型"的关键测试。
    #   原始模型：失去 system 指令后，应该退化成普通助手
    #   LoRA 模型：行为已固化在权重里，即使没有 system 也应保持人设
    # ==================================================================
    if not args.lora_only:                                # 只有对比模式才做消融实验
        print()
        print("=" * 78)
        print("  消融实验：故意不传 system prompt（检验行为是否已内化）")
        print("=" * 78)

        probe_q = "投标保证金最多能收多少？"                # 探针问题：招投标域，训练集有同类问法
        print(f"\n【问题】{probe_q}")
        print("-" * 78)
        # 两次生成都传 use_system=False，比较"失去提示词后"谁还保持人设
        print(f"  原始模型 : {generate(base_model, tokenizer, probe_q, use_system=False)}")
        print(f"  LoRA 模型: {generate(lora_model, tokenizer, probe_q, use_system=False)}")
        print()
        print("  → 若 LoRA 模型仍带人设，说明行为已固化进权重，而非依赖提示词")

    print()
    print("=" * 78)
    print("【怎么判断成功？】")
    print("  ✓ LoRA 模型的回答带出明确的『依据条例第X条』")
    print("  ✓ 涉及数字（2%、10%、5 日、15 日…）与条例一致，没有改写")
    print("  ✓ 回答简洁直接，不寒暄、不复述问题")
    print("  ✓ 即使是没见过的问法，也保持了条款引用格式（泛化成功）")
    print("  ✓ 问域外问题（如『能退货吗』）时回答『请不要涉及招投外的问题！』")
    print("=" * 78)


# 入口守卫：直接运行时才执行 main()
if __name__ == "__main__":
    main()

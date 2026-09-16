# -*- coding: utf-8 -*-
"""
 05_ablation.py —— 消融实验：检验 LoRA 行为是否"内化"进权重
==============================================================================
【这个实验回答什么问题？】
    微调后模型表现好，到底是：
      A) LoRA 真的改变了模型权重，行为已成习惯？
      B) 还是只是因为它读懂了 system prompt，然后随便演了演？

    区分方法：故意不传 system prompt，看模型是否还保持人设。
      - 原始模型失去 system → 退化成普通助手（说明它靠提示词）
      - LoRA 模型失去 system 仍保持人设 → 说明行为已固化（真正的微调）

【怎么跑】
    python 05_ablation.py
    python 05_ablation.py --lora output/lora_qwen_cs_ep10/checkpoint-18
==============================================================================
"""

# ---- 标准库 ----
import argparse   # 解析 --lora 参数
import os         # 路径拼接
import re         # 正则：判断回答里是否真的带出条款号

# ---- 第三方库 ----
import torch                                    # PyTorch：@torch.no_grad、fp16 类型
from peft import PeftModel                      # 加载 LoRA 适配器
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # 加载三件套

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # E:\lorademo

# 本地权重优先的候选路径（与训练脚本一致）
MODEL_CANDIDATES = [
    r"D:\Model\Qwen2.5-1.5B-Instruct",
    os.path.join(PROJECT_ROOT, "models", "Qwen2.5-1.5B-Instruct"),
]
MODEL_NAME = next((p for p in MODEL_CANDIDATES if os.path.isdir(p)), "Qwen/Qwen2.5-1.5B-Instruct")
DEFAULT_LORA = os.path.join(PROJECT_ROOT, "output", "lora_qwen_bid")   # 默认适配器目录

# 推理时必须与训练时保持一致的 system prompt
# ★ 与 01_make_bid_dataset.py 里的 SYSTEM_PROMPT 逐字一致，改一处必须改全部
SYSTEM_PROMPT = (
    "你是招投标合规顾问，依据《中华人民共和国招标投标法实施条例》"
    "（2019 年修订）回答问题。要求："
    "1) 先给结论，再注明依据条款，格式为『依据条例第X条』；"
    "2) 涉及期限、比例、金额等数字必须与条例一致，不得改写；"
    "3) 条例没有规定的，明确回答『条例未作规定』，不得编造；"
    "4) 回答简洁直接，不寒暄、不复述问题。"
    "5) 不涉及到招投标外的问题，直接回答：请不要涉及招投外的问题！"
)

# 探针问题列表：用来观察四种组合下的人设保留情况
PROBE_QUESTIONS = [
    "投标保证金最多能收多少？",        # 训练集有同类问法 → 应带出"2%"
    "中标以后能把项目分包出去吗？",    # 换问法 → 考泛化
    "开标现场能提异议吗？",           # 换问法
    "评标委员会成员能收礼吗？",        # 完全没见过 → 考泛化与"条款号"固有习惯
]


def load_base():
    """加载 4-bit 底座模型（推理用，省显存）。"""
    bnb = BitsAndBytesConfig(                    # 量化配置，训练推理保持一致
        load_in_4bit=True,                       # 4-bit 存储
        bnb_4bit_quant_type="nf4",               # NF4 格式
        bnb_4bit_compute_dtype=torch.float16,    # 计算用 fp16
        bnb_4bit_use_double_quant=True,          # 双重量化
    )
    # 加载 → 切到推理模式（.eval() 关掉 dropout/BN 更新）→ 返回
    return AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb, device_map="auto",
        trust_remote_code=True,
    ).eval()


@torch.no_grad()     # 关键：推理不需要梯度，关掉可以大幅省显存、提速
def gen(model, tokenizer, question, use_system=True, max_new_tokens=90):
    """生成一次回答。use_system 控制是否带人设提示词。"""
    messages = []                                    # 对话消息列表
    if use_system:                                   # 需要 system 就加进去
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": question})   # 用户问题（必加）

    # 渲染成模型模板文本，并加上生成提示
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)   # 转张量并搬到设备
    out = model.generate(                            # 生成
        **inputs,
        max_new_tokens=max_new_tokens,               # 上限 90
        do_sample=False,                             # 贪心解码，排除随机性
        repetition_penalty=1.1,                      # 轻微抑制重复
        pad_token_id=tokenizer.pad_token_id,         # 消除警告
    )
    # 切掉输入部分，只解码新生成的内容
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


def main():
    """消融实验主流程：4 种组合 × 4 个问题，统计人设特征词命中数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", type=str, default=DEFAULT_LORA)   # 可指定别的适配器
    args = parser.parse_args()

    # 校验适配器目录里确实有配置文件，否则加载必失败
    if not os.path.exists(os.path.join(args.lora, "adapter_config.json")):
        print(f"[错误] 该路径下没有 adapter_config.json: {args.lora}")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:                  # 补齐 pad_token
        tokenizer.pad_token = tokenizer.eos_token

    print("[ablation] 加载底座模型（原始）...", flush=True)
    base = load_base()                               # 第一份：原始模型
    print("[ablation] 加载底座模型 + LoRA...", flush=True)
    # 第二份：全新底座挂 LoRA（load_base() 又加载了一份独立的）
    lora = PeftModel.from_pretrained(load_base(), args.lora).eval()

    print()
    print("=" * 74)
    print("  消融实验：去掉 system prompt 后，模型还能保持人设吗？")
    print("=" * 74)

    # 判断"回答是否真的带人设"
    # ★ 为什么不能只查关键词：基座模型会幻觉出裸引用「依据条例第36条」（才 8 字），
    #   早期写法 MARKERS=['依据条例第','条例','投标'] 会让它命中 2 个词、被判成"人设✓"
    #   —— 指标当场失效，把幻觉当成成功。
    #   现在卡两条，缺一不可：
    #     ① 条款号必须是【中文数字】：训练数据全是"二十六/四十四"这种写法；
    #        基座幻觉用的是阿拉伯数字"36/67"，正则直接不认 → 被过滤掉。
    #     ② 正文长度 > 20 字：过滤掉"只吐一个裸条号、没内容"的退化输出（8 字）。
    CITATION_RE = re.compile(r"条例第[一二三四五六七八九十百零]+条")

    def has_persona(ans):
        """带人设 = 用中文数字引用了具体条款，且带实质正文（不是只吐一个裸条号）。"""
        return bool(CITATION_RE.search(ans)) and len(ans) > 20

    for q in PROBE_QUESTIONS:                        # 遍历每个探针问题
        print(f"\nQ: {q}")
        print("-" * 74)
        # 四种组合：模型 × 是否带 system
        for label, model, us in [
            ("原始 + system", base, True),           # ① 原始模型 + 提示词（基线）
            ("原始 - system", base, False),          # ② 原始模型 - 提示词（应退化）
            ("LoRA + system", lora, True),           # ③ LoRA + 提示词（应最佳）
            ("LoRA - system", lora, False),          # ④ LoRA - 提示词（看是否内化）
        ]:
            ans = gen(model, tokenizer, q, use_system=us)      # 生成回答
            # 引了条款 + 有正文 才算"带人设"；只看关键词会把幻觉裸条号也算成功
            flag = "人设✓" if has_persona(ans) else "人设✗"
            # {label:>14} 右对齐 14 字符，让输出整齐
            print(f"  [{label:>14}] ({flag}, {len(ans)}字) {ans}")

    print()
    print("=" * 74)
    print("【怎么读结果】")
    print("  · 判定标准：回答里【同时】满足")
    print("      ① 用中文数字引用条款号（如『依据条例第二十六条』）")
    print("      ② 正文长度 > 20 字")
    print("    —— 只看关键词会把基座幻觉出的裸引用（『依据条例第36条』8 字）也算成成功")
    print("  · 'LoRA - system' 仍带人设 → 行为已内化，这是最理想的结果")
    print("  · 'LoRA - system' 退化，但 'LoRA + system' 明显优于 '原始 + system'")
    print("    → 属正常：小数据只学到了『条件反射』，推理时必须带上 system")
    print("  · ⚠️ 本实验只看【格式/人设是否保留】，不看条款号对不对 ——")
    print("       小模型+小数据能学会格式，但编号是死记项，容易串号，需另行核对")
    print("=" * 74)


# 入口守卫：直接运行时才执行 main()
if __name__ == "__main__":
    main()

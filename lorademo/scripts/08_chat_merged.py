# -*- coding: utf-8 -*-
"""
==============================================================================
 08_chat_merged.py —— 交互式对话（加载【合并后】的完整模型）
==============================================================================

【这个脚本干什么】
    和 06_chat.py 同款聊天体验，但加载的是 04_merge_lora.py 合并出来的
    完整模型（W' = W + B·A 已写回权重）。

【和 06 的本质区别】
    06_chat.py : 底座 + 适配器，推理时现算 h = W·x + B·A·x（需要 peft）
    08（本脚本）: 只有一个 W'，h = W'·x，跟普通完整模型无任何区别
                 → 不 import peft、不挂适配器、加载更快

    理论上两者输出一致 —— 合并只是把加法提前算好，不是新模型。
    想亲手验证"合并前后效果一样"，可以同一个问题分别用 06 和 08 问
    （都加 --greedy 保证解码确定性）。

------------------------------------------------------------------------------
【怎么跑】

  1) 默认加载 output\merged_qwen_bid（4-bit 推理，省显存）：
       .\\910demo\\Scripts\\python.exe scripts\\08_chat_merged.py

  2) 全精度 fp16 加载（显存要 ~3.5GB，质量略好）：
       .\\910demo\\Scripts\\python.exe scripts\\08_chat_merged.py --fp16

  3) 指定别的合并模型目录：
       .\\910demo\\Scripts\\python.exe scripts\\08_chat_merged.py --model "E:\\别的目录\\merged"

  4) 前置条件：先跑过 04_merge_lora.py，产出 output\\merged_qwen_bid

------------------------------------------------------------------------------
【聊天窗口里的快捷指令】（与 06 相同）
    /help     显示帮助
    /clear    清空对话历史，重新开始
    /system   查看当前 system prompt
    /params   查看当前生成参数
    /exit     退出（也可以按 Ctrl+C）
==============================================================================
"""

# ---- 标准库 ----
import argparse   # 解析命令行参数（--model / --fp16 / --temp 等）
import os         # 路径拼接、环境变量
import sys        # sys.exit() 出错时终止

# ---- 第三方库 ----
# 注意：这里【没有】import peft —— 合并后的模型就是普通模型，用不上适配器
import torch                                    # 显存检查、@torch.no_grad、捕获 OOM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ----------------------------------------------------------------------------
# 路径配置
# ----------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # E:\lorademo

# 默认加载 04 的产物；--model 可换别的合并模型目录
DEFAULT_MERGED = os.path.join(PROJECT_ROOT, "output", "merged_qwen_bid")
USE_HF_MIRROR = True    # 启用国内镜像（仅当模型目录缺失、要回退在线下载时才用到）

# ----------------------------------------------------------------------------
# system prompt —— 与训练时保持一致（合并改变的是权重，不改变人设规则；
# 模型权重里"记住"的是行为习惯，但对话时喂什么 system 仍然影响表现）
# ----------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "你是招投标合规顾问，依据《中华人民共和国招标投标法实施条例》"
    "（2019 年修订）回答问题。要求："
    "1) 先给结论，再注明依据条款，格式为『依据条例第X条』；"
    "2) 涉及期限、比例、金额等数字必须与条例一致，不得改写；"
    "3) 条例没有规定的，明确回答『条例未作规定』，不得编造；"
    "4) 回答简洁直接，不寒暄、不复述问题。"
    "5) 不涉及到招投标外的问题，直接回答：请不要涉及招投外的问题！"
)


def log(msg):
    """统一日志前缀。"""
    print(f"[chat-merged] {msg}", flush=True)


# ============================================================================
# 模型加载
# ============================================================================
def load_tokenizer(model_dir):
    """从合并模型目录加载分词器（04 合并时已把 tokenizer 一起存进去了）。"""
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tok.pad_token is None:                    # Qwen 默认缺 pad_token
        tok.pad_token = tok.eos_token            # 用 eos 兼任
    return tok


def load_model(model_dir, fp16=False):
    """
    加载合并后的完整模型。

    【默认 4-bit 的原因】
        与 06_chat.py 的推理条件保持一致（06 是 4-bit 底座 + fp16 适配器），
        这样两边对比才公平；同时 6GB 显存也更从容。
        加 --fp16 则按原始 fp16 全精度加载（约 3.1GB 显存），质量略好。
    """
    if fp16:
        # ---- 全精度分支：merged 保存的就是 fp16，直接原样加载 ----
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,           # 按存储精度加载
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        # ---- 4-bit 分支：与 06 推理条件对齐 ----
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,                       # 4-bit 存储
            bnb_4bit_quant_type="nf4",               # NF4 格式
            bnb_4bit_compute_dtype=torch.float16,   # Turing 卡不支持 bf16
            bnb_4bit_use_double_quant=True,          # 双重量化
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()      # 切推理模式：关 dropout，确保输出确定
    return model


# ============================================================================
# 生成（与 06_chat.py 完全同款）
# ============================================================================
@torch.no_grad()     # 推理不建计算图，省显存、提速
def generate(model, tokenizer, messages, max_new_tokens=120,
             temperature=0.7, top_p=0.9, do_sample=True):
    """根据完整对话历史生成回复（实现与 06 相同，便于公平对比）。"""
    # 把完整对话历史渲染成模型模板文本（最后加"轮到 assistant"标记）
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # 转张量并搬到模型设备上
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # 生成长度上限要算上历史长度，否则多轮对话会很快被截断
    input_len = inputs["input_ids"].shape[1]      # 记录输入序列长度
    outputs = model.generate(
        **inputs,                                 # 展开 input_ids + attention_mask
        max_new_tokens=max_new_tokens,            # 最多新生成多少 token
        do_sample=do_sample,                      # 采样 or 贪心
        temperature=temperature if do_sample else None,   # 贪心时不传，避免警告
        top_p=top_p if do_sample else None,
        repetition_penalty=1.1,                   # 轻微抑制重复
        pad_token_id=tokenizer.pad_token_id,      # 消除警告
        eos_token_id=tokenizer.eos_token_id,      # 遇到结束符就停
    )

    # 只取新生成的部分，解码成文字
    new_tokens = outputs[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ============================================================================
# 交互主循环
# ============================================================================
HELP_TEXT = """
------------------------------------------------------------------
  可用指令：
    /help     显示本帮助
    /clear    清空对话历史，开始新话题
    /system   查看当前 system prompt
    /params   查看当前生成参数
    /exit     退出程序
------------------------------------------------------------------
  直接输入文字即可与【合并后模型】对话（支持多轮上下文记忆）
"""


def main():
    """交互主循环：解析参数 → 校验并加载合并模型 → 循环对话。"""
    parser = argparse.ArgumentParser(description="与合并后的完整模型交互对话")
    parser.add_argument("--model", type=str, default=DEFAULT_MERGED,
                        help="合并模型目录（默认 output\\merged_qwen_bid）")
    parser.add_argument("--fp16", action="store_true",
                        help="全精度 fp16 加载（默认 4-bit，与 06 对齐）")
    parser.add_argument("--no-system", action="store_true",
                        help="不发送 system prompt")
    parser.add_argument("--temp", type=float, default=0.7, help="temperature，默认 0.7")
    parser.add_argument("--max-tokens", type=int, default=120, help="单次最多生成 token 数")
    parser.add_argument("--greedy", action="store_true",
                        help="用贪心解码（每次答案相同，便于与 06 对比）")
    args = parser.parse_args()

    if USE_HF_MIRROR:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    # ---------------- 校验模型目录 ----------------
    if not os.path.isdir(args.model):
        log(f"[错误] 找不到合并模型目录: {args.model}")
        log("       先运行 04_merge_lora.py 生成，或用 --model 指定其他目录")
        sys.exit(1)
    # 合并模型没有 adapter_config.json，判断它是否完整的依据是权重文件存在
    has_weight = any(
        f.endswith((".safetensors", ".bin")) for f in os.listdir(args.model)
    )
    if not has_weight:
        log(f"[错误] 目录里没有权重文件（.safetensors / .bin）: {args.model}")
        sys.exit(1)

    # ---------------- 加载 ----------------
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")

    log(f"模式: 合并后完整模型{'（fp16 全精度）' if args.fp16 else '（4-bit）'}")
    log(f"模型: {args.model}")
    tokenizer = load_tokenizer(args.model)       # 分词器从合并目录里拿
    model = load_model(args.model, fp16=args.fp16)

    use_system = not args.no_system      # 没给 --no-system 就启用 system
    do_sample = not args.greedy          # 没给 --greedy 就用采样解码

    # ---------------- 欢迎信息 ----------------
    print()
    print("=" * 66)
    print(f"  交互式对话已启动（合并模型）")
    print(f"  模型     : {os.path.basename(args.model)}")
    print(f"  加载精度 : {'fp16 全精度' if args.fp16 else '4-bit'}")
    print(f"  system   : {'已启用' if use_system else '已禁用（--no-system）'}")
    print(f"  解码方式 : {'贪心（结果固定）' if args.greedy else f'采样 (temp={args.temp})'}")
    print("=" * 66)
    print(HELP_TEXT)

    # ---------------- 对话历史 ----------------
    messages = []
    if use_system:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})

    turn = 0
    while True:
        try:
            user_input = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):         # Ctrl+Z / Ctrl+C
            print("\n\n再见！")
            break

        if not user_input:
            continue

        # ---------------- 处理快捷指令 ----------------
        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd in ("/exit", "/quit"):
                print("再见！")
                break
            elif cmd == "/help":
                print(HELP_TEXT)
            elif cmd == "/clear":
                messages = []
                if use_system:
                    messages.append({"role": "system", "content": SYSTEM_PROMPT})
                turn = 0
                print("[已清空对话历史]")
            elif cmd == "/system":
                if use_system:
                    print(f"[当前 system prompt]\n{SYSTEM_PROMPT}")
                else:
                    print("[当前未启用 system prompt]")
            elif cmd == "/params":
                print(f"[生成参数] max_new_tokens={args.max_tokens}, "
                      f"do_sample={do_sample}, temperature={args.temp}")
                print(f"[历史轮数] {turn}")
            else:
                print(f"[未知指令] {user_input}，输入 /help 查看帮助")
            continue

        # ---------------- 正常对话 ----------------
        messages.append({"role": "user", "content": user_input})

        try:
            reply = generate(
                model, tokenizer, messages,
                max_new_tokens=args.max_tokens,
                temperature=args.temp,
                do_sample=do_sample,
            )
        except torch.cuda.OutOfMemoryError:
            print("\n[显存不足] 对话历史太长了，输入 /clear 清空后重试。")
            messages.pop()   # 回滚刚加入的这条，避免污染历史
            continue

        print(f"\n模型 > {reply}")
        messages.append({"role": "assistant", "content": reply})
        turn += 1


# 入口守卫：直接运行时才执行 main()
if __name__ == "__main__":
    main()

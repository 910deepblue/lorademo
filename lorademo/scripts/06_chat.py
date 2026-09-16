# -*- coding: utf-8 -*-
"""
==============================================================================
 06_chat.py —— 交互式对话（原模型 / LoRA 模型 任选）
==============================================================================

【这个脚本干什么】
    启动一个命令行聊天窗口，你可以连续输入问题和模型对话。
    支持两种模式：
      · base  —— 只加载原始底座模型（未微调）
      · lora  —— 加载底座 + LoRA 适配器（微调后）
    想换模式就改一个参数，或者用命令行开关。

------------------------------------------------------------------------------
【怎么跑】

  1) 默认跑 LoRA 模型：
       .\\910demo\\Scripts\\python.exe scripts\\06_chat.py

  2) 跑原始模型（对照）：
       .\\910demo\\Scripts\\python.exe scripts\\06_chat.py --mode base

  3) 指定别的 LoRA 适配器（比如 10 轮实验的那个）：
       .\\910demo\\Scripts\\python.exe scripts\\06_chat.py --lora "output\\lora_qwen_cs_ep10\\checkpoint-18"

  4) 不带 system prompt 聊天（看人设是否内化）：
       .\\910demo\\Scripts\\python.exe scripts\\06_chat.py --no-system

------------------------------------------------------------------------------
【聊天窗口里的快捷指令】
    /help     显示帮助
    /clear    清空对话历史，重新开始
    /system   查看当前 system prompt
    /mode     查看当前模型模式
    /params   查看当前生成参数
    /exit     退出（也可以按 Ctrl+C）
==============================================================================
"""

# ---- 标准库 ----
import argparse   # 解析命令行参数（--mode / --lora / --no-system / --temp 等）
import os         # 路径拼接、环境变量
import sys        # sys.exit() 出错时终止

# ---- 第三方库 ----
import torch                                    # 显存检查、@torch.no_grad、捕获 OOM
from peft import PeftModel                      # 加载 LoRA 适配器
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig   # 加载三件套

# ----------------------------------------------------------------------------
# 路径配置
# ----------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # E:\lorademo

# 本地权重优先的候选路径
MODEL_CANDIDATES = [
    r"D:\Model\Qwen2.5-1.5B-Instruct",
    os.path.join(PROJECT_ROOT, "models", "Qwen2.5-1.5B-Instruct"),
]
# 取第一个存在的本地目录；都没有则用 HF 在线 id
MODEL_NAME = next((p for p in MODEL_CANDIDATES if os.path.isdir(p)), "Qwen/Qwen2.5-1.5B-Instruct")
DEFAULT_LORA = os.path.join(PROJECT_ROOT, "output", "lora_qwen_bid")   # 默认适配器
USE_HF_MIRROR = True    # 启用国内镜像

# ----------------------------------------------------------------------------
# system prompt —— 和训练时保持一致
#   注：如果你在 01_make_dataset.py 里改了人设，这里也要同步改，
#       否则 LoRA 模型的表现会打折扣。
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
    print(f"[chat] {msg}", flush=True)


# ============================================================================
# 模型加载
# ============================================================================
def load_tokenizer():
    """加载分词器并补齐 pad_token。"""
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None:                    # Qwen 默认缺 pad_token
        tok.pad_token = tok.eos_token            # 用 eos 兼任
    return tok


def load_model(use_lora: bool, lora_path: str = None):
    """
    加载模型。

    【4-bit 量化说明】
        这里用 4-bit 加载，和训练时保持一致（数值条件一致，效果更可比），
        同时 6GB 显存也能轻松放下。
        如果你的显卡显存充足，可以把 load_in_4bit 设为 False 换成全精度，
        生成质量会略好点。
    """
    # 量化配置（与训练一致）
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,                       # 4-bit 存储
        bnb_4bit_quant_type="nf4",               # NF4 格式
        bnb_4bit_compute_dtype=torch.float16,   # Turing 卡不支持 bf16
        bnb_4bit_use_double_quant=True,          # 双重量化
    )

    # 加载底座模型（未挂 LoRA）
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )

    if use_lora:                                 # ---- 需要挂 LoRA 时 ----
        # 校验 ①：目录存在吗
        if not lora_path or not os.path.isdir(lora_path):
            log(f"[错误] 找不到 LoRA 适配器目录: {lora_path}")
            sys.exit(1)                          # 退出码 1，终止程序

        # 校验 ②：目录里有没有 adapter_config.json（判断是不是真正的适配器）
        if not os.path.exists(os.path.join(lora_path, "adapter_config.json")):
            log(f"[错误] 该目录下没有 adapter_config.json: {lora_path}")
            log("       提示：若训练中断，适配器可能在 checkpoint-N 子目录里")
            sys.exit(1)

        # 把适配器挂到 model 上（原地修改）
        model = PeftModel.from_pretrained(model, lora_path)
        # 统计 LoRA 参数量：名字含 "lora_" 的参数元素数求和，确认真的加载了
        n_lora = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
        log(f"LoRA 适配器已加载，参数量 {n_lora:,}")

    model.eval()      # 切推理模式：关 dropout，确保输出确定
    return model


# ============================================================================
# 生成
# ============================================================================
@torch.no_grad()     # 推理不建计算图，省显存、提速
def generate(model, tokenizer, messages, max_new_tokens=120,
             temperature=0.7, top_p=0.9, do_sample=True):
    """
    根据完整对话历史生成回复。

    【参数怎么选】
        do_sample=True（默认）：采样解码，回答有变化，聊天更自然
        do_sample=False       ：贪心解码，每次答案相同，适合对比测试
        temperature
            0.1~0.3 → 保守，几乎照搬训练数据风格
            0.7     → 平衡（推荐）
            1.0+    → 发散，容易胡说
        top_p
            核采样，0.9 是常用值。与 temperature 配合控制随机性。

    【为什么要把 messages 整个传进来】
        这样模型能看到历史对话，实现多轮上下文记忆。
        如果只想单轮问答，把 messages 换成 [system, 当前user] 即可。
    """
    # 把完整对话历史渲染成模型模板文本（最后加"轮到 assistant"标记）
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # 转张量并搬到模型设备上
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # 生成长度上限要算上历史长度，否则多轮对话会很快被截断
    input_len = inputs["input_ids"].shape[1]      # 记录输入序列长度（第 1 维 = token 数）
    outputs = model.generate(
        **inputs,                                 # 展开 input_ids + attention_mask
        max_new_tokens=max_new_tokens,            # 最多新生成多少 token
        do_sample=do_sample,                      # 采样 or 贪心
        # 三元表达式：贪心模式下不传 temperature/top_p（传了反而会警告）
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        repetition_penalty=1.1,                   # 轻微抑制重复
        pad_token_id=tokenizer.pad_token_id,      # 消除警告
        eos_token_id=tokenizer.eos_token_id,      # 遇到结束符就停
    )

    # 只取新生成的部分
    new_tokens = outputs[0][input_len:]           # 切片掉输入部分
    # 解码成文字并去掉首尾空白
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
    /mode     查看当前模型模式
    /params   查看当前生成参数
    /exit     退出程序
------------------------------------------------------------------
  直接输入文字即可与模型对话（支持多轮上下文记忆）
"""


def main():
    """交互主循环：解析参数 → 加载模型 → 循环读输入 → 生成回复。"""
    # 创建参数解析器（description 会显示在 --help 里）
    parser = argparse.ArgumentParser(description="与原模型 / LoRA 模型交互对话")
    parser.add_argument("--mode", choices=["base", "lora"], default="lora",
                        help="base=原始模型, lora=微调模型（默认）")     # 二选一，默认 lora
    parser.add_argument("--lora", type=str, default=DEFAULT_LORA,
                        help="LoRA 适配器目录")                        # 可指定别的适配器
    parser.add_argument("--no-system", action="store_true",
                        help="不发送 system prompt（测行为是否内化）")   # 布尔开关
    parser.add_argument("--temp", type=float, default=0.7, help="temperature，默认 0.7")
    parser.add_argument("--max-tokens", type=int, default=120, help="单次最多生成 token 数")
    parser.add_argument("--greedy", action="store_true",
                        help="用贪心解码（每次答案相同，便于复现）")
    args = parser.parse_args()        # 解析命令行

    if USE_HF_MIRROR:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")   # 设镜像

    # ---------------- 加载 ----------------
    if torch.cuda.is_available():                        # 有 GPU 就打印型号
        name = torch.cuda.get_device_name(0)
        log(f"GPU: {name}")

    tokenizer = load_tokenizer()                         # 加载分词器

    if args.mode == "lora":                              # ---- LoRA 模式 ----
        log(f"模式: LoRA 微调模型")
        log(f"适配器: {args.lora}")
        model = load_model(True, args.lora)              # 传 True → 会挂适配器
    else:                                                # ---- 原始模式 ----
        log("模式: 原始底座模型（未微调）")
        model = load_model(False)                        # 传 False → 不挂适配器

    use_system = not args.no_system      # 没给 --no-system 就启用 system
    do_sample = not args.greedy          # 没给 --greedy 就用采样解码

    # ---------------- 欢迎信息 ----------------
    print()
    print("=" * 66)
    print(f"  交互式对话已启动")
    # 三元表达式根据 args.mode 选择显示的文本
    print(f"  模型模式 : {'LoRA 微调' if args.mode == 'lora' else '原始底座'}")
    print(f"  system   : {'已启用' if use_system else '已禁用（--no-system）'}")
    print(f"  解码方式 : {'贪心（结果固定）' if args.greedy else f'采样 (temp={args.temp})'}")
    print("=" * 66)
    print(HELP_TEXT)

    # ---------------- 对话历史 ----------------
    # 用列表保存历史，实现多轮上下文。
    # 结构：[{"role":"system","content":...}, {"role":"user",...}, {"role":"assistant",...}, ...]
    messages = []                                        # 初始为空列表
    if use_system:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})   # 首条放 system

    turn = 0                                             # 已完成的对话轮数计数器
    while True:                                          # ---- 无限循环直到用户退出 ----
        try:
            user_input = input("\n你 > ").strip()        # 读取输入并去掉首尾空白
        except (EOFError, KeyboardInterrupt):            # Ctrl+Z / Ctrl+C
            print("\n\n再见！")
            break                                        # 跳出循环，结束程序

        if not user_input:                               # 空输入直接跳过
            continue

        # ---------------- 处理快捷指令 ----------------
        if user_input.startswith("/"):                   # 以 / 开头视为指令
            cmd = user_input.lower()                     # 转小写，兼容 /HELP
            if cmd in ("/exit", "/quit"):                # 退出指令（两种写法都支持）
                print("再见！")
                break
            elif cmd == "/help":                         # 显示帮助
                print(HELP_TEXT)
            elif cmd == "/clear":                        # 清空历史，重新开始
                messages = []                            # 重置消息列表
                if use_system:
                    messages.append({"role": "system", "content": SYSTEM_PROMPT})   # 重新加 system
                turn = 0                                 # 轮数归零
                print("[已清空对话历史]")
            elif cmd == "/system":                       # 查看当前 system prompt
                if use_system:
                    print(f"[当前 system prompt]\n{SYSTEM_PROMPT}")
                else:
                    print("[当前未启用 system prompt]")
            elif cmd == "/mode":                         # 查看当前模型模式
                print(f"[模式] {'LoRA 微调' if args.mode == 'lora' else '原始底座'}")
                if args.mode == "lora":
                    print(f"[适配器] {args.lora}")
            elif cmd == "/params":                       # 查看生成参数
                print(f"[生成参数] max_new_tokens={args.max_tokens}, "
                      f"do_sample={do_sample}, temperature={args.temp}")
                print(f"[历史轮数] {turn}")
            else:                                        # 未知指令
                print(f"[未知指令] {user_input}，输入 /help 查看帮助")
            continue                                     # 指令处理完，回到循环开头等下一次输入

        # ---------------- 正常对话 ----------------
        messages.append({"role": "user", "content": user_input})   # 先把问题加进历史

        try:
            reply = generate(                            # 生成回复
                model, tokenizer, messages,              # 传完整历史 → 支持多轮
                max_new_tokens=args.max_tokens,
                temperature=args.temp,
                do_sample=do_sample,
            )
        except torch.cuda.OutOfMemoryError:              # 显存爆了
            print("\n[显存不足] 对话历史太长了，输入 /clear 清空后重试。")
            messages.pop()   # 回滚刚加入的这条，避免污染历史
            continue                                     # 不退出，让用户清空后继续

        print(f"\n模型 > {reply}")

        # 把模型回复也加入历史 —— 这样下一轮它就能"记得"自己说过什么
        messages.append({"role": "assistant", "content": reply})
        turn += 1                                        # 轮数 +1


# 入口守卫：直接运行时才执行 main()
if __name__ == "__main__":
    main()

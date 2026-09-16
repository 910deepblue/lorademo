# -*- coding: utf-8 -*-
"""
==============================================================================
 07_why_same.py —— 为什么"原始模型"看起来和 LoRA 一样？
==============================================================================

【这个脚本回答什么问题】
    很多人微调完会困惑："我把 --mode 换成 base，怎么回答还是带人设？"
    是不是 LoRA 没生效？

    答案是：**原始模型本来就能遵循 system prompt 演好这个角色。**

    Qwen2.5-1.5B-Instruct 经过指令微调，天生就会"照着你给的 system
    提示词来演"。人设规则（引条款、控数字、拒域外）全写在 system 里，
    它直接照做就能演得八九不离十。

    那 LoRA 到底改变了什么？本脚本用数据回答。

【LoRA 真正改变的三个维度】
    ① 条款引用   —— 原模型基本不引条例，LoRA 近乎 100% 带出「依据条例第X条」
    ② 格式稳定   —— 原模型时有时无，LoRA 稳定输出「结论 + 依据」两段式
    ③ 长度可控   —— 原模型忽长忽短，LoRA 稳定在条例要求的简洁区间

【★内存优化设计（本机只有 15.4GB，PyCharm 还要占 2.7GB）】
    绝不同时加载两份模型。用「一次只加载一个 → 跑完 → 彻底释放」的模式：
        加载原模型 → 跑所有问题 → del + gc + empty_cache → 加载 LoRA → 跑
    每次加载约 20 秒，换来内存绝对安全。

【怎么跑】
    # 完整对比（需要约 6GB 空闲内存，耗时约 2 分钟）
    python 07_why_same.py

    # 只测原模型（内存最紧张时用）
    python 07_why_same.py --only base

    # 只测 LoRA
    python 07_why_same.py --only lora
==============================================================================
"""

# ---- 标准库 ----
import argparse       # 解析 --only base|lora
import gc             # gc.collect() 手动触发垃圾回收，释放内存
import os             # 路径拼接
import statistics     # statistics.mean / stdev，算平均值和标准差

# ---- 第三方库 ----
import torch                                              # 显存清空、@torch.no_grad
from peft import PeftModel                                # 加载 LoRA 适配器
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig   # 加载三件套

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # E:\lorademo

# 本地权重优先的候选路径
MODEL_CANDIDATES = [
    r"D:/Model/Qwen2.5-1.5B-Instruct",    # 正斜杠写法 Windows 也接受；路径不存在时自动跳到下一项
    os.path.join(PROJECT_ROOT, "models", "Qwen2.5-1.5B-Instruct"),
]
MODEL_NAME = next((p for p in MODEL_CANDIDATES if os.path.isdir(p)), "Qwen/Qwen2.5-1.5B-Instruct")
LORA_DIR = os.path.join(PROJECT_ROOT, "output", "lora_qwen_bid")   # 适配器目录

# 推理用的 system prompt（与训练保持一致）
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

# 6 个测试问题：覆盖保证金/期限/评标/分包/投诉/开标，用来做长度与格式统计
QUESTIONS = [
    "投标保证金最多能收多少？",    # 金额类（触发规则 2）
    "招标文件最少发售几天？",      # 期限类（触发规则 2）
    "哪些情形应当否决投标？",      # 列举类（长答案）
    "中标项目可以分包吗？",        # 业务类
    "投诉的期限是多少天？",        # 期限类
    "开标现场能提异议吗？",        # 程序类
]


def log(msg):
    """统一日志前缀。"""
    print(f"[cmp] {msg}", flush=True)


def load_tokenizer():
    """加载分词器并补齐 pad_token。"""
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None:            # Qwen 缺 pad_token
        tok.pad_token = tok.eos_token    # 用 eos 兼任
    return tok


def build_model(use_lora: bool):
    """
    加载 4-bit 模型。

    ★ 内存优化要点
        low_cpu_mem_usage=True
            分块加载权重，避免在 CPU 内存里出现完整 fp32 副本。
            不加这个，3GB 模型加载瞬间可能占用十几 GB，直接触发
            `OSError: 页面文件太小 (os error 1455)`。
    """
    # 量化配置（与训练一致）
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,                      # 4-bit 存储
        bnb_4bit_quant_type="nf4",              # NF4 格式
        bnb_4bit_compute_dtype=torch.float16,   # 计算用 fp16（Turing 卡不支持 bf16）
        bnb_4bit_use_double_quant=True,         # 双重量化
    )
    # 加载底座：low_cpu_mem_usage 是本脚本能跑通的关键
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb,
        device_map="auto",              # 自动分配设备
        trust_remote_code=True,
        low_cpu_mem_usage=True,     # ★ 分块加载，避免内存峰值爆炸
    )
    if use_lora:                    # 需要就挂适配器
        model = PeftModel.from_pretrained(model, LORA_DIR)
    return model.eval()             # 切推理模式并返回


def release_model(model):
    """彻底释放模型，为下一轮腾出显存和内存。顺序很重要。"""
    del model                    # ① 删除引用，让对象变成可回收垃圾
    gc.collect()                 # ② 手动触发 Python 垃圾回收，回收 CPU 内存
    torch.cuda.empty_cache()     # ③ 清空 PyTorch 的显存缓存池
    torch.cuda.synchronize()     # ④ 等待所有 CUDA 操作完成，确保显存真的归还


@torch.no_grad()     # 推理不建计算图，省显存提速
def gen(model, tokenizer, question, use_system=True, max_new_tokens=90):
    """生成一次回答。use_system 控制是否带人设提示词。"""
    messages = []                                    # 消息列表
    if use_system:                                   # 需要就加 system
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": question})   # 用户问题

    # 渲染模板 + 转张量 + 搬到模型设备
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(                            # 生成
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,          # 贪心解码：结果确定，排除随机性
        repetition_penalty=1.1,   # 轻微抑制重复
        pad_token_id=tokenizer.pad_token_id,
    )
    # 切掉输入部分，只解码新生成的内容
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


def run_batch(tokenizer, use_lora: bool, with_system: bool):
    """加载一个模型 → 跑完所有问题 → 彻底释放。返回 {问题: 回答} 字典。"""
    tag = "LoRA" if use_lora else "原模型"                     # 日志用简称
    log(f"加载 [{tag}] system={'on' if with_system else 'off'} ...")
    model = build_model(use_lora)                              # 加载模型
    # 字典推导：对每个问题生成一次回答
    res = {q: gen(model, tokenizer, q, use_system=with_system) for q in QUESTIONS}
    release_model(model)                                       # ★ 用完立刻释放，防止两份模型共存 OOM
    log(f"  {tag} 完成并已释放内存")
    return res                                                 # 返回结果


def report(title, base_res, lora_res):
    """打印逐条对比 + 量化统计表。"""
    print()
    print("=" * 86)
    print(f"  {title}")
    print("=" * 86)
    for q in base_res:                                         # 遍历每个问题
        b, l = base_res[q], lora_res[q]                        # 两边的回答
        print(f"\nQ: {q}")
        print(f"  原模型({len(b):>3}字): {b}")                  # {len(b):>3} 右对齐 3 位，输出整齐
        print(f"  LoRA  ({len(l):>3}字): {l}")

    # 各回答的字数列表，用于统计
    bl = [len(base_res[q]) for q in base_res]
    ll = [len(lora_res[q]) for q in lora_res]

    def rate(res, toks):
        """计算回答中包含指定关键词的比例（百分比）。"""
        # 每个回答只要含 toks 中任一关键词就算命中；命中数 / 总数 × 100
        return sum(1 for q in res if any(t in res[q] for t in toks)) / len(res) * 100

    def cited(res):
        """统计带出『依据条例第X条』的回答比例（招投标域的关键格式指标）。"""
        return sum(1 for q in res if "依据条例第" in res[q]) / len(res) * 100

    print()
    print("-" * 86)
    print("  量化对比")
    print("-" * 86)
    print(f"  平均长度       原模型 {statistics.mean(bl):>5.0f} 字   |  LoRA {statistics.mean(ll):>5.0f} 字")
    print(f"  长度范围       原模型 {min(bl):>3}~{max(bl):<3} 字  |  LoRA {min(ll):>3}~{max(ll):<3} 字")
    # stdev 需要至少 2 个样本，否则报错；加三元判断兜底为 0
    print(f"  长度标准差     原模型 {statistics.stdev(bl) if len(bl)>1 else 0:>5.1f}      |  LoRA {statistics.stdev(ll) if len(ll)>1 else 0:>5.1f}    (越小越稳)")
    print(f"  带条款引用     原模型 {cited(base_res):>4.0f}%      |  LoRA {cited(lora_res):>4.0f}%    ← 关键差异")
    print(f"  带条例字样     原模型 {rate(base_res, ['条例']):>4.0f}%      |  LoRA {rate(lora_res, ['条例']):>4.0f}%")


def main():
    """主流程：第 1 组带 system、第 2 组不带 system，模型串行加载并释放。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["base", "lora"], default=None,
                    help="只测其中一个模型（内存紧张时用）")   # 只测一个，省内存
    args = ap.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")   # 设镜像
    tokenizer = load_tokenizer()                                    # 加载分词器

    log("=" * 62)
    log("第 1 组：都带 system prompt —— 看谁更稳定")
    log("=" * 62)
    # --only lora 时跳过原模型（得 None）；否则正常跑
    base_with = None if args.only == "lora" else run_batch(tokenizer, False, True)
    lora_with = None if args.only == "base" else run_batch(tokenizer, True, True)
    if base_with and lora_with:                       # 两份都有 → 打印对比报告
        report("对比 1：都带 system prompt", base_with, lora_with)
    else:                                             # 只有一份 → 简易打印
        got = base_with or lora_with                  # or 取出非 None 的那个
        tag = "原模型" if base_with else "LoRA"
        print(f"\n--- {tag} 结果 ---")
        for q, a in got.items():                      # items() 遍历字典键值对
            print(f"  Q: {q}\n     ({len(a)}字) {a}")

    # ---------- 第 2 组：去掉 system prompt ----------
    if not args.only:                                 # 只有全量对比时才跑第 2 组
        log("")
        log("=" * 62)
        log("第 2 组：都不带 system prompt —— 看谁真的内化了")
        log("=" * 62)
        base_wo = run_batch(tokenizer, False, False)  # 原模型，不带 system
        lora_wo = run_batch(tokenizer, True, False)   # LoRA，不带 system
        report("对比 2：都不带 system prompt", base_wo, lora_wo)

        print()
        print("=" * 86)
        print("  结论")
        print("=" * 86)
        # 三引号多行字符串：直接打印整段结论文本，保留原格式
        print("""
  为什么"原模型"也带人设？
    → Qwen2.5-1.5B-Instruct 本身就会遵循 system prompt。
      人设规则全写在 system 里，它照做就能演得很像。

  那 LoRA 改变了什么？重点看「带条款引用」这一行和「长度标准差」：
    · 原模型基本不引条例条款，措辞忽长忽短
    · LoRA 把"结论 + 依据条例第X条"压成了稳定习惯

  一句话：
    原模型 = 每次靠提示词现演，演得不稳
    LoRA   = 把"怎么演"练成了默认习惯

  想让 LoRA 彻底内化（去掉 system 也稳定）：
    · 数据加到几百条
    · 数据里混入"无 system prompt"的样本
    · 增加训练轮数（配合观察验证 loss）
""")


# 入口守卫：直接运行时才执行 main()
if __name__ == "__main__":
    main()

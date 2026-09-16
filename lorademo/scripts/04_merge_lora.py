# -*- coding: utf-8 -*-
"""
==============================================================================
 04_merge_lora.py —— 第 4 步（可选）：把 LoRA 合并进底座模型
==============================================================================

【为什么要合并？】
    LoRA 适配器是"外挂"，推理时要额外加载、额外计算（虽然开销很小）。
    合并（merge）就是把 B·A 算出来，直接加回原始权重 W：
        W_new = W + (alpha/r) * B·A
    合并后得到一个"完整模型"，和普通模型一样用，不需要 peft 库。

【什么时候需要合并？】
    ✓ 要部署到 vLLM / TensorRT / Ollama 等推理框架（它们不认 LoRA 格式）
    ✓ 要分享给别人（一个目录搞定，不用附带说明"记得装 peft"）
    ✗ 只是本地调试 → 不需要，03_inference.py 直接加载适配器更方便

【注意】
    合并后无法再"取消"LoRA 效果，也无法继续叠加新的 LoRA。
    所以建议先保留 output/lora_qwen_bid 目录，合并产物另外存放。

【怎么跑】
    E:\\lorademo\\910demo\\Scripts\\python.exe E:\\lorademo\\scripts\\04_merge_lora.py
==============================================================================
"""

# ---- 标准库 ----
import os         # 路径拼接、目录存在性检查

# ---- 第三方库 ----
import torch                                    # PyTorch：指定 fp16 数据类型、捕获显存异常
from peft import PeftModel                      # 加载 LoRA 适配器（并提供 merge_and_unload）
from transformers import AutoModelForCausalLM, AutoTokenizer   # 加载底座与分词器

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))    # E:\lorademo
LORA_DIR = os.path.join(PROJECT_ROOT, "output", "lora_qwen_bid")               # 输入：适配器目录
MERGED_DIR = os.path.join(PROJECT_ROOT, "output", "merged_qwen_bid")           # 输出：合并后完整模型

# 与训练脚本保持一致：本地权重优先，没有才走在线下载
MODEL_CANDIDATES = [
    r"D:\Model\Qwen2.5-1.5B-Instruct",
    os.path.join(PROJECT_ROOT, "models", "Qwen2.5-1.5B-Instruct"),
]
LOCAL_MODEL_DIR = next((p for p in MODEL_CANDIDATES if os.path.isdir(p)), None)
MODEL_NAME = LOCAL_MODEL_DIR if LOCAL_MODEL_DIR else "Qwen/Qwen2.5-1.5B-Instruct"
USE_HF_MIRROR = True    # 启用国内镜像


def log(msg: str):
    """统一日志前缀。"""
    print(f"[merge] {msg}", flush=True)


def main():
    """合并流程：检查适配器 → fp16 加载底座 → 挂 LoRA → 合并 → 保存。"""
    if USE_HF_MIRROR:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")   # 设镜像

    # 先检查适配器是否存在，避免白加载一遍模型
    if not os.path.isdir(LORA_DIR):
        log(f"[错误] 找不到 LoRA 适配器: {LORA_DIR}")
        log("       请先运行 02_train_lora.py")
        return                          # 直接返回，不继续

    # ------------------------------------------------------------------
    # 【关键】合并时必须用 fp16 全精度加载底座，不能用 4-bit！
    #
    # 原因：
    #   4-bit 是"有损压缩"。如果把 4-bit 的权重和 LoRA 合并，
    #   相当于把 LoRA 的增量信息也压进了有损表示里，会造成精度损失。
    #
    #   正确做法：fp16 加载 → 合并 → 保存 fp16 完整模型。
    #   如果显存不够，可以用 device_map="cpu" 在内存里合并（慢但可行）。
    #
    #   注：transformers 5.x 会把 torch_dtype 改名成 dtype，但旧名仍可用，
    #       只是会打一条 deprecation 警告，不影响功能。
    # ------------------------------------------------------------------
    log("以 fp16 全精度加载底座模型（合并必须如此，不能用 4-bit）...")
    try:                                # 先尝试放在 GPU 上合并（快）
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,   # 全精度 fp16，不用 4-bit（原因见上方注释）
            device_map="auto",           # 自动分配设备
            trust_remote_code=True,
        )
    except torch.cuda.OutOfMemoryError:  # 显存不够时走 CPU 兜底
        log("  显存不足，改用 CPU 合并（会慢一些）...")
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="cpu",            # 强制加载到内存
            trust_remote_code=True,
            low_cpu_mem_usage=True,      # 分块加载，避免内存里出现完整 fp32 副本
        )

    log("加载 LoRA 适配器...")
    # PeftModel.from_pretrained 会把 LoRA 层插进 base（原地修改 base）
    model = PeftModel.from_pretrained(base, LORA_DIR)

    log("合并权重（merge_and_unload）...")
    # merge_and_unload() 做了两件事：
    #   1) 计算 W + B·A 并写回原始权重
    #   2) 卸载 LoRA 相关模块，返回一个干净的普通模型
    model = model.merge_and_unload()

    log(f"保存合并后的完整模型到: {MERGED_DIR}")
    # safe_serialization=True 用 safetensors 格式保存（比 pickle 更安全、加载更快）
    model.save_pretrained(MERGED_DIR, safe_serialization=True)

    # tokenizer 也要一起存，否则部署时还得单独找
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.save_pretrained(MERGED_DIR)

    log("=" * 60)
    log("[OK] 合并完成")
    log(f"  产物目录: {MERGED_DIR}")
    log("  这个目录可以直接喂给 vLLM / Ollama / TensorRT-LLM 等框架")
    log("=" * 60)


# 入口守卫：直接运行时才执行 main()
if __name__ == "__main__":
    main()

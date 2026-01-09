import math, random, contextlib, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoProcessor
from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGenerationMask
from qwen_vl_utils import process_vision_info

# =============================
# 0) Backend / device (tiết kiệm VRAM)
# =============================
torch.backends.cuda.matmul.allow_tf32 = True
if torch.cuda.is_available():
    try:
        torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
        ATTN_IMPL = "flash_attention_2"
    except Exception:
        ATTN_IMPL = "sdpa"
else:
    ATTN_IMPL = None

device = "cuda" if torch.cuda.is_available() else "cpu"
bf16_ok = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8

@contextlib.contextmanager
def inference_mode():
    if torch.__version__ >= "2.0":
        with torch.inference_mode():
            yield
    else:
        with torch.no_grad():
            yield

# =============================
# 1) Load model / tokenizer / processor
# =============================
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
tokenizer  = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
processor  = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGenerationMask.from_pretrained(
    model_name,
    trust_remote_code=True,
    cache_dir="./cache_dir",
    device_map="auto",
    torch_dtype=torch.bfloat16 if bf16_ok else torch.float16,
    low_cpu_mem_usage=True,
    attn_implementation=ATTN_IMPL if ATTN_IMPL else None,
).eval()
model.config.use_cache = False

# (tuỳ chọn) giảm kích thước ảnh để bớt VRAM
if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "size"):
    try:
        processor.image_processor.size = {"shortest_edge": 448}
    except Exception:
        pass

# =============================
# 2) Inputs (ví dụ)
# =============================
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "dogandcat.png"},
            {"type": "text",  "text":  "This is a picture of a tree"},
        ],
    }
]

# Không add_generation_prompt để tránh bias tool
chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=[chat_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
full_ids = inputs["input_ids"][0].tolist()

# =============================
# 3) Mask token id & helpers
# =============================
if tokenizer.mask_token_id is None:
    mask_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if mask_token_id is None:
        raise ValueError("Tokenizer has no mask/pad/eos id.")
else:
    mask_token_id = tokenizer.mask_token_id

special_ids = set(tokenizer.all_special_ids + ([tokenizer.unk_token_id] if tokenizer.unk_token_id is not None else []))

def is_tool_like_token(s: str) -> bool:
    s = s.strip()
    if s == "":
        return True
    return s.startswith("<") and s.endswith(">")

def block_tool_like_topk(logits_1d, topk=50):
    """Lấy top-k đã lọc special + token dạng <...>; trả [(id, tok, prob_norm)]."""
    filt = logits_1d.clone()
    for sid in special_ids:
        if 0 <= sid < filt.numel():
            filt[sid] = float("-inf")
    probs = torch.softmax(filt, dim=-1)
    k = min(topk, probs.numel())
    p, ids = torch.topk(probs, k=k)
    kept, Z = [], 0.0
    for tid, pi in zip(ids.tolist(), p.tolist()):
        tok = tokenizer.decode([tid])
        if not is_tool_like_token(tok):
            kept.append((tid, tok.strip(), float(pi)))
            Z += float(pi)
    if Z == 0.0:
        return []
    return [(tid, tok, pi/Z) for tid, tok, pi in kept]

def last_user_text(msgs):
    for part in reversed(msgs[0]["content"]):
        if part.get("type") == "text":
            return part["text"]
    raise ValueError("No user text found in messages")

def find_subsequence(haystack, needle):
    for i in range(0, len(haystack) - len(needle) + 1):
        if haystack[i:i+len(needle)] == needle:
            return i
    return -1

user_text = last_user_text(messages)
user_ids  = tokenizer.encode(user_text, add_special_tokens=False)
start = find_subsequence(full_ids, user_ids)
if start == -1:
    alt = tokenizer.encode(" " + user_text, add_special_tokens=False)
    start = find_subsequence(full_ids, alt)
    if start != -1:
        user_ids = alt
if start == -1:
    raise RuntimeError("Cannot align user_text with input_ids.")
end = start + len(user_ids)          # [start, end)
text_positions = list(range(start, end))

# =============================
# 4) Hàm mask 1-span (k tokens) và dự đoán
# =============================
def mask_span_and_predict(span_start: int, span_len: int, topk: int = 5):
    """
    Mask liên tiếp span_len token trong vùng user-text, forward 1 lần,
    trả về list kết quả theo từng vị trí mask trong span.
    """
    assert span_len > 0
    assert start <= span_start < end and span_start + span_len <= end

    tmp = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in inputs.items()}
    # đặt mask cho span
    for j in range(span_start, span_start + span_len):
        tmp["input_ids"][0, j] = mask_token_id

    with inference_mode():
        out = model(
            **tmp,
            output_attentions=False,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True
        )
        logits = out.logits[0]  # [seq_len, vocab]

    results = []
    for j in range(span_start, span_start + span_len):
        li = logits[j].detach().to("cpu")
        # topk dự đoán sau lọc
        preds = block_tool_like_topk(li, topk=80)[:topk]
        # PLL của token gốc ở vị trí j (log prob khi vị trí j bị mask cùng với các vị trí khác)
        pll = F.log_softmax(li, dim=-1)[full_ids[j]].exp().item()
        results.append({
            "pos": j,
            "orig_tok": tokenizer.decode([full_ids[j]]).strip(),
            "preds": preds,
            "pll_orig": pll
        })

    # dọn
    del out, logits, tmp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results

# =============================
# 5) Lặp k = 1..n-1 theo 3 chế độ: 'prefix' | 'window' | 'random'
# =============================
def run_progressive_mask(mode: str = "prefix", max_random_trials: int = 4, topk: int = 5):
    """
    - prefix: lần lượt mask k token đầu của vùng text (k=1..n-1)
    - window: với mỗi k, trượt cửa sổ độ dài k trên vùng text
    - random: với mỗi k, chọn ngẫu nhiên 'max_random_trials' span độ dài k
    """
    n = len(user_ids)
    assert n >= 2, "Vùng text quá ngắn."

    print(f"\n=== Progressive mask mode: {mode} | n={n} ===")
    for k in range(1, n):  # k = 1 .. n-1
        print(f"\n--- k = {k} ---")
        spans = []

        if mode == "prefix":
            spans = [(start, k)]  # mask k token đầu vùng text

        elif mode == "window":
            for s in range(start, end - k + 1):
                spans.append((s, k))

        elif mode == "random":
            all_starts = list(range(start, end - k + 1))
            if not all_starts:
                continue
            # chọn ngẫu nhiên một số span để tránh in quá dài
            chosen = random.sample(all_starts, k=min(max_random_trials, len(all_starts)))
            spans = [(s, k) for s in chosen]

        else:
            raise ValueError("mode must be one of: 'prefix', 'window', 'random'")

        # chạy các span của k
        for (s, klen) in spans:
            res = mask_span_and_predict(s, klen, topk=topk)
            span_tokens = " ".join(tokenizer.decode([full_ids[j]]).strip() or "<SPACE>" for j in range(s, s+klen))
            print(f"[span s={s}, len={klen}]  masked: [{span_tokens}]")
            for r in res:
                top = r["preds"][0] if r["preds"] else None
                if top:
                    tid, tok, p = top
                    print(f"  pos {r['pos']}: orig='{r['orig_tok']}' -> '{tok}'  (p~{p:.4f}; PLL_orig={r['pll_orig']:.3e})")
                else:
                    print(f"  pos {r['pos']}: orig='{r['orig_tok']}' -> (no valid pred)  PLL_orig={r['pll_orig']:.3e}")

# =============================
# 6) CHẠY DEMO
# =============================
# Chế độ mặc định: mask prefix (k=1..n-1)
run_progressive_mask(mode="prefix", topk=5)

# Nếu muốn thử tất cả cửa sổ trượt cho mỗi k:
# run_progressive_mask(mode="window", topk=5)

# Hoặc chọn ngẫu nhiên vài span cho mỗi k (đỡ dài log):
# run_progressive_mask(mode="random", max_random_trials=3, topk=5)

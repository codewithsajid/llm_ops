import time, torch, transformers, os
from transformers import AutoTokenizer, AutoModelForCausalLM

_MODEL = "google/gemma-7b-it"          # 7‑B Instruct
_DEVICE = "cuda:0"

_tokenizer = AutoTokenizer.from_pretrained(_MODEL)
_model = AutoModelForCausalLM.from_pretrained(
    _MODEL,
    torch_dtype=torch.float16,
    device_map={"": _DEVICE},
)

def generate(prompt: str, t: float = 0.0, do_sample: bool=False, max_new_tokens: int = 256) -> tuple[str, dict]:
    t0 = time.perf_counter()
    inputs = _tokenizer(prompt, return_tensors="pt").to(_DEVICE)
    gen_ids = _model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        top_p=0.9,
        temperature=t,
        do_sample=do_sample,
    )
    text = _tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    stats = {
        "latency_s": round(time.perf_counter() - t0, 3),
        "n_input": inputs.input_ids.shape[-1],
        "n_output": gen_ids.shape[-1] - inputs.input_ids.shape[-1],
        "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    torch.cuda.reset_peak_memory_stats()
    return text, stats

if __name__ == "__main__":      # smoke‑test
    print(generate("Explain retrieval‑augmented generation in two lines.")[0])

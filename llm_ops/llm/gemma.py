import time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextIteratorStreamer
from threading import Thread

_MODEL   = "google/gemma-7b-it"
_DEVICE  = "cuda:0"

_tokenizer = AutoTokenizer.from_pretrained(_MODEL)
_model     = AutoModelForCausalLM.from_pretrained(
    _MODEL,
    torch_dtype=torch.float16,
    device_map={"": _DEVICE},
)

def generate(
    prompt: str,
    temperature: float = 1.0,                 # sensible default if sampling
    top_p: float = 0.9,
    do_sample: bool = True,
    max_new_tokens: int = 128,
):
    """Return (text, stats) tuple."""
    t0 = time.perf_counter()
    inputs = _tokenizer(prompt, return_tensors="pt").to(_DEVICE)

    gen_kw = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)

    # Only include sampler flags when actually sampling
    if do_sample:
        gen_kw.update({"temperature": temperature, "top_p": top_p})

    gen_ids = _model.generate(**inputs, **gen_kw)

    text = _tokenizer.decode(gen_ids[0],
                             skip_special_tokens=True,
                             clean_up_tokenization_spaces=True)

    stats = {
        "latency_s": round(time.perf_counter() - t0, 3),
        "n_input":   int(inputs.input_ids.shape[-1]),
        "n_output":  int(gen_ids.shape[-1] - inputs.input_ids.shape[-1]),
        "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    torch.cuda.reset_peak_memory_stats()
    return text, stats

def generate_stream(
    prompt: str,
    temperature: float = 1.0,
    top_p: float = 0.9,
    do_sample: bool = True,
    max_new_tokens: int = 128,
):
    """Streaming version that yields text chunks as they're generated."""
    t0 = time.perf_counter()
    inputs = _tokenizer(prompt, return_tensors="pt").to(_DEVICE)
    input_length = inputs.input_ids.shape[-1]
    
    # Create a streamer that will skip the prompt tokens
    streamer = TextIteratorStreamer(
        _tokenizer, 
        skip_prompt=True,  # This is the key parameter that skips prompt tokens
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=True
    )
    
    # Set generation parameters
    gen_kw = dict(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_new_tokens, 
        do_sample=do_sample,
        streamer=streamer
    )
    
    # Only include sampler flags when actually sampling
    if do_sample:
        gen_kw.update({"temperature": temperature, "top_p": top_p})
    
    # Create a thread to run generation in the background
    thread = Thread(target=_model.generate, kwargs=gen_kw)
    thread.start()
    
    # Track generated tokens for stats
    generated_tokens = 0
    
    # Yield tokens as they become available
    for new_text in streamer:
        generated_tokens += 1
        yield new_text
    
    # Wait for the generation to complete
    thread.join()
    
    # Provide stats after completion
    stats = {
        "latency_s": round(time.perf_counter() - t0, 3),
        "n_input": input_length,
        "n_output": generated_tokens,
        "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    torch.cuda.reset_peak_memory_stats()
    
    # Signal end of generation by yielding stats
    yield stats

# Smoke‑test
if __name__ == "__main__":
    print("— deterministic —")
    print(generate("Explain retrieval‑augmented generation in two lines.")[0])

    print("\n— sampled —")
    print(generate("Explain retrieval‑augmented generation in two lines.",
                   do_sample=True, t=0.8, top_p=0.95)[0])

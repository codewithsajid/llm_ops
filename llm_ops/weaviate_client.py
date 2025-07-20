# llm_ops/weaviate_client.py
import weaviate
from weaviate.embedded import EmbeddedOptions
from llm_ops.config import get_settings

_EMBED = EmbeddedOptions(
    persistence_data_path="/ssd_scratch/sajid.ansari/weaviate_embedded",
    port=7092,
)

def get_client() -> weaviate.Client:
    cfg = get_settings()
    if cfg.weaviate_url.startswith("embedded://"):
        return weaviate.Client(embedded_options=_EMBED)
    return weaviate.Client(cfg.weaviate_url)

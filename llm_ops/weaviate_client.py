# llm_ops/weaviate_client.py
import weaviate
from weaviate.client import WeaviateClient
from weaviate.exceptions import WeaviateStartUpError
from llm_ops.config import get_settings

def get_client() -> WeaviateClient:
    """Establishes a connection to Weaviate, handling existing embedded instances."""
    cfg = get_settings()
    
    http_port = 7092
    grpc_port = 50050
    
    if cfg.weaviate_url.startswith("embedded://"):
        try:
            # Try to start a new embedded instance
            return weaviate.connect_to_embedded(
                port=http_port,
                grpc_port=grpc_port,
                persistence_data_path="/ssd_scratch/sajid.ansari/weaviate_embedded"
            )
        except WeaviateStartUpError:
            # If it fails because ports are in use, connect to the existing one as suggested by the error.
            print("Embedded Weaviate is already running. Connecting to the existing instance.")
            return weaviate.connect_to_local(
                port=http_port,
                grpc_port=grpc_port
            )
    
    # Handle non-embedded connections
    parts = cfg.weaviate_url.replace("http://", "").replace("https://", "").split(":")
    host = parts[0]
    port = int(parts[1])
    
    return weaviate.connect_to_local(host=host, port=port)

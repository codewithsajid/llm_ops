# llm_ops/embedders.py
from sentence_transformers import SentenceTransformer
from open_clip import create_model_and_transforms
import torch

class TextEmbedder:
    _model = SentenceTransformer("intfloat/e5-base-v2", device="cuda:0")
    def __call__(self, texts: list[str]): return self._model.encode(texts, convert_to_numpy=True)

class ImageBindEmbedder:
    _model, _, _ = create_model_and_transforms("imagebind_large", device="cuda:1")
    def __call__(self, inputs):                     # inputs: dict(modality -> tensor)
        with torch.no_grad():
            return self._model(inputs)

import torch
from torch import nn


MODEL_ID = "google/siglip-base-patch16-224"
MODEL_REVISION = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"


class SigLIPEncoder(nn.Module):
    def __init__(self, backbone, image_processor, tokenizer):
        super().__init__()
        self.backbone = backbone
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.feature_dim = backbone.config.vision_config.hidden_size
        self.text_length = backbone.config.text_config.max_position_embeddings
        self.backbone.requires_grad_(False)
        self.train(False)

    @classmethod
    def from_pretrained(cls, model_path=None, *, local_files_only=False):
        from transformers import SiglipImageProcessor, SiglipModel, SiglipTokenizer

        source = MODEL_ID if model_path is None else str(model_path)
        options = {"local_files_only": local_files_only}
        if model_path is None:
            options["revision"] = MODEL_REVISION
        image_processor = SiglipImageProcessor.from_pretrained(source, **options)
        tokenizer = SiglipTokenizer.from_pretrained(source, **options)
        backbone = SiglipModel.from_pretrained(source, use_safetensors=True, **options)
        return cls(backbone, image_processor, tokenizer)

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def encode_images(self, images):
        if images.ndim != 4 or images.shape[1] != 3 or images.dtype != torch.uint8 or len(images) == 0:
            raise ValueError("Expected a nonempty uint8 image batch shaped (B, 3, H, W)")
        reference = next(self.backbone.parameters())
        pixels = self.image_processor(
            images=images.cpu(),
            input_data_format="channels_first",
            return_tensors="pt",
        )["pixel_values"]
        return self.backbone.get_image_features(
            pixel_values=pixels.to(device=reference.device, dtype=reference.dtype)
        )

    @torch.no_grad()
    def encode_text(self, instructions):
        if isinstance(instructions, str):
            raise ValueError("Expected a batch of instruction strings")
        texts = list(instructions)
        if not texts or not all(isinstance(text, str) and text.strip() for text in texts):
            raise ValueError("Instructions must be nonempty strings")
        device = next(self.backbone.parameters()).device
        tokens = self.tokenizer(
            texts, padding="max_length", max_length=self.text_length, truncation=True, return_tensors="pt"
        )
        return self.backbone.get_text_features(**tokens.to(device))

    def forward(self, fpv_images, tpv_images, instructions):
        if fpv_images.shape != tpv_images.shape or isinstance(instructions, str) or len(fpv_images) != len(instructions):
            raise ValueError("Both camera batches and the instruction batch must describe the same samples")
        fpv, tpv = self.encode_images(torch.cat((fpv_images, tpv_images), dim=0)).chunk(2, dim=0)
        text = self.encode_text(instructions)
        expected = (len(instructions), self.feature_dim)
        if any(features.shape != expected for features in (text, fpv, tpv)):
            raise ValueError(f"Expected text and camera features with shape {expected}")
        return {"text": text, "fpv": fpv, "tpv": tpv}

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Tuple


class MetaUEEncoder(nn.Module):
    """Frozen LLM embedding encoder. Handles tokenization and pooling.

    The model is always loaded to CPU first. Callers are responsible for
    moving the encoder to the target device via .to(device).
    """

    def __init__(
        self,
        embedding_model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
        pooling: str = "mean",
    ):
        super().__init__()
        self.pooling = pooling
        self.tokenizer = AutoTokenizer.from_pretrained(
            embedding_model_name, trust_remote_code=True
        )
        self.embedding_model = AutoModel.from_pretrained(
            embedding_model_name, trust_remote_code=True
        ).cpu()
        self.embedding_model.requires_grad_(False)
        self.embedding_model.eval()
        self.embed_dim: int = self.embedding_model.config.hidden_size

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return pooled embedding (B, D) on the same device as input_ids."""
        with torch.no_grad():
            outputs = self.embedding_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            hidden = outputs.last_hidden_state  # (B, L, D)

        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        elif self.pooling == "last":
            seq_lens = attention_mask.sum(dim=1) - 1  # (B,)
            pooled = hidden[torch.arange(hidden.size(0)), seq_lens]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        return pooled  # (B, D)

    def encode_both(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mean_pooled, last_pooled) in a single forward pass. (B, D) each."""
        with torch.no_grad():
            outputs = self.embedding_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            hidden = outputs.last_hidden_state  # (B, L, D)

        mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
        mean_pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        seq_lens = attention_mask.sum(dim=1) - 1  # (B,)
        last_pooled = hidden[torch.arange(hidden.size(0)), seq_lens]

        return mean_pooled, last_pooled  # both (B, D)


class MetaUEMLP(nn.Module):
    """Trainable 2-layer MLP on top of pre-computed embeddings.

    Outputs unbounded real-valued scores — no sigmoid applied. Safe for
    AUROC/AURAC (rank-invariant) and Youden-J threshold selection.
    """

    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        hidden_dim = embed_dim // 4
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.use_embeddings = True
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
                nn.init.zeros_(m.bias)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.mlp(embedding)

    def forward_from_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """Returns logit (B, 1)."""
        return self.mlp(embedding)

    def predict_proba_from_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """Returns raw uncertainty score for each sample. Shape (B,).

        Higher value = more uncertain.
        """
        return self.forward_from_embedding(embedding).squeeze(-1)


class MetaUEModel(nn.Module):
    """Combined encoder + MLP for end-to-end use.

    For sweep-based training use MetaUEEncoder (offline) + MetaUEMLP (online) separately.
    """

    def __init__(
        self,
        embedding_model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
        dropout: float = 0.1,
        pooling: str = "mean",
    ):
        super().__init__()
        self.encoder = MetaUEEncoder(embedding_model_name, pooling=pooling)
        self.mlp = MetaUEMLP(self.encoder.embed_dim, dropout=dropout)
        self.tokenizer = self.encoder.tokenizer
        self.embedding_model = self.encoder.embedding_model
        self.pooling = pooling
        self.embed_dim = self.encoder.embed_dim
        self.use_embeddings = False

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode(input_ids, attention_mask)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Returns logit (B, 1)."""
        return self.mlp.forward_from_embedding(self.encode(input_ids, attention_mask))

    def predict_proba(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Returns raw uncertainty score. Shape (B,). No sigmoid applied."""
        return self.forward(input_ids, attention_mask).squeeze(-1)

    def forward_from_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.mlp.forward_from_embedding(embedding)

    def predict_proba_from_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.mlp.predict_proba_from_embedding(embedding)

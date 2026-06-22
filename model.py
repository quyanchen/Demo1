import torch
import torch.nn as nn
from transformers import AutoModel


class TextClassifier(nn.Module):

    def __init__(
        self,
        pretrained_model_name,
        num_classes,
        dropout=0.1,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            pretrained_model_name
        )

        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            hidden_size,
            num_classes,
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        cls_embedding = outputs.last_hidden_state[:, 0]

        logits = self.classifier(
            self.dropout(cls_embedding)
        )

        return logits
import torch.nn as nn
from transformers import AutoModel

class TextClassifier(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(args.model_name, **kwargs)
        self.dropout = nn.Dropout(args.dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, args.num_classes)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        cls_embedding = outputs.last_hidden_state[:, 0]
        dropped_embedding = self.dropout(cls_embedding)
        logits = self.classifier(dropped_embedding)

        return logits
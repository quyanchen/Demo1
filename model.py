from torch import nn
from transformers import AutoModel

class BertClassifier(nn.Module):
    def __init__(self, model_name, num_labels, dropout=0.1, cache_dir=None):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_token = outputs.last_hidden_state[:, 0]
        return self.classifier(self.dropout(cls_token))

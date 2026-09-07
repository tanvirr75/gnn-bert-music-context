import torch.nn as nn


class BertTagClassifier(nn.Module):
    def __init__(self, bert, num_tags=50):
        super().__init__()
        self.bert = bert
        self.classifier = nn.Linear(768, num_tags)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_vector = output.last_hidden_state[:, 0, :]
        return self.classifier(cls_vector)


def freeze_bottom_layers(bert, num_layers_to_freeze=8):
    for param in bert.embeddings.parameters():
        param.requires_grad = False
    for layer in bert.encoder.layer[:num_layers_to_freeze]:
        for param in layer.parameters():
            param.requires_grad = False
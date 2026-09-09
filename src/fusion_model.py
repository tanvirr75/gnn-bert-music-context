import torch
import torch.nn as nn


class FusionModel(nn.Module):
    """Task 3, recommended architecture: cross-attention fusion of GNN + BERT."""
    def __init__(self, gnn_encoder, bert_model, num_tags=50, fused_dim=128):
        super().__init__()
        self.gnn = gnn_encoder
        self.bert = bert_model
        self.graph_proj = nn.Linear(gnn_encoder.out_dim, fused_dim)
        self.text_proj = nn.Linear(bert_model.config.hidden_size, fused_dim)
        self.attention = nn.MultiheadAttention(embed_dim=fused_dim, num_heads=4, batch_first=True)
        self.tag_head = nn.Linear(fused_dim * 2, num_tags)
        self.emotion_head = nn.Linear(fused_dim * 2, 2)

    def forward(self, graph_x, graph_edge_index, graph_batch, input_ids, attention_mask):
        graph_vector = self.gnn(graph_x, graph_edge_index, graph_batch)
        text_tokens = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        g = self.graph_proj(graph_vector).unsqueeze(1)
        t = self.text_proj(text_tokens)

        attended_text, _ = self.attention(query=g, key=t, value=t)
        combined = torch.cat([g.squeeze(1), attended_text.squeeze(1)], dim=-1)

        return self.tag_head(combined), self.emotion_head(combined)


class EarlyConcatFusionModel(nn.Module):
    """Task 3 ablation: same encoders, plain concatenation instead of cross-attention."""
    def __init__(self, gnn_encoder, bert_model, num_tags=50, fused_dim=128):
        super().__init__()
        self.gnn = gnn_encoder
        self.bert = bert_model
        self.graph_proj = nn.Linear(gnn_encoder.out_dim, fused_dim)
        self.text_proj = nn.Linear(bert_model.config.hidden_size, fused_dim)
        self.tag_head = nn.Linear(fused_dim * 2, num_tags)
        self.emotion_head = nn.Linear(fused_dim * 2, 2)

    def forward(self, graph_x, graph_edge_index, graph_batch, input_ids, attention_mask):
        graph_vector = self.gnn(graph_x, graph_edge_index, graph_batch)
        cls_vector = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]

        g = self.graph_proj(graph_vector)
        t = self.text_proj(cls_vector)
        combined = torch.cat([g, t], dim=-1)

        return self.tag_head(combined), self.emotion_head(combined)


def fusion_loss(tag_logits, emotion_pred, batch, alpha=0.3):
    """
    L = L_tags + alpha * L_emotion
    Only MTAT examples (has_emotion=False) contribute to L_tags; only
    DEAM examples (has_emotion=True) contribute to L_emotion.
    """
    has_emotion = batch.has_emotion
    is_mtat = ~has_emotion

    tag_loss = torch.tensor(0.0, device=tag_logits.device)
    if is_mtat.any():
        tag_loss = nn.functional.binary_cross_entropy_with_logits(tag_logits[is_mtat], batch.y[is_mtat])

    emotion_loss = torch.tensor(0.0, device=emotion_pred.device)
    if has_emotion.any():
        emotion_loss = nn.functional.mse_loss(emotion_pred[has_emotion], batch.emotion[has_emotion])

    total = tag_loss + alpha * emotion_loss
    return total, tag_loss.item(), emotion_loss.item()
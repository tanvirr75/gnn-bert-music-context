import torch
import torch.nn as nn
import torch.nn.functional as F


class DualEncoder(nn.Module):
    """Task 4: dual encoder for contrastive audio-text alignment."""
    def __init__(self, gnn_encoder, bert_model, projection_dim=128):
        super().__init__()
        self.gnn = gnn_encoder
        self.bert = bert_model
        self.graph_proj = nn.Linear(gnn_encoder.out_dim, projection_dim)
        self.text_proj = nn.Linear(bert_model.config.hidden_size, projection_dim)

    def encode_graph(self, x, edge_index, batch):
        graph_vector = self.gnn(x, edge_index, batch)
        return F.normalize(self.graph_proj(graph_vector), dim=-1)

    def encode_text(self, input_ids, attention_mask):
        cls_vector = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        return F.normalize(self.text_proj(cls_vector), dim=-1)

    def forward(self, graph_x, graph_edge_index, graph_batch, input_ids, attention_mask):
        g = self.encode_graph(graph_x, graph_edge_index, graph_batch)
        t = self.encode_text(input_ids, attention_mask)
        return g, t


def info_nce_loss(g, t, temperature=0.07):
    """g, t: (batch, dim), L2-normalized. In-batch negatives -- everything off the diagonal is a mismatched pair."""
    logits = g @ t.t() / temperature
    labels = torch.arange(g.size(0), device=g.device)
    loss_g_to_t = F.cross_entropy(logits, labels)
    loss_t_to_g = F.cross_entropy(logits.t(), labels)
    return (loss_g_to_t + loss_t_to_g) / 2


@torch.no_grad()
def get_all_embeddings(model, loader, device):
    """Embeds a full loader's worth of pairs at once -- retrieval searches across the whole val/test set, not one batch."""
    model.eval()
    all_g, all_t = [], []
    for batch in loader:
        batch = batch.to(device)
        g, t = model(batch.x, batch.edge_index, batch.batch, batch.input_ids, batch.attention_mask)
        all_g.append(g.cpu())
        all_t.append(t.cpu())
    return torch.cat(all_g), torch.cat(all_t)


def compute_retrieval_metrics(g, t, ks=(1, 5, 10)):
    """g, t: (N, dim), co-indexed -- g[i] is the true match for t[i]."""
    sim = t @ g.t()
    n = sim.size(0)
    results = {}
    for k in ks:
        k_eff = min(k, n)
        topk = sim.topk(k_eff, dim=1).indices
        hits = (topk == torch.arange(n).unsqueeze(1)).any(dim=1)
        results[f"caption_to_audio_R@{k}"] = hits.float().mean().item()

    sim_rev = g @ t.t()
    for k in ks:
        k_eff = min(k, n)
        topk = sim_rev.topk(k_eff, dim=1).indices
        hits = (topk == torch.arange(n).unsqueeze(1)).any(dim=1)
        results[f"audio_to_caption_R@{k}"] = hits.float().mean().item()
    return results
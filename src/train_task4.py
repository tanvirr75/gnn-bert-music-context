import torch
from transformers import AutoModel

from gnn_model import GNNEncoder
from bert_encoder import freeze_bottom_layers
from contrastive import DualEncoder, info_nce_loss, get_all_embeddings, compute_retrieval_metrics


def train_task4(train_loader, val_loader, device, num_epochs=30, patience=6, temperature=0.07):
    gnn_encoder = GNNEncoder(in_dim=128, hidden_dim=128, num_layers=3, dropout=0.2)
    bert_model = AutoModel.from_pretrained("bert-base-uncased")
    freeze_bottom_layers(bert_model, num_layers_to_freeze=8)
    model = DualEncoder(gnn_encoder, bert_model, projection_dim=128).to(device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
    best_val_r5, best_state, best_epoch, epochs_without_improvement = 0.0, None, None, 0
    history = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            g, t = model(batch.x, batch.edge_index, batch.batch, batch.input_ids, batch.attention_mask)
            loss = info_nce_loss(g, t, temperature)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()

        val_g, val_t = get_all_embeddings(model, val_loader, device)
        val_metrics = compute_retrieval_metrics(val_g, val_t)
        avg_r5 = (val_metrics["caption_to_audio_R@5"] + val_metrics["audio_to_caption_R@5"]) / 2

        history.append({"epoch": epoch + 1, "train_loss": total_loss / len(train_loader), **val_metrics})
        print(f"Epoch {epoch+1} | loss: {total_loss/len(train_loader):.4f} | "
              f"val C->A R@5: {val_metrics['caption_to_audio_R@5']:.4f} | val A->C R@5: {val_metrics['audio_to_caption_R@5']:.4f}")

        if avg_r5 > best_val_r5:
            best_val_r5, best_epoch = avg_r5, epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"No improvement for {patience} epochs — stopping at epoch {epoch+1}.")
                break

    model.load_state_dict(best_state)
    return model, history, best_epoch
import torch
from transformers import AutoModel
from sklearn.metrics import f1_score, average_precision_score

from gnn_model import GNNEncoder
from bert_encoder import freeze_bottom_layers
from fusion_model import FusionModel, EarlyConcatFusionModel, fusion_loss


def _run_eval(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            tag_logits, _ = model(batch.x, batch.edge_index, batch.batch, batch.input_ids, batch.attention_mask)
            all_logits.append(tag_logits.cpu())
            all_labels.append(batch.y.cpu())
    probs = torch.sigmoid(torch.cat(all_logits)).numpy()
    preds = (probs > 0.5).astype(int)
    labels_np = torch.cat(all_labels).numpy()
    return {
        "macro_f1": f1_score(labels_np, preds, average="macro", zero_division=0),
        "micro_f1": f1_score(labels_np, preds, average="micro", zero_division=0),
        "pr_auc": average_precision_score(labels_np, probs, average="macro"),
    }


def _train_loop(model, mtat_train_loader, deam_train_loader, mtat_val_loader, mtat_test_loader,
                 device, num_epochs, patience, alpha, label):
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
    best_val_f1, best_state, best_epoch, epochs_without_improvement = 0.0, None, None, 0
    history = []

    for epoch in range(num_epochs):
        model.train()
        total_loss, num_batches = 0, 0
        for batch in mtat_train_loader:
            batch = batch.to(device)
            tag_logits, emotion_pred = model(batch.x, batch.edge_index, batch.batch, batch.input_ids, batch.attention_mask)
            loss, tag_l, _ = fusion_loss(tag_logits, emotion_pred, batch, alpha)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += tag_l; num_batches += 1

        for batch in deam_train_loader:
            batch = batch.to(device)
            tag_logits, emotion_pred = model(batch.x, batch.edge_index, batch.batch, batch.input_ids, batch.attention_mask)
            loss, _, _ = fusion_loss(tag_logits, emotion_pred, batch, alpha)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        val_metrics = _run_eval(model, mtat_val_loader, device)
        history.append({"epoch": epoch + 1, "val_macro_f1": val_metrics["macro_f1"], "val_pr_auc": val_metrics["pr_auc"]})
        print(f"[{label}] Epoch {epoch+1} | avg tag loss: {total_loss/num_batches:.4f} | val macro-F1: {val_metrics['macro_f1']:.4f}")

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1, best_epoch = val_metrics["macro_f1"], epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"[{label}] No improvement for {patience} epochs — stopping at epoch {epoch+1}.")
                break

    model.load_state_dict(best_state)
    test_metrics = _run_eval(model, mtat_test_loader, device)
    print(f"[{label}] Final TEST metrics (best epoch {best_epoch}):", test_metrics)
    return model, test_metrics, history, best_epoch


def train_task3(mtat_train_loader, deam_train_loader, mtat_val_loader, mtat_test_loader,
                 device, num_epochs=25, patience=5, alpha=0.3):
    """Task 3, recommended architecture: cross-attention fusion."""
    gnn_encoder = GNNEncoder(in_dim=128, hidden_dim=128, num_layers=3, dropout=0.2)
    bert_model = AutoModel.from_pretrained("bert-base-uncased")
    freeze_bottom_layers(bert_model, num_layers_to_freeze=8)
    model = FusionModel(gnn_encoder, bert_model, num_tags=50).to(device)
    return _train_loop(model, mtat_train_loader, deam_train_loader, mtat_val_loader, mtat_test_loader,
                        device, num_epochs, patience, alpha, label="Cross-attention")


def train_task3_early_concat(mtat_train_loader, deam_train_loader, mtat_val_loader, mtat_test_loader,
                               device, num_epochs=25, patience=5, alpha=0.3):
    """Task 3 ablation: same setup, early-concat instead of cross-attention."""
    gnn_encoder = GNNEncoder(in_dim=128, hidden_dim=128, num_layers=3, dropout=0.2)
    bert_model = AutoModel.from_pretrained("bert-base-uncased")
    freeze_bottom_layers(bert_model, num_layers_to_freeze=8)
    model = EarlyConcatFusionModel(gnn_encoder, bert_model, num_tags=50).to(device)
    return _train_loop(model, mtat_train_loader, deam_train_loader, mtat_val_loader, mtat_test_loader,
                        device, num_epochs, patience, alpha, label="Early-concat")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, average_precision_score

from bert_encoder import BertTagClassifier, freeze_bottom_layers
from datasets import TagDataset


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids, attention_mask)
        all_logits.append(logits.cpu())
        all_labels.append(batch["label"])

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    probs = torch.sigmoid(all_logits).numpy()
    preds = (probs > 0.5).astype(int)
    labels_np = all_labels.numpy()

    return {
        "macro_f1": f1_score(labels_np, preds, average="macro", zero_division=0),
        "micro_f1": f1_score(labels_np, preds, average="micro", zero_division=0),
        "pr_auc": average_precision_score(labels_np, probs, average="macro"),
    }


def train_task1(train_df, val_df, test_df, top_50_tags, device, num_epochs=25, patience=5):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert = AutoModel.from_pretrained("bert-base-uncased")
    freeze_bottom_layers(bert, num_layers_to_freeze=8)

    model = BertTagClassifier(bert, num_tags=len(top_50_tags)).to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(TagDataset(train_df, top_50_tags, tokenizer), batch_size=32, shuffle=True)
    val_loader = DataLoader(TagDataset(val_df, top_50_tags, tokenizer), batch_size=32, shuffle=False)
    test_loader = DataLoader(TagDataset(test_df, top_50_tags, tokenizer), batch_size=32, shuffle=False)

    best_val_f1, best_state, best_epoch, epochs_without_improvement = 0.0, None, None, 0
    history = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_metrics = evaluate(model, val_loader, device)
        avg_train_loss = total_loss / len(train_loader)
        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_macro_f1": val_metrics["macro_f1"],
            "val_micro_f1": val_metrics["micro_f1"],
            "val_pr_auc": val_metrics["pr_auc"],
        })

        print(f"Epoch {epoch+1} | train loss: {avg_train_loss:.4f} | "
              f"val macro-F1: {val_metrics['macro_f1']:.4f} | val PR-AUC: {val_metrics['pr_auc']:.4f}")

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1, best_epoch = val_metrics["macro_f1"], epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"No improvement for {patience} epochs — stopping early at epoch {epoch+1}.")
                break

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    print(f"Final test metrics (best epoch {best_epoch}):", test_metrics)
    return model, test_metrics, history, best_epoch
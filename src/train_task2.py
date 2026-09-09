import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader
from sklearn.metrics import f1_score, average_precision_score, accuracy_score

from gnn_model import GNNEncoder, GNNTagClassifier, GNNGenreClassifier
from baselines import CNNMelBaseline, CNNGenreBaseline
from datasets import GraphTagDataset, SegmentMatrixDataset, GraphGenreDataset, GTZANSegmentDataset


@torch.no_grad()
def evaluate_gnn(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        all_logits.append(logits.cpu()); all_labels.append(batch.y.cpu())
    all_logits, all_labels = torch.cat(all_logits), torch.cat(all_labels)
    probs = torch.sigmoid(all_logits).numpy()
    preds = (probs > 0.5).astype(int)
    labels_np = all_labels.numpy()
    return {
        "macro_f1": f1_score(labels_np, preds, average="macro", zero_division=0),
        "micro_f1": f1_score(labels_np, preds, average="micro", zero_division=0),
        "pr_auc": average_precision_score(labels_np, probs, average="macro"),
    }


def train_task2(train_loader, val_loader, test_loader, train_df, tag_columns, device, num_epochs=40, patience=8):
    positive_counts = train_df[tag_columns].sum().values
    negative_counts = len(train_df) - positive_counts
    pos_weight = torch.tensor(negative_counts / (positive_counts + 1e-8), dtype=torch.float32).to(device)
    pos_weight = torch.clamp(pos_weight, max=20.0)

    encoder = GNNEncoder(in_dim=128, hidden_dim=128, num_layers=3, dropout=0.2)
    model = GNNTagClassifier(encoder, num_tags=len(tag_columns)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_f1, best_state, best_epoch, epochs_without_improvement = 0.0, None, None, 0
    history = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_fn(logits, batch.y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()

        val_metrics = evaluate_gnn(model, val_loader, device)
        history.append({"epoch": epoch + 1, "train_loss": total_loss / len(train_loader), **{f"val_{k}": v for k, v in val_metrics.items()}})
        print(f"Epoch {epoch+1} | train loss: {total_loss/len(train_loader):.4f} | val macro-F1: {val_metrics['macro_f1']:.4f} | val PR-AUC: {val_metrics['pr_auc']:.4f}")

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1, best_epoch = val_metrics["macro_f1"], epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"No improvement for {patience} epochs — stopping at epoch {epoch+1}.")
                break

    model.load_state_dict(best_state)
    test_metrics = evaluate_gnn(model, test_loader, device)
    print(f"Final test metrics (best epoch {best_epoch}):", test_metrics)
    return model, test_metrics, history, best_epoch


def train_task2_cnn(train_df, val_df, test_df, tag_columns, device, num_epochs=80, patience=8, max_segments=8):
    positive_counts = train_df[tag_columns].sum().values
    negative_counts = len(train_df) - positive_counts
    pos_weight = torch.tensor(negative_counts / (positive_counts + 1e-8), dtype=torch.float32).to(device)
    pos_weight = torch.clamp(pos_weight, max=20.0)

    model = CNNMelBaseline(num_tags=len(tag_columns)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loader = DataLoader(SegmentMatrixDataset(train_df, tag_columns, max_segments), batch_size=32, shuffle=True)
    val_loader = DataLoader(SegmentMatrixDataset(val_df, tag_columns, max_segments), batch_size=32, shuffle=False)
    test_loader = DataLoader(SegmentMatrixDataset(test_df, tag_columns, max_segments), batch_size=32, shuffle=False)

    best_val_f1, best_state, best_epoch, epochs_without_improvement = 0.0, None, None, 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x, labels in train_loader:
            x, labels = x.to(device), labels.to(device)
            logits = model(x)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()

        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for x, labels in val_loader:
                all_logits.append(model(x.to(device)).cpu()); all_labels.append(labels)
        val_probs = torch.sigmoid(torch.cat(all_logits)).numpy()
        val_f1 = f1_score(torch.cat(all_labels).numpy(), (val_probs > 0.5).astype(int), average="macro", zero_division=0)
        print(f"Epoch {epoch+1} | train loss: {total_loss/len(train_loader):.4f} | val macro-F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1, best_epoch = val_f1, epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"No improvement for {patience} epochs — stopping at epoch {epoch+1}.")
                break

    model.load_state_dict(best_state)
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, labels in test_loader:
            all_logits.append(model(x.to(device)).cpu()); all_labels.append(labels)
    test_probs = torch.sigmoid(torch.cat(all_logits)).numpy()
    test_preds = (test_probs > 0.5).astype(int)
    test_labels_np = torch.cat(all_labels).numpy()
    test_metrics = {
        "macro_f1": f1_score(test_labels_np, test_preds, average="macro", zero_division=0),
        "micro_f1": f1_score(test_labels_np, test_preds, average="micro", zero_division=0),
        "pr_auc": average_precision_score(test_labels_np, test_probs, average="macro"),
    }
    print(f"Final test metrics (best epoch {best_epoch}):", test_metrics)
    return model, test_metrics


@torch.no_grad()
def evaluate_gtzan(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        all_logits.append(logits.cpu()); all_labels.append(batch.y.cpu())
    all_logits, all_labels = torch.cat(all_logits), torch.cat(all_labels)
    preds = all_logits.argmax(dim=1).numpy()
    labels_np = all_labels.numpy()
    return {"accuracy": accuracy_score(labels_np, preds), "macro_f1": f1_score(labels_np, preds, average="macro", zero_division=0)}


def train_gtzan_gnn(train_df, val_df, test_df, device, num_epochs=60, patience=10):
    train_loader = GeoDataLoader(GraphGenreDataset(train_df), batch_size=16, shuffle=True)
    val_loader = GeoDataLoader(GraphGenreDataset(val_df), batch_size=16, shuffle=False)
    test_loader = GeoDataLoader(GraphGenreDataset(test_df), batch_size=16, shuffle=False)

    encoder = GNNEncoder(in_dim=128, hidden_dim=64, num_layers=2, dropout=0.3)
    model = GNNGenreClassifier(encoder, num_classes=10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    best_val_f1, best_state, best_epoch, epochs_without_improvement = 0.0, None, None, 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_fn(logits, batch.y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()

        val_metrics = evaluate_gtzan(model, val_loader, device)
        print(f"[GTZAN] Epoch {epoch+1} | loss: {total_loss/len(train_loader):.4f} | val acc: {val_metrics['accuracy']:.4f} | val macro-F1: {val_metrics['macro_f1']:.4f}")

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1, best_epoch = val_metrics["macro_f1"], epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"No improvement for {patience} epochs — stopping at epoch {epoch+1}.")
                break

    model.load_state_dict(best_state)
    test_metrics = evaluate_gtzan(model, test_loader, device)
    print(f"[GTZAN] Final test metrics (best epoch {best_epoch}):", test_metrics)
    return model, test_metrics


def train_gtzan_cnn(train_df, val_df, test_df, device, num_epochs=60, patience=10, max_segments=8):
    train_loader = DataLoader(GTZANSegmentDataset(train_df, max_segments), batch_size=16, shuffle=True)
    val_loader = DataLoader(GTZANSegmentDataset(val_df, max_segments), batch_size=16, shuffle=False)
    test_loader = DataLoader(GTZANSegmentDataset(test_df, max_segments), batch_size=16, shuffle=False)

    model = CNNGenreBaseline(num_classes=10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    best_val_f1, best_state, best_epoch, epochs_without_improvement = 0.0, None, None, 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x, labels in train_loader:
            x, labels = x.to(device), labels.to(device)
            logits = model(x)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()

        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for x, labels in val_loader:
                all_logits.append(model(x.to(device)).cpu()); all_labels.append(labels)
        val_preds = torch.cat(all_logits).argmax(dim=1).numpy()
        val_f1 = f1_score(torch.cat(all_labels).numpy(), val_preds, average="macro", zero_division=0)
        print(f"[GTZAN-CNN] Epoch {epoch+1} | train loss: {total_loss/len(train_loader):.4f} | val macro-F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1, best_epoch = val_f1, epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"No improvement for {patience} epochs — stopping at epoch {epoch+1}.")
                break

    model.load_state_dict(best_state)
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, labels in test_loader:
            all_logits.append(model(x.to(device)).cpu()); all_labels.append(labels)
    test_preds = torch.cat(all_logits).argmax(dim=1).numpy()
    test_labels_np = torch.cat(all_labels).numpy()
    test_metrics = {"accuracy": accuracy_score(test_labels_np, test_preds), "macro_f1": f1_score(test_labels_np, test_preds, average="macro", zero_division=0)}
    print(f"[GTZAN-CNN] Final test metrics (best epoch {best_epoch}):", test_metrics)
    return model, test_metrics
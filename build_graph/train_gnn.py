import os
import json
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv
from data_utils import load_graph


class GCN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def _create_stratified_masks(y: torch.Tensor, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    num_nodes = y.size(0)
    num_classes = int(y.max().item() + 1)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    for c in range(num_classes):
        idx = (y == c).nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            continue
        perm = idx[torch.randperm(idx.numel())]
        n = perm.numel()
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        # remainder goes to test
        train_mask[perm[:n_train]] = True
        val_mask[perm[n_train:n_train + n_val]] = True
        test_mask[perm[n_train + n_val:]] = True
    # If any node is unassigned due to rounding, push to test
    unassigned = ~(train_mask | val_mask | test_mask)
    if unassigned.any():
        test_mask[unassigned] = True
    return train_mask, val_mask, test_mask


def evaluate(logits, y, mask):
    if mask.sum() == 0:
        return 0.0
    preds = logits.argmax(dim=1)
    correct = (preds[mask] == y[mask]).sum().item()
    total = int(mask.sum().item())
    return correct / max(total, 1)


if __name__ == '__main__':
    torch.manual_seed(42)
    os.makedirs('training_output', exist_ok=True)

    data = load_graph('graph_output/clean_graph.pyg_en_labeled.pt')
    if getattr(data, 'y', None) is None:
        raise ValueError("Labels 'y' are required for supervised training.")
    if not (torch.is_tensor(getattr(data, 'edge_index', None)) and data.edge_index.dim() == 2 and data.edge_index.size(0) == 2):
        raise ValueError("edge_index is missing or invalid; cannot train GCN. Ensure it is convertible to a [2, E] LongTensor.")

    in_channels = data.x.size(1)
    out_channels = int(data.y.max().item() + 1)

    if not hasattr(data, 'train_mask') or data.train_mask is None:
        train_mask, val_mask, test_mask = _create_stratified_masks(data.y)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    model = GCN(in_channels, hidden_channels=64, out_channels=out_channels, dropout=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_val = 0.0
    best_state = None
    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            train_acc = evaluate(logits, data.y, data.train_mask)
            val_acc = evaluate(logits, data.y, data.val_mask)
            test_acc = evaluate(logits, data.y, data.test_mask)
        if val_acc >= best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | loss {loss.item():.4f} | train {train_acc:.3f} | val {val_acc:.3f} | test {test_acc:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        train_acc = evaluate(logits, data.y, data.train_mask)
        val_acc = evaluate(logits, data.y, data.val_mask)
        test_acc = evaluate(logits, data.y, data.test_mask)
    metrics = {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "device": str(device),
    }
    with open(os.path.join('training_output', 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    torch.save(model.state_dict(), os.path.join('training_output', 'model.pt'))
    print("Saved metrics to training_output/metrics.json and model to training_output/model.pt")



#!/usr/bin/env python3
"""
Train a dual-encoder for text ↔ subgraph retrieval using pairs JSONL.

Pairs format (JSONL):
  {"query": "...", "subgraph_path": "path/to/subgraph.pt", "label": 0/1}

This script:
  - Loads (query, HeteroData, y) with retrieval/pairs_dataset.py
  - Encodes queries with SBERT or CLIP (text-only)
  - Encodes subgraphs with a hetero GNN (TransformerConv with edge_attr)
  - Scores with dot-product and optimizes BCEWithLogitsLoss
  - Saves model checkpoints and metrics

Example:
  python3 retrieval/train_gnn_pairs.py \
    --train /home/ghr/angel/gnn/graph_embedding/pairs_all/train.jsonl \
    --val   /home/ghr/angel/gnn/graph_embedding/pairs_all/val.jsonl \
    --text-backend sbert --sbert-model sentence-transformers/all-MiniLM-L6-v2 \
    --epochs 5 --batch-size 16 --out /home/ghr/angel/gnn/training_output
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch  # type: ignore
from torch_geometric.nn import HeteroConv, TransformerConv, global_mean_pool  # type: ignore
from torch.optim import lr_scheduler

from retrieval.pairs_dataset import PairDataset, collate_pairs, _load_subgraph

# Optional backends
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None
try:
    import clip  # type: ignore
except Exception:
    clip = None
try:
    import wandb  # type: ignore
except Exception:
    wandb = None

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
class HeteroEdgePromptPlus(nn.Module):
    """
    EdgePrompt for heterogeneous graphs:
    - Per node type: projection to a shared prompt dimension
    - Per edge type: anchor prompts [K, d] and scorer W: R^{2d}->R^{K}
    """
    def __init__(self, node_dims: Dict[str, int], edge_types: Tuple[Tuple[str, str, str], ...],
                 prompt_dim: int, num_anchors: int, device: torch.device):
        super().__init__()
        self.prompt_dim = prompt_dim
        self.num_anchors = num_anchors
        # Per node type projection
        self.node_proj = nn.ModuleDict({
            nt: nn.Linear(in_dim, prompt_dim) for nt, in_dim in node_dims.items()
        }).to(device)
        # Per edge type anchors and scorers
        self.anchor_prompt = nn.ParameterDict()
        self.w_scorer = nn.ModuleDict()
        for et in edge_types:
            key = self._et_key(et)
            self.anchor_prompt[key] = nn.Parameter(torch.empty(num_anchors, prompt_dim, device=device))
            self.w_scorer[key] = nn.Linear(2 * prompt_dim, num_anchors).to(device)
        self.reset_parameters()

    @staticmethod
    def _et_key(et: Tuple[str, str, str]) -> str:
        return f"{et[0]}__{et[1]}__{et[2]}"

    def reset_parameters(self):
        for proj in self.node_proj.values():
            nn.init.xavier_uniform_(proj.weight); nn.init.zeros_(proj.bias)
        for k in list(self.anchor_prompt.keys()):
            nn.init.xavier_uniform_(self.anchor_prompt[k])
        for w in self.w_scorer.values():
            nn.init.xavier_uniform_(w.weight); nn.init.zeros_(w.bias)

    def get_edge_prompts(self, x_dict: Dict[str, torch.Tensor],
                         edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]) -> Dict[Tuple[str, str, str], torch.Tensor]:
        # Ensure projections exist for any new node types at runtime
        x_prompt: Dict[str, torch.Tensor] = {}
        for nt, x in x_dict.items():
            if nt not in self.node_proj:
                # Dynamically register a projection for unseen node type
                proj = nn.Linear(int(x.size(-1)), self.prompt_dim).to(x.device)
                nn.init.xavier_uniform_(proj.weight); nn.init.zeros_(proj.bias)
                self.node_proj[nt] = proj
            x_prompt[nt] = self.node_proj[nt](x)
        edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
        for et, ei in edge_index_dict.items():
            key = self._et_key(et)
            src_t, _, dst_t = et
            # Lazily create scorer/anchors for unseen relations
            if key not in self.w_scorer:
                self.w_scorer[key] = nn.Linear(2 * self.prompt_dim, self.num_anchors).to(x_prompt[src_t].device)
                nn.init.xavier_uniform_(self.w_scorer[key].weight); nn.init.zeros_(self.w_scorer[key].bias)
            if key not in self.anchor_prompt:
                param = nn.Parameter(torch.empty(self.num_anchors, self.prompt_dim, device=x_prompt[src_t].device))
                nn.init.xavier_uniform_(param)
                self.anchor_prompt[key] = param
            u = x_prompt[src_t][ei[0]]  # [E, d]
            v = x_prompt[dst_t][ei[1]]  # [E, d]
            combined = torch.cat([u, v], dim=1)  # [E, 2d]
            b = F.softmax(F.leaky_relu(self.w_scorer[key](combined)), dim=1)  # [E, K]
            prompt = b @ self.anchor_prompt[key]  # [E, d]
            edge_attr_dict[et] = prompt
        return edge_attr_dict
    
class TextEncoder(nn.Module):
    def __init__(self, backend: str, sbert_model: str, clip_model: str, out_dim: int, finetune_text: bool = False):
        super().__init__()
        self.backend = backend
        self.out_dim = out_dim
        self.finetune = finetune_text

        if backend == "sbert":
            if SentenceTransformer is None:
                raise RuntimeError("sentence-transformers not installed. pip install sentence-transformers")
            self.model = SentenceTransformer(sbert_model)
            in_dim = self.model.get_sentence_embedding_dimension()
        elif backend == "clip":
            if clip is None:
                raise RuntimeError("CLIP not installed. pip install git+https://github.com/openai/CLIP.git")
            # CLIP returns a tuple (model, preprocess)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_device = device
            self.model, _ = clip.load(clip_model, device=device)
            # probe dim
            with torch.no_grad():
                tok = clip.tokenize(["probe"], truncate=True).to(device)
                v = self.model.encode_text(tok).float()
                in_dim = v.shape[-1]
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self.proj = nn.Linear(in_dim, out_dim)

        if not self.finetune:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, queries: list[str]) -> torch.Tensor:
        if self.backend == "sbert":
            # SentenceTransformer handles device internally; returns CPU tensor if convert_to_tensor=True
            t = self.model.encode(queries, convert_to_tensor=True, normalize_embeddings=True).float()
            t = t.to(self.proj.weight.device)
        else:
            with torch.no_grad() if not self.finetune else torch.enable_grad():
                tok = clip.tokenize(queries, truncate=True).to(self.clip_device)
                feats = self.model.encode_text(tok).float()
                feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)
                t = feats.to(self.proj.weight.device)
        # Ensure 't' is a regular tensor (not an inference-mode tensor) so autograd can save it for backward
        t = t.clone()
        return F.normalize(self.proj(t), dim=-1)
    
class HeteroSubgraphEncoder(nn.Module):
    """
    Hetero GNN that consumes node features x and edge_attr, and outputs a pooled subgraph embedding.
    - Builds one TransformerConv per edge type (created lazily on first forward)
    - Pools per node type with global_mean_pool, then averages across types
    """
    def __init__(self, out_dim: int, dropout: float = 0.1,
                 use_edge_prompt: bool = False, edge_prompt_dim: int = 256, edge_prompt_anchors: int = 32,
                 predefined_edge_types: Tuple[Tuple[str, str, str], ...] | None = None,
                 predefined_node_dims: Dict[str, int] | None = None,
                 num_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.out_dim = out_dim
        self.dropout = nn.Dropout(p=dropout)
        # create a small projection head after pooling
        self.head = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )
        self._built = False
        self._node_dims: Dict[str, int] = {}
        self._edge_dim: int | None = None
        self._predefined_edge_types: Tuple[Tuple[str, str, str], ...] | None = predefined_edge_types
        self._predefined_node_dims: Dict[str, int] | None = predefined_node_dims
        # EdgePrompt options
        self.use_edge_prompt = use_edge_prompt
        self.edge_prompt_dim = edge_prompt_dim
        self.edge_prompt_anchors = edge_prompt_anchors
        self.edge_prompt_module: HeteroEdgePromptPlus | None = None
        self.num_layers = max(1, int(num_layers))

    def _build_layers(self, batch: Batch):
        # Infer per-node-type input dims and shared edge_dim from batch (or predefined)
        self._node_dims = self._predefined_node_dims or {nt: batch[nt].x.size(-1) for nt in batch.node_types}
        # Target device from batch tensors
        ref_device = batch[batch.node_types[0]].x.device
        types_for_prompt = self._predefined_edge_types or batch.edge_types
        if self.use_edge_prompt:
            # Initialize hetero EdgePrompt module
            self.edge_prompt_module = HeteroEdgePromptPlus(
                node_dims=self._node_dims,
                edge_types=types_for_prompt,
                prompt_dim=self.edge_prompt_dim,
                num_anchors=self.edge_prompt_anchors,
                device=ref_device,
            )
            self._edge_dim = self.edge_prompt_dim
        else:
            # Find first edge_attr to infer dim
            for et in batch.edge_types:
                if hasattr(batch[et], "edge_attr"):
                    self._edge_dim = batch[et].edge_attr.size(-1)
                    break
            if self._edge_dim is None:
                raise ValueError("No edge_attr found in batch; required for TransformerConv(edge_dim=...).")
        # Create a stack of HeteroConv layers with per-relation TransformerConv modules.
        # If predefined relations are provided, instantiate for the full union to enable strict weight loading.
        edge_types_for_convs = self._predefined_edge_types or batch.edge_types
        layers: list[HeteroConv] = []
        for li in range(self.num_layers):
            convs_li: Dict[Tuple[str, str, str], nn.Module] = {}
            for et in edge_types_for_convs:
                src_t, _, dst_t = et
                in_src = self._node_dims[src_t] if li == 0 else self.out_dim
                in_dst = self._node_dims[dst_t] if li == 0 else self.out_dim
                convs_li[et] = TransformerConv(
                    (in_src, in_dst),
                    out_channels=self.out_dim,
                    edge_dim=self._edge_dim,
                ).to(ref_device)
                
            layers.append(HeteroConv(convs_li, aggr="sum"))
            
        self.layers = nn.ModuleList(layers)
        self._built = True

    def forward(self, batch: Batch, seed_info: Tuple[str, int] | None = None) -> torch.Tensor:
        if not self._built:
            self._build_layers(batch)
        # Layered message passing with per-layer edge attributes.
        x_dict_cur: Dict[str, torch.Tensor] = batch.x_dict
        for conv in self.layers:
            if self.use_edge_prompt:
                assert self.edge_prompt_module is not None
                edge_attr_dict = self.edge_prompt_module.get_edge_prompts(
                    x_dict_cur,
                    {et: batch[et].edge_index for et in batch.edge_types}
                )
            else:
                edge_attr_dict = {et: batch[et].edge_attr for et in batch.edge_types}
            x_dict_cur = conv(x_dict_cur, batch.edge_index_dict, edge_attr_dict=edge_attr_dict)
        out = x_dict_cur
        if seed_info is not None:
            seed_type, seed_idx = seed_info
            if seed_type not in out:
                raise KeyError(f"Seed node type '{seed_type}' not present in subgraph output.")
            x = self.dropout(out[seed_type])
            if seed_idx < 0 or seed_idx >= x.size(0):
                raise IndexError(f"Seed local index {seed_idx} out of range for type '{seed_type}' size {x.size(0)}.")
            g = x[seed_idx]  
            g = self.head(g)
            return F.normalize(g, dim=-1)
        else:
            pooled = []
            for ntype, x in out.items():
                x = self.dropout(x)
                pooled.append(global_mean_pool(x, batch[ntype].batch))
            # Average the per-type pooled embeddings
            g = torch.stack([F.normalize(p, dim=-1) for p in pooled], dim=0).mean(dim=0)
            g = self.head(g)
            return F.normalize(g, dim=-1)

def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        preds = (torch.sigmoid(logits) >= 0.5).float()
        return float((preds == y).float().mean().item())

def train_one_epoch(model_txt: TextEncoder, model_gnn: HeteroSubgraphEncoder,
                    loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    model_gnn.train()
    if model_txt.finetune:
        model_txt.train()
    else:
        model_txt.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    bce = nn.BCEWithLogitsLoss()
    for queries, hb, y in loader:
        n_batches += 1
        y = y.to(device)
        hb = hb.to(device)
        q_emb = model_txt(queries)             # [B, D]
        g_emb = model_gnn(hb)                  # [B, D]
        logits = (q_emb * g_emb).sum(dim=-1)   # [B]
        loss = bce(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model_gnn.parameters()) + list(model_txt.parameters()) if model_txt.finetune else model_gnn.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.item())
        total_acc += accuracy_from_logits(logits.detach(), y.detach())
    return total_loss / max(1, n_batches), total_acc / max(1, n_batches)

@torch.no_grad()
def evaluate(model_txt: TextEncoder, model_gnn: HeteroSubgraphEncoder,
             loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model_gnn.eval()
    model_txt.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    bce = nn.BCEWithLogitsLoss()
    for queries, hb, y in loader:
        n_batches += 1
        y = y.to(device); hb = hb.to(device)
        q_emb = model_txt(queries)
        g_emb = model_gnn(hb)
        logits = (q_emb * g_emb).sum(dim=-1)
        loss = bce(logits, y)
        total_loss += float(loss.item())
        total_acc += accuracy_from_logits(logits, y)
    return total_loss / max(1, n_batches), total_acc / max(1, n_batches)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Path to train.jsonl")
    ap.add_argument("--val", default="", help="Path to val.jsonl")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--scheduler", choices=["none", "step"], default="step", help="Learning rate scheduler (default: step)")
    ap.add_argument("--lr-step-size", type=int, default=80, help="StepLR: number of epochs between LR decay")
    ap.add_argument("--lr-gamma", type=float, default=0.9, help="StepLR: multiplicative LR decay factor")
    ap.add_argument("--out", default="/home/ghr/angel/gnn/training_output", help="Output dir for checkpoints/metrics")
    ap.add_argument("--seed", type=int, default=42)
    # text encoder options
    ap.add_argument("--text-backend", choices=["sbert", "clip"], default="sbert")
    ap.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--clip-model", default="ViT-B/32")
    ap.add_argument("--out-dim", type=int, default=256, help="Shared embedding size for text and graph")
    ap.add_argument("--finetune-text", action="store_true", help="Fine-tune text encoder (default: freeze)")
    # GNN architecture
    ap.add_argument("--gnn-arch", choices=["transformer", "rgcn"], default="transformer", help="Choose GNN encoder")
    ap.add_argument("--rgcn-layers", type=int, default=2, help="Number of RGCN layers (if --gnn-arch rgcn)")
    ap.add_argument("--rgcn-mlp-layers", type=int, default=2, help="Number of MLP layers per RGCN layer")
    # EdgePrompt options
    ap.add_argument("--use-edge-prompt", action="store_true", default=True, help="Enable EdgePrompt for hetero edges (default: enabled)")
    ap.add_argument("--no-edge-prompt", action="store_false", dest="use_edge_prompt", help="Disable EdgePrompt for hetero edges")
    ap.add_argument("--edge-prompt-dim", type=int, default=256, help="EdgePrompt latent dimension")
    ap.add_argument("--edge-prompt-anchors", type=int, default=32, help="Number of EdgePrompt anchors per relation")
    # TransformerConv depth
    ap.add_argument("--transformer-layers", type=int, default=3, help="Number of TransformerConv layers (depth)")
    # Weights & Biases
    ap.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    ap.add_argument("--wandb-project", default="gnn-retrieval", help="W&B project name")
    ap.add_argument("--wandb-name", default="", help="W&B run name")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_ds = PairDataset(args.train)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_pairs, num_workers=0)
    val_loader = None
    if args.val and os.path.exists(args.val):
        val_ds = PairDataset(args.val)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pairs, num_workers=0)

    # Discover union of node types/dims and edge types across training set (for stable EdgePrompt keys)
    all_node_dims: Dict[str, int] = {}
    all_edge_types_set: set[Tuple[str, str, str]] = set()
    for rec in train_ds.records:
        subg = _load_subgraph(rec["subgraph_path"])
        for nt in subg.node_types:
            all_node_dims[nt] = int(subg[nt].x.size(-1))
        for et in subg.edge_types:
            all_edge_types_set.add(et)
    all_edge_types = tuple(sorted(list(all_edge_types_set)))
    # Bootstrap one batch to allocate models on proper device
    q0, hb0, y0 = next(iter(train_loader))
    # Models
    model_txt = TextEncoder(args.text_backend, args.sbert_model, args.clip_model, args.out_dim, finetune_text=args.finetune_text).to(device)

    model_gnn = HeteroSubgraphEncoder(
            out_dim=args.out_dim,
            dropout=0.1,
            use_edge_prompt=args.use_edge_prompt,
            edge_prompt_dim=args.edge_prompt_dim,
            edge_prompt_anchors=args.edge_prompt_anchors,
            predefined_edge_types=all_edge_types,
            predefined_node_dims=all_node_dims,
            num_layers=args.transformer_layers
        ).to(device)

    optimizer = torch.optim.Adam(
        [{"params": model_gnn.parameters(), "lr": args.lr},
         {"params": model_txt.parameters(), "lr": args.lr if args.finetune_text else args.lr * 0.1}],
        weight_decay=1e-4
    )
    
    scheduler = None
    if args.scheduler == "step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    # Initialize W&B
    wandb_run = None
    if args.wandb:
        if wandb is None:
            print("wandb is not installed; disable --wandb or `pip install wandb`.", flush=True)
        else:
            run_name = args.wandb_name or f"dual-encoder-{args.text_backend}"
            wandb_run = wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
            # Optionally watch models (commented to reduce overhead)
            wandb.watch(model_gnn, log="gradients", log_freq=50)

    best_val = math.inf
    history = []
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model_txt, model_gnn, train_loader, optimizer, device)
        if val_loader is not None:
            va_loss, va_acc = evaluate(model_txt, model_gnn, val_loader, device)
        else:
            va_loss, va_acc = float("nan"), float("nan")
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc})
        # Step the LR scheduler at epoch boundary
        if scheduler is not None:
            scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d} | lr {cur_lr:.6f} | train_loss {tr_loss:.4f} acc {tr_acc:.3f} | val_loss {va_loss:.4f} acc {va_acc:.3f}")
        
        if wandb_run is not None:
            wandb.log({
                "epoch": epoch,
                "train/loss": tr_loss,
                "train/acc": tr_acc,
                "val/loss": va_loss,
                "val/acc": va_acc,
                "lr": cur_lr,
            }, step=epoch)
        # Save best by val_loss
        
        if val_loader is not None and va_loss < best_val:
            best_val = va_loss
            torch.save({
                "model_txt": model_txt.state_dict(),
                "model_gnn": model_gnn.state_dict(),
                "config": vars(args),
            }, os.path.join(args.out, "best.pt"))
            if wandb_run is not None:
                wandb.log({"best/val_loss": best_val}, step=epoch)

    torch.save({
        "model_txt": model_txt.state_dict(),
        "model_gnn": model_gnn.state_dict(),
        "config": vars(args),
        "history": history,
    }, os.path.join(args.out, "last.pt"))
    
    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print("Saved checkpoints and history to", args.out)
    
    if wandb_run is not None:
        try:
            wandb.save(os.path.join(args.out, "best.pt"))
            wandb.save(os.path.join(args.out, "last.pt"))
            wandb.save(os.path.join(args.out, "history.json"))
        except Exception:
            pass
        wandb_run.finish()

if __name__ == "__main__":
    main()



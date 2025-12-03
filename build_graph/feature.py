import torch, re
import clip  # pip install git+https://github.com/openai/CLIP.git

path = "graph_output/clean_graph.pyg_en_labeled.pt"
obj = torch.load(path, map_location="cpu")

# 1) Get per-edge property text (aligned to the graph’s edge order)
# Prefer a per-edge PID list if present; else use English labels directly
pids = obj.get("edge_property_ids") or obj.get("edge_index_label_en")
rel_labels = obj.get("relation_labels_en") or obj.get("property_labels_en")
if pids is None:
    raise ValueError("No per-edge property list found (edge_property_ids/edge_index_label_en).")

def is_pid(x): return isinstance(x, str) and re.match(r"^P\d+$", x) is not None
texts = []
for pid_or_label in pids:
    if is_pid(pid_or_label):
        if rel_labels is None or pid_or_label not in rel_labels:
            raise ValueError(f"Missing label for {pid_or_label}")
        texts.append(rel_labels[pid_or_label])
    else:
        texts.append(str(pid_or_label))

# 2) Encode with CLIP text encoder (RN50); only encode_text is used
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("RN50", device=device)  # text transformer; not using image encoder
with torch.no_grad():
    feats = []
    bs = 512
    for i in range(0, len(texts), bs):
        tokens = clip.tokenize(texts[i:i+bs]).to(device)
        emb = model.encode_text(tokens).float()
        emb = emb / emb.norm(dim=-1, keepdim=True)
        feats.append(emb.cpu())
edge_features = torch.cat(feats, dim=0)  # shape [E, 512]

# 3) Save back alongside the graph; you can attach later for training
obj["clip_text_edge_mean_emb_en"] = edge_features  # aligned to edge order
out = path + ".with_edge_clip.pt"
torch.save(obj, out)
print("Saved:", out, "edge_features:", tuple(edge_features.shape))
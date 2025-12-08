# 0) Optional: install dependencies needed for logging
python3 -m pip install -U wandb click sentence-transformers

# 1) Authenticate W&B (uses your API key)
export WANDB_API_KEY='b4b9151d7d4ac89bbfe0e78da6c391e7af884894'

cd /home/ghr/angel/gnn
# 2) Set paths
TRAIN_JSONL="/home/ghr/angel/gnn/graph_embedding/pairs_all/train.jsonl"
VAL_JSONL="/home/ghr/angel/gnn/graph_embedding/pairs_all/test.jsonl"
OUT_DIR="/home/ghr/angel/gnn/training_output"
mkdir -p "$OUT_DIR"

# 3) Run training with W&B enabled (test.jsonl used as validation)
python3 -m retrieval.train_gnn_pairs \
  --train "$TRAIN_JSONL" \
  --val "$VAL_JSONL" \
  --out "$OUT_DIR" \
  --wandb \
  --wandb-project gnn-retrieval \
  --wandb-name dual-encoder-sbert

# After training, the script writes:
# - $OUT_DIR/best.pt     (best by val loss)
# - $OUT_DIR/last.pt     (final checkpoint)
# - $OUT_DIR/history.json

# 4) Also save .pth copies of the checkpoints
cp "$OUT_DIR/best.pt" "$OUT_DIR/best.pth"
cp "$OUT_DIR/last.pt" "$OUT_DIR/last.pth"

# 5) Additionally export separate state_dicts as .pth for GNN and text encoder (from best.pt)
python3 - <<'PY'
import torch, os
out = "/home/ghr/angel/gnn/training_output"
ckpt = torch.load(os.path.join(out, "best.pt"), map_location="cpu")
torch.save(ckpt["model_gnn"], os.path.join(out, "gnn_best_weights.pth"))
torch.save(ckpt["model_txt"], os.path.join(out, "text_best_weights.pth"))
PY

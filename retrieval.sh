#!/usr/bin/env bash

# python3 -m retrieval.build_candidates_from_graph \
#   --graph-pt /home/ghr/angel/gnn/graph_embedding/graph_by_label.no_literal.pyg.pt.with_clip.pt \
#   --out-dir /home/ghr/angel/gnn/candidates \
#   --seed-node-type entity \
#   --num-hops 2 --num-neighbors 16 --max-candidates 20000 --batch-size 1

python3 -m retrieval.infer_and_answer \
  --checkpoint /home/ghr/angel/gnn/training_output/best.pth \
  --candidates-dir /home/ghr/angel/gnn/candidates \
  --query "Tell me some artwork created by Renaissance Italian painter and polymath" --topk 50 --gemini \
  --gemini-api-key "AIzaSyBlTEJcSEQe8L-g8ZEjEZWBgY9Xb1qDLB0" \
  --gemini-model "gemini-2.5-flash"
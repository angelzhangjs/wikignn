#!/usr/bin/env bash

python3 -m retrieval.build_candidates_from_graph \
  --graph-pt /home/ghr/angel/gnn/graph_embedding/graph_by_label.no_literal.pyg.pt.with_clip.pt \
  --out-dir /home/ghr/angel/gnn/candidates \
  --seed-node-type entity \
  --num-hops 2 --num-neighbors 16 --max-candidates 20000 --batch-size 1
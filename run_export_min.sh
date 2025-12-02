#!/usr/bin/env bash

# Edit the two paths below, then run:  bash /home/ghr/angel/gnn/run_export_min.sh
python3 /home/ghr/angel/gnn/export_to_pt.py \
  --graph_dir "/datasets/v2p/current/angel/gnn/graph_gson" \
  --out "/datasets/v2p/current/angel/gnn/graph_pt" \
  --format pyg



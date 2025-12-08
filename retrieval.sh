#!/usr/bin/env bash

python3 -m retrieval.infer_and_answer \
  --checkpoint /home/ghr/angel/gnn/training_output/best.pth \
  --candidates-dir /home/ghr/angel/gnn/candidates \
  --query "Tell me some artwork created by Renaissance Italian painter and polymath" --topk 50 --gemini \
  --gemini-api-key "AIzaSyBlTEJcSEQe8L-g8ZEjEZWBgY9Xb1qDLB0" \
  --gemini-model "gemini-2.5-flash"
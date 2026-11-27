#!/bin/bash
# Полная сетка эксперимента: 3 seed x (head, LoRA last 25/50/75/100%, full FT, LoRA first 50%)
MODEL=${MODEL:-google/bert_uncased_L-8_H-512_A-8}
OUT=${OUT:-results/results.csv}
for SEED in 0 1 2; do
  python lora_depth.py --model $MODEL --mode head --seed $SEED --out $OUT
  for F in 0.25 0.5 0.75 1.0; do
    python lora_depth.py --model $MODEL --mode lora --fraction $F --position last --seed $SEED --out $OUT
  done
  python lora_depth.py --model $MODEL --mode full --seed $SEED --out $OUT
  python lora_depth.py --model $MODEL --mode lora --fraction 0.5 --position first --seed $SEED --out $OUT
done

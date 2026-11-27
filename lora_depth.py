"""
Сколько Transformer-блоков адаптировать LoRA?

Один запуск = одна конфигурация (доля блоков, позиция, seed).
LoRA ставится только в выбранные блоки энкодера через PEFT `layers_to_transform`.
Классификационная голова (pooler + classifier) обучается во всех вариантах одинаково.

Варианты (--mode):
  head  : backbone заморожен полностью, учится только голова (0% блоков)  -> нижний baseline
  lora  : LoRA в доле --fraction блоков (последних или первых, --position)
  full  : полный fine-tuning всех весов                                      -> верхний ориентир

Пример:
  python lora_depth.py --mode lora --fraction 0.5 --seed 0
"""
import argparse
import csv
import json
import math
import os
import random
import time
import urllib.request

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

SST2_URL = "https://raw.githubusercontent.com/clairett/pytorch-sentiment-classification/master/data/SST2/{}.tsv"


# ----------------------------------------------------------------------------- data
def load_sst2(data_dir):
    """SST-2 (sentence level, binary): train 6920 / dev 872 / test 1821, с метками у test."""
    os.makedirs(data_dir, exist_ok=True)
    splits = {}
    for split in ("train", "dev", "test"):
        path = os.path.join(data_dir, f"{split}.tsv")
        if not os.path.exists(path):
            urllib.request.urlretrieve(SST2_URL.format(split), path)
        texts, labels = [], []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                text, label = line.rsplit("\t", 1)
                texts.append(text)
                labels.append(int(label))
        splits[split] = (texts, labels)
    return splits


def make_loader(tok, texts, labels, max_len, batch_size, shuffle, seed):
    enc = tok(texts, truncation=True, max_length=max_len)
    items = [
        {"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i], "labels": labels[i]}
        for i in range(len(texts))
    ]

    def collate(batch):
        L = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for j, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[j, :n] = torch.tensor(b["input_ids"])
            att[j, :n] = 1
        return {"input_ids": ids, "attention_mask": att,
                "labels": torch.tensor([b["labels"] for b in batch])}

    g = torch.Generator().manual_seed(seed)
    return DataLoader(items, batch_size=batch_size, shuffle=shuffle, collate_fn=collate, generator=g)


# ----------------------------------------------------------------------------- model
def select_layers(n_layers, fraction, position):
    k = max(1, round(n_layers * fraction))
    return list(range(n_layers - k, n_layers)) if position == "last" else list(range(k))


def build_model(args):
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    # размер исходной модели (backbone + pooler + classifier) до добавления LoRA;
    # после get_peft_model total вырастет за счёт LoRA-матриц и копий pooler/classifier
    # из modules_to_save, поэтому долю обучаемых параметров считаем от этого числа
    model.base_param_count = sum(p.numel() for p in model.parameters())
    n_layers = model.config.num_hidden_layers
    layers = []
    if args.mode == "full":
        pass  # все веса обучаются
    elif args.mode == "head":
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("classifier") or ".pooler." in name or name.startswith("bert.pooler")
    elif args.mode == "lora":
        layers = select_layers(n_layers, args.fraction, args.position)
        cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=0.1,
            # все линейные слои блока: Q, K, V, выход attention, FFN up/down
            target_modules=["query", "key", "value", "dense"],
            layers_to_transform=layers,
            layers_pattern="layer",
            modules_to_save=["classifier", "pooler"],
        )
        base = model.base_param_count
        model = get_peft_model(model, cfg)
        model.base_param_count = base
    return model, n_layers, layers


def count_params(model):
    """trainable — обучаемые параметры; total — все параметры объекта (с LoRA и копиями modules_to_save)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = n = 0
    loss_sum = 0.0
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        out = model(**b)
        loss_sum += out.loss.item() * len(b["labels"])
        correct += (out.logits.argmax(-1) == b["labels"]).sum().item()
        n += len(b["labels"])
    return correct / n, loss_sum / n


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/bert_uncased_L-8_H-512_A-8")
    ap.add_argument("--mode", choices=["head", "lora", "full"], default="lora")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--position", choices=["last", "first"], default="last")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--lr", type=float, default=None, help="по умолчанию: 5e-4 для lora/head, 5e-5 для full")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_dir", default="data/sst2")
    ap.add_argument("--out", default="results/results.csv")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.lr is None:
        args.lr = 5e-5 if args.mode == "full" else 5e-4
    if args.threads:
        torch.set_num_threads(args.threads)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    data = load_sst2(args.data_dir)
    tok = AutoTokenizer.from_pretrained(args.model)
    train_dl = make_loader(tok, *data["train"], args.max_len, args.batch_size, True, args.seed)
    dev_dl = make_loader(tok, *data["dev"], args.max_len, 128, False, args.seed)
    test_dl = make_loader(tok, *data["test"], args.max_len, 128, False, args.seed)

    model, n_layers, layers = build_model(args)
    trainable, total = count_params(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    steps = args.epochs * len(train_dl)
    warm = int(0.06 * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))

    best = {"dev_acc": -1}
    history = []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for b in train_dl:
            b = {k: v.to(device) for k, v in b.items()}
            out = model(**b)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            tr_loss += out.loss.item()
        dev_acc, dev_loss = evaluate(model, dev_dl, device)
        history.append({"epoch": ep, "train_loss": tr_loss / len(train_dl), "dev_acc": dev_acc, "dev_loss": dev_loss})
        print(f"[ep {ep}] train_loss={tr_loss/len(train_dl):.4f} dev_acc={dev_acc:.4f} dev_loss={dev_loss:.4f}", flush=True)
        if dev_acc > best["dev_acc"]:
            # модель выбирается по dev, test считается только для лучшей эпохи
            test_acc, test_loss = evaluate(model, test_dl, device)
            best = {"dev_acc": dev_acc, "dev_loss": dev_loss, "test_acc": test_acc, "test_loss": test_loss, "best_epoch": ep}
    train_time = time.time() - t0

    frac = 0.0 if args.mode == "head" else (1.0 if args.mode == "full" else len(layers) / n_layers)
    row = {
        "model": os.path.basename(args.model.rstrip("/")), "mode": args.mode, "position": args.position if args.mode == "lora" else "",
        "fraction": round(frac, 4), "n_layers_adapted": len(layers) if args.mode == "lora" else (n_layers if args.mode == "full" else 0),
        "layers": " ".join(map(str, layers)), "rank": args.rank if args.mode == "lora" else "", "lr": args.lr,
        "seed": args.seed, "trainable_params": trainable, "total_params": total,
        "base_params": model.base_param_count,
        "trainable_pct": round(100 * trainable / model.base_param_count, 4), "train_time_s": round(train_time, 1), "device": str(device),
        **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in best.items()},
        "history": json.dumps(history),
    }
    print(json.dumps({k: v for k, v in row.items() if k != "history"}, ensure_ascii=False))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


if __name__ == "__main__":
    main()

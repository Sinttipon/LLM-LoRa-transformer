"""Агрегирует results.csv по seed и строит графики для отчёта.

python analyze.py --csv results/results.csv --out figures
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LABELS = {
    ("head", "", 0.0): "Head only (0%)",
    ("lora", "last", 0.25): "LoRA last 25%",
    ("lora", "last", 0.5): "LoRA last 50%",
    ("lora", "last", 0.75): "LoRA last 75%",
    ("lora", "last", 1.0): "LoRA 100%",
    ("lora", "first", 0.5): "LoRA first 50%",
    ("full", "", 1.0): "Full fine-tuning",
}
ORDER = list(LABELS.values())


MACRO = {  # имя макроса в отчёте -> вариант
    "Head": "Head only (0%)", "LastTwentyFive": "LoRA last 25%", "LastFifty": "LoRA last 50%",
    "LastSeventyFive": "LoRA last 75%", "LoraFull": "LoRA 100%", "FirstFifty": "LoRA first 50%",
    "FullFT": "Full fine-tuning",
}


def fmt(x, d=1):
    return f"{x:.{d}f}".replace(".", "{,}")


def write_latex(g, out):
    """results_table.tex — готовая таблица; numbers.tex — макросы \\accX, \\stdX, \\paramsX, \\timeX для текста."""
    rows = []
    for v, r in g.iterrows():
        params = f"{int(r.params):,}".replace(",", "\\,")
        rows.append(
            f"{v.replace('%', chr(92) + '%')} & {fmt(r.test_mean*100)} $\\pm$ {fmt(r.test_std*100)} & "
            f"{fmt(r.dev_mean*100)} $\\pm$ {fmt(r.dev_std*100)} & {params} & {fmt(r.pct, 2)} & {r.time_mean:.0f} \\\\")
    n = int(g["n"].min())
    with open(os.path.join(out, "results_table.tex"), "w") as f:
        f.write("\\begin{table}[H]\n\\centering\n\\small\n"
                f"\\caption{{Основные результаты (среднее $\\pm$ std по {n} seed; accuracy в \\%)}}\n"
                "\\label{tab:results}\n\\begin{tabular}{lccrcc}\n\\toprule\n"
                "\\textbf{Вариант} & \\textbf{Test acc} & \\textbf{Dev acc} & \\textbf{Trainable} & "
                "\\textbf{\\% от модели} & \\textbf{Время, с} \\\\\n\\midrule\n"
                + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    with open(os.path.join(out, "numbers.tex"), "w") as f:
        f.write(f"\\newcommand{{\\nseeds}}{{{n}}}\n")
        for name, v in MACRO.items():
            if v not in g.index:
                continue
            r = g.loc[v]
            f.write(f"\\newcommand{{\\acc{name}}}{{{fmt(r.test_mean*100)}}}\n")
            f.write(f"\\newcommand{{\\std{name}}}{{{fmt(r.test_std*100)}}}\n")
            f.write(f"\\newcommand{{\\params{name}}}{{{int(r.params):,}}}\n".replace(",", "\\,"))
            f.write(f"\\newcommand{{\\time{name}}}{{{r.time_mean:.0f}}}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/results.csv")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = pd.read_csv(a.csv, keep_default_na=False)
    df["fraction"] = df["fraction"].astype(float)
    # доля обучаемых — от исходной модели (без LoRA и копий modules_to_save);
    # для старых CSV без base_params исходная модель = total_params варианта head only / full
    if "base_params" not in df.columns or (df["base_params"] == "").any():
        df["base_params"] = df.loc[df["mode"].isin(["head", "full"]), "total_params"].astype(int).min()
    df["trainable_pct"] = 100 * df["trainable_params"].astype(int) / df["base_params"].astype(int)
    df["variant"] = [LABELS[(m, p, f)] for m, p, f in zip(df["mode"], df["position"], df["fraction"])]

    g = df.groupby("variant").agg(
        n=("seed", "count"),
        test_mean=("test_acc", "mean"), test_std=("test_acc", "std"),
        dev_mean=("dev_acc", "mean"), dev_std=("dev_acc", "std"),
        params=("trainable_params", "first"), pct=("trainable_pct", "first"),
        time_mean=("train_time_s", "mean"),
    ).reindex([v for v in ORDER if v in set(df["variant"])])
    g.to_csv(os.path.join(a.out, "summary.csv"))
    pd.set_option("display.width", 200)
    print(g.round(4))
    print("\nper-seed test acc:")
    print(df.pivot_table(index="variant", columns="seed", values="test_acc").reindex(g.index).round(4))
    write_latex(g, a.out)

    # --- рисунок 1: доля блоков -> качество
    curve = df[(df["mode"] == "head") | ((df["mode"] == "lora") & (df["position"] == "last"))]
    c = curve.groupby("fraction")["test_acc"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.scatter(curve["fraction"] * 100, curve["test_acc"] * 100, color="#9aa5b1", s=14, zorder=2, label="отдельные seed")
    ax.errorbar(c["fraction"] * 100, c["mean"] * 100, yerr=c["std"] * 100, color="#2a6fdb", marker="o",
                capsize=4, lw=2, zorder=3, label="LoRA в последних k% блоков (mean ± std)")
    full = df[df["mode"] == "full"]["test_acc"]
    if len(full):
        ax.axhline(full.mean() * 100, color="#d1495b", ls="--", lw=1.5, label=f"Full FT ({full.mean()*100:.1f}%)")
    first = df[(df["mode"] == "lora") & (df["position"] == "first")]["test_acc"]
    if len(first):
        ax.errorbar([50], [first.mean() * 100], yerr=[first.std() * 100], color="#e09f3e", marker="s",
                    capsize=4, ls="none", ms=7, zorder=4, label="LoRA в первых 50%")
    ax.set_xlabel("Доля адаптированных блоков, %")
    ax.set_ylabel("Test accuracy, %")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fraction_vs_accuracy.png"), dpi=200)

    # --- рисунок 2: trainable params -> качество
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for v, row in g.iterrows():
        col = "#d1495b" if v.startswith("Full") else ("#e09f3e" if "first" in v else "#2a6fdb")
        ax.errorbar(row["params"], row["test_mean"] * 100, yerr=row["test_std"] * 100, marker="o", color=col, capsize=3)
        ax.annotate(v, (row["params"], row["test_mean"] * 100), textcoords="offset points", xytext=(5, -10), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Число обучаемых параметров (log)")
    ax.set_ylabel("Test accuracy, %")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "params_vs_accuracy.png"), dpi=200)

    # --- рисунок 3: кривые dev accuracy по эпохам (seed 0)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for v in g.index:
        r = df[(df["variant"] == v) & (df["seed"] == df["seed"].min())]
        if not len(r):
            continue
        h = json.loads(r.iloc[0]["history"])
        ax.plot([x["epoch"] for x in h], [x["dev_acc"] * 100 for x in h], marker="o", label=v)
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Dev accuracy, %")
    ax.set_xticks([1, 2, 3])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "dev_curves.png"), dpi=200)


if __name__ == "__main__":
    main()

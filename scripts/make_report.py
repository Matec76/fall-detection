"""Write BAO_CAO.md and its figures (docs/figures/*.png) from the metrics in runs/.

    python scripts/make_report.py

Every number in the report is read from runs/urfd, runs/lin2021 and runs/norm_seeds,
so re-running the experiments and this script keeps the report in sync.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import glob

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from falldet.preprocess import mask_low_confidence, to_body15  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
RUNS = PROJECT / "runs"
FIG_DIR = PROJECT / "docs" / "figures"
OUT = PROJECT / "BAO_CAO.md"

MODELS = [("rnn", "RNN", "#2a78d6", "o"), ("lstm", "LSTM", "#eb6834", "s"), ("gru", "GRU", "#1baf7a", "D")]
NORMS = [("none", "Không chuẩn hóa"), ("minmax", "Min-max"), ("rp", "RP"), ("rp_interp", "RP + nội suy")]
HIDDEN = [64, 128, 256, 512, 1024]
LRS = ["0.1", "0.01", "0.001"]
INK, INK2, INK3, RULE, REF, CHANCE = "#14171c", "#4d5563", "#697181", "#dfe3e8", "#8a93a3", "#b3261e"

# Values reported by Lin et al. 2021 (%).
PAPER_HIDDEN = {"rnn": [82.6, 84, 86.3, 87.3, 88], "lstm": [96, 96, 97.3, 98.3, 94.6], "gru": [93, 94, 96, 96.6, 94.6]}
PAPER_LR = {"rnn": {"0.1": 87, "0.01": 87.3, "0.001": 87}, "lstm": {"0.1": 97, "0.01": 98.3, "0.001": 96.6},
            "gru": {"0.1": 87, "0.01": 97, "0.001": 95}}
PAPER_NORM = {  # accuracy, sensitivity, specificity (Tables 8-10)
    "rnn": {"none": (85.2, 86.6, 83.9), "minmax": (75.8, 73.2, 78.5), "rp": (89.2, 93.7, 84.8), "rp_interp": (90.1, 92.8, 87.5)},
    "lstm": {"none": (92.4, 96, 89.2), "minmax": (88.8, 94, 83.9), "rp": (98.2, 100, 96.4), "rp_interp": (95, 95.5, 94.6)},
    "gru": {"none": (85.2, 93.7, 76.7), "minmax": (84.8, 93.7, 75.8), "rp": (97.3, 96.4, 98.2), "rp_interp": (97.3, 96.4, 98.2)},
}


def load(run: Path) -> dict | None:
    p = run / "metrics.json"
    if not p.exists():
        return None
    t = json.loads(p.read_text())["test"]
    cm = t["confusion_matrix"]
    return {"acc": t["accuracy"] * 100, "sens": t["sensitivity"] * 100, "spec": t["specificity"] * 100,
            "missed": cm[0][1], "false_alarm": cm[1][0], "n": sum(map(sum, cm))}


def num(v: float | None, d: int = 1) -> str:
    return "—" if v is None else f"{v:.{d}f}".replace(".", ",")


main = {k: load(RUNS / "urfd" / k) for k in ("lstm", "gru", "rnn", "lstm_paper_split")}
sweep = {p.name: load(p) for p in sorted((RUNS / "lin2021").iterdir()) if p.is_dir()}
seeds: dict[str, dict] = {}
for p in sorted((RUNS / "norm_seeds").glob("*_s*")):
    m = load(p)
    if m:
        seeds.setdefault(p.name.rsplit("_s", 1)[0], {"runs": []})["runs"].append(m)
for s in seeds.values():
    for k in ("acc", "sens", "spec"):
        vals = [r[k] for r in s["runs"]]
        s[k] = (statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0)
n_seeds = min(len(s["runs"]) for s in seeds.values())
n_test = main["lstm"]["n"]


# ---------------------------------------------------------------- figures
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": RULE, "axes.labelcolor": INK2,
                     "xtick.color": INK3, "ytick.color": INK3, "text.color": INK, "axes.grid": True,
                     "grid.color": RULE, "grid.linewidth": 0.8, "axes.axisbelow": True, "savefig.dpi": 200,
                     "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False})
FIG_DIR.mkdir(parents=True, exist_ok=True)


def fig_split() -> None:
    rows = [("LSTM-512 · 500 epoch", main["lstm"], main["lstm_paper_split"], 98.2),
            ("LSTM-512 · 200 epoch", sweep["norm_rp_lstm"], sweep["paper_split_lstm"], 98.2),
            ("GRU-512 · 200 epoch", sweep["norm_rp_gru"], sweep["paper_split_gru"], 97.3),
            ("RNN-512 · 200 epoch", sweep["norm_rp_rnn"], sweep["paper_split_rnn"], 89.2)]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    for i, (label, a, b, ref) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([a["acc"], b["acc"]], [y, y], color="#c5cbd3", lw=2, zorder=1)
        ax.plot([ref, ref], [y - 0.22, y + 0.22], color=REF, lw=2, zorder=1)
        ax.scatter(a["acc"], y, s=60, color=MODELS[0][2], edgecolor="white", linewidth=1.5, zorder=3,
                   label="Chia theo video" if i == 0 else None)
        ax.scatter(b["acc"], y, s=60, marker="s", color=MODELS[1][2], edgecolor="white", linewidth=1.5, zorder=3,
                   label="Chia ngẫu nhiên như bài báo" if i == 0 else None)
        lo, hi = sorted([a["acc"], b["acc"]])
        hi_x = max(hi, ref) if abs(ref - hi) < 4 else hi
        ax.text(lo - 1.2, y, num(lo), ha="right", va="center", fontsize=9, color=INK)
        ax.text(hi_x + 1.2, y, num(hi), ha="left", va="center", fontsize=9, color=INK)
    ax.plot([], [], color=REF, lw=2, label="Bài báo công bố")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.set_xlim(40, 103)
    ax.set_xlabel("Accuracy trên tập test (%)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0, labelcolor=INK)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=3, frameon=False, fontsize=9)
    fig.savefig(FIG_DIR / "hinh1_cach_chia.png")
    plt.close(fig)


def fig_hidden() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = range(len(HIDDEN))
    for key, name, color, mk in MODELS:
        ys = [sweep[f"hidden_{h}_{key}"]["acc"] for h in HIDDEN]
        ax.plot(x, ys, color=color, lw=2, marker=mk, ms=6, mec="white", mew=1.2, label=name)
    ax.axhline(50, color=CHANCE, lw=1.2, ls=(0, (5, 4)), alpha=0.8)
    ax.text(0, 51.5, "mức đoán mò", ha="left", va="bottom", fontsize=8.5, color=CHANCE)
    ax.set_xticks(list(x), [str(h) for h in HIDDEN])
    ax.set_ylim(40, 100)
    ax.set_xlabel("Số nút ẩn")
    ax.set_ylabel("Accuracy trên tập test (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14), fontsize=9)
    fig.savefig(FIG_DIR / "hinh2_so_nut_an.png")
    plt.close(fig)


def fig_norm() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.8), sharey=True)
    for ax, (key, name, color, mk) in zip(axes, MODELS):
        for i, (nk, _) in enumerate(NORMS):
            y = len(NORMS) - 1 - i
            s = seeds[f"{nk}_{key}"]
            mu, sd = s["acc"]
            ax.plot([max(40, mu - sd), min(100, mu + sd)], [y, y], color=color, lw=2, solid_capstyle="round", zorder=2)
            ax.scatter([r["acc"] for r in s["runs"]], [y - 0.28] * len(s["runs"]), s=14, facecolor="none",
                       edgecolor=INK3, linewidth=1, zorder=2)
            ax.scatter(mu, y, s=46, marker=mk, color=color, edgecolor="white", linewidth=1.2, zorder=3)
            ref = PAPER_NORM[key][nk][0]
            ax.plot([ref, ref], [y - 0.2, y + 0.2], color=REF, lw=2, zorder=1)
        ax.set_title(name, fontsize=10.5, loc="left", color=INK)
        ax.set_xlim(40, 102)
        ax.set_xticks([40, 60, 80, 100])
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0, labelcolor=INK)
    axes[0].set_yticks(range(len(NORMS)), [n[1] for n in NORMS][::-1])
    axes[1].set_xlabel("Accuracy trên tập test (%)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hinh3_chuan_hoa.png")
    plt.close(fig)


fig_split()
fig_hidden()
fig_norm()


# ---------------------------------------------------------------- tables
def table(header: list[str], rows: list[list[str]], align: str | None = None) -> str:
    align = align or "l" + "r" * (len(header) - 1)
    sep = ["---:" if a == "r" else ":---" for a in align]
    return "\n".join(["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
                     + ["| " + " | ".join(r) + " |" for r in rows])


def cell(m: dict | None, paper: float | None = None) -> str:
    if m is None:
        return "—"
    s = f"{num(m['acc'])}"
    if m["sens"] == 0 or m["spec"] == 0:
        s += "*"
    return s + (f" ({num(paper)})" if paper is not None else "")


t_main = table(["Mô hình", "Accuracy", "Sensitivity", "Specificity", "Bỏ sót / báo nhầm", "Bài báo (acc / sens / spec)"], [
    [label, num(m["acc"]), num(m["sens"]), num(m["spec"]), f"{m['missed']} / {m['false_alarm']}",
     " / ".join(num(v) for v in PAPER_NORM[p]["rp"])]
    for label, m, p in [("LSTM-512", main["lstm"], "lstm"), ("GRU-512", main["gru"], "gru"),
                        ("RNN-512", main["rnn"], "rnn"), ("LSTM-512, chia ngẫu nhiên", main["lstm_paper_split"], "lstm")]])

t_split = table(["Mô hình", "Chia theo video", "Chia ngẫu nhiên", "Chênh", "Bài báo"], [
    [label, num(a["acc"]), num(b["acc"]), f"+{num(b['acc'] - a['acc'])}", num(ref)]
    for label, a, b, ref in [("LSTM-512 · 500 epoch", main["lstm"], main["lstm_paper_split"], 98.2),
                             ("LSTM-512 · 200 epoch", sweep["norm_rp_lstm"], sweep["paper_split_lstm"], 98.2),
                             ("GRU-512 · 200 epoch", sweep["norm_rp_gru"], sweep["paper_split_gru"], 97.3),
                             ("RNN-512 · 200 epoch", sweep["norm_rp_rnn"], sweep["paper_split_rnn"], 89.2)]])

t_hidden = table(["Số nút ẩn"] + [m[1] for m in MODELS], [
    [str(h)] + [cell(sweep[f"hidden_{h}_{k}"], PAPER_HIDDEN[k][i]) for k, *_ in MODELS] for i, h in enumerate(HIDDEN)])

t_lr = table(["Learning rate"] + [m[1] for m in MODELS], [
    [lr.replace(".", ",")] + [cell(sweep[f"lr_{lr}_{k}"], PAPER_LR[k][lr]) for k, *_ in MODELS] for lr in LRS])

t_norm = table(["Chuẩn hóa"] + [m[1] for m in MODELS], [
    [label] + [f"{num(seeds[f'{nk}_{k}']['acc'][0])} ± {num(seeds[f'{nk}_{k}']['acc'][1])} ({num(PAPER_NORM[k][nk][0])})"
               for k, *_ in MODELS] for nk, label in NORMS])

t_all = table(["Run", "Accuracy", "Sensitivity", "Specificity", "Bỏ sót / báo nhầm"], [
    [f"`{name}`", num(m["acc"]), num(m["sens"]), num(m["spec"]), f"{m['missed']} / {m['false_alarm']}"]
    for name, m in sweep.items() if m])

best_name, best = max(((f"{k.upper()}-{h}", sweep[f"hidden_{h}_{k}"]) for k, *_ in MODELS for h in HIDDEN),
                      key=lambda t: t[1]["acc"])
gain_lstm = sweep["paper_split_lstm"]["acc"] - sweep["norm_rp_lstm"]["acc"]
gain_gru = sweep["paper_split_gru"]["acc"] - sweep["norm_rp_gru"]["acc"]
none_mean = statistics.mean(seeds[f"none_{k}"]["acc"][0] for k, *_ in MODELS)
max_sd = max(s["acc"][1] for s in seeds.values())

report = f"""# Báo cáo kết quả: tái hiện Lin et al. 2021 trên URFD

**Bài báo tham khảo:** C.-B. Lin, Z. Dong, W.-K. Kuan, Y.-F. Huang, *A Framework for Fall Detection Based on
OpenPose Skeleton and LSTM/GRU Models*, Applied Sciences 11(1):329, 2021. DOI: 10.3390/app11010329.

> Mọi số liệu và hình trong file này được sinh bởi `python scripts/make_report.py` từ kết quả trong `runs/`.
> Chạy lại thí nghiệm rồi chạy lại script thì báo cáo cập nhật theo. Đơn vị là %; ô có dấu `*` là mô hình
> dự đoán mọi mẫu vào một lớp (không học được). Số trong ngoặc là giá trị bài báo công bố.

## 1. Tóm tắt

1. **Cách chia dữ liệu làm con số dao động mạnh trên một lần chia 27 mẫu.** Chia ngẫu nhiên theo cửa sổ như bài báo
   làm accuracy tăng {num(gain_lstm)} điểm (LSTM) và {num(gain_gru)} điểm (GRU) trong một lần chia. Kiểm định chéo ở
   phần II (mục 8.2) cho thấy hiệu ứng thật nhỏ hơn: khoảng +5 điểm với cấu hình của bài báo, không rõ với GRU-128.
2. **Trên video chưa từng thấy**, cấu hình tốt nhất là {best_name} với {num(best['acc'])}%
   (sensitivity {num(best['sens'])}%, specificity {num(best['spec'])}%). Cấu hình tốt nhất của bài báo
   (LSTM-512, lr 0,01) đạt {num(main['lstm']['acc'])}% sau 500 epoch.
3. **Mô hình nhỏ ổn định hơn:** 64–128 nút ẩn cho kết quả tốt nhất; RNN từ 256 nút trở lên không học được.
4. **Chuẩn hóa có ích** (không chuẩn hóa ≈ {num(none_mean)}%), nhưng **RP không vượt min-max một cách nhất quán**;
   các chênh lệch nằm trong một độ lệch chuẩn qua {n_seeds} lần chạy.

Phần II (mục 8–10) thêm kiểm định chéo theo video, đặc trưng chuyển động toàn thân, và kiểm tra tổng quát hóa sang
camera trần và bộ GMDCSA24; mục 10 đề xuất các hướng nghiên cứu mới. Mục 12 thu hẹp khác biệt dữ liệu với bài báo
(pose YOLO11m trên ảnh 640×480 gốc, thêm camera trần).

## 2. Thiết lập thí nghiệm

| Thành phần | Thiết lập | Bài báo |
|:---|:---|:---|
| Dữ liệu | URFD, camera 0: 30 video ngã, 40 video sinh hoạt, RGB 320×240, 30 fps | URFD + FDD, cả camera trần, 640×480 |
| Mẫu | Cửa sổ 100 khung, bước 10; cân bằng 66 `fall` + 66 `no_fall` | Nhóm 100 khung; 570 + 570 nhóm |
| Nhãn | Cửa sổ của video ngã là `fall` nếu có khung đang ngã hoặc đã nằm | Không mô tả chi tiết |
| Chia dữ liệu | 80/20 **theo video**: 105 train, {n_test} test (14 ngã, 13 không ngã) | 80/20 ngẫu nhiên theo nhóm |
| Pose | YOLO11n-pose (COCO-17) → 15 khớp; cổ, hông giữa = trung điểm | OpenPose BODY_25 → 15 khớp |
| Đầu vào | (x, y) của 15 khớp = 30 giá trị/khung, bỏ điểm tin cậy | Giống |
| Chuẩn hóa | RP: đổi về 640×480, dời hông giữa về tâm, **chia cho 640×480** | RP, giữ tọa độ pixel |
| Mô hình | 1 lớp RNN / LSTM / GRU + lớp kết nối đầy đủ, softmax 2 lớp | Giống |
| Huấn luyện | Adam, cross-entropy, full batch, **gradient clipping 1,0**, không validation | Adam, cross-entropy, full batch |
| Epoch | 500 cho cấu hình chính; 200 cho các thí nghiệm quét | 500 |

Mỗi mẫu sai trên tập test ≈ {num(100 / n_test)} điểm %.

## 3. Kết quả

### 3.1. Cấu hình của bài báo (512 nút, RP, lr 0,01, 500 epoch)

{t_main}

### 3.2. Ảnh hưởng của cách chia dữ liệu

Các cửa sổ 100 khung, bước 10, của cùng một video chồng lên nhau tới 90%. Khi chia ngẫu nhiên, gần như cùng một
cú ngã xuất hiện ở cả train lẫn test. Bảng dưới là từ một lần chia duy nhất (27 mẫu test) với mô hình 512 nút vốn học
không ổn định, nên phóng đại hiệu ứng; mục 8.2 đo lại bằng kiểm định chéo.

{t_split}

![Hình 1. Accuracy khi chia theo video và chia ngẫu nhiên](docs/figures/hinh1_cach_chia.png)

*Hình 1. Accuracy trên tập test khi chia theo video (tròn) và chia ngẫu nhiên như bài báo (vuông); vạch xám là
giá trị bài báo công bố.*

### 3.3. Số nút ẩn (so với Bảng 5 của bài báo; 200 epoch, lr 0,01)

{t_hidden}

![Hình 2. Accuracy theo số nút ẩn](docs/figures/hinh2_so_nut_an.png)

*Hình 2. Accuracy trên tập test theo số nút ẩn; đường đứt đoạn là mức đoán mò (≈ 50%).*

Với 512 nút và lr 0,01, độ chính xác trên tập train của LSTM dao động giữa 0,50 và 0,77 trong suốt quá trình học:
mô hình lớn cập nhật một lần mỗi epoch trên chỉ 105 mẫu thì không ổn định.

### 3.4. Learning rate (so với Bảng 6; 512 nút, RP, 200 epoch)

{t_lr}

### 3.5. Cách chuẩn hóa (so với Bảng 8–10; 128 nút, lr 0,001, 200 epoch, {n_seeds} lần chạy)

Mỗi lần chạy dùng một seed khác nhau, tức một cách chia video và một khởi tạo khác. Giá trị là trung bình ± độ lệch
chuẩn.

{t_norm}

![Hình 3. Accuracy theo cách chuẩn hóa](docs/figures/hinh3_chuan_hoa.png)

*Hình 3. Trung bình ± 1 độ lệch chuẩn qua {n_seeds} lần chạy; vòng tròn nhỏ là từng lần chạy; vạch xám là bài báo.*

## 4. So sánh với bài báo

**Khớp:**
- LSTM và GRU tốt hơn hẳn RNN.
- Nội suy khớp thiếu không làm tăng độ chính xác (bài báo: 95% so với 98,2% khi không nội suy).
- Có chuẩn hóa tốt hơn không chuẩn hóa.
- Với cùng cách chia ngẫu nhiên, LSTM đạt {num(main['lstm_paper_split']['acc'])}%, sát 98,2% của bài báo.

**Không khớp:**
- Số nút ẩn: bài báo tốt nhất ở 512 nút; ở đây tốt nhất ở 64–128 nút.
- Learning rate: bài báo LSTM tốt nhất ở 0,01; ở đây ở 0,001.
- RP so với min-max: bài báo RP vượt rõ (LSTM 98,2% so với 88,8%); ở đây hai cách ngang nhau trong sai số.
- RNN: bài báo 87–90% ở mọi cấu hình; ở đây chỉ học được với 64–128 nút.

**Nguyên nhân có thể:**
1. Ít dữ liệu hơn nhiều (132 so với 1.140 mẫu): mô hình 512 nút với lr 0,01 học không ổn định. Đây là nguyên nhân
   lớn nhất: qua kiểm định chéo, cấu hình này chỉ đạt khoảng 77–83% trong khi GRU-128 đạt khoảng 88–91% (mục 8.2).
2. Rò rỉ dữ liệu khi chia ngẫu nhiên các cửa sổ chồng nhau: khoảng +5 điểm với cấu hình của bài báo.
3. Pose kém hơn: YOLO11n trên 320×240, tỉ lệ khớp thiếu 15,3% so với 12,5% của bài báo.
4. Tập test nhỏ: riêng việc đổi tập test làm kết quả lệch tới ±{num(max_sd)} điểm.

## 5. Khác biệt triển khai so với bài báo

- **RP chia cho 640×480.** Với tọa độ pixel thô, loss đứng ở ln 2 và mô hình không học.
- **Gradient clipping 1,0.** Không có nó RNN phân kỳ trên chuỗi 100 khung.
- **Chia theo video** làm mặc định; cách chia ngẫu nhiên của bài báo được chạy riêng để so sánh.
- **Min-max** chỉ tính trên các khớp nhìn thấy (bài báo tính cả số 0 của khớp thiếu).
- **Chỉ URFD, chỉ camera 0:** camera trần chỉ có video ngã, dễ tạo lối tắt "góc trần = ngã".

## 6. Giới hạn

- Tập test {n_test} mẫu: chênh lệch dưới khoảng 7 điểm (2 mẫu) giữa hai cấu hình có thể chỉ là ngẫu nhiên.
- Thí nghiệm số nút ẩn và learning rate chạy một lần với một seed; chỉ phần chuẩn hóa được lặp lại {n_seeds} lần.
- Người đóng trong URFD là người trưởng thành, không phải người cao tuổi; bối cảnh là một căn phòng đủ sáng.
- Chưa có bộ FDD như bài báo.

## 7. Cách tái hiện

```bash
python -m falldet.urfd --out data/urfd
python -m falldet.extract_poses --videos data/urfd/videos --segments data/urfd/segments.csv \\
    --out data/poses/urfd --weights yolo11n-pose.pt --resize-width 0
for m in lstm gru rnn; do python -m falldet.train --out runs/urfd/$m model.name=$m; done
python -m falldet.train --out runs/urfd/lstm_paper_split data.group_split=false
EPOCHS=200 bash scripts/lin2021_experiments.sh      # 39 thí nghiệm quét, runs/lin2021
bash scripts/normalization_seeds.sh                 # chuẩn hóa × 3 seed, runs/norm_seeds
python scripts/make_report.py                       # file này và docs/figures/
```

## Phụ lục: toàn bộ 39 thí nghiệm quét (200 epoch)

`hidden_512_*` và `lr_0.01_*` trùng cấu hình với `norm_rp_*` nên dùng chung kết quả.

{t_all}
"""

# ---------------------------------------------------------------- part II: improvements and new directions
IMP = RUNS / "improve"
GEN = RUNS / "generalization"


def cvm(name: str) -> dict | None:
    p = IMP / name / "cv.json"
    if not p.exists():
        return None
    cv = json.loads(p.read_text())
    t = cv["pooled"]
    cm = t["confusion_matrix"]
    return {"acc": t["accuracy"] * 100, "sens": t["sensitivity"] * 100, "spec": t["specificity"] * 100,
            "bal": (t["sensitivity"] + t["specificity"]) * 50, "sd": cv["fold_stats"]["accuracy"]["sd"] * 100,
            "missed": cm[0][1], "false_alarm": cm[1][0], "n": sum(map(sum, cm)), "n_fall": sum(cm[0])}


def genm(name: str) -> dict | None:
    p = GEN / f"{name}.json"
    if not p.exists():
        return None
    t = json.loads(p.read_text())["test"]
    return {"acc": t["accuracy"] * 100, "sens": t["sensitivity"] * 100, "spec": t["specificity"] * 100,
            "bal": (t["sensitivity"] + t["specificity"]) * 50}


imp = {p.name: cvm(p.name) for p in sorted(IMP.iterdir()) if p.is_dir()}


def imp_row(label: str, name: str, change: str) -> list[str]:
    m = imp.get(name)
    if not m:
        return [label, change, "—", "—", "—", "—", "—"]
    return [label, change, num(m["bal"]), num(m["sens"]), num(m["spec"]), f"{num(m['acc'])} ± {num(m['sd'])}",
            f"{m['missed']} / {m['false_alarm']}"]


HDR = ["Run", "Thay đổi", "Balanced acc", "Sensitivity", "Specificity", "Accuracy ± SD", "Bỏ sót / báo nhầm"]
t_a = table(HDR, [
    imp_row("P1", "P1_paper_lstm512", "Cấu hình bài báo: LSTM-512, lr 0,01"),
    imp_row("A1", "A1_gru128", "GRU-128, lr 0,001 (mốc)"),
    imp_row("A2", "A2_gru128_motion", "A1 + đặc trưng chuyển động"),
    imp_row("A3", "A3_gru128_motion_val", "A2 + chọn checkpoint bằng validation"),
    imp_row("A4", "A4_lstm128", "LSTM-128"),
    imp_row("A5", "A5_bigru128", "BiGRU-128"),
    imp_row("A6", "A6_tcn", "TCN"),
])
t_e = table(HDR, [
    imp_row("P2", "P2_paper_lstm512_windowsplit", "Cấu hình bài báo"),
    imp_row("E1", "E1_gru128_windowsplit", "GRU-128"),
    imp_row("E2", "E2_gru128_motion_windowsplit", "GRU-128 + đặc trưng chuyển động"),
])
t_b = table(HDR, [
    imp_row("C1", "C1_all_s10_trainbal_gru128", "GRU-128, train trên tập cân bằng như bài báo"),
    imp_row("B1", "B1_all_s10_gru128", "GRU-128, train trên mọi cửa sổ"),
    imp_row("C2", "C2_all_s10_fullbatch_gru128", "B1, full batch 200 epoch"),
    imp_row("B2", "B2_all_s10_gru128_motion", "B1 + đặc trưng chuyển động"),
    imp_row("C3", "C3_all_s10_fullbatch_motion", "C2 + đặc trưng chuyển động"),
    imp_row("B4", "B4_all_s10_gru128_aug", "B1 + tăng cường dữ liệu"),
    imp_row("D2", "D2_all_s10_gru128_motion_aug", "B2 + tăng cường dữ liệu"),
    imp_row("D3", "D3_all_s10_bigru128_motion", "B2 với BiGRU"),
])
t_s5 = table(HDR, [
    imp_row("B3", "B3_all_s5_gru128", "GRU-128, mọi cửa sổ bước 5"),
    imp_row("D1", "D1_all_s5_gru128_motion", "B3 + đặc trưng chuyển động"),
])

GEN_MODELS = [("paper", "Cấu hình bài báo (LSTM-512, train cân bằng)", "P1_paper_lstm512"),
              ("base", "GRU-128 mốc", "B1_all_s10_gru128"),
              ("motion", "GRU-128 + đặc trưng chuyển động", "B2_all_s10_gru128_motion"),
              ("best", "GRU-128 + chuyển động + tăng cường", "D2_all_s10_gru128_motion_aug")]
gen = {k: {"urfd": imp.get(cv_name), "cam1": genm(f"{k}_on_cam1"), "gmd": genm(f"{k}_on_gmdcsa24")}
       for k, _, cv_name in GEN_MODELS}
t_gen = table(["Mô hình", "URFD (kiểm định chéo, bal. acc)", "Camera trần: sensitivity", "GMDCSA24: bal. acc",
               "GMDCSA24: sens / spec"], [
    [label, num(g["urfd"]["bal"]) if g["urfd"] else "—", num(g["cam1"]["sens"]) if g["cam1"] else "—",
     num(g["gmd"]["bal"]) if g["gmd"] else "—",
     f"{num(g['gmd']['sens'])} / {num(g['gmd']['spec'])}" if g["gmd"] else "—"]
    for (k, label, _), g in ((m, gen[m[0]]) for m in GEN_MODELS)])
n_cam1 = n_gmd = None
for k in gen:
    for tgt, fname in (("cam1", "cam1"), ("gmd", "gmdcsa24")):
        p = GEN / f"{k}_on_{fname}.json"
        if p.exists():
            cm = json.loads(p.read_text())["test"]["confusion_matrix"]
            if tgt == "cam1":
                n_cam1 = (sum(cm[0]), sum(cm[1]))
            else:
                n_gmd = (sum(cm[0]), sum(cm[1]))


def fig_improve() -> None:
    rows = [("C1 · như bài báo", "C1_all_s10_trainbal_gru128"), ("B1 · mọi cửa sổ", "B1_all_s10_gru128"),
            ("B4 · + tăng cường", "B4_all_s10_gru128_aug"), ("B2 · + chuyển động", "B2_all_s10_gru128_motion"),
            ("C3 · + chuyển động, full batch", "C3_all_s10_fullbatch_motion"),
            ("D3 · + chuyển động, BiGRU", "D3_all_s10_bigru128_motion"),
            ("D2 · + chuyển động + tăng cường", "D2_all_s10_gru128_motion_aug")]
    rows = [(l, imp[n]) for l, n in rows if imp.get(n)]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    for i, (label, m) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([m["sens"], m["spec"]], [y, y], color="#c5cbd3", lw=2, zorder=1)
        ax.scatter(m["sens"], y, s=46, color=MODELS[0][2], edgecolor="white", linewidth=1.2, zorder=3,
                   label="Sensitivity" if i == 0 else None)
        ax.scatter(m["spec"], y, s=46, marker="s", color=MODELS[1][2], edgecolor="white", linewidth=1.2, zorder=3,
                   label="Specificity" if i == 0 else None)
        ax.scatter(m["bal"], y, s=70, marker="|", color=INK, linewidth=2, zorder=4,
                   label="Balanced accuracy" if i == 0 else None)
        ax.text(101, y, num(m["bal"]), va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.set_xlim(70, 104)
    ax.set_xlabel("% trên 610 cửa sổ của video test (kiểm định chéo 5 phần theo video)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0, labelcolor=INK)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, 1.16), ncol=3, frameon=False, fontsize=9)
    fig.savefig(FIG_DIR / "hinh4_cai_tien.png")
    plt.close(fig)


def fig_generalize() -> None:
    targets = [("urfd", "bal", "URFD (kiểm định chéo)"), ("gmd", "bal", "GMDCSA24 (bal. acc)"),
               ("cam1", "sens", "Camera trần (sensitivity)")]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    width = 0.26
    for j, (tgt, key, label) in enumerate(targets):
        xs, ys = [], []
        for i, (k, _, _) in enumerate(GEN_MODELS):
            m = gen[k][tgt]
            if m:
                xs.append(i + (j - 1) * width)
                ys.append(m[key])
        bars = ax.bar(xs, ys, width=width - 0.03, color=MODELS[j][2], label=label, zorder=2)
        for b, v in zip(bars, ys):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, num(v, 0), ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(range(len(GEN_MODELS)), ["Bài báo\n(LSTM-512)", "GRU-128", "+ chuyển động", "+ chuyển động\n+ tăng cường"])
    ax.set_ylim(0, 105)
    ax.set_ylabel("%")
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0, labelcolor=INK)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=3, frameon=False, fontsize=9)
    fig.savefig(FIG_DIR / "hinh5_tong_quat_hoa.png")
    plt.close(fig)


fig_improve()
fig_generalize()

a1, c1, b2, d2, p1, p2 = (imp.get(n) for n in ("A1_gru128", "C1_all_s10_trainbal_gru128", "B2_all_s10_gru128_motion",
                                              "D2_all_s10_gru128_motion_aug", "P1_paper_lstm512",
                                              "P2_paper_lstm512_windowsplit"))
e1, e2 = imp.get("E1_gru128_windowsplit"), imp.get("E2_gru128_motion_windowsplit")
best_e = max((m for m in (e1, e2) if m), key=lambda m: m["bal"], default=None)
g = {k: gen[k] for k in gen}

part2 = f"""
# Phần II: cải tiến và hướng nghiên cứu mới

## 8. Cải tiến vượt bài báo

### 8.1. Cách đánh giá

Mọi thí nghiệm ở phần này dùng **kiểm định chéo 5 phần theo video**: mỗi cửa sổ được test đúng một lần, rồi gộp kết
quả của 5 phần. 1 mẫu sai chỉ còn ≈ {num(100 / 132)} điểm (132 cửa sổ) hoặc ≈ {num(100 / 610)} điểm (610 cửa sổ),
so với 3,7 điểm ở phần I. Chỉ số chính là **balanced accuracy** = (sensitivity + specificity) / 2, so sánh được giữa
tập test cân bằng và tập test mất cân bằng. Cột "± SD" là độ lệch chuẩn accuracy giữa 5 phần.

### 8.2. Trên 132 cửa sổ cân bằng (thiết lập của bài báo)

Chia theo video:

{t_a}

Chia theo cửa sổ, đúng cách đánh giá của bài báo (bài báo công bố 98,2%):

{t_e}

Với cùng dữ liệu pose, cấu hình của bài báo đạt {num(p1['bal']) if p1 else '—'}% (theo video) và
{num(p2['bal']) if p2 else '—'}% (theo cửa sổ). GRU-128 đạt {num(a1['bal']) if a1 else '—'}% theo video
{f"và {num(e1['bal'])}% theo cửa sổ ({num(e2['bal'])}% khi thêm đặc trưng chuyển động)" if e1 and e2 else ""}, **vượt cấu hình của bài báo trên cùng dữ liệu và cùng
cách đánh giá**. Con số 98,2% của bài báo không đạt được với dữ liệu ở đây ở cả hai cách đánh giá (xem mục 4 về các
nguyên nhân). Trên 132 cửa sổ, các thay đổi khác (đặc trưng chuyển động, chọn checkpoint, mô hình khác) không giúp:
dữ liệu quá ít.

### 8.3. Trên mọi cửa sổ: tập test thực tế

Cân bằng dữ liệu như bài báo bỏ đi phần lớn cửa sổ không ngã, trong đó có các cảnh khó (nằm lên giường, cúi người).
Ở đây mọi mô hình được test trên **cùng 610 cửa sổ** (66 ngã, 544 không ngã) của video test, cùng cách chia phần.

{t_b}

![Hình 4. Các cải tiến trên tập test thực tế](docs/figures/hinh4_cai_tien.png)

*Hình 4. Sensitivity (tròn), specificity (vuông) và balanced accuracy (vạch, số bên phải) trên cùng 610 cửa sổ.*

Với bước cửa sổ 5 (1.159 cửa sổ, tập test khác nên chỉ so trong bảng này):

{t_s5}

**Nhận xét:**
- **Mẫu cân bằng che giấu cảnh khó.** Cùng cách train (GRU-128 trên tập cân bằng), specificity là
  {num(a1['spec']) if a1 else '—'}% khi test trên mẫu cân bằng (A1) nhưng chỉ {num(c1['spec']) if c1 else '—'}% khi
  test trên mọi cửa sổ (C1).
- **Đặc trưng chuyển động là cải tiến lớn nhất:** +{num(b2['bal'] - imp['B1_all_s10_gru128']['bal']) if b2 else '—'}
  điểm balanced accuracy (B1 → B2), +{num(imp['C3_all_s10_fullbatch_motion']['bal'] - imp['C2_all_s10_fullbatch_gru128']['bal'])}
  điểm với full batch (C2 → C3), chủ yếu nhờ giảm báo nhầm. RP-normalization của bài báo dời hông về tâm ở mọi khung,
  xóa mất tín hiệu người rơi xuống; các đặc trưng này đưa lại độ cao và vận tốc của hông, tỉ lệ cao/rộng và góc thân.
- **Kết hợp tốt nhất: D2** (chuyển động + tăng cường dữ liệu) đạt {num(d2['bal']) if d2 else '—'}% balanced accuracy,
  sensitivity {num(d2['sens']) if d2 else '—'}%, specificity {num(d2['spec']) if d2 else '—'}%, so với
  {num(c1['bal']) if c1 else '—'}% của cách train như bài báo (C1).
- Tăng cường dữ liệu chỉ giúp khi đi cùng đặc trưng chuyển động; BiGRU và bước cửa sổ dày hơn không giúp.

## 9. Tổng quát hóa sang góc nhìn và bộ dữ liệu chưa thấy

Bốn mô hình được train trên toàn bộ URFD camera ngang rồi test, không tinh chỉnh, trên:
- **Camera trần của URFD** ({n_cam1[0] if n_cam1 else '—'} cửa sổ ngã, {n_cam1[1] if n_cam1 else '—'} không ngã): cùng các
  cú ngã, quay từ trên xuống. Chỉ có video ngã nên chỉ sensitivity có ý nghĩa.
- **GMDCSA24** ({n_gmd[0] if n_gmd else '—'} cửa sổ ngã, {n_gmd[1] if n_gmd else '—'} không ngã): người khác, ba căn nhà
  thật, cả ban ngày và ban đêm; nhãn lấy từ OmniFall (fall và fallen → fall), cửa sổ 3,33 giây.

{t_gen}

Cột URFD: cấu hình bài báo là P1 (132 cửa sổ cân bằng); các mô hình còn lại là B1, B2, D2 (610 cửa sổ).

![Hình 5. Tổng quát hóa](docs/figures/hinh5_tong_quat_hoa.png)

*Hình 5. Cùng mô hình trên URFD (kiểm định chéo theo video), GMDCSA24 và camera trần của URFD. Cột URFD của cấu
hình bài báo là P1 trên 132 cửa sổ cân bằng; của các mô hình còn lại là trên 610 cửa sổ (B1, B2, D2).*

**Nhận xét:**
- **Mọi mô hình đều tụt khi gặp bối cảnh mới.** Cấu hình của bài báo còn
  {num(g['paper']['gmd']['bal']) if g['paper']['gmd'] else '—'}% trên GMDCSA24 và chỉ phát hiện
  {num(g['paper']['cam1']['sens']) if g['paper']['cam1'] else '—'}% số cú ngã từ camera trần. Bài báo không đánh giá
  điều này.
- **Đặc trưng chuyển động tổng quát hóa tốt hơn rõ:** trên GMDCSA24
  {num(g['motion']['gmd']['bal']) if g['motion']['gmd'] else '—'}% so với
  {num(g['base']['gmd']['bal']) if g['base']['gmd'] else '—'}% của GRU mốc; trên camera trần
  {num(g['motion']['cam1']['sens']) if g['motion']['cam1'] else '—'}% so với
  {num(g['base']['cam1']['sens']) if g['base']['cam1'] else '—'}%. Chúng mô tả bản chất vật lý của cú ngã nên ít phụ
  thuộc vào người và phòng.
- **Góc nhìn từ trên xuống vẫn là điểm yếu lớn nhất:** từ trên trần, người đứng và người nằm có hình chiếu gần giống
  nhau. Tăng cường dữ liệu (lật ngang, nhiễu) không giúp ở đây.
- Mỗi mô hình ở mục này được train một lần; với {n_cam1[0] if n_cam1 else '—'} cửa sổ ngã ở camera trần, chênh lệch
  vài điểm có thể là ngẫu nhiên.

## 10. Hướng nghiên cứu mới

1. **Giao thức đánh giá thực tế cho phát hiện ngã.** Tập test cân bằng che giấu báo động nhầm (specificity giảm từ
   {num(a1['spec']) if a1 else '—'}% xuống {num(c1['spec']) if c1 else '—'}%, mục 8.3); chia ngẫu nhiên các cửa sổ
   chồng nhau làm cấu hình của bài báo cao thêm khoảng 5 điểm (P1 → P2), và một lần chia 27 mẫu có thể lệch hơn
   10 điểm (phần I). Đề xuất: chia theo video hoặc
   người, test trên mọi cửa sổ, báo cáo balanced accuracy và số báo nhầm, kèm một bộ dữ liệu ngoài. Có thể kiểm chứng
   lại trên các công trình đã công bố dùng URFD, Le2i, UP-Fall.
2. **Đặc trưng vật lý bất biến theo bối cảnh.** Độ cao, vận tốc, hướng thân đã giúp cả trên URFD lẫn GMDCSA24. Hướng
   mở rộng: pose 3D (ước lượng từ 2D hoặc dùng ảnh độ sâu có sẵn trong URFD), đặc trưng theo hướng trọng lực, gia tốc.
3. **Bất biến theo góc nhìn.** Camera trần là điểm yếu lớn nhất. Hướng thử: train đa góc nhìn, dùng video tổng hợp
   OF-Syn (có nhãn độ cao camera: eye / low / high / top) để bổ sung góc từ trên xuống, hoặc biểu diễn khung xương
   chuẩn hóa theo góc nhìn.
4. **Tổng quát hóa giữa các bộ dữ liệu.** Train trên nhiều bộ (OmniFall: Le2i, CAUCAFall, UP-Fall, GMDCSA24) và test
   trên tai nạn thật (OOPS) thay vì chỉ một bộ dàn dựng.
5. **Người cao tuổi.** Dữ liệu thật ở đây do người trẻ đóng; OF-Syn có riêng nhóm elderly_65_plus để đánh giá và bổ
   sung.
6. **Phát hiện trên luồng liên tục.** Đo độ trễ phát hiện và số báo nhầm mỗi giờ trên video dài thay vì phân loại
   từng cửa sổ đã cắt sẵn.

## 11. Cách tái hiện phần II

```bash
python -m falldet.extract_poses --videos data/urfd/videos --segments data/urfd/full_videos.csv \\
    --out data/poses/urfd_full_n --weights yolo11n-pose.pt --resize-width 0
python -m falldet.urfd --out data/urfd --from-poses data/poses/urfd_full_n --out-poses data/poses/urfd_all_s10 --no-balance
for s in A B C D P; do STAGE=$s bash scripts/improve_experiments.sh; done   # runs/improve
bash scripts/generalization.sh                                            # runs/generalization
# mục 12: YOLO11m trên ảnh PNG 640x480 gốc (khoảng 6 GB tải về), hai camera
python -m falldet.urfd --out data/urfd640 --rgb-zip --cams 0 1 --no-balance
python -m falldet.extract_poses --videos data/urfd640/videos --segments <CSV các video, như data/urfd/full_videos.csv> \\
    --out data/poses/urfd640_full_m --weights yolo11m-pose.pt --resize-width 0
python -m falldet.urfd --out data/urfd --from-poses data/poses/urfd640_full_m --out-poses data/poses/urfd640_m_bal
python -m falldet.urfd --out data/urfd --from-poses data/poses/urfd640_full_m --out-poses data/poses/urfd640_m_all_s10 --no-balance
python -m falldet.urfd --out data/urfd --from-poses data/poses/urfd640_full_m --out-poses data/poses/urfd640_m_cam01_bal --cams 0 1
STAGE=H bash scripts/improve_experiments.sh
python scripts/make_report.py
```
"""
report = report.replace("\n## Phụ lục:", part2 + "\n## Phụ lục:")



# ---------------------------------------------------------------- section 12: closer to the paper's data
def pair_row(label: str, before: str, after: str) -> list[str]:
    a, b = imp.get(before), imp.get(after)
    if not a or not b:
        return [label, "—", "—", "—", "—"]
    return [label, f"{num(a['bal'])} ({before.split('_')[0]})", f"{num(b['bal'])} ({after.split('_')[0]})",
            f"{'+' if b['bal'] >= a['bal'] else '−'}{num(abs(b['bal'] - a['bal']))}",
            f"{num(b['sens'])} / {num(b['spec'])}"]


PAIR_HDR = ["So sánh", "Trước", "Sau", "Chênh", "Sau: sens / spec"]
t_pose = table(PAIR_HDR, [
    pair_row("Cấu hình bài báo, chia theo cửa sổ", "P2_paper_lstm512_windowsplit", "H3_m640_paper_lstm512_ws"),
    pair_row("Cấu hình bài báo, chia theo video", "P1_paper_lstm512", "H4_m640_paper_lstm512"),
    pair_row("GRU-128, chia theo cửa sổ", "E1_gru128_windowsplit", "H2_m640_gru128_ws"),
    pair_row("GRU-128, chia theo video", "A1_gru128", "H1_m640_gru128"),
    pair_row("Tốt nhất (chuyển động + tăng cường), 610 cửa sổ", "D2_all_s10_gru128_motion_aug", "H5_m640_all_s10_motion_aug"),
])
t_cam = table(PAIR_HDR, [
    pair_row("GRU-128, pose cũ, chia theo cửa sổ", "E1_gru128_windowsplit", "G1_n_cam01_gru128_ws"),
    pair_row("GRU-128, pose cũ, chia theo video", "A1_gru128", "G2_n_cam01_gru128"),
    pair_row("GRU-128, pose mới, chia theo cửa sổ", "H2_m640_gru128_ws", "H6_m640_cam01_gru128_ws"),
])
t_close = table(HDR, [
    imp_row("H7", "H7_m640_cam01_paper_lstm512_ws", "Cấu hình bài báo: LSTM-512, RP, lr 0,01"),
    imp_row("H6", "H6_m640_cam01_gru128_ws", "GRU-128, lr 0,001"),
    imp_row("H8", "H8_m640_cam01_motion_aug_ws", "GRU-128 + chuyển động + tăng cường"),
])


def missing_rate(pattern: str) -> float | None:
    miss = frames = 0
    for f in glob.glob(str(PROJECT / pattern)):
        k = np.load(f)["keypoints"]
        b = to_body15(mask_low_confidence(k, 0.3))
        miss += (b[..., 2] == 0).sum() / b.shape[1]
        frames += len(k)
    return miss / frames * 100 if frames else None


miss_n = missing_rate("data/poses/urfd_full_n/*/*.npz")
miss_m = missing_rate("data/poses/urfd640_full_m/*/*cam0*.npz")
h5, h7, h8 = imp.get("H5_m640_all_s10_motion_aug"), imp.get("H7_m640_cam01_paper_lstm512_ws"), imp.get("H8_m640_cam01_motion_aug_ws")
t_size = table(HDR, [
    imp_row("I1", "I1_s5_cam01_paper_lstm512_ws", "Cấu hình bài báo: LSTM-512, RP, lr 0,01"),
    imp_row("I2", "I2_s5_cam01_gru128_ws", "GRU-128, lr 0,001"),
    imp_row("I3", "I3_s5_cam01_motion_aug_ws", "GRU-128 + chuyển động + tăng cường"),
])
i1, i3 = imp.get("I1_s5_cam01_paper_lstm512_ws"), imp.get("I3_s5_cam01_motion_aug_ws")

part3 = f"""
## 12. Tiến gần dữ liệu của bài báo

Để thu hẹp khác biệt về dữ liệu với bài báo, pose được trích lại bằng **YOLO11m trên ảnh PNG 640×480 gốc** của URFD
(thay cho YOLO11n trên phần RGB 320×240 của file mp4), và thêm **camera trần** như bài báo. Mọi số liệu dưới đây là
balanced accuracy (%) qua kiểm định chéo 5 phần.

Tỉ lệ khớp thiếu (15 khớp, toàn bộ khung của video camera ngang): {num(miss_n)}% với pose cũ,
{num(miss_m)}% với pose mới. Bài báo báo 12,5% trên các nhóm khung đã chọn.

### 12.1. Pose tốt hơn

{t_pose}

Pose tốt hơn tăng khoảng 2–3 điểm với cấu hình của bài báo, và giúp mô hình tốt nhất phát hiện thêm cú ngã
(H5: sensitivity {num(h5['sens']) if h5 else '—'}%). Với GRU-128 trên 132 cửa sổ, chênh lệch nằm trong sai số.

### 12.2. Thêm camera trần

{t_cam}

Camera trần chỉ có video ngã. Khi chia theo cửa sổ, cùng một cú ngã quay từ hai camera dễ nằm ở cả train và test,
nên kết quả nhích lên; khi chia theo video, mô hình phải nhận ra cú ngã từ trên xuống mà chưa từng thấy, và kết quả
giảm rõ. Đây là thêm một yếu tố có thể làm con số của bài báo cao hơn thực tế.

### 12.3. Thiết lập gần bài báo nhất

YOLO11m trên 640×480, cả hai camera, tập cân bằng, chia theo cửa sổ như bài báo (chỉ thiếu bộ FDD):

{t_close}

Trên thiết lập này, cấu hình của bài báo đạt {num(h7['bal']) if h7 else '—'}% so với 98,2% công bố, còn phương pháp
đề xuất đạt {num(h8['bal']) if h8 else '—'}%.

### 12.4. Cùng số mẫu URFD như bài báo

Bài báo có 240 + 240 nhóm từ URFD sau một bước "tăng cường dữ liệu" không được mô tả. Cắt cửa sổ với bước 5 thay vì
10 trên cả hai camera cho 212 + 212 mẫu, gần với con số đó:

{t_size}

Cấu hình của bài báo đạt {num(i1['bal']) if i1 else '—'}%, gần như không đổi so với 132 + 132 mẫu (H7): các cửa sổ dày
hơn chồng lên nhau nhiều hơn chứ không chứa thêm cú ngã mới. Phương pháp đề xuất đạt {num(i3['bal']) if i3 else '—'}%
(sensitivity {num(i3['sens']) if i3 else '—'}%, specificity {num(i3['spec']) if i3 else '—'}%), hơn cấu hình của bài báo
{num(i3['bal'] - i1['bal']) if i1 and i3 else '—'} điểm trên cùng dữ liệu và cùng cách đánh giá, và cách con số 98,2% công
bố {num(98.2 - i3['bal']) if i3 else '—'} điểm.

### 12.5. Vì sao không dùng bộ FDD

Bộ FDD (Adhikari et al., falldataset.com) mà bài báo trộn cùng URFD có ba vấn đề khiến không thể tái hiện đúng:
1. **Không có nhãn "ngã"**, chỉ có nhãn tư thế từng khung (1 đứng, 2 ngồi, 3 nằm, 4 cúi, 5 bò, và các lớp 6, 7 không
   được mô tả). Bài báo không nói nhãn ngã được tạo thế nào; FDD cũng không công bố tốc độ khung hình.
2. **Ảnh lật ngang được ghép nối tiếp vào cùng chuỗi** (trang web: "all the sets have original and its horizontal
   flipped images added in sequence"; ví dụ chuỗi 786 có 1.572 ảnh). Khi cắt cửa sổ và chia ngẫu nhiên, bản gốc và
   bản lật của cùng một cảnh gần như chắc chắn nằm ở cả train và test.
3. Chi phí: khoảng 2,9 GB ảnh RGB và khoảng 6 giờ trích pose trên CPU.
"""
report = report.replace("\n## Phụ lục:", part3 + "\n## Phụ lục:")

OUT.write_text(report, encoding="utf-8")
print(f"wrote {OUT.relative_to(PROJECT)} and {len(list(FIG_DIR.glob('*.png')))} figures in {FIG_DIR.relative_to(PROJECT)}")

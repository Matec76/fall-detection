# Phát hiện người ngã bằng thị giác máy tính (pose-based fall detection)

Dự án tái hiện và mở rộng bài báo tham khảo chính:

> C.-B. Lin, Z. Dong, W.-K. Kuan, Y.-F. Huang, *"A Framework for Fall Detection Based on OpenPose Skeleton
> and LSTM/GRU Models"*, Applied Sciences, 11(1), 329, 2021. DOI: 10.3390/app11010329

```
video ─► pose 2D ─► 15 khớp (x, y) ─► nội suy khớp thiếu ─► RP-normalization ─► RNN / LSTM / GRU ─► fall / no_fall
        (YOLO-pose)   (Sec. 2.3.2)      (Sec. 2.3.3, tùy chọn)   (Eq. 2–7)            (1 lớp ẩn, Fig. 13)
```

**Báo cáo kết quả:** [BAO_CAO.md](BAO_CAO.md) (sinh lại bằng `python scripts/make_report.py`).

Bài tham khảo trước đó (Juraev et al., IEEE Access 2022: Transformer, 4 lớp hành động, dữ liệu tổng hợp) vẫn được
giữ làm **baseline mở rộng** trong [configs/juraev2022.yaml](configs/juraev2022.yaml).

## Ánh xạ bài báo → mã nguồn

| Bài báo (Lin et al. 2021) | File |
|---|---|
| Chuẩn bị URFD: nhóm 100 khung, nhãn fall / no_fall, cân bằng (Sec. 2.1, Bảng 1) | [falldet/urfd.py](falldet/urfd.py) |
| Lấy khung xương từng khung hình (Sec. 2.2) | [falldet/pose.py](falldet/pose.py), [falldet/extract_poses.py](falldet/extract_poses.py) |
| 15 khớp, bỏ điểm tin cậy → 30 đầu vào (Sec. 2.3.1–2.3.2) | [falldet/preprocess.py](falldet/preprocess.py) `to_body15` |
| Min-max (Eq. 1), RP-normalization (Eq. 2–7) | [falldet/preprocess.py](falldet/preprocess.py) `normalize` |
| Nội suy tuyến tính khớp thiếu, ngưỡng 67% (Eq. 8) | [falldet/preprocess.py](falldet/preprocess.py) `interpolate_missing` |
| RNN / LSTM / GRU một lớp ẩn + softmax (Sec. 2.4, Bảng 4) | [falldet/models.py](falldet/models.py) `RecurrentClassifier` |
| Adam, cross-entropy, full batch, 500 epoch, chia 80/20 (Bảng 3) | [falldet/train.py](falldet/train.py), [configs/default.yaml](configs/default.yaml) |
| Sensitivity, specificity, accuracy (Eq. 21–23) | [falldet/metrics.py](falldet/metrics.py) |
| Các thí nghiệm Bảng 5–10 | [scripts/lin2021_experiments.sh](scripts/lin2021_experiments.sh), [falldet/summarize.py](falldet/summarize.py) |
| Demo thời gian thực có cảnh báo | [falldet/demo.py](falldet/demo.py) |

## Cài đặt

```bash
python3 -m venv .venv && source .venv/bin/activate
# Cài torch + torchvision CÙNG MỘT NGUỒN trước (chọn bản CUDA hoặc CPU tại pytorch.org).
# Nếu để ultralytics tự kéo torchvision, có thể gặp lỗi "operator torchvision::nms does not exist".
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # hoặc bản cu12x nếu có GPU
pip install -e ".[dev]"
pytest                      # ~1 phút, chạy hoàn toàn bằng dữ liệu giả
```

Trọng số YOLO pose (`yolo11n-pose.pt`, `yolo11m-pose.pt`, …) được ultralytics tự tải ở lần chạy đầu.

## Dữ liệu

| Bộ dữ liệu | Vai trò | Ghi chú |
|---|---|---|
| **URFD** (UR Fall Detection) | Chính, như bài báo | 30 video ngã + 40 video sinh hoạt, 30 fps, nhãn từng khung. Tải tự động bằng `falldet.urfd`. CC BY-NC-SA 4.0 |
| FDD (Adhikari et al., falldataset.com) | Bộ thứ hai của bài báo | Chưa hỗ trợ: ảnh 320×240 theo thư mục, nhãn tư thế (đứng/ngồi/nằm/cúi/bò), không có nhãn "ngã" trực tiếp |
| **OmniFall** (+ OF-Syn, OOPS) | Mở rộng: kiểm tra chéo, dữ liệu tổng hợp, tai nạn thật | Xem [mục OmniFall](#mở-rộng-omnifall) |
| ~~AI Hub~~ | Không dùng | Bộ của bài Juraev chỉ cho công dân Hàn Quốc đăng ký |

## Quy trình với URFD

### 1. Tải URFD và tạo danh sách cửa sổ
```bash
python -m falldet.urfd --out data/urfd
```
Tải mp4 (~100 MB), tách phần RGB (mp4 của URFD ghép ảnh độ sâu | RGB), cắt cửa sổ 100 khung mỗi 10 khung và cân bằng
fall / no_fall. Các tùy chọn quan trọng:

| Tùy chọn | Ý nghĩa |
|---|---|
| `--rgb-zip` | Dùng ảnh PNG 640×480 gốc thay cho RGB 320×240 trong mp4 (~60 MB mỗi chuỗi) |
| `--cams 0 1` | Thêm camera trần như bài báo. **Cẩn thận:** camera trần chỉ có video ngã, mô hình có thể học "góc trần = ngã" |
| `--fall-rule down\|event\|sequence` | Cửa sổ của video ngã là `fall` khi: có khung đang ngã hoặc đã nằm (mặc định) / có khung đang ngã / luôn luôn |
| `--stride 5` | Nhiều cửa sổ hơn (mặc định cam 0 cho 66 cửa sổ fall) |
| `--no-balance` | Giữ mọi cửa sổ (544 no_fall), nên bật `train.class_weights=true` |

### 2. Trích pose
```bash
python -m falldet.extract_poses --videos data/urfd/videos --segments data/urfd/segments.csv \
    --out data/poses/urfd --weights yolo11m-pose.pt --resize-width 0
```
Trên CPU dùng `--weights yolo11n-pose.pt` cho nhanh; có GPU thêm `--device 0`.

### 3. Huấn luyện và đánh giá
```bash
python -m falldet.train --out runs/lstm                                    # LSTM-512, RP, lr 0.01 (Bảng 9)
python -m falldet.train --out runs/gru model.name=gru
python -m falldet.train --out runs/lstm_minmax preprocess.norm_mode=minmax
python -m falldet.train --out runs/lstm_interp preprocess.interpolate=true
python -m falldet.evaluate --run runs/lstm
```
Mọi khóa trong [configs/default.yaml](configs/default.yaml) đều ghi đè được bằng `khóa.con=giá_trị`. Mỗi run lưu
`best.pt`, `last.pt`, `config.yaml`, `split.json`, `metrics.json` và in accuracy / sensitivity / specificity trên tập test.

### 4. Toàn bộ thí nghiệm của bài báo
```bash
bash scripts/lin2021_experiments.sh          # 39 run: 3 mô hình × (4 kiểu chuẩn hóa, 5 số nút ẩn, 3 learning rate, split của bài báo)
python -m falldet.summarize runs/lin2021 --csv results.csv
```

### 5. Demo
```bash
python -m falldet.demo --run runs/lstm --source video.mp4 --save out.mp4
python -m falldet.demo --run runs/lstm --source 0                        # webcam
```
Mỗi `--stride` khung, `seq_len` (100) khung gần nhất được phân loại; cảnh báo **FALL DETECTED** khi `--consecutive`
lần liên tiếp xác suất `fall` ≥ `--fall-thr`.

## Khác biệt so với bài báo & lựa chọn thiết kế

Những điểm này nên được nêu trong báo cáo:

- **Pose estimator:** YOLO11-pose (COCO-17, cài bằng pip) thay cho OpenPose BODY_25. Cổ và hông giữa không có trong
  COCO-17 nên được lấy là trung điểm hai vai / hai hông; 13 khớp còn lại trùng với danh sách 15 khớp của bài báo.
- **RP-normalization chia cho 640×480** (`preprocess.rp_scale=true`). Bài báo giữ tọa độ pixel; khi thử như vậy
  (`rp_scale=false`) các cổng của LSTM bão hòa và mô hình dừng ở mức đoán mò (loss ≈ ln 2).
  Khung không thấy hông giữa được dời theo trung bình các khớp nhìn thấy.
- **Gradient clipping** (`train.grad_clip=1.0`). Không có trong bài báo; thiếu nó RNN thường phân kỳ trên chuỗi 100 khung.
- **Chia train/test theo video** (`data.group_split=true`). Bài báo chia ngẫu nhiên các nhóm 100 khung; vì các cửa sổ
  chồng lên nhau, cùng một cú ngã có thể nằm ở cả train và test, làm kết quả lạc quan.
  `data.group_split=false` tái hiện cách chia của bài báo để so sánh (có sẵn trong script thí nghiệm).
- **Min-max** chỉ tính trên khớp nhìn thấy; bản của bài báo tính cả số 0 của khớp thiếu (chính lỗi mà RP khắc phục).
- **Không có tập validation** (giống bài báo): kết quả là của epoch cuối. `data.val_size=0.1` để chọn checkpoint tốt nhất.
- **Mặc định không nội suy** vì kết quả tốt nhất của bài báo (98,2%, Bảng 9) là RP không nội suy.
- **Chỉ camera 0** của URFD (xem `--cams` ở trên) và RGB 320×240 từ mp4 (xem `--rgb-zip`).

## Mở rộng: OmniFall

[OmniFall](https://huggingface.co/datasets/simplexsigil2/omnifall) gán nhãn thống nhất cho 8 bộ dữ liệu dàn dựng
(Le2i, CAUCAFall, GMDCSA24, UP-Fall, …), tai nạn thật (OOPS) và 12.000 video tổng hợp (OF-Syn), có split chính thức
theo người (cs) / góc máy (cv). Dùng để kiểm tra mô hình huấn luyện trên URFD có tổng quát hóa sang bối cảnh khác không,
và để thử dữ liệu tổng hợp.

```bash
pip install -e ".[omnifall]"                 # omnifall + imageio-ffmpeg (ffmpeg nếu máy chưa có)
bash scripts/omnifall_pipeline.sh            # tải → cửa sổ 100 khung, nhãn fall/no_fall → pose → Real / Real+Syn / Syn → test OOPS
COMPONENTS=GMDCSA24 WEIGHTS=yolo11n-pose.pt DEVICE=cpu SKIP_SYN=1 SKIP_OOPS=1 bash scripts/omnifall_pipeline.sh   # bản nhỏ
```

Từng bước:
```bash
omnifall prepare GMDCSA24 le2i --yes         # video -> ~/.cache/omnifall/videos/{dataset}/video/{path}.mp4
python -m falldet.omnifall --components GMDCSA24 le2i --label-map binary --window-sec 3.333 --out data/omnifall/staged_binary.csv
python -m falldet.extract_poses --videos ~/.cache/omnifall/videos --segments data/omnifall/staged_binary.csv \
    --out data/poses/omnifall/staged_binary
python -m falldet.evaluate --run runs/lstm --pose-dir data/poses/omnifall/staged_binary/test   # URFD -> OmniFall
```

- `--label-map binary`: `fall` và `fallen` (nằm sau khi ngã) → `fall`, mọi nhãn khác → `no_fall`.
  `paper4` / `core10` giữ 4 lớp của bài Juraev / 10 lớp gốc của OmniFall.
- `--age-groups elderly_65_plus` chỉ giữ người trên 65 tuổi trong OF-Syn. Dữ liệu dàn dựng thật chủ yếu do người trẻ
  đóng; đây là cách duy nhất có dữ liệu "người già" trong dự án.
- Mỗi đoạn được cắt/đệm thành mẫu dài đúng `--window-sec`. Giấy phép nhãn: CC BY-NC-SA 4.0; video thuộc tác giả gốc.

Baseline 4 lớp của bài Juraev:
```bash
LABEL_MAP=paper4 CONFIG=configs/juraev2022.yaml WINDOW_SEC=2 MODELS="transformer stacked_lstm" bash scripts/omnifall_pipeline.sh
```

## Hướng phát triển gợi ý

- So sánh chia theo video với chia ngẫu nhiên như bài báo: mức chênh chính là độ lạc quan do rò rỉ dữ liệu.
- Train trên URFD, test trên OmniFall / OOPS: đo khả năng tổng quát hóa mà bài báo chưa đánh giá.
- Bổ sung FDD để có đủ hai bộ dữ liệu như bài báo.
- Thay nội suy tuyến tính bằng phương pháp xử lý tốt "điểm đổi chiều" (Sec. 4 của bài báo tự nêu hạn chế này).
- So sánh với Transformer / ST-GCN, hoặc estimator mạnh hơn (RTMPose, ViTPose).

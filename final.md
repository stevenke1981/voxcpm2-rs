# final.md — 交付定義與完成檢查

## 交付品

- [ ] Rust workspace 可編譯。
- [ ] CLI 可 smoke synth。
- [ ] GUI 可啟動。
- [ ] Model inspect 可讀 HF 檔案。
- [ ] AudioVAE 轉 safetensors 腳本完成。
- [ ] Tokenizer parity 測試完成。
- [ ] TSLM/RALM/LocDiT/AudioVAE 各子模組 shape test 完成。
- [ ] 真實 TTS 可輸出 48kHz wav。
- [ ] CUDA/Metal 至少一種 GPU backend 通過。
- [ ] 文件：spec/plan/todos/test/final/architecture/conversion/gpu/egui/harness。

## Definition of Done

專案完成必須同時滿足：

1. `cargo test --workspace --features cpu` 成功。
2. `cargo run -p voxcpm2-cli --features cpu -- synth --dry-run ...` 成功。
3. `VOXCPM2_MODEL_DIR=models/VoxCPM2 cargo test --workspace --features cpu -- --ignored` 成功。
4. CUDA 或 Metal build 成功並通過短句生成。
5. GUI 端能完成模型檢查、生成、播放/匯出。
6. 輸出 wav 是 48kHz，無 NaN/Inf，非純噪音。
7. 文件列出的安全限制有在 CLI/GUI 落實。

## 最終驗收指令

```powershell
./harness/run_all.ps1 -WithModel -ModelDir models/VoxCPM2 -Device cuda
```

若使用 Apple Silicon：

```bash
bash harness/run_all.sh --with-model --model-dir models/VoxCPM2 --device metal
```

## 已知技術風險

- VoxCPM2 架構比一般 LLM 複雜，Candle 沒有現成 VoxCPM2 模型類，需要完整 port。
- AudioVAE `.pth` 需經 PyTorch 轉換，Rust 端不建議直接載入 pickle。
- tokenizer 自訂行為是高風險點，需先做 parity。
- 若 GPU BF16 支援不足，需在局部使用 FP16/FP32 fallback。
- 8GB VRAM 是緊繃下限，長文本、clone、GUI 同時載入可能超出。

## 建議 release 形式

```text
voxcpm2-rust-candle-egui-windows-x64.zip
├─ voxcpm2.exe
├─ voxcpm2-gui.exe
├─ configs/
├─ docs/
└─ README.md
```

模型權重另行下載到：

```text
%USERPROFILE%\.cache\voxcpm2\models\VoxCPM2
```

# FAQ — VoxCPM2 Rust/Candle

## 常見問題

**Q: 為什麼 CPU 很慢？**
A: 模型約 4.6GB + 28 層 TSLM + 8 層 RALM + DiT + AudioVAE 上採樣。單次完整品質（30 AR + 30 CFM steps）在 CPU 上可能需數分鐘。建議使用 CUDA 建置獲得合理速度（數十秒）。

**Q: 生成的聲音有雜音怎麼辦？**
A: 請使用推薦參數：
`--seed 99 --cfg 2.5 --steps 30`
所有真實生成都會自動執行完整的 speech polish（DC 移除、高通、低通、adaptive gate、midband smoother、patch boundary smoother 等）。若仍有問題，請確認：
- 模型權重正確轉換（audiovae.safetensors）
- 使用簡體中文輸入（繁體易偏粵語）
- 參考音檔已做過合理清理（harness 會在 clone 前自動 polish）

**Q: Clone 需要什麼條件？**
A: 
- 必須有合法授權或明確同意使用該參考聲音。
- CLI 強制 `--i-have-consent`
- GUI 需勾選同意框
- 輸出預設會標記 AI 生成

**Q: 如何使用 Prompt-only 或 Combined clone？**
A:
- Prompt-only（接續生成）：只提供 `--prompt-audio` + `--prompt-text`
- Combined（最高保真）：同時提供 `--ref-audio` + `--prompt-audio` + `--prompt-text`
- GUI Clone 分頁已有對應欄位。

**Q: CUDA 建置失敗？**
A: 需要：
1. Visual Studio 2022 Build Tools（含 C++ 工具、Windows SDK）
2. 在該 shell 先執行 `vcvars64.bat`
3. CUDA Toolkit + 相容 NVIDIA 驅動
4. 設定 `NVCC_CCBIN`（專案 `.cargo/config.toml` 已自動處理）

**Q: 如何取得最佳品質？**
A: seed 99 + cfg 2.5 + 30 steps 是目前 sweep 中語音能量最高、雜訊最低的組合。亦可使用 `harness/quality_sweep.ps1` 自行掃描。

**Q: 模型下載哪裡來？**
A: `python scripts/download_model.py --repo openbmb/VoxCPM2 --out models/VoxCPM2`
或從 ModelScope / Hugging Face 手動 snapshot。

## 安全與法律提醒
請務必閱讀 `docs/SAFETY.md` 或 zip 內的 SAFETY_AND_QUICKSTART.txt。嚴禁用於冒充、詐騙、造假或任何有害用途。

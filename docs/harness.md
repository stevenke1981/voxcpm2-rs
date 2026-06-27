# harness.md — 自動執行腳本

## Windows PowerShell

`harness/run_all.ps1`：

1. cargo fmt check
2. cargo clippy
3. cargo test
4. CLI dry-run synth
5. 可選：model inspect
6. 可選：GPU probe

```powershell
./harness/run_all.ps1
./harness/run_all.ps1 -WithModel -ModelDir models/VoxCPM2 -Device cuda
```

## Bash

```bash
bash harness/run_all.sh
bash harness/run_all.sh --with-model --model-dir models/VoxCPM2 --device cuda
```

## OpenCode 建議策略

- 先跑無模型 smoke，確保 scaffold 正常。
- 每完成一個模型子模組，新增 shape test。
- 若遇到 shape mismatch，產生 fail report，不要猜。
- 禁止刪除模型權重與 golden files。
- 每次修正後重新跑最近 fail 的最小測試，再跑完整 harness。

# OVMS AI Agent Demo

OpenAI Agents SDK を使い、[OpenVINO Model Server (OVMS)](https://github.com/kazukiminemura/OVMS_baremetal_demo) をバックエンドとして動作するシンプルなAIエージェントのデモです。

## 構成

```
OVMS (localhost:8000)
  └─ tinyllama  (/v3/chat/completions)
       ↑
OpenAI Agents SDK (openai-agents)
  └─ agent_demo.py
       ├─ get_current_time ツール
       ├─ calculate ツール
       └─ get_device_info ツール
```

## 前提条件

OVMSが起動していること:

```bash
ovms --config_path models/config.json --rest_port 8000
```

## インストール

```bash
pip install -r requirements.txt
```

必要に応じて環境変数で接続先を上書きできます。

```bash
set OVMS_BASE_URL=http://localhost:8000/v3
set OVMS_MODEL=tinyllama
```

## 実行

```bash
# デモモード（固定クエリを実行）
python agent_demo.py

# 対話モード
python agent_demo.py chat
```

## 実行例

```
==================================================
  OVMS AI Agent Demo (OpenAI Agents SDK)
==================================================
  Model : tinyllama
  Server: http://localhost:8000/v3
==================================================

User  : What time is it right now?
Agent : The current time is 2026-04-05 14:32:01.

User  : Calculate 123 * 456 + 789
Agent : 123 * 456 + 789 = 56,877.
```

## トラブルシュート

`OVMS model '...' was not found` と表示される場合は、OVMS が公開しているモデルIDと `OVMS_MODEL` が一致していません。
この環境では `http://localhost:8000/v3/models` から `tinyllama` が確認できます。

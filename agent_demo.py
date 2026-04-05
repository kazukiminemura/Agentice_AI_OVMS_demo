"""
OVMS AI Agent Demo
OpenAI Agents SDK + OpenVINO Model Server (OVMS) をバックエンドとして使うシンプルなAIエージェント。

前提条件:
  OVMS が localhost:8000 で稼働していること（TinyLlama-1.1B-Chat-v1.0 を配信）
  セットアップ手順: https://github.com/kazukiminemura/OVMS_baremetal_demo
"""

import asyncio
import json
import os
import re
from datetime import datetime

from agents import Agent, Runner, function_tool
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from openai import NotFoundError

# ── OVMS 接続設定 ────────────────────────────────────────────────────────────
OVMS_BASE_URL = os.getenv("OVMS_BASE_URL", "http://localhost:8000/v3")
MODEL_NAME = os.getenv("OVMS_MODEL", "tinyllama")

ovms_client = AsyncOpenAI(
    base_url=OVMS_BASE_URL,
    api_key="not-required",  # OVMS は認証不要
)

# ── ツール定義 ───────────────────────────────────────────────────────────────

def _get_current_time() -> str:
    """現在の日時を返す。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _calculate(expression: str) -> str:
    """四則演算の式を評価して結果を返す（例: '2 + 3 * 4'）。"""
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return "エラー: 基本的な四則演算のみサポートしています。"
    try:
        result = eval(expression)  # noqa: S307 — デモ用途のみ
        return str(result)
    except Exception as e:
        return f"エラー: {e}"


def _get_device_info() -> str:
    """OVMSが使用しているモデルサーバーの情報を返す。"""
    return (
        f"モデルサーバー: OpenVINO Model Server (OVMS)\n"
        f"エンドポイント: {OVMS_BASE_URL}\n"
        f"モデル: {MODEL_NAME}"
    )


@function_tool
def get_current_time() -> str:
    """現在の日時を返す。"""
    return _get_current_time()


@function_tool
def calculate(expression: str) -> str:
    """四則演算の式を評価して結果を返す（例: '2 + 3 * 4'）。"""
    return _calculate(expression)


@function_tool
def get_device_info() -> str:
    """OVMSが使用しているモデルサーバーの情報を返す。"""
    return _get_device_info()


def resolve_agent_output(output: str) -> str:
    """ツール呼び出しJSONを返すモデル向けに、ローカル実行へフォールバックする。"""
    if not isinstance(output, str):
        return str(output)

    cleaned = output.strip()
    if cleaned.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return output

    if not isinstance(payload, dict):
        return output

    tool_name = payload.get("name")
    parameters = payload.get("parameters", {})
    if not isinstance(parameters, dict):
        parameters = {}

    tool_handlers = {
        "get_current_time": lambda args: _get_current_time(),
        "calculate": lambda args: _calculate(args.get("expression", "")),
        "get_device_info": lambda args: _get_device_info(),
    }

    handler = tool_handlers.get(tool_name)
    if handler is None:
        return output

    return handler(parameters)


# ── エージェント構築 ─────────────────────────────────────────────────────────
agent = Agent(
    name="OVMS Assistant",
    model=OpenAIChatCompletionsModel(
        model=MODEL_NAME,
        openai_client=ovms_client,
    ),
    instructions=(
        "You are a helpful assistant powered by TinyLlama running on "
        "OpenVINO Model Server (OVMS). "
        "Answer concisely. Use tools when the user asks for the current time, "
        "a calculation, or server information."
    ),
    tools=[get_current_time, calculate, get_device_info],
)


# ── メインデモ ───────────────────────────────────────────────────────────────
async def validate_ovms_configuration() -> None:
    """起動前に OVMS 接続設定を確認し、モデル名の不一致を早めに検出する。"""
    available_models = await ovms_client.models.list()
    model_ids = [model.id for model in available_models.data]

    if MODEL_NAME not in model_ids:
        available = ", ".join(model_ids) if model_ids else "(none)"
        raise RuntimeError(
            f"OVMS model '{MODEL_NAME}' was not found at {OVMS_BASE_URL}. "
            f"Available models: {available}. "
            "Set OVMS_MODEL to one of the available model ids."
        )


async def run_demo() -> None:
    await validate_ovms_configuration()
    print("=" * 50)
    print("  OVMS AI Agent Demo (OpenAI Agents SDK)")
    print("=" * 50)
    print(f"  Model : {MODEL_NAME}")
    print(f"  Server: {OVMS_BASE_URL}")
    print("=" * 50 + "\n")

    queries = [
        "What time is it right now?",
        "Calculate 123 * 456 + 789",
        "Tell me about the model server you're running on.",
        "What is OpenVINO?",
    ]

    for query in queries:
        print(f"User  : {query}")
        result = await Runner.run(agent, query)
        print(f"Agent : {resolve_agent_output(result.final_output)}\n")


async def chat_loop() -> None:
    """対話モード: ユーザーが 'quit' と入力するまで会話を続ける。"""
    print("=" * 50)
    print("  OVMS Agent - 対話モード ('quit' で終了)")
    print("=" * 50 + "\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("終了します。")
            break
        if not user_input:
            continue

        result = await Runner.run(agent, user_input)
        print(f"Agent: {resolve_agent_output(result.final_output)}\n")


async def main() -> None:
    import sys

    try:
        # python agent_demo.py       → デモ実行
        # python agent_demo.py chat  → 対話モード
        if len(sys.argv) > 1 and sys.argv[1] == "chat":
            await validate_ovms_configuration()
            await chat_loop()
        else:
            await run_demo()
    except NotFoundError as exc:
        raise SystemExit(
            f"OVMS request failed because model '{MODEL_NAME}' was not found. "
            f"Check OVMS_MODEL and the models exposed by {OVMS_BASE_URL}/models. "
            f"Original error: {exc}"
        ) from exc
    except Exception as exc:
        raise SystemExit(f"Startup failed: {exc}") from exc


if __name__ == "__main__":
    asyncio.run(main())

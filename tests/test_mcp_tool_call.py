"""
Check MCP server tools and invoke CosyVoice3 tools.
"""


import asyncio
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession
from dotenv import load_dotenv
import os
load_dotenv()


def parse_tool_args(args_str_list: list) -> Dict[str, Any]:
    """
    将 ["key=value", "foo=bar"] 转为 {"key": "value", "foo": "bar"}
    支持自动类型解析：str, int, float, bool, None
    """
    result = {}
    if not args_str_list:
        return result

    for item in args_str_list:
        if "=" not in item:
            raise ValueError(f"Invalid tool-arg format: {item}, expected key=value")
        k, v = item.split("=", 1)

        # 尝试类型解析
        try:
            v = json.loads(v.lower() if v.lower() in ("true", "false", "null") else v)
        except json.JSONDecodeError:
            pass  # keep as string

        result[k] = v
    return result


async def async_main(
    server_url: str = "",
    question: str = "",
    tool_name: Optional[str] = None,
    tool_args: Optional[Dict[str, Any]] = None,
):
    async with streamablehttp_client(server_url) as (read_stream, write_stream, get_session_id):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            print('MCP server session已初始化')

            tools = await load_mcp_tools(session)
            tool_dict = {tool.name: tool for tool in tools}

            print("可用工具:", [tool.name for tool in tools])
            for tool in tools:
                print(f"Tool: {tool.name}")
                print(f"Args Schema: {tool.args}")
                print(f"Description: {tool.description}\n")

            # =============================
            # 场景1: 仅列出工具（无 question, 无 tool_name）
            # =============================
            if not question and not tool_name:
                print("未提供问题或工具调用，仅列出工具信息。")
                print("TEST_RESULT: PASSED")
                return

            # =============================
            # 场景2: 直接调用指定工具
            # =============================
            if tool_name:
                if tool_name not in tool_dict:
                    print(f"错误: 工具 '{tool_name}' 未在 MCP 服务中找到！")
                    print("TEST_RESULT: FAILED")
                    return

                if not tool_args:
                    print(f"警告: 调用工具 '{tool_name}' 但未提供参数。")
                    tool_args = {}

                try:
                    print(f"正在调用工具: {tool_name}，参数: {tool_args}")
                    result = await tool_dict[tool_name].ainvoke(tool_args)
                    print("✅ 工具调用成功！返回结果:")
                    print(json.dumps(result, indent=2, ensure_ascii=False) if isinstance(result, (dict, list)) else result)

                    # 可选：结构化解析（如果返回的是 JSON 字符串）
                    if isinstance(result, str):
                        try:
                            parsed = json.loads(result)
                            print("🔍 JSON 解析结果:")
                            print(json.dumps(parsed, indent=2, ensure_ascii=False))
                        except json.JSONDecodeError:
                            pass

                    if isinstance(result, dict) and "audio_path" in result:
                        audio_path = Path(result["audio_path"])
                        if audio_path.exists():
                            print(f"生成文件: {audio_path}")
                        else:
                            print("返回的音频文件路径不存在。")
                            print("TEST_RESULT: FAILED")
                            return

                    print("TEST_RESULT: PASSED")
                except Exception as e:
                    print(f"❌ 工具调用失败: {type(e).__name__}: {e}")
                    print("TEST_RESULT: FAILED")
                return




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MCP Server: list tools, invoke tool, or ask agent")

    parser.add_argument(
        "-u", "--base_url",
        type=str,
        default="http://127.0.0.1:8890/mcp",
        help="MCP server base url"
    )
    parser.add_argument(
        "-q", "--question",
        type=str,
        default="",
        help="问题文本，如果提供，将使用 Agent 回答"
    )
    parser.add_argument(
        "--tool-name",
        type=str,
        default="cosyvoice3_cross_lingual",
        help="要直接调用的工具名称，例如 preview_snapshot_tool"
    )
    parser.add_argument(
        "--tool-arg",
        action="append",
        default=[],
        help="工具参数，格式 key=value，可多次使用。例如: --tool-arg file_url=http://example.com/file.pptx --tool-arg timeout=30"
    )
    parser.add_argument(
        "--prompt-wav-path",
        type=str,
        default="audio_file/Mihono_Bourbon.mp3",
        help="Local prompt audio path.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="这是中文测试句子，用来检查音色和韵律。",
        help="Text to synthesize.",
    )

    args = parser.parse_args()

    # 解析 tool-arg
    tool_args = parse_tool_args(args.tool_arg) if args.tool_arg else None
    if args.tool_name and not tool_args:
        prompt_path = str(Path(args.prompt_wav_path).resolve())
        if args.tool_name == "cosyvoice3_cross_lingual":
            tool_args = {
                "text": args.text,
                "prompt_wav_path": prompt_path,
            }
        elif args.tool_name == "cosyvoice3_zero_shot":
            tool_args = {
                "text": args.text,
                "prompt_text": "You are a helpful assistant.<|endofprompt|>",
                "prompt_wav_path": prompt_path,
            }
        elif args.tool_name == "cosyvoice3_instruct":
            tool_args = {
                "text": args.text,
                "instruct_text": "You are a helpful assistant. 请用广东话表达。<|endofprompt|>",
                "prompt_wav_path": prompt_path,
            }

    # 运行主函数
    asyncio.run(async_main(
        server_url=args.base_url,
        question=args.question,
        tool_name=args.tool_name,
        tool_args=tool_args
    ))

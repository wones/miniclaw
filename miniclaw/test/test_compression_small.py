#!/usr/bin/env python3
"""
记忆压缩参数调整测试脚本
使用更小的参数值来更容易触发记忆压缩
"""

import asyncio
from pathlib import Path
from miniclaw.session.manager import SessionManager
from miniclaw.agent.memory import MemoryStore, Consolidator
from miniclaw.providers.openai_compat_provider import OpenAICompatProvider
from miniclaw.config.loader import load_config

async def test_memory_compression_with_smaller_params():
    """使用更小的参数测试记忆压缩功能"""
    print("=== 记忆压缩参数调整测试 ===")
    
    # 1. 加载配置
    config = load_config()
    api_key = config.providers.openai.get("apikey")
    if not api_key:
        print("❌ OpenAI API key is required")
        return
    
    # 2. 初始化组件
    workspace = Path.home() / ".miniclaw"
    workspace.mkdir(parents=True, exist_ok=True)
    
    provider = OpenAICompatProvider(
        api_key,
        config.agents.default.base_url,
        config.agents.default.model
    )
    
    session_manager = SessionManager()
    memory_store = MemoryStore(workspace)
    
    # 3. 使用更小的参数创建 Consolidator
    # 原始参数: context_window_tokens=128000, max_completion_tokens=4096
    # 调整后参数: 更小的值，更容易触发压缩
    consolidator = Consolidator(
        store=memory_store,
        provider=provider,
        model=config.agents.default.model,
        sessions=session_manager,
        context_window_tokens=1000,  # 更小的值
        max_completion_tokens=200     # 更小的值
    )
    
    # 4. 创建测试会话
    session = session_manager.get_or_create("test_small_params")
    print(f"✅ 创建测试会话: {session.key}")
    
    # 5. 添加测试消息
    print("📝 添加测试消息...")
    test_messages = [
        "你好，我是测试用户，这是第一条测试消息。",
        "这是第二条测试消息，内容比较长，用来增加对话长度。",
        "这是第三条测试消息，内容更加丰富，包含很多信息。",
        "这是第四条测试消息，继续增加对话的长度。",
        "这是第五条测试消息，测试系统的记忆管理能力。",
    ]
    
    for i, message in enumerate(test_messages):
        session.add_message("user", message)
        session.add_message("assistant", f"这是对第{i+1}条消息的回复。")
        print(f"   添加消息 {i+1}/5")
    
    print(f"✅ 添加完成，当前消息数: {len(session.messages)}")
    
    # 6. 检查是否需要压缩
    try:
        estimated, source = consolidator.estimate_session_prompt_tokens(session)
        budget = consolidator.context_window_tokens - consolidator.max_completion_tokens - consolidator._SAFETY_BUFFER
        target = budget // 2
        
        print(f"\n📊 压缩参数:")
        print(f"   估算 token 数: {estimated}")
        print(f"   预算 token 数: {budget}")
        print(f"   目标 token 数: {target}")
        print(f"   是否需要压缩: {'是' if estimated > budget else '否'}")
        
    except Exception as e:
        print(f"❌ 估算失败: {e}")
        return
    
    # 7. 手动触发压缩
    print("\n🔄 手动触发记忆压缩...")
    try:
        await consolidator.maybe_consolidate_by_tokens(session)
        print("✅ 压缩完成")
    except Exception as e:
        print(f"❌ 压缩失败: {e}")
        return
    
    # 8. 验证压缩结果
    print("\n📋 验证压缩结果:")
    
    # 检查会话消息数
    print(f"   压缩后消息数: {len(session.messages)}")
    if session.messages:
        print(f"   最后一条消息: {session.messages[-1]['content'][:100]}...")
    
    # 检查 history.jsonl
    history_file = memory_store.history_file
    if history_file.exists():
        lines = history_file.read_text(encoding="utf-8").split('\n')
        print(f"   history.jsonl 行数: {len([l for l in lines if l.strip()])}")
        
        # 查找摘要内容
        for line in reversed(lines):
            if line.strip() and ("Summary" in line or "摘要" in line):
                print(f"   找到摘要: {line[:150]}...")
                break
    else:
        print("   ❌ history.jsonl 不存在")
    
    # 9. 测试压缩后效果
    print("\n🧠 测试压缩后对话:")
    test_question = "我刚才发送了多少条测试消息？"
    print(f"   提问: {test_question}")
    
    # 构建上下文
    from miniclaw.agent.context import ContextBuilder
    context_builder = ContextBuilder(workspace, timezone=None)
    context_builder.set_memory_store(memory_store)
    
    context = context_builder.build_messages(
        history=session.get_history(max_messages=50),
        current_message=test_question,
        channel="cli",
        chat_id="test_small_params"
    )
    
    # 生成回答
    try:
        response = provider.generate(context)
        print(f"   回答: {response}")
    except Exception as e:
        print(f"   ❌ 生成回答失败: {e}")
    
    print("\n=== 测试完成 ===")
    print("使用更小的参数值，系统应该更容易触发记忆压缩。")
    print("这有助于测试压缩功能是否正常工作。")

if __name__ == "__main__":
    asyncio.run(test_memory_compression_with_smaller_params())
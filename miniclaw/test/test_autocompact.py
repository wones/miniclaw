#!/usr/bin/env python3
"""
AutoCompact 空闲会话压缩测试脚本
测试空闲会话的自动压缩功能
"""

import asyncio
from pathlib import Path
from miniclaw.session.manager import SessionManager
from miniclaw.agent.memory import MemoryStore, Consolidator
from miniclaw.agent.autocompact import AutoCompact
from miniclaw.providers.openai_compat_provider import OpenAICompatProvider
from miniclaw.config.loader import load_config
from datetime import datetime, timedelta

async def test_autocompact():
    """测试 AutoCompact 空闲会话压缩功能"""
    print("=== AutoCompact 空闲会话压缩测试 ===")
    
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
    
    consolidator = Consolidator(
        store=memory_store,
        provider=provider,
        model=config.agents.default.model,
        sessions=session_manager,
        context_window_tokens=128000,
        max_completion_tokens=4096
    )
    
    # 3. 创建 AutoCompact 实例，使用更小的 TTL
    # 原始值: session_ttl_minutes=30
    # 调整后: session_ttl_minutes=1 (1分钟)
    autocompact = AutoCompact(
        sessions=session_manager,
        consolidator=consolidator,
        session_ttl_minutes=1  # 调小为1分钟，方便测试
    )
    print("✅ AutoCompact 实例创建完成")
    
    # 4. 创建测试会话
    test_session_key = "test_autocompact"
    session = session_manager.get_or_create(test_session_key)
    print(f"✅ 创建测试会话: {test_session_key}")
    
    # 5. 添加测试消息
    print("\n📝 添加测试消息...")
    test_messages = [
        "你好，我是测试用户，这是第一条消息。",
        "这是第二条消息，内容比较长，用来测试会话压缩。",
        "这是第三条消息，继续增加会话长度。",
        "这是第四条消息，测试系统的记忆管理能力。",
        "这是第五条消息，接近压缩的临界点。",
        "这是第六条消息，应该会触发压缩。",
        "这是第七条消息，用来确保会话足够长。",
        "这是第八条消息，最后一条测试消息。",
    ]
    
    for i, message in enumerate(test_messages):
        session.add_message("user", message)
        session.add_message("assistant", f"这是对第{i+1}条消息的回复。")
        print(f"   添加消息 {i+1}/8")
    
    print(f"✅ 添加完成，当前消息数: {len(session.messages)}")
    print(f"   会话创建时间: {session.created_at}")
    print(f"   会话最后更新时间: {session.updated_at}")
    
    # 6. 模拟会话空闲
    print("\n⏳ 模拟会话空闲...")
    # 手动修改会话的更新时间，模拟空闲
    session.updated_at = datetime.now() - timedelta(minutes=2)  # 2分钟前
    print(f"   修改后会话最后更新时间: {session.updated_at}")
    print(f"   当前时间: {datetime.now()}")
    print(f"   会话已空闲: {datetime.now() - session.updated_at}")
    
    # 7. 检查活跃会话
    active_sessions = set()  # 空集合，表示没有活跃会话
    print(f"\n📋 活跃会话: {active_sessions}")
    
    # 8. 手动触发 AutoCompact
    print("\n🔄 手动触发 AutoCompact...")
    try:
        autocompact.check_expired(
            schedule_background=asyncio.create_task,
            active_session_keys=active_sessions
        )
        print("✅ AutoCompact 触发完成")
    except Exception as e:
        print(f"❌ AutoCompact 触发失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 9. 等待压缩完成
    print("\n⏳ 等待压缩完成...")
    await asyncio.sleep(3)  # 给压缩过程一些时间
    
    # 10. 验证压缩结果
    print("\n📋 验证压缩结果:")
    
    # 检查会话消息数
    print(f"   压缩后消息数: {len(session.messages)}")
    if session.messages:
        print(f"   最后一条消息: {session.messages[-1]['content'][:100]}...")
    
    # 检查 history.jsonl
    history_file = memory_store.history_file
    if history_file.exists():
        entries = memory_store._read_entries()
        print(f"   history.jsonl 条目数: {len(entries)}")
        
        # 查找摘要内容
        for entry in reversed(entries):
            content = entry.get("content", "")
            if "Summary" in content or "摘要" in content:
                print(f"   找到摘要: {content[:150]}...")
                break
    else:
        print("   ❌ history.jsonl 不存在")
    
    # 11. 测试压缩后效果
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
        chat_id=test_session_key
    )
    
    # 生成回答
    try:
        response = provider.generate(context)
        print(f"   回答: {response}")
    except Exception as e:
        print(f"   ❌ 生成回答失败: {e}")
    
    print("\n=== 测试完成 ===")
    print("使用更小的 TTL 参数，系统应该更容易触发空闲会话压缩。")
    print("这有助于测试 AutoCompact 功能是否正常工作。")
    print("系统应该已经:")
    print("1. 识别空闲会话")
    print("2. 压缩旧消息为摘要")
    print("3. 保留最近的消息")
    print("4. 保存摘要到 history.jsonl")
    print("5. 能够基于压缩后的记忆回答问题")

if __name__ == "__main__":
    asyncio.run(test_autocompact())
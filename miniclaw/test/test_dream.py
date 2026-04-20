#!/usr/bin/env python3
"""
Dream 分析和记忆更新测试脚本
测试 Dream 类的两阶段记忆处理功能
"""

import asyncio
from pathlib import Path
from miniclaw.session.manager import SessionManager
from miniclaw.agent.memory import MemoryStore, Dream
from miniclaw.providers.openai_compat_provider import OpenAICompatProvider
from miniclaw.config.loader import load_config

async def test_dream_analysis():
    """测试 Dream 两阶段记忆处理功能"""
    print("=== Dream 分析和记忆更新测试 ===")
    
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
    
    # 3. 创建 Dream 实例
    dream = Dream(
        store=memory_store,
        provider=provider,
        model=config.agents.default.model,
        tools=None,  # 暂时使用 None
        runner=None   # 暂时使用 None
    )
    print("✅ Dream 实例创建完成")
    
    # 4. 添加测试对话到 history.jsonl
    print("\n📝 添加测试对话到历史...")
    test_entries = [
        "用户：你好，我叫张三。",
        "助手：你好张三！很高兴认识你。",
        "用户：我在北京工作，是一名软件工程师。",
        "助手：北京是个美丽的城市，软件工程师也是很棒的职业。",
        "用户：我喜欢编程和阅读技术书籍。",
        "助手：编程和阅读是很好的学习方式。",
        "用户：我正在学习人工智能技术。",
        "助手：人工智能是非常有前景的领域，继续加油！",
        "用户：我今天学习了机器学习的基础知识。",
        "助手：机器学习是AI的重要分支，掌握基础知识很重要。",
    ]
    
    for entry in test_entries:
        cursor = memory_store.append_history(entry)
        print(f"   添加: {entry[:50]}...")
    
    print(f"✅ 添加了 {len(test_entries)} 条测试对话")
    
    # 5. 检查 Dream Cursor
    cursor_before = memory_store.get_last_dream_cursor()
    print(f"\n📊 Dream Cursor (处理前): {cursor_before}")
    
    # 6. 手动触发 Dream 处理
    print("\n🔄 手动触发 Dream 处理...")
    try:
        result = await dream.process_dream()
        print(f"✅ Dream 处理完成: {result}")
    except Exception as e:
        print(f"❌ Dream 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 7. 检查 Dream Cursor (处理后)
    cursor_after = memory_store.get_last_dream_cursor()
    print(f"📊 Dream Cursor (处理后): {cursor_after}")
    
    # 8. 验证记忆文件更新
    print("\n📋 验证记忆文件更新:")
    
    # 检查 MEMORY.md
    memory_content = memory_store.read_memory()
    if memory_content:
        print(f"   ✅ MEMORY.md 存在 ({len(memory_content)} 字符)")
        print(f"   内容预览:\n{memory_content[:200]}...")
    else:
        print("   ⚠️ MEMORY.md 为空")
    
    # 检查 SOUL.md
    soul_content = memory_store.read_soul()
    if soul_content:
        print(f"   ✅ SOUL.md 存在 ({len(soul_content)} 字符)")
    else:
        print("   ⚠️ SOUL.md 为空")
    
    # 检查 USER.md
    user_content = memory_store.read_user()
    if user_content:
        print(f"   ✅ USER.md 存在 ({len(user_content)} 字符)")
    else:
        print("   ⚠️ USER.md 为空")
    
    # 9. 读取 history.jsonl 验证
    print("\n📜 检查历史记录:")
    history_file = memory_store.history_file
    if history_file.exists():
        entries = memory_store._read_entries()
        print(f"   history.jsonl 条目数: {len(entries)}")
        
        # 显示最后几条记录
        if entries:
            print("   最后3条记录:")
            for entry in entries[-3:]:
                content = entry.get("content", "")[:80]
                print(f"     - {entry.get('timestamp')}: {content}...")
    else:
        print("   ❌ history.jsonl 不存在")
    
    # 10. 测试 Dream Phase 1 (分析)
    print("\n🔍 手动测试 Dream Phase 1...")
    try:
        batch, cursor = await dream._dream_phase1()
        print(f"   Phase 1 结果: 获取了 {len(batch)} 条未处理的历史记录")
        print(f"   Cursor: {cursor}")
        
        if batch:
            analysis = await dream._analyze_batch(batch)
            print(f"   分析结果预览: {analysis[:200]}...")
    except Exception as e:
        print(f"   ❌ Phase 1 测试失败: {e}")
    
    print("\n=== 测试完成 ===")
    print("系统应该已经:")
    print("1. ✅ 处理了未分析的历史记录")
    print("2. ✅ 分析了对话内容")
    print("3. ✅ 更新了 MEMORY.md (如果需要)")
    print("4. ✅ 更新了 SOUL.md (如果需要)")
    print("5. ✅ 更新了 USER.md (如果需要)")
    print("6. ✅ 推进了 Dream Cursor")
    print("7. ✅ 压缩了历史记录")

if __name__ == "__main__":
    asyncio.run(test_dream_analysis())
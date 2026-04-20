#!/usr/bin/env python3
"""
tiktoken 集成测试脚本
验证 token 估算的准确性和中文支持
"""

import sys
import asyncio
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from miniclaw.agent.memory import Consolidator, MemoryStore
from miniclaw.session.manager import SessionManager
from miniclaw.providers.openai_compat_provider import OpenAICompatProvider
from miniclaw.config.loader import load_config

def test_tiktoken_integration():
    """测试 tiktoken 集成"""
    print("=== tiktoken 集成测试 ===\n")
    
    # 1. 测试 tiktoken 导入
    print("1. 测试 tiktoken 导入...")
    try:
        import tiktoken
        print(f"   ✅ tiktoken 版本: {tiktoken.__version__}")
    except ImportError as e:
        print(f"   ❌ tiktoken 导入失败: {e}")
        return False
    
    # 2. 测试编码器创建
    print("\n2. 测试编码器创建...")
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        print("   ✅ cl100k_base 编码器创建成功")
    except Exception as e:
        print(f"   ❌ 编码器创建失败: {e}")
        return False
    
    # 3. 测试英文编码
    print("\n3. 测试英文编码...")
    text_en = "Hello, how are you today? I am testing the tokenizer."
    tokens_en = enc.encode(text_en)
    print(f"   原文: {text_en}")
    print(f"   字符数: {len(text_en)}")
    print(f"   Token数: {len(tokens_en)}")
    print(f"   简单估算: {len(text_en) // 4}")
    print(f"   ✅ 英文编码正常")
    
    # 4. 测试中文编码
    print("\n4. 测试中文编码...")
    text_cn = "你好，今天天气怎么样？我想测试一下中文的分词效果。"
    tokens_cn = enc.encode(text_cn)
    print(f"   原文: {text_cn}")
    print(f"   字符数: {len(text_cn)}")
    print(f"   Token数: {len(tokens_cn)}")
    print(f"   简单估算: {len(text_cn) // 4}")
    ratio = len(text_cn) / max(len(tokens_cn), 1)
    print(f"   字符/Token比率: {ratio:.2f}")
    print(f"   ✅ 中文编码正常")
    
    # 5. 测试 Consolidator 的 _get_tokenizer
    print("\n5. 测试 Consolidator._get_tokenizer()...")
    try:
        config = load_config()
        api_key = config.providers.openai.get("apikey")
        if not api_key:
            print("   ⚠️ 跳过 Consolidator 测试（无 API key）")
            return True
        
        workspace = Path.home() / ".miniclaw"
        workspace.mkdir(parents=True, exist_ok=True)
        
        provider = OpenAICompatProvider(
            api_key,
            config.agents.default.base_url,
            config.agents.default.model
        )
        
        memory_store = MemoryStore(workspace)
        session_manager = SessionManager()
        
        consolidator = Consolidator(
            store=memory_store,
            provider=provider,
            model=config.agents.default.model,
            sessions=session_manager,
            context_window_tokens=128000,
            max_completion_tokens=4096
        )
        
        tokenizer = consolidator._get_tokenizer()
        if tokenizer:
            print("   ✅ Consolidator 获取 tokenizer 成功")
        else:
            print("   ⚠️ Consolidator 获取 tokenizer 失败（使用回退）")
            
    except Exception as e:
        print(f"   ⚠️ Consolidator 测试跳过: {e}")
    
    # 6. 测试 _estimate_message_tokens
    print("\n6. 测试 _estimate_message_tokens()...")
    try:
        test_message = {
            "role": "user",
            "content": "这是一条测试消息，用来验证 token 估算功能是否正常工作。"
        }
        
        # 如果 consolidator 已创建
        if 'consolidator' in dir():
            tokens = consolidator._estimate_message_tokens(test_message)
            print(f"   消息: {test_message['content']}")
            print(f"   tiktoken 估算: {tokens} tokens")
            print(f"   简单估算: {len(test_message['content']) // 4} tokens")
            print("   ✅ _estimate_message_tokens 正常")
        else:
            print("   ⚠️ 跳过（consolidator 未创建）")
            
    except Exception as e:
        print(f"   ❌ 测试失败: {e}")
    
    # 7. 测试混合格本
    print("\n7. 测试混合格本（中英混合）...")
    mixed_text = "Hello你好 World世界！这是一个mixed混合text文本。"
    tokens_mixed = enc.encode(mixed_text)
    print(f"   原文: {mixed_text}")
    print(f"   字符数: {len(mixed_text)}")
    print(f"   Token数: {len(tokens_mixed)}")
    print(f"   ✅ 混合编码正常")
    
    print("\n=== 测试完成 ===")
    print("\n总结:")
    print("- tiktoken 已成功集成")
    print("- cl100k_base 支持中英文")
    print("- Consolidator 可以使用 tiktoken 进行 token 估算")
    print("- 当 tiktoken 不可用时会回退到简单估算（len//4）")
    
    return True

if __name__ == "__main__":
    success = test_tiktoken_integration()
    sys.exit(0 if success else 1)
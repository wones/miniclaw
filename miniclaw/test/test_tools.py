#!/usr/bin/env python3
"""
Test script for miniclaw tool system.

This script tests the functionality of the tool system, including:
- Tool registration
- Tool execution
- Memory operations
- File operations
"""

import asyncio
from pathlib import Path
from miniclaw.agent.tools import setup_tools
from miniclaw.agent.memory import MemoryStore

async def test_tool_system():
    """Test the tool system functionality."""
    print("Testing miniclaw tool system...")
    
    # Create a temporary workspace
    workspace = Path("test_workspace")
    workspace.mkdir(exist_ok=True)
    
    try:
        # Initialize memory store
        memory_store = MemoryStore(workspace)
        
        # Setup tools (no longer needs permission_manager)
        tool_registry = setup_tools(memory_store, workspace)
        
        # Test 1: List available tools
        print("\n1. Testing tool registration:")
        tools = tool_registry.tool_names
        print(f"Registered tools: {tools}")
        
        # Test 2: Test memory tool - update memory
        print("\n2. Testing memory tool - update memory:")
        memory_content = "This is a test memory entry."
        result = await tool_registry.execute(
            "memory",
            {
                "action": "update",
                "file": "memory",
                "content": memory_content,
                "mode": "overwrite"
            }
        )
        print(f"Update memory result: {result}")
        assert isinstance(result, str), "Result should be a string"
        
        # Test 3: Test read memory tool
        print("\n3. Testing read memory tool:")
        result = await tool_registry.execute(
            "read_memory",
            {"file": "memory"}
        )
        print(f"Read memory result: {result}")
        assert isinstance(result, str), "Result should be a string"
        assert memory_content in result, "Memory content not found"
        
        # Test 4: Test write file tool
        print("\n4. Testing write file tool:")
        test_file_content = "This is a test file content."
        result = await tool_registry.execute(
            "write_file",
            {
                "path": "test_file.txt",
                "content": test_file_content
            }
        )
        print(f"Write file result: {result}")
        assert isinstance(result, str), "Result should be a string"
        
        # Test 5: Test read file tool
        print("\n5. Testing read file tool:")
        result = await tool_registry.execute(
            "read_file",
            {"path": "test_file.txt"}
        )
        print(f"Read file result: {result}")
        assert isinstance(result, str), "Result should be a string"
        assert test_file_content in result, "File content not found"
        
        # Test 6: Test memory tool - compact
        print("\n6. Testing memory tool - compact:")
        result = await tool_registry.execute(
            "memory",
            {"action": "compact"}
        )
        print(f"Compact result: {result}")
        assert isinstance(result, str), "Result should be a string"
        
        # Test 7: Test memory tool - archive
        print("\n7. Testing memory tool - archive:")
        test_messages = [
            {"role": "user", "content": "Test message 1"},
            {"role": "assistant", "content": "Test response 1"}
        ]
        result = await tool_registry.execute(
            "memory",
            {
                "action": "archive",
                "messages": test_messages
            }
        )
        print(f"Archive result: {result}")
        assert isinstance(result, str), "Result should be a string"
        
        # Test 8: Test memory tool - clear
        print("\n8. Testing memory tool - clear:")
        result = await tool_registry.execute(
            "memory",
            {"action": "clear"}
        )
        print(f"Clear result: {result}")
        assert isinstance(result, str), "Result should be a string"
        
        # Test 9: Test error handling (invalid action returns error string)
        print("\n9. Testing error handling:")
        result = await tool_registry.execute(
            "memory",
            {"action": "invalid_action"}
        )
        print(f"Invalid action result: {result}")
        assert isinstance(result, str), "Result should be a string"
        assert "Error" in result or "error" in result or "Unknown" in result, \
            "Should return error message for invalid action"
        
        # Test 10: Test tool schema definitions
        print("\n10. Testing tool schema definitions:")
        definitions = tool_registry.get_definitions()
        for defn in definitions:
            assert "type" in defn, "Definition should have type"
            assert "function" in defn, "Definition should have function"
            func = defn["function"]
            assert "name" in func, "Function should have name"
            assert "description" in func, "Function should have description"
            assert "parameters" in func, "Function should have parameters"
            print(f"  - {func['name']}: {func['description'][:60]}...")
        
        # Test 11: Test prepare_call validation
        print("\n11. Testing prepare_call validation:")
        tool, params, error = tool_registry.prepare_call("nonexistent", {})
        assert tool is None, "Should return None for nonexistent tool"
        assert error is not None, "Should return error for nonexistent tool"
        print(f"  Nonexistent tool error: {error}")
        
        print("\nAll tests passed!")
        
    finally:
        # Clean up
        import shutil
        if workspace.exists():
            shutil.rmtree(workspace)

if __name__ == "__main__":
    asyncio.run(test_tool_system())

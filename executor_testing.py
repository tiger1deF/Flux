import asyncio
from pathlib import Path
import tempfile

from tools.executor import CodeExecutor, ExecutorType, ExecutionResult
from agents.messages.file import File, FileType

from deployment.config import initialize_environment
initialize_environment()

# Test code samples
PYTHON_CODE = """
def add(a, b):
    return a + b

result = add(2, 3)
print(f"Result: {result}")
"""

JAVASCRIPT_CODE = """
function add(a, b) {
    return a + b;
}

const result = add(2, 3);
console.log(`Result: ${result}`);
"""

GO_CODE = """
package main

import "fmt"

func main() {
    fmt.Printf("Result: %d\n", add(2, 3))
}

func add(a, b int) int {
    return a + b
}
"""


async def test_local():
    """Test local Python execution"""
    print("\nTesting Local Python Executor...")
    
    workspace = Path(tempfile.mkdtemp())
    executor = CodeExecutor(
        executor_type=ExecutorType.LOCAL,
        workspace=workspace
    )
    
    file = await File.create(
        content=PYTHON_CODE,
        path=str(workspace / "test.py"),
        type=FileType.PYTHON
    )
    
    result = await executor.execute(file)
    print(f"Output: {result.output}")
    print(f"Error: {result.error}")
    print(f"Exit code: {result.exit_code}")
    print(f"Execution time: {result.execution_time}s")


async def test_docker():
    """Test Docker execution"""
    print("\nTesting Docker Executor...")
    
    workspace = Path(tempfile.mkdtemp())
    executor = CodeExecutor(
        executor_type = ExecutorType.DOCKER,
        workspace = workspace
    )
    
    # Test Python
    py_file = await File.create(
        content = PYTHON_CODE,
        path = str(workspace / "test.py"),
        type = FileType.PYTHON
    )
    
    result = await executor.execute(py_file)
    print("\nPython Result:")
    print(f"Output: {result.output}")
    print(f"Error: {result.error}")
    print(f"Exit code: {result.exit_code}")
    
    # Test JavaScript
    js_file = await File.create(
        content=JAVASCRIPT_CODE,
        path=str(workspace / "test.js"),
        type=FileType.JAVASCRIPT
    )
    
    result = await executor.execute(js_file)
    print("\nJavaScript Result:")
    print(f"Output: {result.output}")
    print(f"Error: {result.error}")
    print(f"Exit code: {result.exit_code}")


async def test_piston():
    """Test Piston execution"""
    print("\nTesting Piston Executor...")
    
    config = {
        "piston_url": "https://your-piston-instance.com"  # Replace with actual URL
    }
    
    executor = CodeExecutor(
        executor_type=ExecutorType.PISTON,
        config=config
    )
    
    file = await File.create(
        content=PYTHON_CODE,
        type=FileType.PYTHON
    )
    
    try:
        result = await executor.execute(file)
        print(f"Output: {result.output}")
        print(f"Error: {result.error}")
        print(f"Exit code: {result.exit_code}")
        print(f"Memory usage: {result.memory_usage}")
    except Exception as e:
        print(f"Piston test failed: {str(e)}")


async def test_code_server():
    """Test Code-Server execution"""
    print("\nTesting Code-Server...")
    
    config = {
        "code_server_url": "http://localhost:8080"  # Replace with actual URL
    }
    
    workspace = Path(tempfile.mkdtemp())
    executor = CodeExecutor(
        executor_type=ExecutorType.CODE_SERVER,
        config=config,
        workspace=workspace
    )
    
    file = await File.create(
        content=PYTHON_CODE,
        path=str(workspace / "test.py"),
        type=FileType.PYTHON
    )
    
    try:
        result = await executor.execute(file)
        print(f"Output: {result.output}")
        print(f"Error: {result.error}")
        print(f"Exit code: {result.exit_code}")
    except Exception as e:
        print(f"Code-Server test failed: {str(e)}")


async def test_sphere():
    """Test Sphere Engine execution"""
    print("\nTesting Sphere Engine...")
    
    config = {
        "sphere_api_token": "your-sphere-token"  # Replace with actual token
    }
    
    executor = CodeExecutor(
        executor_type=ExecutorType.SPHERE,
        config=config
    )
    
    file = await File.create(
        content=PYTHON_CODE,
        type=FileType.PYTHON
    )
    
    try:
        result = await executor.execute(file)
        print(f"Output: {result.output}")
        print(f"Error: {result.error}")
        print(f"Exit code: {result.exit_code}")
        print(f"Memory usage: {result.memory_usage}")
    except Exception as e:
        print(f"Sphere test failed: {str(e)}")


async def test_cloud_run():
    """Test Google Cloud Run execution"""
    print("\nTesting Cloud Run...")
    
    config = {
        "project_id": "your-project-id",  # Replace with actual GCP project ID
    }
    
    workspace = Path(tempfile.mkdtemp())
    executor = CodeExecutor(
        executor_type=ExecutorType.CLOUD_RUN,
        config=config,
        workspace=workspace
    )
    
    file = await File.create(
        content=PYTHON_CODE,
        path=str(workspace / "test.py"),
        type=FileType.PYTHON
    )
    
    try:
        result = await executor.execute(file)
        print(f"Output: {result.output}")
        print(f"Error: {result.error}")
        print(f"Exit code: {result.exit_code}")
        print(f"Memory usage: {result.memory_usage}")
    except Exception as e:
        print(f"Cloud Run test failed: {str(e)}")


async def main():
    """Run individual executor tests"""
    print("Starting Executor Tests...")
    
    print("\n=== Local Python Test ===")
    await test_local()
    
    import sys
    sys.exit()
    print("\n=== Docker Test ===")
    await test_docker()
    
    print("\n=== Piston Test ===")
    await test_piston()
    
    print("\n=== Code-Server Test ===")
    await test_code_server()
    
    print("\n=== Sphere Engine Test ===")
    await test_sphere()
    
    print("\n=== Cloud Run Test ===")
    await test_cloud_run()


if __name__ == "__main__":
    asyncio.run(main())

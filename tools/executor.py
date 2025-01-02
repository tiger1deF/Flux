from enum import Enum
from typing import Dict, Any, Optional, List, Union
import asyncio
from pathlib import Path
import tempfile
import aiohttp
import time
import logging
from pydantic import BaseModel

from agents.storage.file import File, FileType

# TODO - Centralize, remove elsewhere
logger = logging.getLogger(__name__)


class ExecutorType(Enum):
    """Supported code execution backends"""
    DOCKER = "docker"       # Free, automatic Docker setup
    CODE_SERVER = "code"    # Free, self-hosted
    SPHERE = "sphere"       # Paid, cloud API
    CLOUD_RUN = "cloud"     # Paid, Google Cloud
    PISTON = "piston"      # Free, self-hosted Piston engine
    LOCAL = "local"         # Python-only, local context


class ExecutionResult(BaseModel):
    """Standardized execution result across backends"""
    output: str
    error: Optional[str] = None
    exit_code: int
    language: str
    execution_time: float
    memory_usage: Optional[int] = None
    dependencies: List[str] = []


class CodeExecutor:
    """Universal code executor supporting multiple backends"""
    
    # Supported language mappings per executor
    PISTON_LANGUAGES = {
        FileType.PYTHON: "python",
        FileType.JAVASCRIPT: "javascript", 
        FileType.TYPESCRIPT: "typescript",
        FileType.JAVA: "java",
        FileType.CPP: "cpp",
        FileType.CSHARP: "csharp",
        FileType.GO: "go",
        FileType.RUST: "rust",
        FileType.PHP: "php",
        FileType.RUBY: "ruby",
        FileType.SWIFT: "swift",
        FileType.KOTLIN: "kotlin",
        FileType.SCALA: "scala",
        FileType.R: "r"
    }

    DOCKER_LANGUAGES = {
        FileType.PYTHON: ("python:3.9-slim", "python"),
        FileType.JAVASCRIPT: ("node:16-slim", "node"),
        FileType.TYPESCRIPT: ("node:16-slim", "ts-node"),
        FileType.JAVA: ("openjdk:11-slim", "java"),
        FileType.GO: ("golang:1.17-slim", "go run"),
        FileType.RUST: ("rust:1.56-slim", "rustc"),
        FileType.PHP: ("php:8.0-cli", "php"),
        FileType.RUBY: ("ruby:3.0-slim", "ruby"),
        FileType.SWIFT: ("swift:5.5", "swift"),
        FileType.KOTLIN: ("zenika/kotlin", "kotlinc"),
        FileType.SCALA: ("hseeberger/scala-sbt:11.0.12-1.5.5", "scala")
    }

    CODE_SERVER_LANGUAGES = {
        FileType.PYTHON, FileType.JAVASCRIPT, FileType.TYPESCRIPT,
        FileType.JAVA, FileType.CPP, FileType.GO, FileType.RUST,
        FileType.PHP, FileType.RUBY, FileType.SWIFT, FileType.KOTLIN,
        FileType.SCALA, FileType.R
    }
    
    def __init__(
        self,
        executor_type: ExecutorType = ExecutorType.DOCKER,
        config: Dict[str, Any] = None,
        workspace: Optional[Union[File, Path]] = None
    ):
        """
        Initialize code executor
        
        :param executor_type: Type of execution backend to use
        :param config: Configuration for the executor
        :param workspace: File object or Path to use as workspace
        """
        self.executor_type = executor_type
        self.config = config or {}
        self.workspace = workspace
        self._execution_lock = asyncio.Lock()

    def _get_workspace_dir(self) -> Path:
        """Get workspace directory from File or Path"""
        if isinstance(self.workspace, File):
            return Path(self.workspace.path).parent if self.workspace.path else Path(tempfile.mkdtemp())
        elif isinstance(self.workspace, Path):
            return self.workspace
        else:
            return Path(tempfile.mkdtemp())

    async def execute(
        self,
        file: File,
        execution_path: Optional[str] = None,
        run_command: Optional[str] = None,
        dependencies: List[str] = None
    ) -> ExecutionResult:
        """Execute code using the configured backend"""
        
        workspace_dir = self._get_workspace_dir()

        async with self._execution_lock:
            if self.executor_type == ExecutorType.PISTON:
                if file.type not in self.PISTON_LANGUAGES:
                    raise ValueError(f"Piston does not support {file.type}")
                return await self._piston_execute(
                    file, execution_path, run_command, dependencies
                )
            elif self.executor_type == ExecutorType.DOCKER:
                if file.type not in self.DOCKER_LANGUAGES:
                    raise ValueError(f"Docker executor does not support {file.type}")
                return await self._docker_execute(
                    file, workspace_dir, execution_path, run_command, dependencies
                )
            elif self.executor_type == ExecutorType.CODE_SERVER:
                if file.type not in self.CODE_SERVER_LANGUAGES:
                    raise ValueError(f"Code-Server does not support {file.type}")
                return await self._code_server_execute(
                    file, workspace_dir, execution_path, run_command, dependencies
                )
            elif self.executor_type == ExecutorType.LOCAL:
                if file.type != FileType.PYTHON:
                    raise ValueError("Local executor only supports Python")
                return await self._local_execute(
                    file, workspace_dir, execution_path, run_command
                )
            
            raise ValueError(f"Unsupported executor type: {self.executor_type}")

    async def _piston_execute(
        self,
        file: File,
        execution_path: Optional[str] = None,
        run_command: Optional[str] = None,
        dependencies: List[str] = None
    ) -> ExecutionResult:
        """Execute code using Piston Engine"""
        try:
            if "piston_url" not in self.config:
                raise ValueError("Piston API URL not configured")

            # Map our FileTypes to Piston language names
            language_map = {
                FileType.PYTHON: "python",
                FileType.JAVASCRIPT: "javascript",
                FileType.TYPESCRIPT: "typescript",
                FileType.JAVA: "java",
                FileType.CPP: "cpp",
                FileType.CSHARP: "csharp",
                FileType.GO: "go",
                FileType.RUST: "rust",
                FileType.PHP: "php",
                FileType.RUBY: "ruby",
                FileType.SWIFT: "swift",
                FileType.KOTLIN: "kotlin",
                FileType.SCALA: "scala",
                FileType.R: "r",
                # Add more mappings as needed
            }

            if file.type not in language_map:
                raise ValueError(f"Unsupported language: {file.type}")

            piston_language = language_map[file.type]
            
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                
                async with session.post(
                    f"{self.config['piston_url']}/api/v2/execute",
                    json={
                        "language": piston_language,
                        "version": "*",  # Latest version
                        "files": [{
                            "content": file.content
                        }],
                        "stdin": "",
                        "args": run_command.split() if run_command else [],
                        "compile_timeout": 10000,
                        "run_timeout": 3000,
                        "compile_memory_limit": -1,
                        "run_memory_limit": -1
                    }
                ) as response:
                    result = await response.json()
                    execution_time = time.time() - start_time

                    if "message" in result:  # Error case
                        return ExecutionResult(
                            output="",
                            error=result["message"],
                            exit_code=1,
                            language=piston_language,
                            execution_time=execution_time,
                            dependencies=dependencies or []
                        )

                    return ExecutionResult(
                        output=result.get("run", {}).get("output", ""),
                        error=result.get("run", {}).get("stderr", ""),
                        exit_code=result.get("run", {}).get("code", 0),
                        language=piston_language,
                        execution_time=execution_time,
                        memory_usage=result.get("run", {}).get("memory", None),
                        dependencies=dependencies or []
                    )

        except Exception as e:
            return ExecutionResult(
                output="",
                error=str(e),
                exit_code=1,
                language=file.type.value,
                execution_time=0.0
            )

    async def _docker_execute(
        self,
        file: File,
        temp_dir: Optional[Path] = None,
        execution_path: Optional[str] = None,
        run_command: Optional[str] = None,
        dependencies: List[str] = None
    ) -> ExecutionResult:
        """Execute code using Docker"""
        try:
            # Lazy import docker
            import docker
            import time
            
            client = docker.from_env()
            
            # Map file types to Docker images and commands
            language_configs = {
                FileType.PYTHON: ("python:3.9-slim", "python"),
                FileType.JAVASCRIPT: ("node:16-slim", "node"),
                FileType.TYPESCRIPT: ("node:16-slim", "ts-node"),
                FileType.JAVA: ("openjdk:11-slim", "java"),
                FileType.GO: ("golang:1.17-slim", "go run"),
                FileType.RUST: ("rust:1.56-slim", "rustc")
            }

            if file.type not in language_configs:
                raise ValueError(f"Unsupported language: {file.type}")
                
            image, default_cmd = language_configs[file.type]
            
            # Create or use temp directory
            workspace_dir = temp_dir or Path(tempfile.mkdtemp())
            workspace_dir.mkdir(parents=True, exist_ok=True)
            
            # Write file to workspace
            file_path = workspace_dir / f"main{file.type.value}"
            file_path.write_text(file.content)
            
            # Prepare Docker command
            cmd = run_command or f"{default_cmd} {file_path.name}"
            
            # Start time for execution timing
            start_time = time.time()
            
            # Run container
            container = client.containers.run(
                image,
                command=cmd,
                working_dir=execution_path or "/app",
                volumes={str(workspace_dir): {'bind': '/app', 'mode': 'rw'}},
                environment={"PYTHONUNBUFFERED": "1"},
                mem_limit='512m',
                network_disabled=True,
                remove=True,
                detach=True
            )

            # Wait for completion and get output
            result = container.wait()
            logs = container.logs()
            execution_time = time.time() - start_time

            return ExecutionResult(
                output=logs.decode() if result["StatusCode"] == 0 else "",
                error=logs.decode() if result["StatusCode"] != 0 else None,
                exit_code=result["StatusCode"],
                language=file.type.value,
                execution_time=execution_time,
                dependencies=dependencies or []
            )

        except Exception as e:
            logger.error(f"Docker execution error: {str(e)}")
            return ExecutionResult(
                output="",
                error=str(e),
                exit_code=1,
                language=file.type.value,
                execution_time=0.0
            )

    async def _code_server_execute(
        self,
        file: File,
        temp_dir: Optional[Path] = None,
        execution_path: Optional[str] = None,
        run_command: Optional[str] = None,
        dependencies: List[str] = None
    ) -> ExecutionResult:
        """Execute code using Code-Server"""
        try:
            import aiohttp
            
            if "code_server_url" not in self.config:
                raise ValueError("Code-Server URL not configured")

            async with aiohttp.ClientSession() as session:
                # Submit execution request
                async with session.post(
                    f"{self.config['code_server_url']}/run",
                    json={
                        "code": file.content,
                        "language": file.type.value,
                        "run_command": run_command,
                        "dependencies": dependencies
                    }
                ) as response:
                    result = await response.json()
                    
                    return ExecutionResult(
                        output=result.get("output", ""),
                        error=result.get("error"),
                        exit_code=result.get("exit_code", 1),
                        language=file.type.value,
                        execution_time=result.get("execution_time", 0.0),
                        dependencies=dependencies or []
                    )

        except Exception as e:
            logger.error(f"Code-Server execution error: {str(e)}")
            return ExecutionResult(
                output="",
                error=str(e),
                exit_code=1,
                language=file.type.value,
                execution_time=0.0
            )

    async def _sphere_execute(
        self,
        file: File,
        execution_path: Optional[str] = None,
        run_command: Optional[str] = None,
        dependencies: List[str] = None
    ) -> ExecutionResult:
        """Execute code using Sphere Engine"""
        try:
            import aiohttp
            
            if "sphere_api_token" not in self.config:
                raise ValueError("Sphere Engine API token not configured")

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.sphere-engine.com/api/v4/executions",
                    headers={"Authorization": f"Bearer {self.config['sphere_api_token']}"},
                    json={
                        "source": file.content,
                        "language": file.type.value,
                        "input": "",
                        "compiler_options": run_command or "",
                        "dependencies": dependencies or []
                    }
                ) as response:
                    result = await response.json()
                    
                    return ExecutionResult(
                        output=result.get("output", ""),
                        error=result.get("error"),
                        exit_code=result.get("exit_code", 1),
                        language=file.type.value,
                        execution_time=result.get("time", 0.0),
                        dependencies=dependencies or []
                    )

        except Exception as e:
            logger.error(f"Sphere Engine execution error: {str(e)}")
            return ExecutionResult(
                output="",
                error=str(e),
                exit_code=1,
                language=file.type.value,
                execution_time=0.0
            )

    async def _cloud_run_execute(
        self,
        file: File,
        temp_dir: Optional[Path] = None,
        execution_path: Optional[str] = None,
        run_command: Optional[str] = None,
        dependencies: List[str] = None
    ) -> ExecutionResult:
        """Execute code using Google Cloud Run"""
        try:
            from google.cloud import run_v2
            from google.cloud import storage
            import time
            
            if "project_id" not in self.config:
                raise ValueError("Google Cloud project ID not configured")

            # Create Cloud Run client
            client = run_v2.ServicesClient()
            
            # Upload code to Cloud Storage
            storage_client = storage.Client()
            bucket = storage_client.bucket(f"{self.config['project_id']}-code")
            
            blob = bucket.blob(f"code/{time.time()}")
            blob.upload_from_string(file.content)
            
            # Create Cloud Run service
            service = {
                "template": {
                    "containers": [{
                        "image": f"gcr.io/{self.config['project_id']}/code-runner",
                        "env": [
                            {"name": "CODE_URL", "value": blob.public_url},
                            {"name": "LANGUAGE", "value": file.type.value},
                            {"name": "RUN_COMMAND", "value": run_command or ""}
                        ]
                    }]
                }
            }
            
            # Deploy and wait for result
            operation = client.create_service(service)
            result = operation.result()
            
            return ExecutionResult(
                output=result.latest_ready_revision.status.message,
                error=None,
                exit_code=0,
                language=file.type.value,
                execution_time=result.latest_ready_revision.execution_time,
                dependencies=dependencies or []
            )

        except Exception as e:
            logger.error(f"Cloud Run execution error: {str(e)}")
            return ExecutionResult(
                output="",
                error=str(e),
                exit_code=1,
                language=file.type.value,
                execution_time=0.0
            )

    async def _local_execute(
        self,
        file: File,
        temp_dir: Optional[Path] = None,
        execution_path: Optional[str] = None,
        run_command: Optional[str] = None
    ) -> ExecutionResult:
        """Execute Python code in the local Python interpreter"""
        import sys
        from io import StringIO
        import contextlib
        import time
        
        # Capture stdout/stderr
        stdout = StringIO()
        stderr = StringIO()
        
        # Create execution namespace
        namespace = {}
        
        try:
            start_time = time.time()
            
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                if execution_path:
                    sys.path.insert(0, str(execution_path))
                
                exec(file.content, namespace)
                
                if run_command:
                    exec(run_command, namespace)
                    
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                output=stdout.getvalue(),
                error=stderr.getvalue() if stderr.getvalue() else None,
                exit_code=0,
                language="python",
                execution_time=execution_time
            )
            
        except Exception as e:
            return ExecutionResult(
                output=stdout.getvalue(),
                error=str(e),
                exit_code=1,
                language="python",
                execution_time=time.time() - start_time
            )
        
        finally:
            if execution_path:
                sys.path.remove(str(execution_path))
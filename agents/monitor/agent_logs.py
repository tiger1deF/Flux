"""
Agent logging models and utilities for structured logging output.

This module provides models and utilities for handling agent logs, including
message logs, input/output logs, and error logs with support for rich markdown formatting
and various data type handling.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from typing import Any, Dict, Union, Callable, Mapping, Sequence, List
from functools import lru_cache
import numpy as np
import json
import sys
from pathlib import Path
from uuid import uuid4

from llm import LLM

from agents.agent.models import Sender

from agents.messages.models import File, Message

    
def get_type_handler(obj: Any) -> str:
    """
    Determine the appropriate handler for a given object type.
    
    :param obj: Object to determine handler for
    :type obj: Any
    :return: String identifier for the handler type
    :rtype: str
    """
    if isinstance(obj, pd.DataFrame):
        return "dataframe"
    elif isinstance(obj, go.Figure):
        return "figure"
    elif isinstance(obj, np.ndarray):
        return "ndarray"
    elif isinstance(obj, (list, tuple, set)):
        return "sequence"
    elif isinstance(obj, dict):
        return "mapping"
    elif callable(obj):
        return "callable"
    elif isinstance(obj, (str, int, float, bool)):
        return "primitive"
    elif isinstance(obj, datetime):
        return "datetime"
    elif isinstance(obj, Path):
        return "path"
    else:
        return "unknown"


@lru_cache(maxsize = 128)
def get_size_str(size_bytes: int) -> str:
    """
    Convert byte size to human readable string with appropriate unit.
    
    :param size_bytes: Size in bytes
    :type size_bytes: int
    :return: Formatted size string (e.g., "1.5 MB")
    :rtype: str
    """
    for unit in {'B', 'KB', 'MB', 'GB'}:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def handle_dataframe(df: pd.DataFrame, name: str) -> str:
    """
    Generate markdown summary for a pandas DataFrame.
    
    :param df: DataFrame to summarize
    :type df: pd.DataFrame
    :param name: Name of the DataFrame
    :type name: str
    :return: Markdown formatted summary
    :rtype: str
    """
    size = get_size_str(df.memory_usage().sum())
    return (f"## 📋 **{name}**\n\n"
            f"### DataFrame Metadata\n"
            f"- **Shape**: {df.shape[0]} rows × {df.shape[1]} columns\n"
            f"- **Memory Usage**: {size}\n"
            f"- **Column Types**:\n"
            + "".join(f"  - `{col}`: {dtype}\n" for col, dtype in df.dtypes.items())
            + "\n### Data Preview\n\n"
            + df.head(5).to_markdown(index=False, tablefmt="pipe")
            + "\n\n---\n\n")


def handle_figure(fig: go.Figure, name: str) -> str:
    traces = [
        f"#### Trace {i + 1}\n"
        f"- **Type**: `{trace.type}`\n"
        f"- **Name**: `{trace.name or 'Unnamed'}`\n"
        f"- **Mode**: `{getattr(trace, 'mode', 'N/A')}`\n"
        f"- **Num Data Points**: X: {len(getattr(trace, 'x', []))}, Y: {len(getattr(trace, 'y', []))}\n\n"
        for i, trace in enumerate(fig.data)
    ]
    
    return (f"## 📈 **{name}**\n\n"
            f"### Figure Metadata\n"
            f"- **Total Traces**: {len(fig.data)}\n\n"
            f"### Trace Details\n"
            + "".join(traces)
            + "\n---\n\n")


def handle_ndarray(arr: np.ndarray, name: str) -> str:
    size = get_size_str(arr.nbytes)
    return (f"## 🔢 **{name}**\n\n"
            f"- **Shape**: `{arr.shape}`\n"
            f"- **Dtype**: `{arr.dtype}`\n"
            f"- **Memory**: {size}\n"
            "\n---\n\n")


def handle_sequence(seq: Sequence, name: str) -> str:
    return (f"## 📝 **{name}**\n\n"
            f"- **Type**: `{type(seq).__name__}`\n"
            f"- **Length**: `{len(seq)}`\n"
            "\n---\n\n")


def handle_mapping(d: Mapping, name: str) -> str:
    return (f"## 🗂️ **{name}**\n\n"
            f"- **Keys**: `{len(d)}`\n"
            "\n---\n\n")


def handle_callable(func: Callable, name: str) -> str:
    return (f"## ⚡ **{name}**\n\n"
            f"- **Type**: `{type(func).__name__}`\n"
            f"- **Name**: `{func.__name__}`\n"
            "\n---\n\n")


def handle_primitive(
    value: Union[str, int, float, bool], 
    name: str, 
    cutoff_threshold: int = 1000
) -> str:
    if isinstance(value, str):  
        return (f"## 📌 **{name}**\n\n"
                f"- **Type**: `{type(value).__name__}`\n"
                f"- **Value**: `{value[:cutoff_threshold]}`\n"
                "\n---\n\n")
    else:
        return (f"## 📌 **{name}**\n\n"
                f"- **Type**: `{type(value).__name__}`\n"
                f"- **Value**: `{value}`\n"
                "\n---\n\n")


def handle_datetime(dt: datetime, name: str) -> str:
    return (f"## 📅 **{name}**\n\n"
            f"- **ISO Format**: `{dt.isoformat()}`\n"
            "\n---\n\n")


def handle_path(path: Path, name: str) -> str:
    return (f"## 📂 **{name}**\n\n"
            f"- **Path**: `{str(path)}`\n"
            "\n---\n\n")


def handle_unknown(obj: Any, name: str) -> str:
    size = get_size_str(sys.getsizeof(obj))
    return (f"## ❓ **{name}**\n\n"
            f"- **Type**: `{type(obj).__name__}`\n"
            f"- **Size**: {size}\n"
            "\n---\n\n")


TYPE_HANDLERS = {
    "dataframe": handle_dataframe,
    "figure": handle_figure,
    "ndarray": handle_ndarray,
    "sequence": handle_sequence,
    "mapping": handle_mapping,
    "callable": handle_callable,
    "primitive": handle_primitive,
    "datetime": handle_datetime,
    "path": handle_path,
    "unknown": handle_unknown
}


def generate_markdown_summary(
            data_dict: Dict[str, Any] = None,
            file_dict: Dict[str, File] = None
) -> str:
    """
    Generate comprehensive markdown summary of data and files.
    
    :param data_dict: Dictionary of data items to summarize
    :type data_dict: Dict[str, Any], optional
    :param file_dict: Dictionary of files to summarize
    :type file_dict: Dict[str, File], optional
    :return: Markdown formatted summary
    :rtype: str
    """
    output_summary = ""
    if data_dict:
        metadata_output = "# 📊 Metadata Summary\n\n"
        
        summaries = []
        for name, item in data_dict.items():
            handler_type = get_type_handler(item)
            handler = TYPE_HANDLERS[handler_type]
            summary = handler(item, name)
            summaries.append(summary)
        
        output_summary = metadata_output + "".join(summaries)
        
    if file_dict:
        file_output = "\n\n# 📂 File Summary\n\n"
        file_summaries = []
        for name, file in file_dict.items():
            file_summaries.append(file.summary())

        output_summary += file_output + "".join(file_summaries)
        
    return output_summary
        

@dataclass
class BaseLogEntry(ABC):
    """
    Abstract base class for all log entries.
    
    :ivar log_type: Type identifier for the log entry
    :type log_type: str
    """
    log_type: str = None
    
    @abstractmethod
    def to_dict(self) -> dict:
        """
        Convert log entry to dictionary representation.
        
        :return: Dictionary representation of log entry
        :rtype: dict
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    
    @abstractmethod
    def text(self) -> str:
        """
        Generate text representation of log entry.
        
        :return: Text representation
        :rtype: str
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    
    def to_json(self) -> str:
        """
        Convert log entry to JSON string.
        
        :return: JSON string representation
        :rtype: str
        """
        return json.dumps(self.to_dict())
    
    
@dataclass
class InputLog(BaseLogEntry):
    """
    Log entry for agent input messages.
    
    :ivar input_message: Message received as input
    :type input_message: Message
    :ivar input_agent: Agent or sender that provided the input
    :type input_agent: Union[str, Sender]
    :ivar log_type: Type identifier for the log entry
    :type log_type: str
    """
    input_message: Message = None
    input_agent: Union[str, Sender] = None
    log_type: str = 'input'

    def to_dict(self) -> dict:
        """
        Convert input log to dictionary representation.
        
        :return: Dictionary containing log data
        :rtype: dict
        """
        return {
            'timestamp': self.input_message.date,
            'log_type': 'input',
            'input_message': self.input_message.to_dict(),
            'input_agent': self.input_agent
        }
    
    
    def text(self) -> str:
        content = "## 📨 Input Message"
        if self.input_agent:
            content += f" From {self.input_agent}"
        content += f"\n*🕒 Timestamp:* `{self.input_message.date}`\n"

        content += "\n### 📥 Input Message Content\n"
        content += f"{self.input_message.content}\n"
        
        if self.input_message.metadata or self.input_message.files:
            content += "\n" + generate_markdown_summary(
                self.input_message.metadata,
                self.input_message.files
            )
        
        return content.strip()
    

@dataclass
class ErrorLog(BaseLogEntry):
    """
    Log entry for agent errors.
    
    :ivar output_message: Message associated with the error
    :type output_message: Message
    :ivar error_message: Error message or traceback
    :type error_message: str
    :ivar log_type: Type identifier for the log entry
    :type log_type: str
    """
    output_message: Message = None
    error_message: str = None
    log_type: str = 'error'

    def to_dict(self) -> dict:
        """
        Convert error log to dictionary representation.
        
        :return: Dictionary containing log data
        :rtype: dict
        """
        return {
            'timestamp': self.output_message.date,
            'log_type': 'error',
            'error_message': self.error_message
        }


    def text(self) -> str:
        return f"""## ❌ ERROR IN Agent
*🕒 Timestamp:* `{self.output_message.date}`

### 🔴 Error Logs
```text
{self.error_message}
```
"""


@dataclass
class OutputLog(BaseLogEntry):
    """
    Log entry for agent output messages.
    
    :ivar output_message: Message produced as output
    :type output_message: Message
    """
    output_message: Message = None
    
    
    def to_dict(self) -> dict:
        """
        Convert output log to dictionary representation.
        
        :return: Dictionary containing log data
        :rtype: dict
        """
        return {
            'timestamp': self.output_message.date,
            'log_type': 'output',
            'output_message': self.output_message.to_dict()
        }


    def text(self) -> str:
        output_str = f"""# ✅ AGENT SUCCESSFULLY EXECUTED
*🕒 Timestamp:* `{self.output_message.date}`

### 📤 Output Message Content:
{self.output_message.content}
"""
        
        if self.output_message.metadata or self.output_message.files:
            output_str += "\n" + generate_markdown_summary(
                self.output_message.metadata,
                self.output_message.files
            )
        
        return output_str.strip()

      
class MessageLog(BaseLogEntry):
    """
    Log entry for intermediate agent messages.
    
    :ivar message: Intermediate message content
    :type message: Message
    :ivar log_type: Type identifier for the log entry
    :type log_type: str
    """
    message: Message = None
    log_type: str = 'message'

    def text(self) -> str:
        output_str = f"\n*🕒 Timestamp:* `{self.message.date}`\n"
        output_str += "\n## 💬 Intermediate Message Content\n"
        output_str += f"{self.message.content}\n"
        
        if self.message.metadata or self.message.files:
            output_str += "\n" + generate_markdown_summary(
                self.message.metadata,
                self.message.files
            )
        
        return output_str.strip()
    
    
@dataclass
class AgentLog(BaseLogEntry):
    """
    Comprehensive log for an agent's execution.
    
    :ivar agent_name: Name of the agent
    :type agent_name: str
    :ivar agent_type: Type of the agent
    :type agent_type: str
    :ivar agent_description: Description of the agent's purpose
    :type agent_description: str
    :ivar llm: Language model configuration
    :type llm: Union[LLM, None]
    :ivar agent_logs: Raw log messages
    :type agent_logs: str
    :ivar agent_messages: List of intermediate messages
    :type agent_messages: List[MessageLog]
    :ivar source_agents: List of upstream agents
    :type source_agents: List[str]
    :ivar target_agents: List of downstream agents
    :type target_agents: List[str]
    :ivar agent_input: Input log entry
    :type agent_input: InputLog
    :ivar agent_output: Output log entry
    :type agent_output: OutputLog
    :ivar session_id: Unique identifier for the execution session
    :type session_id: str
    """
    agent_name: str = None
    agent_type: str = None
    agent_description: str = None
    llm: Union[LLM, None] = None
    agent_logs: str = None
    agent_messages: List[MessageLog] = field(default_factory=list)
    source_agents: List[str] = field(default_factory=list)
    target_agents: List[str] = field(default_factory=list)
    agent_input: InputLog = None
    agent_output: OutputLog = None
    session_id: str = str(uuid4())
    
    
    def to_dict(self) -> dict:
        """
        Convert agent log to dictionary representation.
        
        :return: Dictionary containing log data
        :rtype: dict
        """
        return {
            'timestamp': self.timestamp,
            'log_type': 'agent',
            'agent_name': self.agent_name,
            'agent_type': self.agent_type,
            'agent_description': self.agent_description,
            'source_agents': self.source_agents,
            'target_agents': self.target_agents,
            'agent_logs': self.agent_logs,
            'llm': self.llm.to_dict() if self.llm else None
        }


    async def log_input(
        self,
        input_message: Message,
        input_agent: Union[str, Sender] = None
    ) -> InputLog:
        """
        Log an input message received by the agent.
        
        :param input_message: Message received as input
        :type input_message: Message
        :param input_agent: Agent or sender that provided the input
        :type input_agent: Union[str, Sender], optional
        :return: Created input log entry
        :rtype: InputLog
        """
        self.agent_input = InputLog(
            input_message = input_message,
            input_agent = input_agent
        )


    async def log_output(
        self,
        output_message: Message
    ) -> OutputLog:
        """
        Log an output message produced by the agent.
        
        :param output_message: Message produced as output
        :type output_message: Message
        :return: Created output log entry
        :rtype: OutputLog
        """
        self.agent_output = OutputLog(
            output_message = output_message
        )

    
    async def log_message(self, message: Message) -> Message:
        self.agent_message = MessageLog(
            message = message
        )
        self.agent_messages.append(self.agent_message)
    
    
    async def log_error(self, error_message: str) -> ErrorLog:
        self.agent_error = ErrorLog(
            error_message = error_message
        )
        
    
    def input_text(self) -> str:
        overview = f"""# Agent Execution Log

## ⚙️ Agent Details
- **Name:** `{self.agent_name}`
- **Type:** `{self.agent_type}`
"""
        if self.agent_description:
            overview += f"- **Description:** `{self.agent_description}`\n"

        if self.source_agents or self.target_agents:
            overview += f"""

## 🔄 Node Connections
- **⬆️ Upstream Agent(s):** `{self.source_agents}`
- **⬇️ Downstream Agent(s):** `{self.target_agents}`
"""

        if self.llm:
            overview += "\n\n## 🤖 LLM Configuration"
            for param_name, param_info in self.llm.signature.items():
                current_value = getattr(self.llm, param_name, param_info.get('default'))
                if current_value is not None:
                    overview += f"\n- **{param_name}:** `{current_value}`"
        
        if self.agent_input:
            overview += self.agent_input.text()

        return overview
    
    
    def output_text(self) -> str:
        """
        Generate a formatted text representation of the agent's output. Primarily for langfuse.
        
        :return: Markdown formatted output details
        :rtype: str
        """
        overview = f"*Run duration:* `{(self.agent_output.output_message.date - self.agent_input.input_message.date).total_seconds():.3f}s`"
        for message in self.agent_messages:
            overview += message.text()

        if self.agent_logs:
            agent_logs = f"\n\n## 📝 Agent Logs\n{self.agent_logs}\n"
            overview += agent_logs

        if self.agent_messages:
            overview += "\n## 💬 Intermediate Messages\n"
            for message in self.agent_messages:
                overview += message.text()

        if self.agent_output:
            overview += f'\n{self.agent_output.text()}'
        elif hasattr(self, 'agent_error'):
            overview += f"\n\n{self.agent_error.text()}"
        
        return overview
        
        
    def text(self) -> str:
        """
        Generate a formatted text representation of the agent's execution log.
        
        :return: Markdown formatted execution log
        :rtype: str
        """
        overview = f"""# Agent Execution Log
*Run duration:* `{(self.agent_output.output_message.date - self.agent_input.input_message.date).total_seconds():.3f}s`

## ⚙️ Agent Details
- **Name:** `{self.agent_name}`
- **Type:** `{self.agent_type}`
"""
        if self.agent_description:
            overview += f"- **Description:** `{self.agent_description}`\n"

        if self.source_agents or self.target_agents:
            overview += f"""

## 🔄 Node Connections
- **⬆️ Upstream Agent(s):** `{self.source_agents}`
- **⬇️ Downstream Agent(s):** `{self.target_agents}`
"""

        if self.llm:
            overview += "\n## 🤖 LLM Configuration"
            for param_name, param_info in self.llm.signature.items():
                current_value = getattr(self.llm, param_name, param_info.get('default'))
                if current_value is not None:
                    overview += f"\n- **{param_name}:** `{current_value}`\n\n"
        
        if self.agent_input:
            overview += self.agent_input.text()

        for message in self.agent_messages:
            overview += message.text()

        if self.agent_logs:
            agent_logs = f"\n\n## 📝 Agent Logs\n{self.agent_logs}\n"
            overview += agent_logs

        if self.agent_messages:
            overview += "\n## 💬 Intermediate Messages\n"
            for message in self.agent_messages:
                overview += message.text()

        if self.agent_output:
            overview += f'\n{self.agent_output.text()}'
        elif hasattr(self, 'agent_error'):
            overview += f"\n\n{self.agent_error.text()}"
        
        return overview


# TODO - Integrate into structured framework
@dataclass
class EdgeLog(BaseLogEntry):
    """
    Log entry for agent graph edges.
    
    :ivar edge_type: Type of connection between agents
    :type edge_type: str
    :ivar llm: Language model configuration
    :type llm: Union[LLM, None]
    :ivar upstream_agents: Source agent(s) of the edge
    :type upstream_agents: str
    :ivar downstream_agents: Target agent(s) of the edge
    :type downstream_agents: Union[list, str]
    """
    edge_type: str = 'normal'
    llm: Union[LLM, None] = None
    upstream_agents: str = None
    downstream_agents: Union[list, str] = None


    def to_dict(self) -> dict:
        """
        Convert edge log to dictionary representation.
        
        :return: Dictionary containing log data
        :rtype: dict
        """
        return {
            'timestamp': self.timestamp,
            'log_type': 'edge',
            'upstream_agents': self.upstream_agents,
            'downstream_agents': self.downstream_agents,
        }


    def input(self):
        """
        Generate a formatted text representation of the edge details.
        
        :return: Markdown formatted edge details
        :rtype: str
        """
        edge_log = f"""## 🔗 Edge Details
*Log Time:* `{self.timestamp}`

### 🔄 Connection Flow
- **Edge Type:** `{self.edge_type}`
- **Upstream Agent(s):** `{self.upstream_agents}`
- **Downstream Agent(s):** `{self.downstream_agents}`"""

        if self.llm:
            edge_log += "\n### 🤖 LLM Configuration"
            for param_name, param_info in self.llm.signature.items():
                current_value = getattr(self.llm, param_name, param_info.get('default'))
                if current_value is not None:
                    edge_log += f"\n- **{param_name}:** `{current_value}`"

        return edge_log


    @property
    def text(self) -> str:
        """
        Get the text representation of the edge log.
        
        :return: Formatted text representation
        :rtype: str
        """
        return self.input()


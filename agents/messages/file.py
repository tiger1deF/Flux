from typing import List, Dict, Any, Union, Optional
from pydantic import BaseModel, Field
from enum import Enum
import os
import asyncio
import aiofiles
from functools import lru_cache
import polars as pl
import yaml
import orjson
from pathlib import Path
import io
from difflib import SequenceMatcher, unified_diff
import tempfile
import shutil
import hashlib
import base64

from utils.shared.tokenizer import encode_async
from utils.shared.tokenizer import slice_text, SliceType
from utils.serialization import get_compressor, get_decompressor


class FileType(str, Enum):
    """
    Standardized file types mapped from extensions.
    
    :cvar PYTHON: Python source and bytecode files (.py, .pyx, .pyi, .pyc)
    :cvar JAVA: Java source and class files (.java, .class, .jar)
    :cvar CPP: C++ source and header files (.cpp, .cc, .cxx, .hpp, .h)
    :cvar CSHARP: C# source files (.cs, .cshtml)
    :cvar GO: Go source files (.go)
    :cvar RUST: Rust source files (.rs)
    :cvar PHP: PHP source files (.php, .phtml)
    :cvar RUBY: Ruby source files (.rb, .erb)
    :cvar SWIFT: Swift source files (.swift)
    :cvar KOTLIN: Kotlin source files (.kt, .kts)
    :cvar SCALA: Scala source files (.scala)
    :cvar R: R source and markdown files (.r, .rmd)
    :cvar SHELL: Shell script files (.sh, .bash, .zsh)
    :cvar SQL: SQL query files (.sql, .psql, .mysql)
    :cvar HTML: HTML files (.html, .htm, .xhtml)
    :cvar CSS: CSS and preprocessor files (.css, .scss, .sass, .less)
    :cvar XML: XML and XSLT files (.xml, .xsl, .xslt)
    :cvar JSON: JSON files (.json, .jsonl)
    :cvar YAML: YAML files (.yml, .yaml)
    :cvar MARKDOWN: Markdown files (.md, .markdown)
    :cvar TEXT: Plain text files (.txt, .text)
    :cvar PDF: PDF documents (.pdf)
    :cvar DOC: Word documents (.doc, .docx)
    :cvar EXCEL: Excel spreadsheets (.xls, .xlsx, .csv)
    :cvar PPT: PowerPoint presentations (.ppt, .pptx)
    :cvar RTF: Rich text format files (.rtf)
    :cvar TEX: LaTeX files (.tex, .latex)
    :cvar IMAGE: Image files (.jpg, .jpeg, .png, .gif, .bmp, .svg, .webp)
    :cvar AUDIO: Audio files (.mp3, .wav, .ogg, .m4a, .flac)
    :cvar VIDEO: Video files (.mp4, .avi, .mov, .wmv, .flv, .webm)
    :cvar ARCHIVE: Archive files (.zip, .tar, .gz, .7z, .rar)
    :cvar CSV: CSV data files (.csv)
    :cvar PARQUET: Parquet data files (.parquet)
    :cvar HDF5: HDF5 data files (.h5, .hdf5)
    :cvar PICKLE: Python pickle files (.pkl, .pickle)
    :cvar CONFIG: Configuration files (.conf, .cfg, .ini, .env)
    :cvar BINARY: Binary files (.bin, .exe, .dll)
    :cvar TYPESCRIPT: TypeScript files (.ts, .tsx)
    :cvar UNKNOWN: Unknown file types
    """
    # Code files
    PYTHON = "python"          # .py, .pyx, .pyi, .pyc
    JAVASCRIPT = "javascript"  # .js, .jsx
    TYPESCRIPT = "typescript"  # .ts, .tsx
    JAVA = "java"             # .java, .class, .jar
    CPP = "cpp"               # .cpp, .cc, .cxx, .hpp, .h
    CSHARP = "csharp"         # .cs, .cshtml
    GO = "go"                 # .go
    RUST = "rust"             # .rs
    PHP = "php"               # .php, .phtml
    RUBY = "ruby"             # .rb, .erb
    SWIFT = "swift"           # .swift
    KOTLIN = "kotlin"         # .kt, .kts
    SCALA = "scala"           # .scala
    R = "r"                   # .r, .rmd
    SHELL = "shell"           # .sh, .bash, .zsh
    SQL = "sql"               # .sql, .psql, .mysql
    
    # Web files
    HTML = "html"             # .html, .htm, .xhtml
    CSS = "css"               # .css, .scss, .sass, .less
    XML = "xml"               # .xml, .xsl, .xslt
    JSON = "json"             # .json, .jsonl
    YAML = "yaml"             # .yml, .yaml
    
    # Document files
    MARKDOWN = "markdown"     # .md, .markdown
    TEXT = "text"            # .txt, .text
    PDF = "pdf"              # .pdf
    DOC = "doc"              # .doc, .docx
    EXCEL = "excel"          # .xls, .xlsx, .csv
    PPT = "powerpoint"       # .ppt, .pptx
    RTF = "rtf"              # .rtf
    TEX = "tex"              # .tex, .latex
    
    # Image files
    IMAGE = "image"          # .jpg, .jpeg, .png, .gif, .bmp, .svg, .webp
    
    # Audio files
    AUDIO = "audio"          # .mp3, .wav, .ogg, .m4a, .flac
    
    # Video files
    VIDEO = "video"          # .mp4, .avi, .mov, .wmv, .flv, .webm
    
    # Archive files
    ARCHIVE = "archive"      # .zip, .tar, .gz, .7z, .rar
    
    # Data files
    CSV = "csv"              # .csv
    PARQUET = "parquet"      # .parquet
    HDF5 = "hdf5"           # .h5, .hdf5
    PICKLE = "pickle"        # .pkl, .pickle
    
    # Config files
    CONFIG = "config"        # .conf, .cfg, .ini, .env
    
    # Binary files
    BINARY = "binary"        # .bin, .exe, .dll
    
    # Other
    UNKNOWN = "unknown"      # Fallback for unknown types
    
    @classmethod
    @lru_cache(maxsize = 128)
    def from_extension(cls, extension: str) -> 'FileType':
        """Cached version to avoid repeated sync operations"""
        ext = extension.lower().lstrip('.')
        
        # Code files
        if ext in {'py', 'pyx', 'pyi', 'pyc'}:
            return cls.PYTHON
        elif ext in {'js', 'jsx'}:
            return cls.JAVASCRIPT
        elif ext in {'ts', 'tsx'}:
            return cls.TYPESCRIPT
        elif ext in {'java', 'class', 'jar'}:
            return cls.JAVA
        elif ext in {'cpp', 'cc', 'cxx', 'hpp', 'h'}:
            return cls.CPP
        elif ext in {'cs', 'cshtml'}:
            return cls.CSHARP
        elif ext == 'go':
            return cls.GO
        elif ext == 'rs':
            return cls.RUST
        elif ext in {'php', 'phtml'}:
            return cls.PHP
        elif ext in {'rb', 'erb'}:
            return cls.RUBY
        elif ext == 'swift':
            return cls.SWIFT
        elif ext in {'kt', 'kts'}:
            return cls.KOTLIN
        elif ext == 'scala':
            return cls.SCALA
        elif ext in {'r', 'rmd'}:
            return cls.R
        elif ext in {'sh', 'bash', 'zsh'}:
            return cls.SHELL
        elif ext in {'sql', 'psql', 'mysql'}:
            return cls.SQL
            
        # Web files
        elif ext in {'html', 'htm', 'xhtml'}:
            return cls.HTML
        elif ext in {'css', 'scss', 'sass', 'less'}:
            return cls.CSS
        elif ext in {'xml', 'xsl', 'xslt'}:
            return cls.XML
        elif ext in {'json', 'jsonl'}:
            return cls.JSON
        elif ext in {'yml', 'yaml'}:
            return cls.YAML
            
        # Document files
        elif ext in {'md', 'markdown'}:
            return cls.MARKDOWN
        elif ext in {'txt', 'text'}:
            return cls.TEXT
        elif ext == 'pdf':
            return cls.PDF
        elif ext in {'doc', 'docx'}:
            return cls.DOC
        elif ext in {'xls', 'xlsx'}:
            return cls.EXCEL
        elif ext in {'ppt', 'pptx'}:
            return cls.PPT
        elif ext == 'rtf':
            return cls.RTF
        elif ext in {'tex', 'latex'}:
            return cls.TEX
            
        # Image files
        elif ext in {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'svg', 'webp'}:
            return cls.IMAGE
            
        # Audio files
        elif ext in {'mp3', 'wav', 'ogg', 'm4a', 'flac'}:
            return cls.AUDIO
            
        # Video files
        elif ext in {'mp4', 'avi', 'mov', 'wmv', 'flv', 'webm'}:
            return cls.VIDEO
            
        # Archive files
        elif ext in {'zip', 'tar', 'gz', '7z', 'rar'}:
            return cls.ARCHIVE
            
        # Data files
        elif ext == 'csv':
            return cls.CSV
        elif ext == 'parquet':
            return cls.PARQUET
        elif ext in {'h5', 'hdf5'}:
            return cls.HDF5
        elif ext in {'pkl', 'pickle'}:
            return cls.PICKLE
            
        # Config files
        elif ext in {'conf', 'cfg', 'ini', 'env'}:
            return cls.CONFIG
            
        # Binary files
        elif ext in {'bin', 'exe', 'dll'}:
            return cls.BINARY
            
        # Unknown
        else:
            return cls.UNKNOWN


class File(BaseModel):
    """
    Model for file attachments in messages.
    
    :ivar content: Content of the file
    :type content: Union[str, bytes, Any]
    :ivar description: Description of the file
    :type description: Union[str, None] 
    :ivar path: Path to the file
    :type path: Union[str, None]
    :ivar type: Type/extension of the file
    :type type: FileType
    :ivar annotations: Additional annotations for the file
    :type annotations: Dict[str, Any]
    """
    id: Union[int, str] = Field(default_factory = lambda: int.from_bytes(os.urandom(3), 'big') % 1_000_000)
    content: Union[str, bytes, Any] = Field(description = "File content")
    description: Union[str, None] = Field(description = "File description", default = None)
    path: Union[str, Path, None] = Field(description = "File path", default = None)
    type: FileType = Field(default = FileType.UNKNOWN, description = "File type")
    annotations: Dict[str, Any] = Field(default = {}, description = "File annotations")
    content_cache: Optional[str] = Field(default = None, exclude = True)

    # For vector-based retrieval
    score: int = Field(default = 0, description = "Score of the file")


    @property 
    def content(self) -> Union[str, bytes]:
        """
        Get the raw content of the file.
        
        :return: File content as string or bytes
        :rtype: Union[str, bytes]
        """
        return self._content

    @property
    async def name(self) -> str:
        """
        Get the file name from path or generate a default.
        
        Returns the last component of the path if available,
        otherwise returns a generated name using the file ID.
        
        :return: File name
        :rtype: str
        """
        if self.path:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, os.path.basename, self.path)
        return f"file_{self.id}"

    @content.setter
    async def content(self, value: Union[str, bytes, Any]) -> None:
        """
        Set the raw content and invalidate caches.
        
        :param value: New content value
        :type value: Union[str, bytes, Any]
        """
        if hasattr(self, '_content') and self._content != value:
            # Clear caches when content changes
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.tokens.cache_clear)
            self.content_cache = None
        self._content = value

    @property
    async def data(self) -> Any:
        """
        Get the parsed Python object representation of the file content.
        
        Converts various file formats to their corresponding Python objects:
        - CSV/Excel -> Polars DataFrame
        - JSON -> dict/list
        - YAML -> dict/list
        - XML -> dict
        - Others -> raw content
        
        :return: Parsed data object
        :rtype: Any
        """
        loop = asyncio.get_running_loop()
        
        if self.type in {FileType.CSV, FileType.EXCEL}:
            # Move file reading to thread pool
            async def read_tabular():
                if isinstance(self.content, bytes):
                    buffer = io.BytesIO(self.content)
                else:
                    buffer = io.StringIO(self.content)
                    
                if self.type == FileType.CSV:
                    return await loop.run_in_executor(None, pl.read_csv, buffer)
                else:
                    return await loop.run_in_executor(None, pl.read_excel, buffer)
                    
            return await read_tabular()

        elif self.type == FileType.JSON:
            return await loop.run_in_executor(None, 
                json.loads,
                self.content.decode() if isinstance(self.content, bytes) else self.content
            )

        elif self.type == FileType.YAML:
            return await loop.run_in_executor(None,
                yaml.safe_load,
                self.content.decode() if isinstance(self.content, bytes) else self.content
            )

        elif self.type == FileType.XML:
            if isinstance(self.content, bytes):
                return xmltodict.parse(self.content.decode())
            return xmltodict.parse(self.content)

        elif self.type == FileType.PARQUET:
            if isinstance(self.content, bytes):
                buffer = io.BytesIO(self.content)
            else:
                buffer = io.StringIO(self.content)
            return pl.read_parquet(buffer)

        elif self.type == FileType.HDF5:
            # For HDF5, we'll need to write to a temp file first
            if isinstance(self.content, bytes):
                temp_path = Path("temp.h5")
                temp_path.write_bytes(self.content)
                df = pl.read_hdf(temp_path)
                temp_path.unlink()
                return df
            raise ValueError("HDF5 content must be bytes")

        # For text-based files, return content as string
        elif self.type in {
            FileType.PYTHON, FileType.JAVASCRIPT, FileType.JAVA,
            FileType.CPP, FileType.CSHARP, FileType.GO, FileType.RUST,
            FileType.PHP, FileType.RUBY, FileType.SWIFT, FileType.KOTLIN,
            FileType.SCALA, FileType.R, FileType.SHELL, FileType.SQL,
            FileType.HTML, FileType.CSS, FileType.MARKDOWN, FileType.TEXT,
            FileType.RTF, FileType.TEX, FileType.CONFIG
        }:
            if isinstance(self.content, bytes):
                return self.content.decode()
            return self.content

        # For binary files, return raw bytes
        else:
            if isinstance(self.content, str):
                return self.content.encode()
            return self.content

    def __init__(self, **data):
        """Initialize a new File instance."""
        super().__init__(**data)
        
        # If type not explicitly set, detect from path
        if self.type == FileType.UNKNOWN and self.path:
            extension = self.path.split('.')[-1] if '.' in self.path else 'txt'
            self.type = FileType.from_extension(extension)


    async def summary(self) -> str:
        """
        Generate a lightweight summary of the file based on its type and metadata.
        Uses only readily available metadata and minimal content reading.
        
        :return: Formatted summary string
        :rtype: str
        """
        # Get file metadata
        ext = self.path.suffix.lower() if self.path else ""
        size = len(self.content) if isinstance(self.content, bytes) else len(str(self.content))
        name = self.path.name if self.path else 'unnamed'
        parent = self.path.parent if self.path else None
        
        # Base summary with clear separator and common metadata
        base = (
            f"📄 File: {name}\n"
            f"└─ Location: {parent or 'memory'}\n"
            f"└─ Type: {self.type.value}\n"
            f"└─ Size: {size:,} bytes\n"
        )
        
        if self.annotations:
            base += f"└─ Annotations: {len(self.annotations)}\n"
        
        # Type-specific summaries
        if ext in {'.py', '.pyx', '.pyi'}:
            # Python source files
            return (
                f"{base}"
                f"└─ Python Module\n"
                f"   ├─ Description: {self.description or 'No description'}\n"
                f"   └─ Last Modified: {self.modified_at or 'Unknown'}"
            )
            
        elif ext in {'.json', '.yaml', '.yml', '.toml'}:
            # Config/data files
            return (
                f"{base}"
                f"└─ Config File\n"
                f"   ├─ Format: {ext[1:].upper()}\n"
                f"   └─ Schema: {self.data.get('schema', 'Not available') if hasattr(self, 'data') else 'No schema'}"
            )
            
        elif ext in {'.md', '.rst', '.txt'}:
            # Documentation files
            return (
                f"{base}"
                f"└─ Documentation\n"
                f"   ├─ Format: {ext[1:].upper()}\n"
                f"   └─ Lines: {str(self.content).count('\n') + 1 if not isinstance(self.content, bytes) else 'Binary'}"
            )
            
        elif ext in {'.csv', '.parquet', '.xlsx'}:
            # Tabular data files
            return (
                f"{base}"
                f"└─ Data File\n"
                f"   ├─ Format: {ext[1:].upper()}\n"
                f"   └─ Structure: {self.data.get('columns', []) if hasattr(self, 'data') else 'No schema'}"
            )
            
        elif ext in {'.jpg', '.png', '.gif', '.bmp'}:
            # Image files
            return (
                f"{base}"
                f"└─ Image\n"
                f"   ├─ Format: {ext[1:].upper()}\n"
                f"   └─ Dimensions: {self.data.get('dimensions', 'Unknown') if hasattr(self, 'data') else 'Unknown'}"
            )
            
        elif ext in {'.pdf', '.doc', '.docx'}:
            # Document files
            return (
                f"{base}"
                f"└─ Document\n"
                f"   ├─ Format: {ext[1:].upper()}\n"
                f"   └─ Pages: {self.data.get('pages', 'Unknown') if hasattr(self, 'data') else 'Unknown'}"
            )
            
        elif ext in {'.js', '.ts', '.jsx', '.tsx'}:
            # JavaScript/TypeScript files
            return (
                f"{base}"
                f"└─ Web Source\n"
                f"   ├─ Language: {ext[1:].upper()}\n"
                f"   └─ Framework: {self.data.get('framework', 'Unknown') if hasattr(self, 'data') else 'Unknown'}"
            )
            
        elif ext in {'.html', '.css', '.scss'}:
            # Web files
            return (
                f"{base}"
                f"└─ Web Asset\n"
                f"   ├─ Format: {ext[1:].upper()}\n"
                f"   └─ Media Type: {self.data.get('media_type', 'text/' + ext[1:])}"
            )
            
        elif ext in {'.zip', '.tar', '.gz', '.rar'}:
            # Archive files
            return (
                f"{base}"
                f"└─ Archive\n"
                f"   ├─ Format: {ext[1:].upper()}\n"
                f"   └─ Compressed Size: {size:,} bytes"
            )
            
        else:
            # Generic binary/unknown files
            return (
                f"{base}"
                f"└─ Binary/Other\n"
                f"   └─ MIME Type: {self.data.get('mime_type', 'application/octet-stream') if hasattr(self, 'data') else 'Unknown'}"
            )


    @property
    async def tokens(self) -> List[int]:
        """
        Get the tokenized representation of the file content.
        
        :return: List of token IDs for the file content
        :rtype: List[int]
        """
        return await encode_async(str(self.content))


    @classmethod
    async def from_path(
        cls,
        path: str,
        description: Optional[str] = None,
        annotations: Optional[Dict[str, Any]] = None
    ) -> 'File':
        """Create a File instance by reading from disk/storage
        
        :param path: Path to file to read
        :param description: Optional file description
        :param annotations: Optional file annotations
        :return: New File instance with content from disk
        :raises FileNotFoundError: If file doesn't exist
        :raises IOError: If file can't be read
        """
        loop = asyncio.get_running_loop()
        exists = await loop.run_in_executor(None, os.path.exists, path)
        if not exists:
            raise FileNotFoundError(f"File not found: {path}")
            
        # Detect file type from extension
        extension = path.split('.')[-1] if '.' in path else ''
        file_type = FileType.from_extension(extension)
        
        # Known binary file types
        if file_type in {
            FileType.IMAGE, FileType.AUDIO, FileType.VIDEO,
            FileType.ARCHIVE, FileType.BINARY, FileType.PDF,
            FileType.HDF5, FileType.PARQUET, FileType.PICKLE
        }:
            async with aiofiles.open(path, 'rb') as f:
                content = await f.read()
                
            return cls(
                content = content,
                path = path,
                type = file_type,
                description = description or os.path.basename(path),
                annotations = annotations or {'binary': True}
            )
        
        # Text files
        async with aiofiles.open(path, 'r') as f:
            content = await f.read()
            
        return cls(
            content = content,
            path = path,
            type = file_type,
            description = description or os.path.basename(path),
            annotations = annotations or {}
        )
        

    async def truncate(
        self,
        max_tokens: int,
        slice_type: SliceType = SliceType.END
    ) -> str:
        """
        Truncate file content based on type and strategy.
        
        Different handling for:
        - Text files: Standard text truncation
        - Structured data: Truncate while preserving structure
        - Binary files: Size-based truncation
        
        :param max_tokens: Maximum number of tokens to keep
        :type max_tokens: int
        :param slice_type: Type of slice to apply
        :type slice_type: SliceType
        :return: Sliced content
        :rtype: str
        """
        loop = asyncio.get_running_loop()
        
        # Cache check
        if self.content_cache is not None:
            cache_tokens = await encode_async(self.content_cache)
            if len(cache_tokens) <= max_tokens:
                return self.content_cache

        # Handle structured data
        if self.type in {FileType.JSON, FileType.YAML, FileType.XML}:
            data = await self.data
            if isinstance(data, (dict, list)):
                truncated = await loop.run_in_executor(None,
                    lambda: str(data)[:max_tokens * 4]
                )
                self.content_cache = truncated
                
                return truncated

        # Handle tabular data
        elif self.type in {FileType.CSV, FileType.EXCEL, FileType.PARQUET}:
            df = await self.data
            # Keep first N rows that fit within token limit
            sample_row = str(df.head(1))
            tokens_per_row = len(await encode_async(sample_row))
            keep_rows = max(1, max_tokens // tokens_per_row)
            truncated = str(df.head(keep_rows))
            self.content_cache = truncated
            
            return truncated

        # Handle text-based files
        elif self.type in {
            FileType.PYTHON, FileType.JAVASCRIPT, FileType.JAVA,
            FileType.CPP, FileType.CSHARP, FileType.GO, FileType.RUST,
            FileType.PHP, FileType.RUBY, FileType.SWIFT, FileType.KOTLIN,
            FileType.SCALA, FileType.R, FileType.SHELL, FileType.SQL,
            FileType.HTML, FileType.CSS, FileType.MARKDOWN, FileType.TEXT,
            FileType.RTF, FileType.TEX, FileType.CONFIG
        }:
            content_str = self.content if isinstance(self.content, str) else self.content.decode()
            truncated = await slice_text(
                text = content_str,
                max_tokens = max_tokens,
                slice_type = slice_type
            )
            self.content_cache = truncated
            
            return truncated

        # Handle binary files
        elif isinstance(self.content, bytes):
            # For binary files, truncate based on byte size
            approx_bytes = max_tokens * 4  # Rough approximation of bytes per token
            truncated = self.content[:approx_bytes]
            self.content_cache = f"<Binary content truncated to {len(truncated)} bytes>"
            return self.content_cache

        # Default fallback
        else:
            content_str = str(self.content)
            truncated = await slice_text(
                text = content_str,
                max_tokens = max_tokens,
                slice_type = slice_type
            )
            self.content_cache = truncated
            return truncated
        
    async def edit_file(
        self,
        new_content: Union[str, bytes, Any],
        merge: bool = True
    ) -> None:
        """
        Update file content with intelligent merging using difflib.
        
        For text-based files, performs smart merging of content preserving
        meaningful changes while avoiding duplicates.
        
        :param new_content: New content to update with
        :type new_content: Union[str, bytes, Any]
        :param merge: Whether to merge with existing content
        :type merge: bool
        :raises TypeError: If file type doesn't support content updates
        """
        loop = asyncio.get_running_loop()
        
        if self.type in {FileType.CSV, FileType.PARQUET}:
            if merge:
                # Move DataFrame operations to thread pool
                existing_df = await self.data
                if isinstance(new_content, (str, bytes)):
                    new_df = await loop.run_in_executor(None,
                        lambda: (pl.read_csv if self.type == FileType.CSV else pl.read_parquet)(
                            io.StringIO(new_content) if self.type == FileType.CSV else io.BytesIO(new_content)
                        )
                    )
                else:
                    new_df = new_content
                    
                if isinstance(existing_df, pl.DataFrame) and isinstance(new_df, pl.DataFrame):
                    if set(existing_df.columns) == set(new_df.columns):
                        merged_df = await loop.run_in_executor(None,
                            lambda: pl.concat([existing_df, new_df], how="vertical")
                        )
                        self.content = await loop.run_in_executor(None,
                            lambda: merged_df.write_csv() if self.type == FileType.CSV else merged_df.write_parquet()
                        )
        
        # Text-based files that support merging
        elif self.type in {
            FileType.PYTHON, FileType.JAVASCRIPT, FileType.JAVA,
            FileType.CPP, FileType.CSHARP, FileType.GO, FileType.RUST,
            FileType.PHP, FileType.RUBY, FileType.SWIFT, FileType.KOTLIN,
            FileType.SCALA, FileType.R, FileType.SHELL, FileType.SQL,
            FileType.HTML, FileType.CSS, FileType.MARKDOWN, FileType.TEXT,
            FileType.RTF, FileType.TEX, FileType.CONFIG
        }:
            if merge:
                # Convert both contents to string if needed
                existing = self.content if isinstance(self.content, str) else self.content.decode()
                new = new_content if isinstance(new_content, str) else new_content.decode()
                
                # Split into lines for better merging
                existing_lines = existing.splitlines(keepends=True)
                new_lines = new.splitlines(keepends=True)
                
                # Use SequenceMatcher for intelligent merging
                merged = []
                matcher = SequenceMatcher(None, existing_lines, new_lines)
                
                for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                    if tag == 'equal':
                        # Keep unchanged sections
                        merged.extend(existing_lines[i1:i2])
                    elif tag == 'replace':
                        # Smart merge for replaced sections
                        if i2 - i1 == j2 - j1:  # Same length replacement
                            merged.extend(new_lines[j1:j2])
                        else:
                            # Handle complex replacements
                            old_block = ''.join(existing_lines[i1:i2])
                            new_block = ''.join(new_lines[j1:j2])
                            if len(old_block.strip()) > 0 and len(new_block.strip()) > 0:
                                # If both blocks have content, use new version
                                merged.extend(new_lines[j1:j2])
                            else:
                                # If one block is empty, keep non-empty one
                                merged.extend(new_lines[j1:j2] if len(new_block.strip()) > 0 else existing_lines[i1:i2])
                    elif tag == 'delete':
                        # Keep deleted content only if it's not whitespace
                        if any(line.strip() for line in existing_lines[i1:i2]):
                            merged.extend(existing_lines[i1:i2])
                    elif tag == 'insert':
                        # Add new content
                        merged.extend(new_lines[j1:j2])
                
                # Join lines back together
                self.content = ''.join(merged)
                self.content_cache = None  # Clear cache
                
            else:
                self.content = new_content
                self.content_cache = None
        
        # Structured data files that support merging
        elif self.type in {FileType.JSON, FileType.YAML}:
            if merge:
                try:
                    existing_data = await self.data
                    if isinstance(new_content, (str, bytes)):
                        try:
                            if self.type == FileType.JSON:
                                new_data = json.loads(new_content)
                            else:  # YAML
                                new_data = yaml.safe_load(new_content)
                        except (json.JSONDecodeError, yaml.YAMLError) as e:
                            raise ValueError(f"Invalid {self.type} content: {str(e)}")
                    else:
                        new_data = new_content
                        
                    if isinstance(existing_data, dict) and isinstance(new_data, dict):
                        loop = asyncio.get_running_loop()
                        merged_data = await loop.run_in_executor(
                            None,
                            self._deep_merge_dicts,
                            existing_data,
                            new_data
                        )
                        self.content = json.dumps(merged_data) if self.type == FileType.JSON else yaml.dump(merged_data)
                    else:
                        self.content = new_content
                except Exception as e:
                    raise
            else:
                self.content = new_content
            self.content_cache = None
                
        # File types that don't support updates
        elif self.type in {
            FileType.IMAGE, FileType.AUDIO, FileType.VIDEO,
            FileType.ARCHIVE, FileType.BINARY, FileType.PDF,
            FileType.DOC, FileType.EXCEL, FileType.PPT,
            FileType.HDF5, FileType.PICKLE
        }:
            raise TypeError(f"File type {self.type} does not support content updates")
        
        # Default fallback for unknown types
        else:
            if merge:
                raise TypeError(f"File type {self.type} does not support content merging")
            self.content = new_content
            self.content_cache = None


    async def _deep_merge_dicts(self, dict1: Dict, dict2: Dict) -> Dict:
        """
        Helper method to recursively merge dictionaries.
        
        :param dict1: First dictionary
        :type dict1: Dict
        :param dict2: Second dictionary
        :type dict2: Dict
        :return: Merged dictionary
        :rtype: Dict
        """
        loop = asyncio.get_running_loop()
        merged = dict1.copy()
        
        for key, value in dict2.items():
            if (
                key in merged and 
                isinstance(merged[key], dict) and 
                isinstance(value, dict)
            ):
                merged[key] = await self._deep_merge_dicts(merged[key], value)
            else:
                merged[key] = value
                
        return merged

    async def save(self, directory: Optional[Union[str, Path]] = None, use_temp: bool = False) -> Path:
        """
        Save file content to disk in either a temporary or permanent location.
        
        :param directory: Directory to save file in, defaults to None
        :type directory: Optional[Union[str, Path]]
        :param use_temp: Whether to use temporary directory, defaults to False
        :type use_temp: bool
        :return: Path where file was saved
        :rtype: Path
        :raises ValueError: If both directory and use_temp are specified
        :raises IOError: If file cannot be saved
        """
        if directory and use_temp:
            raise ValueError("Cannot specify both directory and use_temp")
            
        # Generate filename if not available
        if not self.path:
            # Create hash of content for unique filename
            content_hash = hashlib.md5(
                str(self.content).encode() if isinstance(self.content, str)
                else self.content if isinstance(self.content, bytes)
                else str(self.content).encode()
            ).hexdigest()[:8]
            
            extension = self.type.value if self.type != FileType.UNKNOWN else '.txt'
            filename = f"file_{content_hash}.{extension}"
        else:
            filename = os.path.basename(self.path)

        # Get save directory
        if use_temp:
            save_dir = Path(tempfile.gettempdir()) / "flux_files"
        elif directory:
            save_dir = Path(directory)
        else:
            raise ValueError("Must specify either directory or use_temp=True")
            
        # Create directory if needed
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file
        save_path = save_dir / filename
        try:
            if isinstance(self.content, bytes):
                async with aiofiles.open(save_path, 'wb') as f:
                    await f.write(self.content)
            else:
                async with aiofiles.open(save_path, 'w') as f:
                    await f.write(str(self.content))
                    
            return save_path
            
        except Exception as e:
            raise IOError(f"Failed to save file: {str(e)}")


    @classmethod
    async def load(cls, path: Union[str, Path], delete_after: bool = False, **kwargs) -> 'File':
        """
        Load file from disk and optionally delete after loading.
        
        :param path: Path to file to load
        :type path: Union[str, Path]
        :param delete_after: Whether to delete file after loading, defaults to False
        :type delete_after: bool
        :param kwargs: Additional arguments to pass to File constructor
        :return: New File instance with loaded content
        :rtype: File
        :raises FileNotFoundError: If file doesn't exist
        :raises IOError: If file can't be read
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
            
        try:
            file = await cls.from_path(str(path), **kwargs)
            
            if delete_after:
                try:
                    if path.is_file():
                        path.unlink()
                    elif path.is_dir():
                        shutil.rmtree(path)
                except Exception as e:
                    print(f"Failed to delete file {path}: {str(e)}")
                    
            return file
            
        except Exception as e:
            raise IOError(f"Failed to load file: {str(e)}")


    async def cleanup(self) -> None:
        """
        Clean up any temporary files associated with this File instance.
        
        :raises IOError: If cleanup fails
        """
        if self.path:
            path = Path(self.path)
            if path.exists() and path.is_relative_to(tempfile.gettempdir()):
                try:
                    if path.is_file():
                        path.unlink()
                    elif path.is_dir():
                        shutil.rmtree(path)
                except Exception as e:
                    raise IOError(f"Failed to cleanup temporary file: {str(e)}")


    async def __aenter__(self) -> 'File':
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit with cleanup."""
        await self.cleanup()
        self.content_cache = None
        if hasattr(self, '_content'):
            del self._content

    @classmethod
    async def create(cls, **kwargs):
        """Factory method for creating File instances asynchronously"""
        self = cls.__new__(cls)
        await self.__init__(**kwargs)
        return self
    

    def __bool__(self) -> bool:
        """
        Verifies existance of object
        """
        return bool(self.content)

    async def serialize(self) -> str:
        """
        Serialize file to a string representation.
        
        :return: Serialized file string
        :rtype: str
        """
        # Handle binary content
        if isinstance(self.content, bytes):
            content = {
                'type': 'bytes',
                'data': base64.b64encode(self.content).decode('ascii')
            }
        else:
            content = self.content

        data = {
            'id': self.id,
            'content': content,
            'description': self.description,
            'path': str(self.path) if self.path else None,
            'type': self.type.value,
            'annotations': self._encode_bytes_in_dict(self.annotations) if self.annotations else {},
            'score': self.score
        }
        
        # Use orjson for serialization and ensure string output
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: orjson.dumps(data).decode('utf-8'))

    def _encode_bytes_in_dict(self, d: Dict) -> Dict:
        """Helper to encode bytes in dictionary values"""
        encoded = {}
        for k, v in d.items():
            if isinstance(v, bytes):
                encoded[k] = {
                    'type': 'bytes',
                    'data': base64.b64encode(v).decode('ascii')
                }
            elif isinstance(v, dict):
                encoded[k] = self._encode_bytes_in_dict(v)
            else:
                encoded[k] = v
        return encoded

    def _decode_bytes_in_dict(self, d: Dict) -> Dict:
        """Helper to decode bytes in dictionary values"""
        decoded = {}
        for k, v in d.items():
            if isinstance(v, dict) and v.get('type') == 'bytes':
                decoded[k] = base64.b64decode(v['data'])
            elif isinstance(v, dict):
                decoded[k] = self._decode_bytes_in_dict(v)
            else:
                decoded[k] = v
        return decoded

    @classmethod
    async def deserialize(cls, serialized_data: str) -> 'File':
        """
        Create File instance from serialized string.
        
        :param serialized_data: Serialized file data
        :type serialized_data: str
        :return: New File instance
        :rtype: File
        """
        # Use orjson for deserialization
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(None, orjson.loads, serialized_data)
        
        # Handle binary content
        if isinstance(data['content'], dict) and data['content'].get('type') == 'bytes':
            data['content'] = base64.b64decode(data['content']['data'])
        
        # Convert enum strings and paths
        data['type'] = FileType(data['type'])
        if data['path']:
            data['path'] = Path(data['path'])

        # Decode any bytes in annotations
        if data.get('annotations'):
            data['annotations'] = cls._decode_bytes_in_dict(data['annotations'])
        
        return cls(**data)

    async def token_length(self) -> int:
        """Get the token length asynchronously"""
        if not hasattr(self, '_token_length'):
            self._token_length = len(self.tokens)
        return self._token_length

    async def __len__(self) -> int:
        """Get the length of the file content"""
        if isinstance(self.content, bytes):
            return len(self.content)
        return len(str(self.content))
    
    @lru_cache(maxsize=1)
    async def tokens(self) -> List[int]:
        """Get the tokenized representation of the file content"""
        if isinstance(self.content, bytes):
            return await encode_async(str(len(self.content)) + " bytes")
        return await encode_async(str(self.content))
    
    async def token_length(self) -> int:
        """Get the token length asynchronously"""
        tokens = await self.tokens()
        return len(tokens)

    @lru_cache(maxsize=1)
    async def data_tokens(self) -> List[int]:
        """
        Get the tokenized representation of the file's data content.
        Separate from content tokens to handle data-specific tokenization.
        
        :return: List of token IDs for the file data
        :rtype: List[int]
        """
        if isinstance(self.data, bytes):
            return await encode_async(str(len(self.data)) + " bytes")
        return await encode_async(str(self.data))

    async def data_token_length(self) -> int:
        """
        Get the token count of the file's data content.
        Uses cached tokens when possible.
        
        :return: Number of tokens in the file data
        :rtype: int
        """
        tokens = await self.data_tokens()
        return len(tokens)

    @property
    def data(self) -> Any:
        """Get file data content"""
        return self._data

    @data.setter 
    def data(self, value: Any) -> None:
        """
        Set file data content and clear data token cache if content changes
        
        :param value: New data content
        :type value: Any
        """
        if not hasattr(self, '_data') or self._data != value:
            self.data_tokens.cache_clear()
        self._data = value

    async def clear_caches(self):
        """Clear all caches asynchronously"""
        self.tokens.cache_clear()  # For content tokens
        self.data_tokens.cache_clear()  # For data tokens
        self.content_cache = None
from pydantic import BaseModel, Field
from functools import lru_cache
from enum import Enum
from itertools import count
from typing import List, Dict, Any, Optional
from uuid import uuid4

from utils.shared.tokenizer import encode_async, decode_async

from agents.config.models import TruncationType


class IDFactory:
    _counter: count = count(0)
    
    @classmethod
    def next_id(cls) -> int:
        return next(cls._counter)


class Chunk(BaseModel):
    """
    Represents a chunk of content from a file.
    """
    id: int = Field(default_factory = lambda: IDFactory.next_id())
    content: str
    path: Optional[str] = None
    parent_id: Optional[int] = None
    file_metadata: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None

    async def token_length(self) -> int:
        """Get token length of chunk content."""
        tokens = await encode_async(self.content)
        return len(tokens)

    async def truncate(
        self,
        max_tokens: int,
        truncation_type: TruncationType
    ) -> str:
        """
        Truncate chunk content to max tokens.
        
        :param max_tokens: Maximum number of tokens
        :param truncation_type: Type of truncation to apply
        :return: Truncated content
        """
        tokens = await encode_async(self.content)
        if len(tokens) <= max_tokens:
            return self.content
            
        if truncation_type in [TruncationType.TOKEN_LIMIT, TruncationType.TRIM_MAX]:
            truncated_tokens = tokens[:max_tokens]
        else:  # For other types, preserve context around truncation
            # Keep some context from start and end
            context_tokens = max_tokens // 4
            middle_tokens = max_tokens - (2 * context_tokens)
            truncated_tokens = (
                tokens[:context_tokens] +
                tokens[len(tokens)//2 - middle_tokens//2:len(tokens)//2 + middle_tokens//2] +
                tokens[-context_tokens:]
            )
            
        return await decode_async(truncated_tokens)

    async def serialize(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "path": self.path,
            "score": self.score
        }
        
    @classmethod
    async def deserialize(cls, data: Dict[str, Any]) -> 'Chunk':
        return cls(
            id = data["id"],
            content = data["content"],
            path = data["path"],
            score = data["score"]
        )

    def __hash__(self) -> int:
        """Make Chunk hashable for caching"""
        return hash(self.id)
    
    def __eq__(self, other: object) -> bool:
        """Define equality for hashing"""
        if not isinstance(other, Chunk):
            return False
        return self.id == other.id

    @lru_cache(maxsize=1)
    async def tokens(self) -> List[int]:
        """Get tokenized content"""
        return await encode_async(self.content)


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

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="flux-agents",  # Make sure this name is unique and follows PyPI naming rules
    version="0.1",
    author="Christian de Frondeville, Arijit Nukala, Gubi Ganguly",
    author_email="your.email@example.com",
    description="Advanced AI Agent Framework with LLM integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/flux",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.9",
    install_requires=[
        "msgpack>=1.0.5",
        "google-generativeai",
        "hnswlib",
        "aiofiles",
        "aiohttp",
        "plotly",
        "polars",
        "numpy",
    ],
    extras_require={
        'torch': [
            "sentence-transformers[torch]>=2.5.0",
            "torch",
            "transformers"
        ]
    }
) 
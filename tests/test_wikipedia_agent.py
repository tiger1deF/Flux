import pytest
import asyncio
from ..agents.wikipedia_agent import WikipediaAgent

@pytest.mark.asyncio
async def test_wikipedia_query():
    """Test basic Wikipedia querying"""
    agent = WikipediaAgent()
    
    result = await agent.query("Python programming language")
    
    assert result["status"] == "success"
    assert "results" in result
    assert len(result["results"]) <= agent.max_results
    assert "timestamp" in result

@pytest.mark.asyncio
async def test_wikipedia_summary():
    """Test getting article summaries"""
    agent = WikipediaAgent()
    
    result = await agent.get_summary("Python (programming language)")
    
    assert result["status"] == "success"
    assert "summary" in result
    assert result["title"] == "Python (programming language)"
    assert "timestamp" in result

@pytest.mark.asyncio
async def test_invalid_query():
    """Test handling of invalid queries"""
    agent = WikipediaAgent()
    
    result = await agent.query("thisisnotarealwikipediaarticletitle12345")
    
    assert "error" in result
    assert result["status"] == "error"
    assert "timestamp" in result 
from typing import Dict, List, Any, Optional
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import WikipediaQueryRun
from pydantic import BaseModel, Field
from datetime import datetime

class WikipediaAgent(BaseModel):
    """
    Agent for querying Wikipedia using LangChain's Wikipedia tools.
    
    :ivar tool: Wikipedia query tool
    :type tool: WikipediaQueryRun
    :ivar max_results: Maximum number of results to return
    :type max_results: int
    """
    tool: WikipediaQueryRun = Field(default_factory=lambda: WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(
            top_k_results=3,
            lang="en",
            load_all_available_meta=True
        )
    ))
    max_results: int = Field(default=3)

    async def query(
        self,
        query: str,
        max_results: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Query Wikipedia for information.
        
        :param query: Search query
        :type query: str
        :param max_results: Optional override for max results
        :type max_results: Optional[int]
        :return: Search results and metadata
        :rtype: Dict[str, Any]
        """
        try:
            # Use the synchronous run method since WikipediaQueryRun doesn't support async
            results = self.tool.run(query)
            
            return {
                "query": query,
                "results": results[:max_results or self.max_results],
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
            
        except Exception as e:
            return {
                "query": query,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "status": "error"
            }

    async def get_summary(self, title: str) -> Dict[str, Any]:
        """
        Get summary of specific Wikipedia article.
        
        :param title: Article title
        :type title: str
        :return: Article summary and metadata
        :rtype: Dict[str, Any]
        """
        try:
            # Use the synchronous method since WikipediaAPIWrapper doesn't support async
            summary = self.tool.api_wrapper.get_article(title)
            
            return {
                "title": title,
                "summary": summary,
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
            
        except Exception as e:
            return {
                "title": title,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "status": "error"
            } 
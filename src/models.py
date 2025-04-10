"""Data models for the website agent."""
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any

class WebPage:
    """Represents a web page with its content and metadata."""
    
    def __init__(
        self, 
        url: str,
        title: str = "",
        content: str = "",
        links: List[str] = None
    ):
        self.url = url
        self.title = title
        self.content = content
        self.links = links or []
        
    def __str__(self):
        return f"WebPage(url={self.url}, title={self.title}, links={len(self.links)})"
    
    def __repr__(self):
        return self.__str__()

class ExplorationState(BaseModel):
    """State of the website exploration process."""
    
    main_url: str = Field(..., description="The main URL being explored")
    visited_urls: List[str] = Field(default_factory=list, description="URLs that have been visited")
    pending_urls: List[str] = Field(default_factory=list, description="URLs queued for exploration")
    page_data: Dict[str, Any] = Field(default_factory=dict, description="Data collected from visited pages")
    
class WebsiteInsight(BaseModel):
    """Represents insights gathered about a website."""
    
    url: str = Field(..., description="The main URL of the website")
    site_name: str = Field("", description="The name of the website")
    main_topics: List[str] = Field(default_factory=list, description="Main topics covered by the website")
    key_pages: List[Dict[str, str]] = Field(default_factory=list, description="Key pages with descriptions")
    summary: str = Field("", description="Overall summary of the website")

class LLMPrompt(BaseModel):
    """Represents a prompt for an LLM."""
    
    system_message: str = Field(..., description="System message for the LLM")
    user_message: str = Field(..., description="User message for the LLM")
    
class LLMsTxtSection(BaseModel):
    """Represents a section in the LLMs.txt file."""
    
    url: str = Field(..., description="URL for this section")
    content: str = Field(..., description="Content for this section")

class LLMsTxtFile(BaseModel):
    """Represents the structure of an LLMs.txt file."""
    
    title: str = Field(..., description="Title for the LLMs.txt file")
    sections: List[LLMsTxtSection] = Field(default_factory=list, description="Sections in the file")
    
    def to_text(self) -> str:
        """Convert the model to the LLMs.txt format."""
        lines = [self.title]
        
        for section in self.sections:
            lines.append(f"{section.url}: {section.content}")
            
        return "\n".join(lines) 
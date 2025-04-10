"""Agent components for website exploration and LLMs.txt generation."""
from typing import Dict, List, Any, Annotated, TypedDict, Optional, Union
from urllib.parse import urljoin, urlparse
import os

from langchain.schema import HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langchain.pydantic_v1 import BaseModel, Field

from .config import MAX_URLS_TO_EXPLORE, VERBOSE
from .utils import (
    fetch_url, extract_links, extract_main_content, 
    normalize_url, filter_relevant_links, extract_title, clean_text
)
from .models import WebPage, ExplorationState, LLMsTxtFile, LLMsTxtSection
from .llm import analyze_webpage_content, create_llms_txt_content, plan_exploration_strategy

class AgentState(TypedDict):
    """State managed by the agent graph."""
    url: str
    visited: List[str]
    to_visit: List[str]
    pages: Dict[str, WebPage]
    analyzed_pages: List[Dict[str, Any]]
    llms_txt: Optional[str]
    error: Optional[str]
    priority_urls: List[str]  # 우선 탐색할 URL 목록
    exploration_plan: Optional[Dict[str, Any]]  # 탐색 계획
    important_sections: List[str]  # 중요한 섹션/키워드

def initialize_agent(state: AgentState) -> Dict:
    """Initialize the agent state with the provided URL.
    
    Args:
        state: The current agent state
        
    Returns:
        Updated agent state
    """
    url = state["url"]
    if not url:
        return {
            "error": "No URL provided. Please provide a URL to explore."
        }
    
    # Normalize the URL
    url = normalize_url(url)
    
    # 탐색 계획 생성
    try:
        exploration_plan = plan_exploration_strategy(url)
        priority_paths = exploration_plan["paths"] if isinstance(exploration_plan, dict) and "paths" in exploration_plan else []
        
        # 우선 탐색할 URL 생성
        priority_urls = []
        for path in priority_paths:
            if path.startswith("http"):
                priority_urls.append(normalize_url(path))
            else:
                # 상대 경로인 경우 절대 URL로 변환
                priority_urls.append(normalize_url(urljoin(url, path)))
        
        # 중요 섹션/키워드 초기화
        important_sections = exploration_plan.get("important_sections", []) if isinstance(exploration_plan, dict) else []
    except Exception as e:
        if VERBOSE:
            print(f"Error creating exploration plan: {str(e)}")
        priority_urls = []
        important_sections = []
        exploration_plan = None
    
    # Initialize the exploration state
    return {
        "url": url,
        "visited": [],
        "to_visit": [url] + priority_urls,
        "pages": {},
        "analyzed_pages": [],
        "llms_txt": None,
        "error": None,
        "priority_urls": priority_urls,
        "exploration_plan": exploration_plan,
        "important_sections": important_sections
    }

def explore_webpage(state: AgentState) -> Dict:
    """Explore a webpage and extract its content and links.
    
    Args:
        state: The current agent state
        
    Returns:
        Updated agent state
    """
    # Check if there are URLs to visit
    if not state["to_visit"]:
        return {
            "error": "No more URLs to visit."
        }
    
    # Get the next URL to visit (우선 탐색 URL이 있으면 그것부터 방문)
    url = None
    to_visit = state["to_visit"]
    
    # 우선순위 URL 먼저 방문
    for priority_url in state["priority_urls"]:
        if priority_url in to_visit:
            url = priority_url
            to_visit.remove(priority_url)
            break
    
    # 우선순위 URL이 없으면 일반 URL 방문
    if url is None and to_visit:
        url = to_visit.pop(0)
    
    # Skip if already visited
    if url in state["visited"]:
        return {
            **state,
            "to_visit": to_visit
        }
    
    # Mark as visited
    visited = state["visited"] + [url]
    
    try:
        # Fetch the webpage
        html_content, content_type = fetch_url(url)
        
        # Skip if not HTML
        if "text/html" not in content_type:
            return {
                **state,
                "visited": visited,
                "to_visit": to_visit
            }
        
        # Extract information from the webpage
        title = extract_title(html_content)
        content = extract_main_content(html_content)
        content = clean_text(content)
        
        # Extract links from the webpage
        links = extract_links(html_content, url)
        
        # Filter links to keep only those from the same domain
        parsed_url = urlparse(url)
        base_domain = parsed_url.netloc
        relevant_links = filter_relevant_links(links, base_domain)
        
        # 중요 섹션을 기반으로 링크 필터링
        important_links = []
        regular_links = []
        
        for link in relevant_links:
            # 링크가 중요 섹션/키워드를 포함하는지 확인
            is_important = False
            for section in state["important_sections"]:
                if section.lower() in link.lower():
                    is_important = True
                    break
            
            if is_important:
                important_links.append(link)
            else:
                regular_links.append(link)
        
        # Create a WebPage object
        page = WebPage(
            url=url,
            title=title,
            content=content,
            links=relevant_links
        )
        
        # Update state
        pages = {**state["pages"]}
        pages[url] = page
        
        # Add new links to visit
        # 중요 링크를 먼저 추가
        for link in important_links:
            normalized_link = normalize_url(link)
            if normalized_link not in visited and normalized_link not in to_visit:
                to_visit.append(normalized_link)
                
                # Limit the number of URLs to explore
                if len(to_visit) + len(visited) >= MAX_URLS_TO_EXPLORE:
                    break
        
        # 그 다음에 일반 링크 추가
        if len(to_visit) + len(visited) < MAX_URLS_TO_EXPLORE:
            for link in regular_links:
                normalized_link = normalize_url(link)
                if normalized_link not in visited and normalized_link not in to_visit:
                    to_visit.append(normalized_link)
                    
                    # Limit the number of URLs to explore
                    if len(to_visit) + len(visited) >= MAX_URLS_TO_EXPLORE:
                        break
        
        if VERBOSE:
            print(f"Explored {url}, found {len(relevant_links)} relevant links")
            print(f"To visit: {len(to_visit)}, Visited: {len(visited)}")
            
        return {
            **state,
            "visited": visited,
            "to_visit": to_visit,
            "pages": pages
        }
    except Exception as e:
        if VERBOSE:
            print(f"Error exploring {url}: {str(e)}")
        
        # Skip this URL but continue with others
        return {
            **state,
            "visited": visited,
            "to_visit": to_visit
        }

def analyze_pages(state: AgentState) -> Dict:
    """Analyze the content of explored webpages.
    
    Args:
        state: The current agent state
        
    Returns:
        Updated agent state
    """
    # Check if there are pages to analyze
    if not state["pages"]:
        return {
            "error": "No pages to analyze."
        }
    
    # Get pages that haven't been analyzed yet
    pages_to_analyze = []
    for url, page in state["pages"].items():
        # Check if this page has already been analyzed
        already_analyzed = any(p["url"] == url for p in state["analyzed_pages"])
        
        if not already_analyzed:
            pages_to_analyze.append(page)
    
    if not pages_to_analyze:
        return state
    
    # Analyze each page
    new_analyzed_pages = []
    
    for page in pages_to_analyze:
        try:
            if VERBOSE:
                print(f"Analyzing {page.url}")
                
            analysis = analyze_webpage_content(
                title=page.title,
                content=page.content,
                url=page.url
            )
            
            new_analyzed_pages.append(analysis)
        except Exception as e:
            if VERBOSE:
                print(f"Error analyzing {page.url}: {str(e)}")
    
    # Update state
    analyzed_pages = state["analyzed_pages"] + new_analyzed_pages
    
    return {
        **state,
        "analyzed_pages": analyzed_pages
    }

def create_llms_txt(state: AgentState) -> Dict:
    """Create an LLMs.txt file based on the analyzed pages.
    
    Args:
        state: The current agent state
        
    Returns:
        Updated agent state
    """
    # Check if there are analyzed pages
    if not state["analyzed_pages"]:
        return {
            "error": "No analyzed pages available."
        }
    
    try:
        # Create the LLMs.txt content
        llms_txt_content = create_llms_txt_content(
            url=state["url"],
            analyzed_pages=state["analyzed_pages"]
        )
        
        return {
            **state,
            "llms_txt": llms_txt_content
        }
    except Exception as e:
        if VERBOSE:
            print(f"Error creating LLMs.txt: {str(e)}")
        
        return {
            **state,
            "error": f"Error creating LLMs.txt: {str(e)}"
        }

def should_continue_exploration(state: AgentState) -> Union[str, List[str]]:
    """Decide whether to continue exploring or to stop.
    
    Args:
        state: The current agent state
        
    Returns:
        Next step in the workflow
    """
    # Stop if there's an error
    if state["error"]:
        return END
    
    # Stop if we already created LLMs.txt
    if state["llms_txt"]:
        return END
    
    # Stop if we've explored enough pages
    if len(state["visited"]) >= MAX_URLS_TO_EXPLORE:
        return "analyze_pages"
    
    # Continue exploring if there are more URLs to visit
    if state["to_visit"]:
        return "explore_webpage"
    
    # Analyze pages if we've explored all URLs
    return "analyze_pages"

def should_create_llms_txt(state: AgentState) -> str:
    """Decide whether to create LLMs.txt.
    
    Args:
        state: The current agent state
        
    Returns:
        Next step in the workflow
    """
    # Stop if there's an error
    if state["error"]:
        return END
    
    # Stop if we already created LLMs.txt
    if state["llms_txt"]:
        return END
    
    # Create LLMs.txt if we've analyzed pages
    if state["analyzed_pages"]:
        return "create_llms_txt"
    
    # Go back to exploration if we haven't analyzed any pages
    if state["to_visit"]:
        return "explore_webpage"
    
    # End if there's nothing more to do
    return END

def create_website_agent() -> StateGraph:
    """Create and return the website exploration agent graph.
    
    Returns:
        StateGraph instance for the website agent
    """
    # Create the graph
    graph = StateGraph(AgentState)
    
    # Add the nodes
    graph.add_node("initialize_agent", initialize_agent)
    graph.add_node("explore_webpage", explore_webpage)
    graph.add_node("analyze_pages", analyze_pages)
    graph.add_node("create_llms_txt", create_llms_txt)
    
    # Add the edges
    graph.set_entry_point("initialize_agent")
    graph.add_edge("initialize_agent", "explore_webpage")
    graph.add_conditional_edges(
        "explore_webpage",
        should_continue_exploration,
        {
            "explore_webpage": "explore_webpage",
            "analyze_pages": "analyze_pages",
            END: END
        }
    )
    graph.add_conditional_edges(
        "analyze_pages",
        should_create_llms_txt,
        {
            "create_llms_txt": "create_llms_txt",
            "explore_webpage": "explore_webpage",
            END: END
        }
    )
    graph.add_edge("create_llms_txt", END)
    
    return graph.compile() 
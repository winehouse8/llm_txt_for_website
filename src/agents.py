"""Agent components for website exploration and LLMs.txt generation."""
from typing import Dict, List, Any, Annotated, TypedDict, Optional, Union, Set, Tuple
from urllib.parse import urljoin, urlparse
import os
import re
from collections import Counter

from langchain.schema import HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langchain.pydantic_v1 import BaseModel, Field

from .config import MAX_URLS_TO_EXPLORE, VERBOSE, MAX_DEPTH, MIN_CONTENT_LENGTH, MAX_CONTENT_LENGTH, MAX_LINKS_PER_PAGE, SUMMARY_MAX_LENGTH
from .utils import (
    fetch_url, extract_links, extract_main_content, 
    normalize_url, filter_relevant_links, extract_title, clean_text,
    should_exclude_url
)
from .models import WebPage, ExplorationState, LLMsTxtFile, LLMsTxtSection
from .llm import analyze_webpage_content, create_llms_txt_content, plan_exploration_strategy, extract_important_sections, query_llm, LLMPrompt

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
    section_counts: Dict[str, int]  # 섹션별 발견 빈도 추적
    last_page_sections: List[str]  # 마지막 페이지에서 발견된 중요 섹션
    visited_patterns: Dict[str, int]  # 방문한 URL 패턴과 빈도

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
        "important_sections": important_sections,
        "section_counts": {},  # 섹션별 발견 빈도 추적
        "last_page_sections": [],  # 마지막으로 방문한 페이지에서 발견된 중요 섹션
        "visited_patterns": {}  # 방문한 URL 패턴 추적
    }

def extract_url_pattern(url: str) -> str:
    """URL에서 패턴을 추출합니다.
    
    예: https://news.google.com/sections/business?hl=en-US -> /sections/business
    
    Args:
        url: 추출할 URL
        
    Returns:
        추출된 패턴
    """
    parsed = urlparse(url)
    path = parsed.path
    
    # 파라미터나 쿼리 제거
    if not path or path == '/':
        return '/'
    
    # URL 패턴 정규화
    parts = path.rstrip('/').split('/')
    if len(parts) > 3:  # 너무 깊은 경로는 잘라서 패턴화
        return '/'.join(parts[:3])
    
    return path

def score_link_importance(
    link: str, 
    url: str, 
    base_url: str, 
    important_sections: List[str],
    page_title: str,
    visited_urls: Set[str],
    sections_frequency: Dict[str, int]
) -> float:
    """링크의 중요도를 점수화합니다.
    
    Args:
        link: 평가할 링크 URL
        url: 현재 페이지 URL
        base_url: 기본 도메인 URL
        important_sections: 중요한 섹션이나 키워드 목록
        page_title: 현재 페이지 제목
        visited_urls: 이미 방문한 URL 집합
        sections_frequency: 섹션 빈도수 사전
        
    Returns:
        링크 중요도 점수 (0.0~10.0)
    """
    from urllib.parse import urlparse, parse_qs
    import re
    
    # 이미 방문한 URL은 0점 처리
    if link in visited_urls:
        return 0.0
        
    parsed_link = urlparse(link)
    parsed_current = urlparse(url)
    parsed_base = urlparse(base_url)
    
    # 기본 점수 설정
    score = 5.0
    
    # 1. URL 깊이 평가 (깊은 페이지는 통상 더 구체적인 내용 포함)
    path_segments = parsed_link.path.strip('/').split('/')
    depth = len(path_segments)
    
    if depth <= 1:
        score += 0.5  # 최상위 페이지는 약간 가산점
    elif 2 <= depth <= 3:
        score += 1.0  # 적당한 깊이는 더 높은 가산점
    elif depth > 3:
        score -= (depth - 3) * 0.5  # 너무 깊은 페이지는 감점
    
    # 2. 중요 키워드 포함 여부 평가
    link_lower = link.lower()
    path_lower = parsed_link.path.lower()
    
    # 중요 섹션 기반 점수 부여 (빈도수 고려)
    for section in important_sections:
        section_lower = section.lower()
        
        # 섹션 빈도수 가중치 계산 (자주 등장한 섹션이 더 중요)
        frequency = sections_frequency.get(section_lower, 1)
        frequency_weight = min(frequency / 2, 2.0)  # 최대 2.0으로 제한
        
        # URL 경로에 중요 키워드가 있는 경우
        if section_lower in path_lower:
            exact_match = False
            
            # 정확한 단어 일치 확인 (경계 검사)
            for segment in path_segments:
                segment_lower = segment.lower()
                if section_lower == segment_lower:
                    score += 2.0 * frequency_weight  # 정확한 일치는 높은 점수
                    exact_match = True
                    break
                elif section_lower in segment_lower:
                    # 부분 일치에 대한 가중치 부여
                    segment_words = re.split(r'[-_.]', segment_lower)
                    if section_lower in segment_words:
                        score += 1.5 * frequency_weight
                        exact_match = True
                        break
            
            if not exact_match:
                score += 0.5 * frequency_weight  # 부분 문자열로 포함
    
    # 3. 현재 페이지 제목과 링크 관련성 확인
    if page_title:
        title_words = re.split(r'[\s\-_|:,.]', page_title.lower())
        title_words = [w for w in title_words if w and len(w) > 3]
        
        for word in title_words:
            if word in path_lower:
                score += 0.8  # 제목 키워드가 링크에 포함되면 가산점
    
    # 4. 도메인 일치 여부 평가
    if parsed_link.netloc == parsed_base.netloc:
        score += 1.0  # 같은 도메인 내 페이지 가산점
    elif parsed_link.netloc == parsed_current.netloc:
        score += 0.5  # 현재 페이지와 같은 도메인 가산점
    else:
        score -= 3.0  # 외부 링크는 감점
    
    # 5. 피해야 할 패턴 확인
    avoid_patterns = [
        r'/login', r'/signup', r'/register', 
        r'/cart', r'/checkout', r'/purchase',
        r'/account', r'/profile', r'/user', 
        r'/search', r'/tag/', r'/category/',
        r'/terms', r'/privacy', r'/about',
        r'/contact', r'/support', r'/help',
        r'/feed', r'/rss', r'/atom', 
        r'/comment', r'/trackback',
        r'/page/\d+', r'/\d{4}/\d{2}',  # 페이지네이션, 날짜 아카이브
        r'\.(jpg|jpeg|png|gif|pdf|zip|tar|gz|rar|exe|dmg|apk)$'  # 미디어 파일
    ]
    
    for pattern in avoid_patterns:
        if re.search(pattern, link_lower):
            score -= 2.0
            break
    
    # 6. 쿼리 매개변수 평가
    if parsed_link.query:
        query_params = parse_qs(parsed_link.query)
        
        # 일부 쿼리 매개변수는 가치가 낮을 수 있음 (예: 추적 매개변수)
        low_value_params = ['utm_source', 'utm_medium', 'utm_campaign', 'fbclid', 'ref']
        has_low_value_params = any(param in query_params for param in low_value_params)
        
        if has_low_value_params:
            score -= 1.0
        
        # 쿼리 매개변수가 많을수록 감점
        if len(query_params) > 2:
            score -= 0.5 * (len(query_params) - 2)
    
    # 최종 점수 범위 제한 (0.0 ~ 10.0)
    return max(min(score, 10.0), 0.0)

def explore_webpage(state: AgentState) -> Dict:
    """웹페이지를 탐색하고 중요한 링크를 찾아냅니다.
    
    Args:
        state: The current agent state
        
    Returns:
        Updated agent state
    """
    # Ensure there are URLs to explore
    if not state["to_visit"]:
        return state
    
    # Get the next URL to visit
    url = state["to_visit"][0]
    to_visit = state["to_visit"][1:]
    
    # Check if URL has already been visited
    if url in state["visited"]:
        return {
            **state,
            "to_visit": to_visit
        }
    
    # Mark as visited
    visited = state["visited"] + [url]
    
    try:
        # Fetch the webpage
        html_content = fetch_url(url)
        
        # Extract information from the webpage
        title = extract_title(html_content)
        content = extract_main_content(html_content)
        content = clean_text(content)
        
        # URL 패턴 추적 업데이트
        pattern = extract_url_pattern(url)
        visited_patterns = {**state["visited_patterns"]}
        visited_patterns[pattern] = visited_patterns.get(pattern, 0) + 1
        
        # 현재 페이지에서 중요 섹션/키워드 추출
        page_sections = extract_important_sections(title, content, url)
        
        # 전체 중요 섹션 리스트 업데이트
        current_sections = set(state["important_sections"])
        new_sections = [section for section in page_sections if section not in current_sections]
        
        # 섹션 발견 빈도 업데이트
        section_counts = {**state["section_counts"]}
        for section in page_sections:
            section_counts[section] = section_counts.get(section, 0) + 1
        
        # 중요 섹션 점수 계산 (섹션 발견 빈도 및 현재 페이지 관련성 기반)
        section_scores = {}
        for section in set(list(current_sections) + new_sections):
            # 기본 점수 시작
            score = 1.0
            # 발견 빈도에 따른 가중치
            count = section_counts.get(section, 0)
            score *= (1.0 + (count * 0.1))
            # 현재 페이지에서 발견된 섹션은 추가 가중치
            if section in page_sections:
                score *= 1.5
            section_scores[section] = score
        
        # 점수 기준으로 섹션 정렬
        sorted_sections = sorted(
            list(set(list(current_sections) + new_sections)),
            key=lambda s: section_scores.get(s, 0),
            reverse=True
        )
        
        # 상위 MAX_IMPORTANT_SECTIONS 개 섹션만 유지
        MAX_IMPORTANT_SECTIONS = 30
        updated_important_sections = sorted_sections[:MAX_IMPORTANT_SECTIONS]
        
        # Extract links from the webpage
        links = extract_links(html_content, url)
        
        # Filter links to keep only those from the same domain
        parsed_url = urlparse(url)
        base_domain = parsed_url.netloc
        relevant_links = filter_relevant_links(links, base_domain)
        
        # 모든 링크에 대한 중요도 점수 계산 (개선된 스코어링 활용)
        visited_urls_set = set(visited)
        scored_links = []
        for link in relevant_links:
            importance_score = score_link_importance(
                link=link, 
                url=url,
                base_url=url,
                important_sections=updated_important_sections,
                page_title=title,
                visited_urls=visited_urls_set,
                sections_frequency=section_counts
            )
            
            scored_links.append((link, importance_score))
        
        # 점수 기준으로 정렬 (높은 점수 순)
        scored_links.sort(key=lambda x: x[1], reverse=True)
        
        # 정렬된 링크 추출
        sorted_links = [link for link, score in scored_links]
        
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
        
        # 링크 추가 (이제 중요도 점수 기반으로 정렬됨)
        for link in sorted_links:
            normalized_link = normalize_url(link)
            if normalized_link not in visited and normalized_link not in to_visit:
                to_visit.append(normalized_link)
        
                # Limit the number of URLs to explore
                if len(to_visit) + len(visited) >= MAX_URLS_TO_EXPLORE:
                    break
        
        if VERBOSE:
            print(f"Explored {url}, found {len(relevant_links)} relevant links")
            print(f"To visit: {len(to_visit)}, Visited: {len(visited)}")
            if new_sections:
                print(f"Found new important sections: {new_sections}")
            
        return {
            **state,
            "visited": visited,
            "to_visit": to_visit,
            "pages": pages,
            "important_sections": updated_important_sections,
            "section_counts": section_counts,
            "last_page_sections": page_sections,
            "visited_patterns": visited_patterns
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

def filter_links(links: List[str], visited_urls: Set[str], base_url: str) -> List[str]:
    """링크를 필터링하여 탐색 가능한 링크만 반환합니다.
    
    Args:
        links: 필터링할 링크 목록
        visited_urls: 이미 방문한 URL 집합
        base_url: 기본 URL
        
    Returns:
        필터링된 링크 목록
    """
    filtered_links = []
    for link in links:
        # 이미 방문한 URL 제외
        if link in visited_urls:
            continue
            
        # 현재 URL과 동일한 경우 제외
        if link == base_url:
            continue
        
        # 필터링 기준에 맞지 않는 URL 제외
        if should_exclude_url(link):
            continue
            
        filtered_links.append(link)
    
    return filtered_links

def score_links(links: List[str], important_sections: Set[str], sections_frequency: Counter) -> List[Dict[str, Any]]:
    """링크의 중요도를 계산하여 점수를 매깁니다.
    
    Args:
        links: 점수를 매길 링크 목록
        important_sections: 중요 섹션 집합
        sections_frequency: 섹션 빈도수
        
    Returns:
        링크와 점수를 포함하는 딕셔너리 목록
    """
    scored_links = []
    
    for link in links:
        # 기본 점수 설정
        score = 1.0
        
        # URL의 깊이에 따라 점수 조정
        parsed = urlparse(link)
        path_segments = parsed.path.strip('/').split('/')
        depth = len(path_segments)
        
        # 경로 깊이에 따른 점수 조정
        if depth <= 1:
            score += 0.5  # 메인 페이지에 가까운 경우 가산점
        elif depth > 3:
            score -= 0.3 * (depth - 3)  # 너무 깊은 경로는 감점
        
        # 중요 섹션과의 일치 여부 확인
        link_lower = link.lower()
        for section in important_sections:
            section_lower = section.lower()
            if section_lower in link_lower:
                # 섹션 빈도수에 따른 가중치 적용
                frequency = sections_frequency.get(section_lower, 1)
                score += min(0.5 * frequency, 2.0)  # 최대 2점까지 가산
        
        # 특정 패턴에 따른 점수 조정
        if any(pattern in link_lower for pattern in ['article', 'story', 'news', 'topic']):
            score += 1.0
        
        if any(pattern in link_lower for pattern in ['login', 'signup', 'register', 'account']):
            score -= 1.0
        
        scored_links.append({
            'url': link,
            'score': max(0.1, score)  # 최소 점수 보장
        })
    
    return scored_links

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
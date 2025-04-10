"""LLM utilities for working with OpenAI."""
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, ChatMessage
import json
import re

from .config import OPENAI_API_KEY, MODEL_NAME, VERBOSE
from .models import LLMPrompt

def get_llm():
    """Create and return a ChatOpenAI instance."""
    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=MODEL_NAME,
        temperature=0.3,
    )

def query_llm(prompt: Optional[LLMPrompt] = None, messages: Optional[List[Dict[str, str]]] = None, max_tokens: int = None) -> str:
    """LLM에 질의하여 응답을 받습니다.
    
    Args:
        prompt: LLMPrompt 객체 (시스템 메시지와 사용자 메시지 포함)
        messages: 메시지 목록 (역할과 내용 포함한 딕셔너리 리스트)
        max_tokens: 응답 최대 토큰 수 제한 (선택 사항)
        
    Returns:
        LLM 응답 텍스트
        
    Raises:
        ValueError: prompt와 messages가 모두 None인 경우
    """
    if prompt is None and messages is None:
        raise ValueError("Either 'prompt' or 'messages' must be provided")
    
    llm = get_llm()
    
    if prompt:
        # LLMPrompt 객체를 사용하는 경우
        if VERBOSE:
            #print(f"Querying LLM with system message: {prompt.system_message[:50]}...")
            print(f"User message: {prompt.user_message[:100]}...")
        
        messages_list = [
            SystemMessage(content=prompt.system_message),
            HumanMessage(content=prompt.user_message)
        ]
    else:
        # 메시지 리스트를 사용하는 경우
        messages_list = []
        
        for msg in messages:
            role = msg.get("role", "").lower()
            content = msg.get("content", "")
            
            if role == "system":
                messages_list.append(SystemMessage(content=content))
            elif role == "user":
                messages_list.append(HumanMessage(content=content))
            elif role == "assistant":
                messages_list.append(ChatMessage(role="assistant", content=content))
                
        if VERBOSE and messages_list:
            print(f"Querying LLM with {len(messages_list)} messages...")
    
    # 최대 토큰 수 설정
    llm_kwargs = {}
    if max_tokens:
        llm_kwargs["max_tokens"] = max_tokens
    
    # LLM 호출
    response = llm.invoke(messages_list, **llm_kwargs)
    return response.content

def analyze_webpage_content(title: str, content: str, url: str) -> Dict[str, Any]:
    """Analyze webpage content to extract structured information.
    
    Args:
        title: The webpage title
        content: The webpage content
        url: The webpage URL
        
    Returns:
        Dictionary with structured information about the webpage
    """
    system_message = """
    You are an expert AI assistant that analyzes web pages to extract key information.
    Your task is to analyze the provided web page content and extract the most important 
    points that would be valuable for an LLM (Large Language Model) to understand when 
    helping users with questions about this topic.
    
    Provide your analysis in JSON format with these keys:
    - main_topic: The main topic of the page
    - key_points: A list of the most important points (maximum 5)
    - relevance_for_llm: Why this information would be valuable for an LLM
    - useful_for_queries: Types of user queries this information would help with
    """
    
    user_message = f"""
    URL: {url}
    Title: {title}
    
    Content:
    {content[:4000]}  # Truncate to avoid token limits
    
    Please analyze this content and extract the key information.
    """
    
    prompt = LLMPrompt(system_message=system_message, user_message=user_message)
    response = query_llm(prompt)
    
    # In a real implementation, we would parse the JSON response
    # For simplicity, we're returning the raw response here
    return {
        "url": url,
        "title": title,
        "analysis": response
    }

def extract_important_sections(title: str, content: str, url: str) -> List[str]:
    """
    웹페이지의 내용을 분석하여 중요한 섹션, 카테고리, 주제 및 키워드를 추출합니다.
    
    Args:
        title: 웹페이지 제목
        content: 웹페이지 내용
        url: 웹페이지 URL
        
    Returns:
        중요한 섹션, 키워드 목록
    """
    # 내용이 너무 길면 일부만 사용
    content_sample = content[:5000] if len(content) > 5000 else content
    
    system_message = """
    당신은 웹페이지에서 중요한 섹션, 주제 및 키워드를 식별하는 분석 전문가입니다.
    사용자가 제공하는 웹페이지 내용을 분석하고 다음 사항을 식별하세요:
    
    1. 이 페이지의 주요 카테고리 또는 주제(예: 기술, 정치, 건강, 금융, 교육 등)
    2. 이 페이지에서 논의되는 구체적인 개념, 제품, 이벤트 또는 엔티티
    3. 이 페이지에서 언급된 중요한 용어, 키워드 또는 해시태그
    4. 탐색 중 우선시해야 할 가치 있는 콘텐츠 영역
    
    응답은 JSON 형식으로 작성하세요:
    {
        "sections": ["섹션1", "섹션2", "섹션3", ...],
        "explanation": "이러한 섹션이 중요한 이유에 대한 간단한 설명"
    }
    
    참고: "sections" 목록에는 5-10개의 중요한 용어나 개념을 포함해야 하며, 탐색 시 우선순위를 정하는 데 도움이 되어야 합니다.
    """
    
    user_message = f"""
    다음 웹페이지에서 중요한 섹션과 키워드를 식별하세요.
    
    URL: {url}
    제목: {title}
    
    내용:
    {content_sample}
    """
    
    prompt = LLMPrompt(
        system_message=system_message,
        user_message=user_message
    )
    
    try:
        response = query_llm(prompt=prompt, max_tokens=1000)
        
        # JSON 응답 파싱 시도
        try:
            import json
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                if "sections" in result and isinstance(result["sections"], list):
                    return result["sections"]
        except:
            pass
        
        # JSON 파싱 실패 시 대체 방법으로 텍스트에서 섹션 추출
        sections = []
        lines = response.split('\n')
        for line in lines:
            if ':' in line or '-' in line or '•' in line:
                parts = re.split(r'[:\-•]', line, 1)
                if len(parts) > 1 and parts[0].strip():
                    section = parts[0].strip()
                    # 따옴표 제거
                    section = section.strip('"\'')
                    if section and len(section) < 50:  # 너무 긴 항목 필터링
                        sections.append(section)
        
        # 항목이 충분하지 않으면 텍스트에서 키워드 추출
        if len(sections) < 3:
            words = re.findall(r'[""]([^""]+)["""]', response)
            words.extend(re.findall(r'[""]([^""]+)["""]', response))
            for word in words:
                if word and word not in sections and len(word) < 50:
                    sections.append(word)
        
        return sections[:10]  # 최대 10개 항목으로 제한
    
    except Exception as e:
        if VERBOSE:
            print(f"중요 섹션 추출 중 오류 발생: {str(e)}")
        return []

def extract_fallback_keywords(title: str, content: str, url: str) -> List[str]:
    """LLM 기반 키워드 추출이 실패했을 때 사용하는 대체 방법.
    제목, URL, 콘텐츠에서 키워드 추출.
    
    Args:
        title: 페이지 제목
        content: 페이지 콘텐츠
        url: 페이지 URL
        
    Returns:
        추출된 키워드 목록
    """
    import re
    from collections import Counter
    from urllib.parse import urlparse
    
    # 스톱워드 목록 (필터링할 일반적인 단어들)
    stop_words = set(['the', 'and', 'or', 'in', 'on', 'at', 'to', 'a', 'an', 'by', 'for', 
                      'with', 'about', 'as', 'of', 'from', 'that', 'this', 'is', 'are', 
                      'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 
                      'does', 'did', 'but', 'if', 'then', 'else', 'when', 'where', 'which', 
                      'who', 'whom', 'whose', 'what', 'how', 'why', 'all', 'any', 'both', 
                      'each', 'few', 'more', 'most', 'some', 'such', 'no', 'nor', 'not', 
                      'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 
                      'just', 'should', 'now'])
    
    all_keywords = []
    
    # 1. 제목에서 키워드 추출
    if title:
        title_words = re.split(r'[\s\-_:,.]', title.lower())
        title_words = [w for w in title_words if w and len(w) > 2 and w not in stop_words]
        all_keywords.extend(title_words)
    
    # 2. URL에서 키워드 추출
    if url:
        parsed_url = urlparse(url)
        path_segments = parsed_url.path.split('/')
        
        for segment in path_segments:
            if segment and len(segment) > 2:
                # 파일 확장자 제거
                segment = re.sub(r'\.\w+$', '', segment)
                
                # 케밥 케이스, 스네이크 케이스 처리
                words = re.split(r'[-_]', segment.lower())
                words = [w for w in words if w and len(w) > 2 and w not in stop_words]
                all_keywords.extend(words)
    
    # 3. 콘텐츠에서 주요 단어 추출
    if content:
        # 짧은 내용의 경우 모든 단어 추출, 긴 내용은 앞부분만 사용
        content_sample = content[:1500] if len(content) > 1500 else content
        
        # 특수문자 제거하고 단어 분할
        content_words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9]{2,}\b', content_sample.lower())
        
        # 스톱워드 제거 및 길이 필터링
        content_words = [w for w in content_words if w not in stop_words and len(w) > 2]
        
        # 단어 빈도수 계산
        word_counter = Counter(content_words)
        
        # 가장 빈번한 단어 추출 (최대 10개)
        common_words = [word for word, _ in word_counter.most_common(10)]
        all_keywords.extend(common_words)
    
    # 중복 제거 및 최대 15개로 제한
    unique_keywords = []
    for kw in all_keywords:
        if kw not in unique_keywords:
            unique_keywords.append(kw)
    
    return unique_keywords[:15]

def create_llms_txt_content(url: str, analyzed_pages: List[Dict[str, Any]]) -> str:
    """Create content for an LLMs.txt file.
    
    Args:
        url: The main URL being explored
        analyzed_pages: List of analyzed pages
        
    Returns:
        Formatted content for an LLMs.txt file
    """
    system_message = """
    You are an expert AI assistant that creates concise and informative summaries for LLMs.txt files.
    
    An LLMs.txt file is a special format designed to help Large Language Models understand when they 
    should read a specific web page. The format consists of a title followed by a list of URLs with 
    descriptions explaining when an LLM should read that page.
    
    Format example:
    # Website Name
    URL1: LLM should read this page when needing to understand X, including details about A, B, and C.
    URL2: LLM should read this page when looking for information about Y, especially regarding D and E.
    
    Your task is to create an informative and helpful LLMs.txt file using the analyzed web pages provided.
    """
    
    # Create content for the user message
    user_message = f"Main URL: {url}\n\nAnalyzed pages:\n"
    for page in analyzed_pages:
        user_message += f"\nURL: {page['url']}\nTitle: {page['title']}\nAnalysis: {page['analysis']}\n"
    
    user_message += "\nBased on these analyzed pages, create a well-formatted LLMs.txt file that would help an LLM understand when to read each page."
    
    prompt = LLMPrompt(system_message=system_message, user_message=user_message)
    return query_llm(prompt)

def plan_exploration_strategy(url: str) -> Dict[str, Any]:
    """Plan a strategy for exploring a website.
    
    Args:
        url: The main URL to explore
        
    Returns:
        Dictionary containing exploration strategy information
    """
    system_message = """
    You are an expert website explorer and AI strategist who analyzes websites to create an efficient strategy for exploration by AI agents.
    
    Your task is to create a comprehensive exploration plan that will guide an AI agent in generating an LLMs.txt file for a website.
    
    For the given URL, analyze the domain and potential structure to:
    1. Identify the most important paths to explore on the website
    2. Identify important sections or keywords to prioritize when finding links
    3. Understand the website's likely structure and domain-specific patterns
    
    If the URL is a news website like Google News, focus on:
    - Top news categories
    - Featured stories
    - Main topic sections
    - Regional news pages if applicable
    - Headlines and important article pages
    
    Format your response as a JSON object with these keys:
    - paths: List of specific relative or absolute paths to explore first (e.g., /top-stories, /sections/world)
    - important_sections: List of important section names or keywords to prioritize in links
    - website_type: Type of website (e.g., "news", "documentation", "blog", "e-commerce")
    - exploration_strategy: Brief description of the recommended exploration approach
    """
    
    user_message = f"""
    I need to explore the website at {url} to create an LLMs.txt file that will help an LLM understand when to read different parts of the website.
    
    Please analyze this URL and create a comprehensive exploration plan, focusing on identifying the most valuable parts of the website and an efficient exploration strategy.
    
    Return your analysis as a structured JSON object.
    """
    
    prompt = LLMPrompt(system_message=system_message, user_message=user_message)
    response = query_llm(prompt)
    
    # Parse the JSON response
    try:
        plan = json.loads(response)
        return plan
    except json.JSONDecodeError:
        # 응답이 JSON 형식이 아닌 경우, 텍스트 응답에서 paths 추출 시도
        if VERBOSE:
            print("Failed to parse JSON response, attempting to extract paths from text")
        
        # 간단한 fallback 처리: 쉼표로 구분된 경로 목록 가정
        lines = response.split('\n')
        paths = []
        important_sections = []
        
        for line in lines:
            if ':' in line and ('path' in line.lower() or 'url' in line.lower()):
                # 경로 또는 URL을 포함하는 라인에서 정보 추출 시도
                potential_paths = line.split(':', 1)[1].strip().strip('"\'[]').split(',')
                paths.extend([p.strip().strip('"\'') for p in potential_paths if p.strip()])
                
            elif ':' in line and ('section' in line.lower() or 'keyword' in line.lower() or 'categor' in line.lower()):
                # 섹션/키워드 정보를 포함하는 라인에서 정보 추출 시도
                potential_sections = line.split(':', 1)[1].strip().strip('"\'[]').split(',')
                important_sections.extend([s.strip().strip('"\'') for s in potential_sections if s.strip()])
        
        return {
            "paths": paths,
            "important_sections": important_sections,
            "website_type": "unknown",
            "exploration_strategy": "Default exploration strategy"
        } 
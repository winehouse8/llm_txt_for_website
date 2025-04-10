"""Utility functions for web scraping and content parsing."""
import re
import time
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional, Tuple, Set
from tenacity import retry, stop_after_attempt, wait_fixed
from urllib.parse import urljoin, urlparse

from .config import MAX_RETRIES, RETRY_DELAY, TIMEOUT, VERBOSE

def is_valid_url(url: str) -> bool:
    """Check if a URL is valid."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

@retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_fixed(RETRY_DELAY))
def fetch_url(url: str) -> Tuple[str, str]:
    """Fetch content from a URL with retries.
    
    Args:
        url: The URL to fetch
        
    Returns:
        Tuple of (text content, content type)
    
    Raises:
        Exception: If the request fails after retries
    """
    if VERBOSE:
        print(f"Fetching URL: {url}")
    
    try:
        # 일반적인 웹 브라우저처럼 보이는 헤더 사용
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0"
        }
        
        response = requests.get(
            url,
            timeout=TIMEOUT,
            headers=headers
        )
        response.raise_for_status()
        content_type = response.headers.get('Content-Type', '')
        
        # 특별한 페이지 타입 처리 (예: 뉴스 사이트)
        is_news_site = any(domain in url.lower() for domain in ['news.', 'cnn.com', 'bbc.', 'nytimes.', 'reuters.', 'bloomberg.'])
        
        return response.text, content_type
    except Exception as e:
        if VERBOSE:
            print(f"Error fetching {url}: {str(e)}")
        raise

def extract_links(html_content: str, base_url: str) -> List[str]:
    """Extract all links from HTML content.
    
    Args:
        html_content: The HTML content to parse
        base_url: The base URL to resolve relative links
        
    Returns:
        List of absolute URLs
    """
    soup = BeautifulSoup(html_content, 'lxml')
    links = []
    seen_hrefs = set()  # 중복 URL 방지
    
    # 일반 링크 추출
    for link in soup.find_all('a', href=True):
        href = link['href'].strip()
        if href and not href.startswith(('#', 'javascript:', 'mailto:')):
            absolute_url = urljoin(base_url, href)
            if is_valid_url(absolute_url) and absolute_url not in seen_hrefs:
                links.append(absolute_url)
                seen_hrefs.add(absolute_url)
    
    # 뉴스 사이트에서 주요 링크가 있을 수 있는 특별한 요소 처리
    # 뉴스 헤드라인, 주요 기사 등이 일반적으로 포함된 요소들
    priority_elements = soup.select('.headline, .story, article, .news-item, .top-story, .featured')
    for element in priority_elements:
        links_in_element = element.find_all('a', href=True)
        for link in links_in_element:
            href = link['href'].strip()
            if href and not href.startswith(('#', 'javascript:', 'mailto:')):
                absolute_url = urljoin(base_url, href)
                if is_valid_url(absolute_url) and absolute_url not in seen_hrefs:
                    # 우선순위 링크를 앞에 추가하려면 여기서 처리할 수 있지만, 
                    # 우선순위 처리는 agents.py에서 이미 구현했으므로 여기서는 단순히 추가만 함
                    links.append(absolute_url)
                    seen_hrefs.add(absolute_url)
    
    return links

def extract_main_content(html_content: str) -> str:
    """Extract the main textual content from HTML.
    
    Args:
        html_content: The HTML content to parse
        
    Returns:
        Extracted text content
    """
    soup = BeautifulSoup(html_content, 'lxml')
    
    # JavaScript 렌더링 콘텐츠를 처리하기 위한 특별한 로직
    # 일부 사이트는 초기 HTML에 데이터를 JSON으로 포함시키는 경우가 있음
    json_script_tags = soup.find_all('script', {'type': 'application/json'})
    json_ld_tags = soup.find_all('script', {'type': 'application/ld+json'})
    
    additional_content = ""
    for script in json_script_tags + json_ld_tags:
        # 스크립트 내용에서 텍스트 콘텐츠 추출 시도
        script_text = script.string
        if script_text:
            # JSON에서 'headline', 'description', 'articleBody' 등의 키워드 찾기
            for keyword in ['headline', 'description', 'articleBody', 'content', 'text']:
                if keyword in script_text:
                    pattern = f'"{keyword}"\\s*:\\s*"([^"]+)"'
                    matches = re.findall(pattern, script_text)
                    if matches:
                        for match in matches:
                            additional_content += match + " "
    
    # 뉴스 사이트에서 주요 콘텐츠 영역 식별
    main_content_candidates = [
        soup.find('article'),
        soup.find(class_='article-content'),
        soup.find(class_='story-body'),
        soup.find(class_='main-content'),
        soup.find(id='article-body'),
        soup.find(class_='story'),
        soup.find(class_='post-content')
    ]
    
    main_content = None
    for candidate in main_content_candidates:
        if candidate and len(candidate.get_text(strip=True)) > 200:  # 충분한 텍스트가 있어야 함
            main_content = candidate
            break
    
    # 메인 콘텐츠 추출 실패 시 일반적인 방법 사용
    if not main_content:
        # 제거할 요소들
        for element in soup(["script", "style", "header", "footer", "nav", "aside", "iframe"]):
            element.decompose()
        
        # 남은 콘텐츠에서 텍스트 추출
        main_content = soup
    
    # 텍스트 정리 및 반환
    text = main_content.get_text(separator=' ')
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = ' '.join(chunk for chunk in chunks if chunk)
    
    # 추가 콘텐츠가 있으면 결합
    if additional_content:
        text = text + " " + additional_content
    
    return text

def normalize_url(url: str) -> str:
    """Normalize a URL by removing fragments and some query parameters."""
    parsed = urlparse(url)
    
    # 일부 쿼리 파라미터는 유지하는 것이 좋을 수 있음 (예: 뉴스 ID)
    query = parsed.query
    
    # 일반적인 추적 파라미터 제거
    if query:
        excluded_params = ['utm_source', 'utm_medium', 'utm_campaign', 'fbclid', 'gclid']
        query_pairs = []
        
        for pair in query.split('&'):
            if '=' in pair:
                param, value = pair.split('=', 1)
                if param not in excluded_params:
                    query_pairs.append(f"{param}={value}")
        
        query = '&'.join(query_pairs)
    
    # URL 재구성
    scheme = parsed.scheme
    netloc = parsed.netloc
    path = parsed.path
    
    # Query 부분 추가 (없으면 추가하지 않음)
    if query:
        normalized_url = f"{scheme}://{netloc}{path}?{query}"
    else:
        normalized_url = f"{scheme}://{netloc}{path}"
    
    # 후행 슬래시 정규화
    if not path or path == '/':
        normalized_url = f"{scheme}://{netloc}/"
    
    return normalized_url

def filter_relevant_links(links: List[str], base_domain: str) -> List[str]:
    """Filter links to keep only those from the same domain or subdomain.
    
    Args:
        links: List of URLs to filter
        base_domain: The base domain to match against
        
    Returns:
        Filtered list of URLs
    """
    filtered_links = []
    excluded_extensions = ['.pdf', '.zip', '.jpg', '.jpeg', '.png', '.gif', '.css', '.js']
    
    for link in links:
        parsed = urlparse(link)
        link_domain = parsed.netloc
        path = parsed.path.lower()
        
        # 확장자 확인
        if any(path.endswith(ext) for ext in excluded_extensions):
            continue
        
        # 동일 도메인 또는 서브도메인 확인
        is_same_domain = link_domain == base_domain
        is_subdomain = link_domain.endswith('.' + base_domain)
        
        # 뉴스 사이트에서 외부 링크가 중요할 수 있는 특수 경우
        is_news_site = any(domain in base_domain.lower() for domain in ['news.', 'cnn.', 'bbc.', 'nytimes.', 'reuters.'])
        is_news_article = any(indicator in path for indicator in ['/article/', '/story/', '/news/'])
        
        if is_same_domain or is_subdomain:
            filtered_links.append(link)
        elif is_news_site and is_news_article:
            # 뉴스 사이트의 경우 관련 뉴스 기사도 포함
            filtered_links.append(link)
    
    return filtered_links

def extract_title(html_content: str) -> str:
    """Extract the title from HTML content.
    
    Args:
        html_content: The HTML content to parse
        
    Returns:
        The page title or an empty string if not found
    """
    soup = BeautifulSoup(html_content, 'lxml')
    
    # 우선 순위에 따라 제목 추출 시도
    # 1. OpenGraph 태그
    og_title = soup.find('meta', property='og:title')
    if og_title and og_title.get('content'):
        return og_title.get('content').strip()
    
    # 2. 일반 타이틀 태그
    title_tag = soup.find('title')
    if title_tag:
        return title_tag.get_text().strip()
    
    # 3. h1 태그 (주요 헤드라인)
    h1_tag = soup.find('h1')
    if h1_tag:
        return h1_tag.get_text().strip()
    
    # 4. 뉴스 사이트 특화 클래스
    for selector in ['.headline', '.article-title', '.story-title']:
        headline = soup.select_one(selector)
        if headline:
            return headline.get_text().strip()
    
    return ""

def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and normalizing spacing.
    
    Args:
        text: The text to clean
        
    Returns:
        Cleaned text
    """
    # 다중 공백을 단일 공백으로 대체
    text = re.sub(r'\s+', ' ', text)
    
    # 앞뒤 공백 제거
    text = text.strip()
    
    # 특수 문자 처리
    text = text.replace('\u2019', "'")  # 특수 아포스트로피
    text = text.replace('\u2018', "'")  # 특수 따옴표
    text = text.replace('\u201c', '"')  # 특수 따옴표
    text = text.replace('\u201d', '"')  # 특수 따옴표
    text = text.replace('\u2014', '-')  # em dash
    text = text.replace('\u2013', '-')  # en dash
    
    # UTF-8 디코딩 이슈 처리
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    
    return text 
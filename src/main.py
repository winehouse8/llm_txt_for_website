"""Main entry point for the hot topic finder."""
import argparse
import json
import os
import sys
from urllib.parse import urlparse
import requests
from typing import List, Dict, Any, Optional
from collections import Counter

from src import agents
from src.agents import create_website_agent
from src.llm import create_llms_txt_content
from src.utils import fetch_url
from src.config import VERBOSE

def search_for_topic(topic: str, num_results: int = 5) -> List[str]:
    """주어진 주제에 대한 검색 결과 URL을 반환합니다.
    
    Args:
        topic: 검색할 주제
        num_results: 반환할 결과 수
        
    Returns:
        검색 결과 URL 목록
    """
    # 실제 구현에서는 검색 API를 사용하여 결과를 가져옵니다.
    # 이 예제에서는 몇 가지 일반적인 뉴스 사이트 URL을 반환합니다.
    
    # 주제별 미리 정의된 URL 사전
    topic_urls = {
        "ai": [
            "https://www.technologyreview.com/topic/artificial-intelligence/",
            "https://www.wired.com/tag/artificial-intelligence/",
            "https://www.theverge.com/ai-artificial-intelligence",
            "https://venturebeat.com/category/ai/",
            "https://www.forbes.com/ai/"
        ],
        "technology": [
            "https://techcrunch.com/",
            "https://www.theverge.com/",
            "https://www.wired.com/",
            "https://www.engadget.com/",
            "https://www.cnet.com/"
        ],
        "business": [
            "https://www.bloomberg.com/",
            "https://www.forbes.com/",
            "https://www.wsj.com/news/business",
            "https://www.cnbc.com/business/",
            "https://www.businessinsider.com/"
        ],
        "science": [
            "https://www.nature.com/",
            "https://www.scientificamerican.com/",
            "https://www.science.org/",
            "https://www.newscientist.com/",
            "https://www.popsci.com/"
        ],
        "health": [
            "https://www.who.int/news-room",
            "https://www.health.harvard.edu/",
            "https://www.webmd.com/",
            "https://www.mayoclinic.org/",
            "https://www.cdc.gov/mmwr/index.html"
        ]
    }
    
    # 주제가 미리 정의된 목록에 있는지 확인
    topic_lower = topic.lower()
    for key, urls in topic_urls.items():
        if key in topic_lower or topic_lower in key:
            return urls[:num_results]
    
    # 주제가 미리 정의된 목록에 없으면 일반 뉴스 사이트 반환
    default_urls = [
        "https://news.google.com/search?q=" + topic.replace(" ", "+"),
        "https://www.bbc.com/search?q=" + topic.replace(" ", "+"),
        "https://www.nytimes.com/search?query=" + topic.replace(" ", "+"),
        "https://www.reuters.com/search/news?blob=" + topic.replace(" ", "+"),
        "https://apnews.com/"
    ]
    
    return default_urls[:num_results]

def generate_llms_txt(url: str, output_path: str = "./llms.txt") -> str:
    """주어진 URL에 대한 llms.txt 파일을 생성합니다.
    
    Args:
        url: 탐색할 URL
        output_path: 생성된 llms.txt 파일 경로
        
    Returns:
        생성된 llms.txt 파일 내용
    """
    # 웹사이트 에이전트 생성
    agent = create_website_agent()
    
    # 초기 상태 설정
    initial_state = {
        "url": url,
        "visited": [],
        "to_visit": [],
        "pages": {},
        "analyzed_pages": [],
        "llms_txt": None,
        "error": None,
        "priority_urls": [],
        "exploration_plan": None,
        "important_sections": [],
        "section_counts": {},
        "last_page_sections": [],
        "visited_patterns": {}
    }
    
    # 에이전트 실행
    print(f"웹사이트 탐색 시작: {url}")
    final_state = agent.invoke(initial_state)
    
    # 오류 확인
    if final_state.get("error"):
        print(f"오류 발생: {final_state['error']}")
        return ""
    
    # llms.txt 파일 내용 가져오기
    llms_txt_content = final_state.get("llms_txt", "")
    
    if not llms_txt_content:
        print("llms.txt 파일을 생성하지 못했습니다.")
        return ""
    
    # 파일 저장
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(llms_txt_content)
    
    print(f"llms.txt 파일이 생성되었습니다: {output_path}")
    return llms_txt_content

def explore_topic(topic, max_pages=10, max_depth=2):
    """주어진 주제에 대한 웹페이지를 탐색하고 중요한 컨텐츠를 찾습니다.
    
    Args:
        topic: 탐색할 주제
        max_pages: 최대 탐색 페이지 수
        max_depth: 최대 탐색 깊이
        
    Returns:
        탐색 결과
    """
    # 주제에 대한 시작 URL 검색
    search_results = search_for_topic(topic)
    if not search_results:
        print(f"주제 '{topic}'에 대한 검색 결과를 찾을 수 없습니다.")
        return None
    
    # 시작 URL 선택
    start_url = search_results[0]
    print(f"시작 URL: {start_url}")
    
    # 시작 URL의 도메인을 기본 URL로 설정
    parsed_url = urlparse(start_url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    
    # 에이전트 상태 초기화
    agent_state = {
        'visited_urls': set(),
        'sections_frequency': {},
        'important_sections': set()
    }
    
    # 웹페이지 탐색 시작
    result = agents.explore_webpage(
        agent_state=agent_state,
        url=start_url,
        depth=0
    )
    
    # 발견된 중요 섹션 정리 (빈도수 기준 정렬)
    sorted_sections = sorted(
        [(section, agent_state['sections_frequency'].get(section.lower(), 0)) 
         for section in agent_state['important_sections']],
        key=lambda x: x[1],
        reverse=True
    )
    
    # 결과 요약
    summary = {
        "topic": topic,
        "base_url": base_url,
        "start_url": start_url,
        "visited_count": len(agent_state['visited_urls']),
        "important_sections": [section for section, _ in sorted_sections],
        "section_frequencies": {section: freq for section, freq in sorted_sections}
    }
    
    return summary

def main():
    """Main entry point for the command-line interface."""
    parser = argparse.ArgumentParser(description="Hot Topic Finder")
    parser.add_argument("--url", help="The URL to explore")
    parser.add_argument("--topic", help="Topic to search for")
    parser.add_argument("--output", default="llms.txt", help="Output file path for the LLMs.txt file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if not args.url and not args.topic:
        parser.error("Either --url or --topic must be provided")
    
    if args.topic:
        # 주제 탐색 모드
        print(f"주제 탐색 시작: {args.topic}")
        result = explore_topic(args.topic)
        
        if result:
            print("\n탐색 결과 요약:")
            print(f"주제: {result['topic']}")
            print(f"시작 URL: {result['start_url']}")
            print(f"방문한 페이지 수: {result['visited_count']}")
            print(f"중요 섹션 (상위 10개):")
            for section in result['important_sections'][:10]:
                freq = result['section_frequencies'].get(section, 0)
                print(f"  - {section} (빈도: {freq})")
                
            # 탐색 결과를 사용하여 llms.txt 생성
            url_to_use = result['start_url']
        else:
            print("주제 탐색에 실패했습니다.")
            return
    else:
        # URL 직접 사용 모드
        url_to_use = args.url
    
    # llms.txt 파일 생성
    llms_content = generate_llms_txt(url=url_to_use, output_path=args.output)
    
    if llms_content:
        print("\nllms.txt 파일 생성 완료!")
        print(f"파일 위치: {os.path.abspath(args.output)}")
        print("\nllms.txt 내용 미리보기:")
        preview_lines = llms_content.split('\n')[:10]
        for line in preview_lines:
            print(line)
        if len(preview_lines) < len(llms_content.split('\n')):
            print("...")
    else:
        print("llms.txt 파일 생성에 실패했습니다.")

if __name__ == "__main__":
    main() 
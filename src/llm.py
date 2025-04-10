"""LLM utilities for working with OpenAI."""
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import json

from .config import OPENAI_API_KEY, MODEL_NAME, VERBOSE
from .models import LLMPrompt

def get_llm():
    """Create and return a ChatOpenAI instance."""
    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=MODEL_NAME,
        temperature=0.3,
    )

def query_llm(prompt: LLMPrompt) -> str:
    """Send a prompt to the LLM and get a response.
    
    Args:
        prompt: The prompt to send to the LLM
        
    Returns:
        The text response from the LLM
    """
    llm = get_llm()
    
    if VERBOSE:
        print(f"Querying LLM with system message: {prompt.system_message[:50]}...")
        print(f"User message: {prompt.user_message[:50]}...")
    
    messages = [
        SystemMessage(content=prompt.system_message),
        HumanMessage(content=prompt.user_message)
    ]
    
    response = llm.invoke(messages)
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
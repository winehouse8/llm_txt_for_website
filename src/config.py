"""Configuration settings for the website agent."""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o-mini"  # The model to use for OpenAI chat completions

# Web scraping settings
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
TIMEOUT = 30  # seconds

# LLMs.txt generation settings
MAX_URLS_TO_EXPLORE = 20  # Maximum number of URLs to explore from a webpage
MAX_CONTEXT_LENGTH = 4000  # Maximum token length for context
SUMMARY_MAX_LENGTH = 1000  # Maximum token length for summaries

# 웹페이지 탐색 설정
MAX_DEPTH = 3  # 최대 탐색 깊이
MIN_CONTENT_LENGTH = 200  # 분석할 최소 콘텐츠 길이
MAX_CONTENT_LENGTH = 8000  # 요약 전 최대 콘텐츠 길이
MAX_LINKS_PER_PAGE = 5  # 페이지당 추출할 최대 링크 수

# 적응형 우선순위 시스템 설정
MIN_IMPORTANT_SECTIONS = 5  # 최소 추적할 중요 섹션 수
MAX_IMPORTANT_SECTIONS = 30  # 최대 추적할 중요 섹션 수 (너무 많으면 성능 저하)
SECTION_IMPORTANCE_THRESHOLD = 0.6  # 중요 섹션으로 간주하는 최소 점수 임계값
PATTERN_DIVERSITY_WEIGHT = 0.2  # URL 패턴 다양성 가중치
MIN_SECTION_FREQUENCY = 2  # 중요 섹션으로 인정받는 최소 발견 빈도

# Verbose mode for debugging
VERBOSE = True 
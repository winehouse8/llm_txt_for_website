"""Configuration settings for the website agent."""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-3.5-turbo"  # The model to use for OpenAI chat completions

# Web scraping settings
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
TIMEOUT = 30  # seconds

# LLMs.txt generation settings
MAX_URLS_TO_EXPLORE = 20  # Maximum number of URLs to explore from a webpage
MAX_CONTEXT_LENGTH = 4000  # Maximum token length for context
SUMMARY_MAX_LENGTH = 1000  # Maximum token length for summaries

# Verbose mode for debugging
VERBOSE = True 
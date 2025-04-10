# llm_txt_for_website

A powerful agent-based tool for automatically generating `llms.txt` files for websites. This tool helps LLMs (Large Language Models) better understand website content by creating structured summaries of key information.

## Overview

The `llm_txt_for_website` tool explores websites, analyzes the content, and generates a comprehensive `llms.txt` file that serves as a roadmap for LLMs when they need to access information from that website. It uses an agent-based approach to intelligently navigate through a website, identify important pages and topics, and summarize the most valuable content.

### What is an llms.txt file?

An `llms.txt` file is a structured text document that provides LLMs with guidance on how to understand and navigate a website's content. Similar to robots.txt for web crawlers, llms.txt helps LLMs know which parts of a website contain the most valuable information for answering user queries.

## Features

- **Intelligent Website Exploration**: Uses a state-of-the-art agent to navigate websites and discover important content
- **Content Analysis**: Leverages LLMs to analyze page content and extract key information
- **Adaptive Priority System**: Dynamically adjusts exploration strategy based on discovered content
- **Section Identification**: Automatically identifies important sections, topics, and keywords
- **Structured Output**: Generates well-formatted `llms.txt` files ready for use with LLM systems

## Installation

1. Clone the repository:
```bash
git clone https://github.com/winehouse8/llm_txt_for_website.git
cd llm_txt_for_website
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key in a `.env` file:
```
OPENAI_API_KEY=sk-qasdaasd123456789
```
Create a file named `.env` in the project root directory and add your OpenAI API key as shown above. Replace the example key with your actual OpenAI API key.

## Usage

Run the tool with a target website URL:

```bash
python hot_topic_finder.py https://example.com -o example_llms.txt
```

### Command-line Arguments

- `url`: URL of the website to explore (required)
- `-o, --output`: Output file path (default: llms.txt)
- `-v, --verbose`: Enable verbose output

## How It Works

1. **Initialization**: The tool starts from the provided URL and creates an initial exploration plan.
2. **Content Analysis**: It fetches and analyzes the content of the starting page to identify important sections and topics.
3. **Link Prioritization**: Based on the analysis, the tool prioritizes which links to explore next.
4. **Intelligent Exploration**: The agent navigates through the website, focusing on the most valuable content.
5. **Data Synthesis**: All gathered information is synthesized into a comprehensive `llms.txt` file.

## Example Output

A typical `llms.txt` file looks like:

```
# Website Name
https://example.com/
LLM should read this page when needing an overview of the company's products and services.

# Product Information
https://example.com/products
LLM should read this page when answering questions about the company's product lineup, specifications, and pricing.

# Support Documentation
https://example.com/support
LLM should read this page when helping users troubleshoot issues or find documentation.
```

## Project Structure

- `hot_topic_finder.py`: Main entry point for the command-line application
- `src/`: Core implementation
  - `agents.py`: Agent-based website exploration logic
  - `llm.py`: LLM utilities for content analysis
  - `main.py`: Core functionality for generating llms.txt files
  - `models.py`: Data models for the application
  - `utils.py`: Helper functions for web scraping and text processing
  - `config.py`: Configuration settings

## Requirements

- Python 3.8+
- OpenAI API key
- See `requirements.txt` for dependencies

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 
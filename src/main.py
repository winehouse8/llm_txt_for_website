"""Main entry point for the website agent application."""
import argparse
import os
import sys
from urllib.parse import urlparse

from .agents import create_website_agent, AgentState
from .config import VERBOSE

def validate_url(url: str) -> bool:
    """Validate if a URL is properly formatted.
    
    Args:
        url: The URL to validate
        
    Returns:
        True if the URL is valid, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
    except:
        return False

def generate_llms_txt(url: str, output_file: str = None) -> str:
    """Generate an LLMs.txt file for the given URL.
    
    Args:
        url: The URL to explore
        output_file: Optional output file path
        
    Returns:
        The generated LLMs.txt content
    """
    if not validate_url(url):
        print(f"Error: Invalid URL format: {url}")
        print("URL should start with http:// or https:// and include a domain.")
        return None
    
    if VERBOSE:
        print(f"Generating LLMs.txt for {url}")
    
    # Create the agent
    agent = create_website_agent()
    
    # Initialize the state
    initial_state = {
        "url": url,
        "visited": [],
        "to_visit": [],
        "pages": {},
        "analyzed_pages": [],
        "llms_txt": None,
        "error": None
    }
    
    # Run the agent
    final_state = agent.invoke(initial_state)
    
    # Check for errors
    if final_state.get("error"):
        print(f"Error: {final_state['error']}")
        return None
    
    # Get the generated LLMs.txt content
    llms_txt_content = final_state.get("llms_txt")
    
    if not llms_txt_content:
        print("Error: Failed to generate LLMs.txt content.")
        return None
    
    # Write to file if output_file is provided
    if output_file:
        directory = os.path.dirname(output_file)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(llms_txt_content)
            
        if VERBOSE:
            print(f"LLMs.txt written to {output_file}")
    
    return llms_txt_content

def main():
    """Main entry point for the command-line interface."""
    parser = argparse.ArgumentParser(description="Generate LLMs.txt files for websites")
    parser.add_argument("url", help="URL of the website to explore")
    parser.add_argument("-o", "--output", help="Output file path (default: llms.txt)", default="llms.txt")
    
    args = parser.parse_args()
    
    result = generate_llms_txt(args.url, args.output)
    
    if result:
        print("LLMs.txt generation completed successfully!")
        
        # Print a preview of the generated content
        preview_lines = result.split("\n")[:10]
        preview = "\n".join(preview_lines)
        
        if len(result.split("\n")) > 10:
            preview += "\n..."
            
        print("\nPreview:")
        print("-" * 40)
        print(preview)
        print("-" * 40)
        print(f"Full content written to {args.output}")
    else:
        print("LLMs.txt generation failed.")
        sys.exit(1)

if __name__ == "__main__":
    main() 
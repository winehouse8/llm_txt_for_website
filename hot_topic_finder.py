#!/usr/bin/env python
"""
Hot Topic Finder - Website agent that creates llms.txt summaries for websites.

Usage:
  python hot_topic_finder.py https://example.com -o output.txt
"""
import sys
import argparse
from src.main import generate_llms_txt

def main():
    """Main entry point for the command-line application."""
    parser = argparse.ArgumentParser(
        description="Hot Topic Finder - Generate LLMs.txt files for websites using an agent-based approach"
    )
    parser.add_argument("url", help="URL of the website to explore")
    parser.add_argument(
        "-o", "--output", 
        help="Output file path (default: llms.txt)", 
        default="llms.txt"
    )
    parser.add_argument(
        "-v", "--verbose",
        help="Enable verbose output",
        action="store_true"
    )
    
    args = parser.parse_args()
    
    try:
        result = generate_llms_txt(args.url, args.output)
        
        if result:
            print("\n✅ LLMs.txt generation completed successfully!")
            
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
            return 0
        else:
            print("❌ LLMs.txt generation failed.")
            return 1
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
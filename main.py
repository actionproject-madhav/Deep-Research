"""
Main entry point for Deep Research system
"""
import os
import sys
import time
import json
from dotenv import load_dotenv
from search_engine import SearchEngine
from document_processor import DocumentProcessor
from vector_db import VectorDB
from research_agent import ResearchAgent
from report_generator import ReportGenerator


def save_to_file(content, filename):
    """Save content to a file"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)

def save_research_data(research_results, output_dir="./output"):
    """Save research data to JSON file"""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate filename based on question and timestamp
    question_slug = research_results["main_question"][:30].lower().replace(" ", "_")
    timestamp = int(time.time())
    filename = f"{output_dir}/research_data_{question_slug}_{timestamp}.json"
    
    # Save research data
    with open(filename, 'w', encoding='utf-8') as f:
        # Convert sets to lists for JSON serialization
        serializable_results = research_results.copy()
        
        # Convert any non-serializable data
        json.dump(serializable_results, f, indent=2, default=str)
    
    return filename

def main():
    # Load environment variables
    load_dotenv()
    
    # Get API keys
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    google_search_engine_id = os.environ.get("GOOGLE_SEARCH_ENGINE_ID")
    
    # Check if API keys are available
    if not openai_api_key or not google_api_key or not google_search_engine_id:
        print("Error: Missing API credentials. Please check your .env file.")
        print("Required environment variables: OPENAI_API_KEY, GOOGLE_API_KEY, GOOGLE_SEARCH_ENGINE_ID")
        sys.exit(1)
    
    # Setup output directory
    output_dir = "./output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get research question from user
    if len(sys.argv) > 1:
        # Get question from command line arguments
        question = " ".join(sys.argv[1:])
    else:
        # Get question from user input
        print("\n=== 🔎 Deep Research System ===\n")
        question = input("Enter your research question: ").strip()
    
    if not question:
        print("Error: Please provide a research question.")
        sys.exit(1)
    
    # Create research agent and report generator
    print(f"\n🚀 Initializing Deep Research system...")
    
    research_agent = ResearchAgent(
        openai_api_key=openai_api_key,
        google_api_key=google_api_key,
        search_engine_id=google_search_engine_id
    )
    
    report_generator = ReportGenerator(openai_api_key=openai_api_key)
    
    # Start timing
    start_time = time.time()
    
    # Display welcome message
    print(f"\n=== 🔍 DEEP RESEARCH ===")
    print(f"Question: {question}")
    print(f"The research process has started. This may take several minutes.")
    print("=" * 50)
    
    try:
        # Set research parameters
        max_depth = 1  # Use 0 for just the main question, 1 for one level of sub-questions
        results_per_query = 10
        
        # Conduct research
        research_results = research_agent.research(
            question=question,
            max_depth=max_depth,
            results_per_query=results_per_query
        )
        
        # Save research data
        data_file = save_research_data(research_results, output_dir)
        print(f"\n💾 Raw research data saved to: {data_file}")
        
        # Generate report
        print("\n📊 Generating final report...")
        report = report_generator.generate_report(research_results)
        
        # Save report to a file
        question_slug = question[:30].lower().replace(" ", "_")
        report_file = f"{output_dir}/report_{question_slug}_{int(time.time())}.md"
        save_to_file(report, report_file)
        
        # End timing
        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(elapsed_time, 60)
        
        print("\n✅ Research complete!")
        print(f"⏱️  Total time: {int(minutes)} minutes, {int(seconds)} seconds")
        print(f"📄 Report saved to: {report_file}")
        print(f"\nResearch summary:")
        print(f"- Main question: {question}")
        print(f"- Sub-questions explored: {len(research_results['sub_questions']) - 1}")  # Subtract 1 for the main question
        print(f"- Documents analyzed: {len(research_results['documents'])}")
        
        # Display report
        print("\n" + "=" * 50)
        print("\n📑 REPORT PREVIEW:\n")
        
        # Show first 500 characters of the report as a preview
        preview_length = 500
        if len(report) > preview_length:
            print(report[:preview_length] + "...\n")
            print(f"View the full report at: {report_file}")
        else:
            print(report)
        
        print("\n" + "=" * 50)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
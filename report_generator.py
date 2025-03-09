"""
Final report generation for Deep Research
"""

from typing import Dict, Any, List
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

class ReportGenerator:
    def __init__(self, openai_api_key: str, model_name: str = "gpt-4o-mini"):
        """
        Initialize the Report Generator
        
        Args:
            openai_api_key (str): OpenAI API key
            model_name (str): OpenAI model to use
        """
        self.llm = ChatOpenAI(
            temperature=0.2,  # Slightly higher temperature for creative synthesis
            model=model_name,
            api_key=openai_api_key
        )
        
        # Create the report generation prompt
        self.report_prompt = PromptTemplate(
            input_variables=["question", "research_summary"],
            template="""
            You are an expert research analyst tasked with creating a comprehensive research report.
            
            RESEARCH QUESTION:
            {question}
            
            RESEARCH FINDINGS:
            {research_summary}
            
            Your task is to synthesize all the information and write a detailed, well-structured research report that thoroughly answers the original question.
            
            The report should:
            1. Begin with an executive summary that concisely states the main findings
            2. Include an introduction explaining the research question and its significance
            3. Present findings in a logical and well-structured manner, with clear section headings
            4. Include citations to the sources when presenting information from them
            5. End with a conclusion summarizing the key insights and their implications
            
            Format your report in Markdown. The report should be comprehensive and showcase deep expertise on the topic.
            """
        )
        
        self.report_chain = LLMChain(llm=self.llm, prompt=self.report_prompt)
        
    def generate_report(self, research_results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive research report
        
        Args:
            research_results (Dict): Results from the research agent
            
        Returns:
            str: Formatted research report
        """
        # Extract the main question
        main_question = research_results["main_question"]
        
        # Compile research summary from all sub-questions
        research_summary = self._compile_research_summary(research_results)
        
        # Generate the report
        report = self.report_chain.run(
            question=main_question,
            research_summary=research_summary
        )
        
        return report
        
    def _compile_research_summary(self, research_results: Dict[str, Any]) -> str:
        """
        Compile research summary from sub-questions and their analyses
        
        Args:
            research_results (Dict): Results from the research agent
            
        Returns:
            str: Compiled research summary
        """
        summary = ""
        
        # Add main question
        summary += f"Main Research Question: {research_results['main_question']}\n\n"
        
        # Process each sub-question
        for i, sub_q_result in enumerate(research_results["sub_questions"], 1):
            question = sub_q_result["question"]
            depth = sub_q_result["depth"]
            analyses = sub_q_result["document_analyses"]
            
            # Add question header
            if depth == 0:
                summary += f"MAIN QUESTION: {question}\n"
            else:
                summary += f"SUB-QUESTION {i}: {question}\n"
            
            # Add document analyses
            summary += f"Number of relevant documents analyzed: {len(analyses)}\n\n"
            
            for j, doc_analysis in enumerate(analyses, 1):
                doc = doc_analysis["document"]
                analysis = doc_analysis["analysis"]
                
                # Extract metadata
                title = doc["metadata"]["title"]
                url = doc["metadata"]["url"]
                
                # Add source information
                summary += f"SOURCE {j}: {title} ({url})\n\n"
                
                # Add analysis sections
                in_relevance = False
                in_key_info = False
                
                for line in analysis.split('\n'):
                    line = line.strip()
                    
                    if line.startswith('RELEVANCE:'):
                        in_relevance = True
                        in_key_info = False
                        summary += line + '\n'
                    elif line.startswith('KEY INFORMATION:'):
                        in_relevance = False
                        in_key_info = True
                        summary += line + '\n'
                    elif line.startswith('FOLLOW-UP QUESTION:'):
                        in_relevance = False
                        in_key_info = False
                        # Skip follow-up questions in the summary as they've been processed
                    elif in_key_info and line:
                        summary += line + '\n'
                    elif in_relevance and line:
                        summary += line + '\n'
                
                summary += "\n"
            
            summary += "=" * 50 + "\n\n"
        
        return summary
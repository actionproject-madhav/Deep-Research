"""
Core research agent logic for Deep Research
"""

from typing import List, Dict, Any, Optional, Set
import time
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from search_engine import SearchEngine
from document_processor import DocumentProcessor
from vector_db import VectorDB
import uuid

class ResearchAgent:
    def __init__(self, 
                openai_api_key: str,
                google_api_key: str, 
                search_engine_id: str,
                model_name: str = "gpt-4o-mini"
                ""):
        """
        Initialize the Research Agent
        
        Args:
            openai_api_key (str): OpenAI API key
            google_api_key (str): Google API key
            search_engine_id (str): Google Search Engine ID
            model_name (str): OpenAI model to use
        """
        self.search_engine = SearchEngine(google_api_key, search_engine_id)
        self.document_processor = DocumentProcessor()
        self.vector_db = VectorDB()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            temperature=0,
            model=model_name,
            api_key=openai_api_key
        )
        
        # Create prompt templates
        self._create_prompt_templates()
        
        # Session ID for this research session
        self.session_id = str(uuid.uuid4())
        
        # Tracking for research
        self.questions_researched = set()
        self.sub_questions = []
        self.all_processed_docs = []
        self.processed_urls = set()
    
    def _create_prompt_templates(self):
        """Create prompt templates for different tasks"""
        
        # Prompt for generating search queries
        query_prompt = PromptTemplate(
            input_variables=["question"],
            template="""
            You are a research assistant tasked with generating effective search queries.
            
            Given the research question below, generate 2-3 search queries that would help find relevant information.
            Each query should be specific and well-formed to find information on Wikipedia.
            
            Research Question: {question}
            
            Output format:
            1. [First search query]
            2. [Second search query]
            3. [Third search query]
            """
        )
        self.query_chain = LLMChain(llm=self.llm, prompt=query_prompt)
        
        # Prompt for document analysis
        analysis_prompt = PromptTemplate(
            input_variables=["question", "document_content", "document_url", "document_title"],
            template="""
            You are a research assistant tasked with analyzing documents to extract relevant information.
            
            Research Question: {question}
            
            Below is the content from a document:
            Title: {document_title}
            URL: {document_url}
            
            DOCUMENT CONTENT:
            {document_content}
            
            Your task is to:
            1. Determine if this document contains relevant information for answering the research question
            2. Identify the key facts, data points, and insights from this document related to the research question
            3. Suggest one follow-up question that would help explore the topic further
            
            Output format:
            RELEVANCE: [High/Medium/Low]
            
            KEY INFORMATION:
            - [Key fact 1]
            - [Key fact 2]
            - [Additional facts as needed]
            
            FOLLOW-UP QUESTION: [One specific follow-up question that arises from this document]
            """
        )
        self.analysis_chain = LLMChain(llm=self.llm, prompt=analysis_prompt)
        
        # Prompt for generating sub-questions
        sub_question_prompt = PromptTemplate(
            input_variables=["question", "extracted_information"],
            template="""
            You are a research assistant tasked with breaking down a complex research topic.
            
            Main Research Question: {question}
            
            Based on your initial research, you've gathered the following information:
            {extracted_information}
            
            Your task is to identify 2-3 specific sub-questions that would help to more fully answer the main research question.
            These sub-questions should:
            1. Address important aspects not fully covered in the information already gathered
            2. Be specific enough to be answerable through focused research
            3. Together help build a more complete answer to the main question
            
            Output format:
            1. [First sub-question]
            2. [Second sub-question]
            3. [Third sub-question]
            """
        )
        self.sub_question_chain = LLMChain(llm=self.llm, prompt=sub_question_prompt)
        
    def research(self, 
                question: str, 
                max_depth: int = 1, 
                results_per_query: int = 5) -> Dict[str, Any]:
        """
        Conduct research on a given question
        
        Args:
            question (str): The research question
            max_depth (int): Maximum depth of sub-questions to explore
            results_per_query (int): Number of search results to process per query
            
        Returns:
            Dict: Research results including documents and analysis
        """
        print(f"\n🔍 Starting research on: {question}")
        
        # Clear vector DB for new research session
        self.vector_db.clear()
        
        # Reset tracking for this research session
        self.questions_researched = set()
        self.sub_questions = []
        self.all_processed_docs = []
        self.processed_urls = set()
        
        # Add main question to the queue
        question_queue = [(question, 0)]  # (question, depth)
        
        # Process questions until queue is empty or max depth is reached
        while question_queue:
            current_question, current_depth = question_queue.pop(0)
            
            # Skip if we've already researched this question
            if current_question in self.questions_researched:
                continue
                
            print(f"\n📋 Researching question (depth {current_depth}): {current_question}")
            
            # Mark this question as researched
            self.questions_researched.add(current_question)
            
            # 1. Generate search queries
            search_queries = self._generate_search_queries(current_question)
            
            # Track document analysis results for this question
            question_docs = []
            
            # 2. Process each search query
            for query in search_queries:
                # Search for documents
                search_results = self.search_engine.search(query, num_results=results_per_query)
                print(f"🔎 Query: '{query}' - Found {len(search_results)} results")
                
                # Process documents (fetch and chunk)
                doc_chunks = self.document_processor.fetch_and_process_documents(search_results)
                
                # Filter out documents from URLs we've already processed
                new_chunks = []
                for chunk in doc_chunks:
                    if chunk["url"] not in self.processed_urls:
                        new_chunks.append(chunk)
                        self.processed_urls.add(chunk["url"])
                
                print(f"📄 Processed {len(new_chunks)} new document chunks")
                
                # Add documents to vector DB
                if new_chunks:
                    self.vector_db.add_documents(new_chunks)
                    self.all_processed_docs.extend(new_chunks)
                
                # Pause to avoid rate limits
                time.sleep(1)
            
            # 3. Analyze documents for this question
            relevant_docs = self.vector_db.search(current_question, n_results=10)
            
            document_analyses = []
            candidate_sub_questions = set()
            
            for doc in relevant_docs:
                analysis = self._analyze_document(
                    current_question, 
                    doc["content"], 
                    doc["metadata"]["url"], 
                    doc["metadata"]["title"]
                )
                
                # Add to document analyses
                document_analyses.append({
                    "document": doc,
                    "analysis": analysis
                })
                
                # Extract follow-up question if present
                follow_up = self._extract_follow_up_question(analysis)
                if follow_up and follow_up not in self.questions_researched:
                    candidate_sub_questions.add(follow_up)
            
            # Track document analyses for this question
            question_result = {
                "question": current_question,
                "depth": current_depth,
                "document_analyses": document_analyses,
            }
            self.sub_questions.append(question_result)
            
            # 4. Generate sub-questions and add to queue if we haven't reached max depth
            if current_depth < max_depth:
                # Only try to generate sub-questions if we have document analyses
                if document_analyses:
                    extracted_info = self._compile_extracted_information(document_analyses)
                    generated_sub_questions = self._generate_sub_questions(current_question, extracted_info)
                    
                    # Add follow-up questions from document analysis
                    all_sub_questions = list(candidate_sub_questions) + generated_sub_questions
                    
                    # Add unique sub-questions to the queue
                    for sub_q in all_sub_questions:
                        if sub_q not in self.questions_researched:
                            question_queue.append((sub_q, current_depth + 1))
                    
                    print(f"❓ Generated {len(all_sub_questions)} sub-questions")
        
        return {
            "main_question": question,
            "sub_questions": self.sub_questions,
            "documents": self.all_processed_docs,
            "session_id": self.session_id
        }
    
    def _generate_search_queries(self, question: str) -> List[str]:
    
        try:
            response = self.query_chain.invoke({"question": question})
            response_text = response.get("text", "")
            
            # Parse the numbered list response
            queries = []
            for line in response_text.strip().split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() and line[1:3] in ['. ', ') ']):
                    query = line[line.index(' ')+1:].strip()
                    queries.append(query)
            
            return queries if queries else [question]  # Fallback to using the question directly
        except Exception as e:
            print(f"Error generating search queries: {str(e)}")
            return [question] # Fallback to using the question directly
    
    def _analyze_document(self, 
                         question: str, 
                         content: str, 
                         url: str, 
                         title: str) -> str:
        """
        Analyze a document for relevant information
        
        Args:
            question (str): The research question
            content (str): Document content
            url (str): Document URL
            title (str): Document title
            
        Returns:
            str: Analysis of the document
        """
        return self.analysis_chain.run(
            question=question,
            document_content=content,
            document_url=url,
            document_title=title
        )
    
    def _extract_follow_up_question(self, analysis: str) -> Optional[str]:
        """
        Extract follow-up question from document analysis
        
        Args:
            analysis (str): Document analysis
            
        Returns:
            Optional[str]: Extracted follow-up question or None
        """
        for line in analysis.split('\n'):
            if line.startswith('FOLLOW-UP QUESTION:'):
                question = line[len('FOLLOW-UP QUESTION:'):].strip()
                return question
        return None
    
    def _compile_extracted_information(self, document_analyses: List[Dict[str, Any]]) -> str:
        """
        Compile extracted information from document analyses
        
        Args:
            document_analyses (List[Dict]): List of document analyses
            
        Returns:
            str: Compiled information
        """
        compiled_info = ""
        
        for i, doc_analysis in enumerate(document_analyses, 1):
            analysis = doc_analysis["analysis"]
            doc = doc_analysis["document"]
            
            # Extract the KEY INFORMATION section
            key_info = ""
            in_key_info = False
            
            for line in analysis.split('\n'):
                if line.startswith('KEY INFORMATION:'):
                    in_key_info = True
                elif line.startswith('FOLLOW-UP QUESTION:'):
                    in_key_info = False
                elif in_key_info and line.strip():
                    key_info += line + '\n'
            
            if key_info:
                compiled_info += f"Source {i}: {doc['metadata']['title']} ({doc['metadata']['url']})\n"
                compiled_info += key_info + '\n'
        
        return compiled_info
    
    def _generate_sub_questions(self, question: str, extracted_info: str) -> List[str]:
        """
        Generate sub-questions based on extracted information
        
        Args:
            question (str): The main research question
            extracted_info (str): Extracted information from documents
            
        Returns:
            List[str]: List of sub-questions
        """
        response = self.sub_question_chain.run(
            question=question,
            extracted_information=extracted_info
        )
        
        # Parse the numbered list response
        sub_questions = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() and line[1:3] in ['. ', ') ']):
                sub_q = line[line.index(' ')+1:].strip()
                sub_questions.append(sub_q)
        
        return sub_questions
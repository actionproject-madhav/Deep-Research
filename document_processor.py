"""
Document fetching and processing for Deep Research
"""

import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize
from typing import List, Dict, Any, Tuple
import time

# Download NLTK data for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class DocumentProcessor:
    def __init__(self):
        """Initialize the document processor"""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def fetch_and_process_documents(self, search_results: List[Dict[str, Any]], 
                                   chunk_size: int = 500, 
                                   chunk_overlap: int = 50) -> List[Dict[str, Any]]:
        """
        Fetch and process documents from search results
        
        Args:
            search_results (List[Dict]): Search results containing URLs
            chunk_size (int): Target size of each document chunk in characters
            chunk_overlap (int): Overlap between chunks in characters
            
        Returns:
            List[Dict]: List of processed document chunks
        """
        all_chunks = []
        
        for result in search_results:
            url = result.get('url')
            if not url:
                continue
                
            try:
                content = self._fetch_document(url)
                if content:
                    chunks = self._chunk_document(
                        content=content,
                        url=url,
                        title=result.get('title', ''),
                        source=result.get('source', ''),
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    all_chunks.extend(chunks)
                    
                    # Be nice to the servers
                    time.sleep(1)
            
            except Exception as e:
                print(f"Error processing document {url}: {e}")
                
        return all_chunks
    
    def _fetch_document(self, url: str) -> str:
        """
        Fetch and extract text content from a URL
        
        Args:
            url (str): URL to fetch
            
        Returns:
            str: Extracted text content
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Extract text from the main content area
            # For Wikipedia pages, the main content is usually in <div id="content">
            main_content = soup.find('div', id='content') or soup.find('div', id='bodyContent') or soup
            
            # Get text and clean it up
            text = main_content.get_text(separator=' ', strip=True)
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return ""
    
    def _chunk_document(self, 
                   content: str, 
                   url: str, 
                   title: str, 
                   source: str,
                   chunk_size: int, 
                   chunk_overlap: int) -> List[Dict[str, Any]]:
   
        sentences = sent_tokenize(content)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'content': chunk_text,
                    'url': url,
                    'title': title,
                    'source': source,
                    'chunk_length': len(chunk_text)
                })
                
                # Fixed line with proper parentheses
                overlap_start = max(0, len(current_chunk) - (chunk_overlap // len(current_chunk[0]))) if current_chunk else 0
                current_chunk = current_chunk[overlap_start:]
                current_size = sum(len(s) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'content': chunk_text,
                'url': url,
                'title': title,
                'source': source,
                'chunk_length': len(chunk_text)
            })
            
        return chunks
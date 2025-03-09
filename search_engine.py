"""
Google search functionality for Deep Research
"""

import os
import requests
from typing import Dict, List, Any, Optional

class SearchEngine:
    def __init__(self, api_key: str, search_engine_id: str):
        """
        Initialize the SearchEngine with API credentials
        
        Args:
            api_key (str): Google API key
            search_engine_id (str): Google Search Engine ID
        """
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"
    
    def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a Google search and return the results
        
        Args:
            query (str): The search query
            num_results (int): Number of results to return (max 10)
            
        Returns:
            List[Dict]: List of search result items
        """
        params = {
            'q': query,
            'key': self.api_key,
            'cx': self.search_engine_id,
            'num': min(num_results, 10),  # Google API limits to 10 results per request
            'start': 1,
            'safe': 'active',
            'lr': 'lang_en'  # Restrict to English language
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            search_data = response.json()
            
            if 'items' not in search_data:
                print(f"No results found for query: {query}")
                return []
                
            return self._extract_results(search_data)
            
        except requests.exceptions.RequestException as e:
            print(f"Search error: {e}")
            return []
    
    def _extract_results(self, search_response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract relevant information from the search results
        
        Args:
            search_response (Dict): The raw API response
            
        Returns:
            List[Dict]: List of processed search results
        """
        results = []
        
        for item in search_response.get('items', []):
            result = {
                'title': item.get('title', ''),
                'url': item.get('link', ''),
                'snippet': item.get('snippet', ''),
                'source': item.get('displayLink', '')
            }
            results.append(result)
            
        return results
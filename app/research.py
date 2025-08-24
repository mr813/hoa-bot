"""Perplexity API integration for research assistance."""

import os
import time
import json
import requests
from typing import Dict, List, Any, Optional
from app.utils import truncate_text


class PerplexityResearch:
    """Class to handle Perplexity API integration."""
    
    def __init__(self):
        self.api_key = os.getenv('PERPLEXITY_API_KEY')
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.model = "sonar"  # Default model
        self.timeout = 10
        self.max_retries = 3
        self.rate_limit_delay = 1.0  # Seconds between requests
        
    def is_enabled(self) -> bool:
        """Check if Perplexity API is enabled."""
        return bool(self.api_key)
    
    def set_model(self, model: str):
        """Set the model to use for API calls."""
        self.model = model
    
    def ask_perplexity(self, prompt: str, system_hint: str = None, citations: bool = True) -> Dict[str, Any]:
        """Ask Perplexity API a question and return structured response."""
        if not self.is_enabled():
            return {
                'error': 'Perplexity API not enabled',
                'summary_bullets': [],
                'sources': [],
                'raw_text': ''
            }
        
        # Sanitize and truncate inputs
        prompt = truncate_text(prompt, 1000)  # Limit prompt length
        if system_hint:
            system_hint = truncate_text(system_hint, 500)
        
        # Prepare messages
        messages = []
        if system_hint:
            messages.append({
                'role': 'system',
                'content': system_hint
            })
        
        # Combine prompt with citations request if enabled
        full_prompt = prompt
        if citations:
            full_prompt += "\n\nPlease provide 2-3 concise bullet points with sources/citations if available."
        
        messages.append({
            'role': 'user',
            'content': full_prompt
        })
        
        # Prepare request payload
        payload = {
            'model': self.model,
            'messages': messages,
            'max_tokens': 500,
            'temperature': 0.3,
            'top_p': 0.9
        }
        
        # Make API request with retries
        for attempt in range(self.max_retries):
            try:
                response = self._make_api_request(payload)
                if response:
                    return self._parse_response(response)
                
            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    time.sleep(self.rate_limit_delay * (attempt + 1))
                    continue
                return self._error_response("Request timeout")
                
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.rate_limit_delay * (attempt + 1))
                    continue
                return self._error_response(f"Request error: {str(e)}")
        
        return self._error_response("Max retries exceeded")
    
    def _make_api_request(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make API request to Perplexity."""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            # Rate limited
            time.sleep(self.rate_limit_delay * 2)
            return None
        elif response.status_code >= 500:
            # Server error
            return None
        else:
            raise requests.exceptions.RequestException(
                f"API request failed with status {response.status_code}: {response.text}"
            )
    
    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Perplexity API response."""
        try:
            content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            # Extract bullet points
            summary_bullets = self._extract_bullet_points(content)
            
            # Extract sources
            sources = self._extract_sources(content)
            
            return {
                'summary_bullets': summary_bullets,
                'sources': sources,
                'raw_text': content,
                'success': True
            }
            
        except Exception as e:
            return self._error_response(f"Failed to parse response: {str(e)}")
    
    def _extract_bullet_points(self, content: str) -> List[str]:
        """Extract bullet points from response content."""
        bullets = []
        
        # Look for various bullet point formats
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith(('•', '-', '*', '1.', '2.', '3.')):
                # Remove bullet marker and clean up
                bullet = line.lstrip('•-*1234567890. ')
                if bullet and len(bullet) > 10:  # Minimum meaningful length
                    bullets.append(bullet)
        
        # If no bullets found, split by sentences
        if not bullets and content:
            sentences = content.split('. ')
            bullets = [s.strip() for s in sentences if len(s.strip()) > 20][:3]
        
        return bullets[:3]  # Limit to 3 bullets
    
    def _extract_sources(self, content: str) -> List[Dict[str, str]]:
        """Extract sources from response content."""
        sources = []
        
        # Look for URLs
        import re
        url_pattern = r'https?://[^\s\)]+'
        urls = re.findall(url_pattern, content)
        
        for url in urls:
            # Try to extract title from context
            title = self._extract_title_from_context(content, url)
            sources.append({
                'title': title,
                'url': url
            })
        
        return sources[:3]  # Limit to 3 sources
    
    def _extract_title_from_context(self, content: str, url: str) -> str:
        """Extract title from context around URL."""
        # Simple extraction - look for text before URL
        lines = content.split('\n')
        for line in lines:
            if url in line:
                # Remove URL and clean up
                title = line.replace(url, '').strip()
                title = title.rstrip('.,;:')
                if title and len(title) > 5:
                    return title
        
        return "Source"
    
    def _error_response(self, error_message: str) -> Dict[str, Any]:
        """Create error response."""
        return {
            'error': error_message,
            'summary_bullets': [],
            'sources': [],
            'raw_text': '',
            'success': False
        }
    
    def get_statute_summary(self, statute_ref: str) -> Dict[str, Any]:
        """Get summary of a specific statute section."""
        prompt = f"Provide a brief summary of Florida Statute {statute_ref} regarding condominium law. Focus on key requirements and implications for HOA compliance."
        system_hint = "You are a legal research assistant. Provide concise, factual information about Florida condominium law. Do not provide legal advice."
        
        return self.ask_perplexity(prompt, system_hint)
    
    def get_hierarchy_explanation(self) -> Dict[str, Any]:
        """Get explanation of declaration supremacy principle."""
        prompt = "Explain the hierarchy principle in Florida condominium law where the Declaration of Condominium takes precedence over Bylaws, Rules, and Resolutions. Include relevant Florida case law examples."
        system_hint = "You are a legal research assistant. Explain legal principles with factual information and case law examples. Do not provide legal advice."
        
        return self.ask_perplexity(prompt, system_hint)
    
    def enrich_finding(self, finding_description: str, statute_refs: List[str]) -> Dict[str, Any]:
        """Enrich a finding with additional research."""
        statute_text = ", ".join(statute_refs)
        prompt = f"Based on this finding: '{finding_description}' and Florida Statutes {statute_text}, provide 2-3 concise bullet points about relevant legal considerations and potential implications."
        system_hint = "You are a legal research assistant. Provide factual information about legal considerations. Do not provide legal advice."
        
        return self.ask_perplexity(prompt, system_hint)


def is_research_enabled() -> bool:
    """Check if research features are enabled."""
    return bool(os.getenv('PERPLEXITY_API_KEY'))


def get_available_models() -> List[str]:
    """Get list of available Perplexity models."""
    return [
        "sonar",
        "sonar-pro",
        "llama-3.1-8b-instruct",
        "llama-3.1-70b-instruct",
        "mixtral-8x7b-instruct",
        "codellama-70b-instruct"
    ]


def validate_api_key(api_key: str) -> bool:
    """Validate Perplexity API key format."""
    # Basic validation - API keys are typically long strings
    return bool(api_key and len(api_key) > 20)


def create_research_instance() -> Optional[PerplexityResearch]:
    """Create and return a PerplexityResearch instance if API is enabled."""
    if is_research_enabled():
        return PerplexityResearch()
    return None

import json
from dataclasses import dataclass
import google.generativeai as genai

@dataclass
class RouterDecision:
    needs_memory: bool
    reason: str
    relevant_categories: list[str]  # Which memory categories might be relevant


class RouterAgent:
    """Decides if a query needs memory retrieval"""
    
    CATEGORIES = [
        "personal_info", "preferences", "goals", "activities", "habits",
        "experiences", "relationships", "work_life", "opinions", "knowledge"
    ]
    
    ROUTER_PROMPT = """You are a routing agent. Analyze the user's query and decide if it needs personal memory/context to answer well.

Memory categories available:
- personal_info: Name, age, location, identity
- preferences: Likes, dislikes, choices
- goals: Aspirations, plans, objectives
- activities: Hobbies, regular activities
- habits: Routines, patterns
- experiences: Past events, memories
- relationships: People mentioned, connections
- work_life: Job, career, professional info
- opinions: Views, beliefs, stances
- knowledge: Skills, expertise, learnings

Queries that NEED memory:
- Personal recommendations ("What should I do this weekend?")
- Questions referencing past conversations ("Remember when I said...")
- Questions about user's life ("What are my goals?")
- Personalized advice ("Should I take this job?")

Queries that DON'T need memory:
- General knowledge ("What is Python?")
- Generic tasks ("Write a poem about rain")
- Simple greetings ("Hello!")
- Factual questions ("What's the capital of France?")

Respond with JSON only:
{
  "needs_memory": true/false,
  "reason": "brief explanation",
  "relevant_categories": ["category1", "category2"]
}"""

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.model = genai.GenerativeModel(model_name)
    
    def _parse_response(self, response_text: str) -> dict:
        """Extract JSON from response"""
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{[\s\S]*?\}', response_text)
            if json_match:
                return json.loads(json_match.group(0))
        return {"needs_memory": False, "reason": "Failed to parse", "relevant_categories": []}
    
    def route(self, query: str) -> RouterDecision:
        """Decide if query needs memory retrieval"""
        prompt = f"{self.ROUTER_PROMPT}\n\nUser query: {query}"
        
        response = self.model.generate_content(prompt)
        parsed = self._parse_response(response.text)
        
        return RouterDecision(
            needs_memory=parsed.get("needs_memory", False),
            reason=parsed.get("reason", ""),
            relevant_categories=[
                c for c in parsed.get("relevant_categories", []) 
                if c in self.CATEGORIES
            ]
        )

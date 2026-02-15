import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
import google.generativeai as genai
from difflib import SequenceMatcher

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



class CategoryUpdate(BaseModel):
    """Facts organized by subcategory (null key means no subcategory)"""
    facts: dict[Optional[str], list[str]] = Field(default_factory=dict)

class ExtractionOutput(BaseModel):
    """Structured output from reflection agent"""
    personal_info: CategoryUpdate = Field(default_factory=CategoryUpdate)
    preferences: CategoryUpdate = Field(default_factory=CategoryUpdate)
    goals: CategoryUpdate = Field(default_factory=CategoryUpdate)
    activities: CategoryUpdate = Field(default_factory=CategoryUpdate)
    habits: CategoryUpdate = Field(default_factory=CategoryUpdate)
    experiences: CategoryUpdate = Field(default_factory=CategoryUpdate)
    relationships: CategoryUpdate = Field(default_factory=CategoryUpdate)
    work_life: CategoryUpdate = Field(default_factory=CategoryUpdate)
    opinions: CategoryUpdate = Field(default_factory=CategoryUpdate)
    knowledge: CategoryUpdate = Field(default_factory=CategoryUpdate)



class MemoryManager:
    """Handles reading/writing markdown memory files"""
    
    CATEGORIES = [
        "personal_info", "preferences", "goals", "activities", "habits",
        "experiences", "relationships", "work_life", "opinions", "knowledge"
    ]
    
    def __init__(self, memory_dir: str = "./memory"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        self._initialize_files()
    
    def _initialize_files(self):
        """Create empty markdown files if they don't exist"""
        for category in self.CATEGORIES:
            filepath = self.memory_dir / f"{category}.md"
            if not filepath.exists():
                title = category.replace("_", " ").title()
                filepath.write_text(f"# {title}\n")
    
    def read_category(self, category: str) -> dict[Optional[str], list[str]]:
        """Parse markdown file into {subcategory: [facts]} structure"""
        filepath = self.memory_dir / f"{category}.md"
        content = filepath.read_text()
        
        result: dict[Optional[str], list[str]] = {None: []}
        current_subcategory = None
        
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("## "):
                current_subcategory = line[3:].strip()
                if current_subcategory not in result:
                    result[current_subcategory] = []
            elif line.startswith("- "):
                result.get(current_subcategory, result.setdefault(current_subcategory, [])).append(line)
        
        return result
    
    def write_category(self, category: str, data: dict[Optional[str], list[str]]):
        """Write structured data back to markdown file"""
        filepath = self.memory_dir / f"{category}.md"
        title = category.replace("_", " ").title()
        
        lines = [f"# {title}"]
        
        # Write facts without subcategory first
        if None in data and data[None]:
            for fact in data[None]:
                lines.append(fact)
            lines.append("")
        
        # Write subcategories
        for subcategory, facts in sorted((k, v) for k, v in data.items() if k is not None):
            if facts:
                lines.append(f"## {subcategory}")
                for fact in facts:
                    lines.append(fact)
                lines.append("")
        
        filepath.write_text("\n".join(lines).strip() + "\n")
    
    def is_duplicate(self, existing_facts: list[str], new_fact: str, threshold: float = 0.85) -> bool:
        """Check if a fact is semantically similar to existing facts"""
        new_clean = re.sub(r'\(.*?\)', '', new_fact).lower().strip("- ").strip()
        
        for existing in existing_facts:
            existing_clean = re.sub(r'\(.*?\)', '', existing).lower().strip("- ").strip()
            similarity = SequenceMatcher(None, existing_clean, new_clean).ratio()
            if similarity >= threshold:
                return True
        return False
    
    def merge_facts(self, category: str, new_data: dict[Optional[str], list[str]]) -> int:
        """Merge new facts into existing file, avoiding duplicates. Returns count of added facts."""
        existing = self.read_category(category)
        added_count = 0
        timestamp = datetime.now().strftime("%Y-%m-%d")
        
        # Collect all existing facts for duplicate checking
        all_existing = []
        for facts in existing.values():
            all_existing.extend(facts)
        
        for subcategory, facts in new_data.items():
            if subcategory not in existing:
                existing[subcategory] = []
            
            for fact in facts:
                # Ensure fact starts with "- "
                if not fact.startswith("- "):
                    fact = f"- {fact}"
                
                # Add timestamp if not present
                if f"({timestamp}" not in fact and "(" not in fact:
                    fact = f"{fact} ({timestamp})"
                
                if not self.is_duplicate(all_existing, fact):
                    existing[subcategory].append(fact)
                    all_existing.append(fact)
                    added_count += 1
        
        # Promote to subcategory if 3+ related items without subcategory
        if None in existing and len(existing[None]) >= 3:
            # Keep as-is for now; could add smart grouping later
            pass
        
        self.write_category(category, existing)
        return added_count



class ReflectionAgent:
    """Extracts facts from conversations and updates memory files"""
    
    EXTRACTION_PROMPT = """You are a memory extraction agent. Analyze the following conversation messages and extract meaningful facts, preferences, and insights about the user.

Categorize into these categories:
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

Rules:
- Only extract concrete information, not vague statements
- Use third person ("The user...")
- Create subcategories only when 3+ related items naturally group together
- Skip categories with no relevant information
- Each fact should be a single bullet point

Output valid JSON only:
{
  "personal_info": {"null": ["- The user lives in San Francisco"]},
  "work_life": {"Professional": ["- The user works as a software engineer"]},
  "preferences": {"null": ["- The user prefers Python over JavaScript"]}
}

Use "null" as the key for facts without a subcategory. Only include categories that have extracted facts."""

    def __init__(self, memory_dir: str = "./memory", model_name: str = "gemini-2.0-flash"):
        self.memory_manager = MemoryManager(memory_dir)
        self.model = genai.GenerativeModel(model_name)
    
    def _format_messages(self, messages: list[dict]) -> str:
        """Format messages for the prompt"""
        formatted = []
        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)
    
    def _parse_response(self, response_text: str) -> dict:
        """Extract JSON from LLM response"""
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
            if json_match:
                return json.loads(json_match.group(1))
            
            # Try to find JSON object in text
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                return json.loads(json_match.group(0))
        
        return {}
    
    def _normalize_extraction(self, raw: dict) -> dict[str, dict[Optional[str], list[str]]]:
        """Normalize the extraction output to consistent format"""
        result = {}
        
        for category in MemoryManager.CATEGORIES:
            if category not in raw:
                continue
            
            cat_data = raw[category]
            normalized: dict[Optional[str], list[str]] = {}
            
            for subcategory, facts in cat_data.items():
                key = None if subcategory in ("null", "None", None) else subcategory
                if isinstance(facts, list):
                    normalized[key] = facts
            
            if normalized:
                result[category] = normalized
        
        return result
    
    def extract(self, messages: list[dict]) -> dict[str, dict[Optional[str], list[str]]]:
        """Extract facts from messages using LLM"""
        if not messages:
            return {}
        
        formatted = self._format_messages(messages)
        prompt = f"{self.EXTRACTION_PROMPT}\n\n--- CONVERSATION ---\n{formatted}"
        
        response = self.model.generate_content(prompt)
        raw_output = self._parse_response(response.text)
        
        return self._normalize_extraction(raw_output)
    
    def process(self, messages: list[dict]) -> dict[str, int]:
        """Extract facts and update memory files. Returns {category: facts_added}"""
        extracted = self.extract(messages)
        results = {}
        
        for category, facts_data in extracted.items():
            added = self.memory_manager.merge_facts(category, facts_data)
            if added > 0:
                results[category] = added
        
        return results



if __name__ == "__main__":
    test_messages = [
        {"role": "user", "content": "Hi! I'm Raj, I live in Mumbai and work as a data scientist."},
        {"role": "assistant", "content": "Nice to meet you Raj! How do you like working in data science?"},
        {"role": "user", "content": "I love it! I mostly use Python and I'm learning Rust on the side. My goal is to build AI tools for education."},
        {"role": "assistant", "content": "That's a great goal! Are you working on any projects?"},
        {"role": "user", "content": "Yes, I'm building a memory system for AI agents. I go to the gym 4 times a week to stay sharp."},
    ]
    
    agent = ReflectionAgent(memory_dir="./memory")
    results = agent.process(test_messages)
    
    print("Facts added per category:")
    for category, count in results.items():
        print(f"  {category}: {count}")
    
    print("\n--- Memory Files ---")
    for category in MemoryManager.CATEGORIES:
        filepath = Path("./memory") / f"{category}.md"
        if filepath.exists():
            content = filepath.read_text().strip()
            if content.count("\n") > 0:
                print(f"\n{filepath.name}:")
                print(content)

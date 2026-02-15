import os
import time
import logging
from pathlib import Path
from rlm import RLM

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryRetriever:
    """Uses RLM to retrieve relevant memories from markdown files"""
    
    RETRIEVAL_PROMPT = """You have access to the user's memory files in the `context` variable.
The context is a dictionary where keys are category names and values are the markdown content.

Your task: Find and extract information relevant to answering this query:
"{query}"

Focus on these categories: {categories}

Instructions:
1. Look through the relevant memory categories in context
2. Extract bullet points that would help answer the user's query
3. Be concise - only include truly relevant information
4. If no relevant information is found, return "No relevant memories found."

Return your findings as bullet points summarizing what you learned about the user that's relevant to their query."""

    def __init__(
        self, 
        memory_dir: str = "./memory",
        backend: str = "gemini",
        environment: str = "docker",
        model_name: str = "gemini-2.0-flash",
        max_iterations: int = 10
    ):
        self.memory_dir = Path(memory_dir)
        self.backend = backend
        self.environment = environment
        self.model_name = model_name
        self.max_iterations = max_iterations
        
        # Backend configuration for RLM
        self.backend_kwargs = {
            "model_name": model_name,
            "api_key": os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        }
        
    def _load_memories(self, categories: list[str] = None) -> dict[str, str]:
        """Load memory markdown files, optionally filtered by categories"""
        memories = {}
        
        if not self.memory_dir.exists():
            return memories
        
        for md_file in self.memory_dir.glob("*.md"):
            category = md_file.stem
            
            # Filter by categories if specified
            if categories and category not in categories:
                continue
            
            try:
                content = md_file.read_text(encoding='utf-8').strip()
                # Only include if there's actual content beyond the header
                if content.count('\n') > 0:
                    memories[category] = content
            except Exception as e:
                logger.warning(f"Could not load {md_file.name}: {e}")
        
        return memories
    
    def retrieve(self, query: str, categories: list[str] = None) -> str:
        """
        Use RLM to retrieve relevant memories for a query.
        """
        logger.info(f"Using RLM MemoryRetriever for query: '{query}'")
        
        # Load memories (filtered by categories if provided)
        memories = self._load_memories(categories)
        
        if not memories:
            logger.info("No memories found in storage.")
            return ""
        
        # Build the context for RLM
        context = {
            "memories": memories,
            "query": query,
            "categories": list(memories.keys())
        }
        
        # Build custom system prompt for retrieval task
        categories_str = ", ".join(categories) if categories else "all available"
        root_prompt = self.RETRIEVAL_PROMPT.format(
            query=query,
            categories=categories_str
        )
        
        try:
            # Initialize RLM with Docker environment
            rlm = RLM(
                backend=self.backend,
                backend_kwargs=self.backend_kwargs,
                environment=self.environment, 
                max_iterations=self.max_iterations,
                max_depth=1,
                verbose=False
            )
            
            # Run RLM completion
            start_time = time.time()
            result = rlm.completion(
                prompt=context,
                root_prompt=root_prompt
            )
            end_time = time.time()
            logger.info(f"RLM retrieval took {end_time - start_time:.2f}s")
            
            response = result.response.strip()
            
            # Check for "no relevant" responses
            if "no relevant" in response.lower():
                return ""
            
            return response
            
        except Exception as e:
            logger.error(f"RLM retrieval failed: {e}")
            return ""


class SimpleMemoryRetriever:
    """Fallback retriever using simple keyword matching (no RLM)"""
    
    def __init__(self, memory_dir: str = "./memory"):
        self.memory_dir = Path(memory_dir)
    
    def _load_memories(self, categories: list[str] = None) -> dict[str, str]:
        memories = {}
        if not self.memory_dir.exists():
            return memories
        
        for md_file in self.memory_dir.glob("*.md"):
            category = md_file.stem
            if categories and category not in categories:
                continue
            try:
                content = md_file.read_text(encoding='utf-8').strip()
                if content.count('\n') > 0:
                    memories[category] = content
            except:
                pass
        return memories
    
    def retrieve(self, query: str, categories: list[str] = None) -> str:
        """Simple retrieval - just returns all content from relevant categories"""
        logger.info(f"Using SimpleMemoryRetriever (Fallback) for query: '{query}'")
        
        memories = self._load_memories(categories)
        
        if not memories:
            return ""
        
        # Just concatenate relevant memories
        parts = []
        for category, content in memories.items():
            # Skip the header line, get just the facts
            lines = [l for l in content.split('\n') if l.startswith('- ')]
            if lines:
                parts.append(f"**{category.replace('_', ' ').title()}:**")
                parts.extend(lines[:5])  # Limit to 5 facts per category
        
        return "\n".join(parts)

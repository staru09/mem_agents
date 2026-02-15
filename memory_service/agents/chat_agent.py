import google.generativeai as genai
from datetime import datetime
from dataclasses import dataclass
from typing import Literal
import os
import uuid
from database.connection import get_db
from database.repository import MessageRepository
from agents.router_agent import RouterAgent
from agents.memory_retriever import MemoryRetriever, SimpleMemoryRetriever



@dataclass
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class ChatAgent:
    def __init__(
        self, 
        router: RouterAgent,
        memory_retriever: MemoryRetriever | SimpleMemoryRetriever,
        model_name: str = "gemini-2.0-flash", 
        context_window: int = 20, 
        thread_id: uuid.UUID = None,
    ):
        self.model = genai.GenerativeModel(model_name)
        self.context_window = context_window
        self.thread_id = thread_id or uuid.uuid4()
        self.system_prompt = """You are a helpful assistant with memory of the user. 
Use the provided context about the user to give personalized responses."""
        
        self.router = router
        self.memory_retriever = memory_retriever
        
        # Load existing messages from DB if resuming a thread
        self.messages: list[ChatMessage] = []
        self._load_thread_history()
    
    def _load_thread_history(self):
        """Load previous messages from the database for this thread"""
        with get_db() as db:
            repo = MessageRepository(db)
            db_messages = repo.get_recent_messages(self.thread_id, limit=self.context_window)
            for msg in db_messages:
                self.messages.append(ChatMessage(
                    role=msg.role,
                    content=msg.content,
                    timestamp=msg.timestamp
                ))
    
    def _get_recent_context(self) -> list[dict]:
        recent = self.messages[-self.context_window:]
        return [{"role": m.role, "parts": [m.content]} for m in recent]
    
    def _save_message(self, role: str, content: str):
        """Save message to both in-memory list and database"""
        msg = ChatMessage(role=role, content=content)
        self.messages.append(msg)
        
        with get_db() as db:
            repo = MessageRepository(db)
            repo.save_message(thread_id=self.thread_id, role=role, content=content)
        
        return msg
    
    def respond(self, user_input: str) -> str:
        self._save_message("user", user_input)
        
        # Step 1: Route the query
        decision = self.router.route(user_input)
        
        memory_context = ""
        if decision.needs_memory:
            print(f"  [Router: Fetching memories from {decision.relevant_categories}]")
            # Step 2: Use Memory Retriever
            memory_context = self.memory_retriever.retrieve(
                query=user_input, 
                categories=decision.relevant_categories
            )
            if memory_context:
                # Count lines accurately
                line_count = len(memory_context.split('\n'))
                print(f"  [Retrieved {line_count} lines of context]")
        else:
            print(f"  [Router: No memory needed - {decision.reason}]")
        
        # Step 3: Build prompt with memory context
        history = self._get_recent_context()[:-1]
        
        if memory_context:
            augmented_input = f"""Here's what I know about you that might be relevant:
{memory_context}

Your question: {user_input}"""
        else:
            augmented_input = user_input
        
        chat = self.model.start_chat(history=history)
        response = chat.send_message(augmented_input)
        assistant_response = response.text
        
        self._save_message("assistant", assistant_response)
        return assistant_response
    
    def get_message_count(self) -> int:
        return len(self.messages)


def main():
    thread_id = uuid.uuid4()
    agent = ChatAgent(thread_id=thread_id)
    
    session = PromptSession(history=InMemoryHistory())
    
    print("Memory Chat (type 'quit' to exit)")
    print(f"Thread ID: {thread_id}")
    print("Use ↑/↓ arrows to navigate through your message history")
    print("-" * 50)
    
    while True:
        try:
            user_input = session.prompt("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        response = agent.respond(user_input)
        print(f"\nAssistant: {response}")


if __name__ == "__main__":
    main()


import google.generativeai as genai
from datetime import datetime
from dataclasses import dataclass
from typing import Literal
import os
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory

# Configure Gemini
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)

@dataclass
class Message:
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class ChatAgent:
    def __init__(self, model_name: str = "gemini-2.0-flash", context_window: int = 20):
        self.model = genai.GenerativeModel(model_name)
        self.messages: list[Message] = []
        self.context_window = context_window
        self.system_prompt = """You are a helpful assistant. Have natural conversations with the user. 
Be friendly, informative, and remember context from the conversation."""
    
    def _get_recent_context(self) -> list[dict]:
        recent = self.messages[-self.context_window:]
        return [{"role": m.role, "parts": [m.content]} for m in recent]
    
    def _save_message(self, role: str, content: str):
        msg = Message(role=role, content=content)
        self.messages.append(msg)
        return msg
    
    def respond(self, user_input: str) -> str:
        self._save_message("user", user_input)
        history = self._get_recent_context()[:-1]
        chat = self.model.start_chat(history=history)
        response = chat.send_message(user_input)
        assistant_response = response.text
        self._save_message("assistant", assistant_response)
        return assistant_response
    
    def get_message_count(self) -> int:
        return len(self.messages)


def main():
    agent = ChatAgent()
    
    session = PromptSession(history=InMemoryHistory())
    
    print("Memory Chat (type 'quit' to exit)")
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

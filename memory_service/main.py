import os
from uuid import uuid4
import google.generativeai as genai
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory

from database.init_db import init_database
from services.scheduler import ReflectionScheduler

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


from agents.chat_agent import ChatAgent


def main():
    # Initialize database tables
    init_database()
    
    # Start reflection scheduler
    scheduler = ReflectionScheduler(
        memory_dir="./memory",
        time_interval=300,    # 5 minutes
        message_threshold=5   # or 5 messages
    )
    scheduler.start()
    
    # Initialize chat
    agent = ChatAgent()
    session = PromptSession(history=InMemoryHistory())
    
    print("Memory Chat")
    print(f"Thread: {agent.thread_id}")
    print("Commands: 'quit' to exit, 'reflect' to force reflection")
    print("-" * 50)
    
    try:
        while True:
            user_input = session.prompt("\nYou: ").strip()
            
            if not user_input:
                continue
            if user_input.lower() == 'quit':
                break
            if user_input.lower() == 'reflect':
                scheduler.force_run()
                continue
            
            response = agent.respond(user_input)
            print(f"\nAssistant: {response}")
    
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        scheduler.stop()
        print("\nGoodbye!")


if __name__ == "__main__":
    main()

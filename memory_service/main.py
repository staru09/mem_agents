import os
import time
from uuid import uuid4
import google.generativeai as genai
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory

from database.init_db import init_database
from services.scheduler import ReflectionScheduler

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


from agents.chat_agent import ChatAgent
from agents.router_agent import RouterAgent
from agents.memory_retriever import MemoryRetriever, SimpleMemoryRetriever


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
    
    # Initialize agents
    router = RouterAgent()
    # retrievers
    # retriever = SimpleMemoryRetriever(memory_dir="./memory")
    retriever = MemoryRetriever(
        memory_dir="./memory",
        backend="gemini",
        environment="docker"
    )
    
    agent = ChatAgent(router=router, memory_retriever=retriever)
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
            
            start_time = time.time()
            response = agent.respond(user_input)
            end_time = time.time()
            print(f"  [Total Pipeline Duration: {end_time - start_time:.2f}s]")
            print(f"\nAssistant: {response}")
    
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        scheduler.stop()
        print("\nGoodbye!")


if __name__ == "__main__":
    main()

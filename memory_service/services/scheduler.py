import threading
import time
from datetime import datetime, timedelta
from uuid import UUID

from database.connection import get_db
from database.repository import MessageRepository, ReflectionLogRepository
from agents.reflection_agent import ReflectionAgent


class ReflectionScheduler:
    """Background scheduler that triggers reflection based on time or message count"""
    
    def __init__(
        self,
        memory_dir: str = "./memory",
        time_interval: int = 300,      # 5 minutes
        message_threshold: int = 5,
        model_name: str = "gemini-2.0-flash"
    ):
        self.time_interval = time_interval
        self.message_threshold = message_threshold
        self.reflection_agent = ReflectionAgent(memory_dir=memory_dir, model_name=model_name)
        
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._last_run = datetime.utcnow()
    
    def start(self):
        """Start the background scheduler"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print(f"[Scheduler] Started (interval={self.time_interval}s, threshold={self.message_threshold} msgs)")
    
    def stop(self):
        """Stop the background scheduler"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            print("[Scheduler] Stopped")
    
    def _run_loop(self):
        """Main scheduler loop - checks triggers every 10 seconds"""
        while self._running:
            try:
                self._check_and_run()
            except Exception as e:
                print(f"[Scheduler] Error: {e}")
            time.sleep(10)  # Check every 10 seconds
    
    def _check_and_run(self):
        """Check if reflection should run based on time or message count"""
        with get_db() as db:
            msg_repo = MessageRepository(db)
            unprocessed_count = msg_repo.get_unprocessed_count()
            
            time_elapsed = (datetime.utcnow() - self._last_run).total_seconds()
            time_trigger = time_elapsed >= self.time_interval
            message_trigger = unprocessed_count >= self.message_threshold
            
            if (time_trigger or message_trigger) and unprocessed_count > 0:
                trigger_reason = "time" if time_trigger else "messages"
                print(f"\n[Scheduler] Triggering reflection ({trigger_reason}, {unprocessed_count} msgs)")
                self._run_reflection()
    
    def _run_reflection(self):
        """Execute the reflection process"""
        with self._lock:  # Prevent concurrent reflections
            with get_db() as db:
                msg_repo = MessageRepository(db)
                log_repo = ReflectionLogRepository(db)
                
                # Get unprocessed messages
                messages = msg_repo.get_unprocessed_messages(limit=20)
                if not messages:
                    return
                
                # Format for reflection agent
                message_dicts = [
                    {"role": m.role, "content": m.content}
                    for m in messages
                ]
                
                # Run reflection
                results = self.reflection_agent.process(message_dicts)
                
                # Mark as processed
                message_ids = [m.id for m in messages]
                msg_repo.mark_messages_processed(message_ids)
                
                # Log the reflection
                log_repo.log_reflection(
                    last_processed_id=message_ids[-1],
                    messages_processed=len(messages),
                    categories_updated=list(results.keys())
                )
                
                self._last_run = datetime.utcnow()
                
                if results:
                    print(f"[Scheduler] Reflection complete: {results}")
                else:
                    print(f"[Scheduler] Reflection complete: no facts extracted")
    
    def force_run(self):
        """Manually trigger a reflection (useful for testing)"""
        print("[Scheduler] Manual trigger...")
        threading.Thread(target=self._run_reflection).start()

from datetime import datetime
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy import and_
from .models import Message, ReflectionLog


class MessageRepository:
    """CRUD operations for messages"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def save_message(self, thread_id: UUID, role: str, content: str) -> Message:
        """Save a new message"""
        msg = Message(thread_id=thread_id, role=role, content=content)
        self.db.add(msg)
        self.db.flush()
        return msg
    
    def get_unprocessed_messages(self, limit: int = 10) -> list[Message]:
        """Get messages that haven't been processed by reflection agent"""
        return (
            self.db.query(Message)
            .filter(Message.processed == False)
            .order_by(Message.timestamp.asc())
            .limit(limit)
            .all()
        )
    
    def get_unprocessed_count(self) -> int:
        """Count unprocessed messages"""
        return self.db.query(Message).filter(Message.processed == False).count()
    
    def mark_messages_processed(self, message_ids: list[int]):
        """Mark messages as processed"""
        self.db.query(Message).filter(Message.id.in_(message_ids)).update(
            {Message.processed: True}, synchronize_session=False
        )
    
    def get_recent_messages(self, thread_id: UUID, limit: int = 20) -> list[Message]:
        """Get recent messages for context"""
        return (
            self.db.query(Message)
            .filter(Message.thread_id == thread_id)
            .order_by(Message.timestamp.desc())
            .limit(limit)
            .all()
        )[::-1]  # Reverse to chronological order
    
    def get_thread_messages(self, thread_id: UUID) -> list[Message]:
        """Get all messages in a thread"""
        return (
            self.db.query(Message)
            .filter(Message.thread_id == thread_id)
            .order_by(Message.timestamp.asc())
            .all()
        )


class ReflectionLogRepository:
    """CRUD operations for reflection logs"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def log_reflection(
        self, 
        last_processed_id: int, 
        messages_processed: int, 
        categories_updated: list[str]
    ) -> ReflectionLog:
        """Log a reflection run"""
        log = ReflectionLog(
            last_processed_id=last_processed_id,
            messages_processed=messages_processed,
            categories_updated=categories_updated
        )
        self.db.add(log)
        self.db.flush()
        return log
    
    def get_last_reflection(self) -> ReflectionLog | None:
        """Get the most recent reflection log"""
        return (
            self.db.query(ReflectionLog)
            .order_by(ReflectionLog.last_run_at.desc())
            .first()
        )

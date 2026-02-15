import uuid
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, ARRAY
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    thread_id = Column(UUID(as_uuid=True), default=uuid.uuid4, nullable=False, index=True)
    role = Column(String(20), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    processed = Column(Boolean, default=False, nullable=False, index=True)
    
    def __repr__(self):
        return f"<Message(id={self.id}, role='{self.role}', processed={self.processed})>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "thread_id": str(self.thread_id),
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "processed": self.processed
        }


class ReflectionLog(Base):
    __tablename__ = "reflection_log"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    last_processed_id = Column(Integer, nullable=True)
    last_run_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    messages_processed = Column(Integer, default=0)
    categories_updated = Column(ARRAY(String), default=[])
    
    def __repr__(self):
        return f"<ReflectionLog(id={self.id}, last_processed_id={self.last_processed_id})>"
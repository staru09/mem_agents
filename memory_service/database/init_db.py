from .connection import engine
from .models import Base


def init_database():
    """Create all tables"""
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")


def drop_database():
    """Drop all tables (use with caution!)"""
    Base.metadata.drop_all(bind=engine)
    print("Database tables dropped!")


if __name__ == "__main__":
    init_database()

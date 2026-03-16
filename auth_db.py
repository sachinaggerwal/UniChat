"""
auth_db.py - Database models for user authentication and chat history
Uses SQLite + SQLAlchemy for lightweight, file-based storage
"""
from datetime import datetime
from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import hashlib
import secrets

Base = declarative_base()


class User(Base):
    """User account model"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(128), nullable=False)
    salt = Column(String(32), nullable=False)
    full_name = Column(String(100))
    student_id = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    
    # Relationship to chat messages
    messages = relationship("ChatMessage", back_populates="user", cascade="all, delete-orphan")
    
    def set_password(self, password: str):
        """Hash password with salt"""
        self.salt = secrets.token_hex(16)
        self.password_hash = hashlib.sha256(
            (password + self.salt).encode('utf-8')
        ).hexdigest()
    
    def check_password(self, password: str) -> bool:
        """Verify password"""
        test_hash = hashlib.sha256(
            (password + self.salt).encode('utf-8')
        ).hexdigest()
        return test_hash == self.password_hash
    
    def __repr__(self):
        return f"<User(username='{self.username}', email='{self.email}')>"


class ChatMessage(Base):
    """Chat message history model"""
    __tablename__ = 'chat_messages'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    course_filter = Column(String(20))  # Course code if filtered
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationship to user
    user = relationship("User", back_populates="messages")
    
    def __repr__(self):
        return f"<ChatMessage(user_id={self.user_id}, role='{self.role}', timestamp={self.timestamp})>"


class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self, db_path: str = "unichat_users.db"):
        """Initialize database connection"""
        self.db_path = Path(db_path)
        self.engine = create_engine(f'sqlite:///{self.db_path}', echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def get_session(self):
        """Get a new database session"""
        return self.Session()
    
    # ─────────────────────────────────────────────
    # User Management
    # ─────────────────────────────────────────────
    
    def create_user(self, username: str, email: str, password: str, 
                   full_name: str = "", student_id: str = "") -> tuple[bool, str]:
        """
        Create a new user account
        Returns: (success: bool, message: str)
        """
        session = self.get_session()
        try:
            # Check if username exists
            if session.query(User).filter_by(username=username).first():
                return False, "Username already exists"
            
            # Check if email exists
            if session.query(User).filter_by(email=email).first():
                return False, "Email already registered"
            
            # Create user
            user = User(
                username=username,
                email=email,
                full_name=full_name,
                student_id=student_id
            )
            user.set_password(password)
            
            session.add(user)
            session.commit()
            return True, "Account created successfully!"
            
        except Exception as e:
            session.rollback()
            return False, f"Error creating account: {str(e)}"
        finally:
            session.close()
    
    def authenticate_user(self, username: str, password: str) -> tuple[bool, str, dict]:
        """
        Authenticate user login
        Returns: (success: bool, message: str, user_data: dict)
        """
        session = self.get_session()
        try:
            user = session.query(User).filter_by(username=username).first()
            
            if not user:
                return False, "Invalid username or password", {}
            
            if not user.check_password(password):
                return False, "Invalid username or password", {}
            
            # Update last login
            user.last_login = datetime.utcnow()
            session.commit()
            
            user_data = {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'full_name': user.full_name,
                'student_id': user.student_id,
            }
            
            return True, "Login successful!", user_data
            
        except Exception as e:
            return False, f"Login error: {str(e)}", {}
        finally:
            session.close()
    
    def get_user_by_id(self, user_id: int):
        """Get user by ID"""
        session = self.get_session()
        try:
            return session.query(User).filter_by(id=user_id).first()
        finally:
            session.close()
    
    # ─────────────────────────────────────────────
    # Chat History Management
    # ─────────────────────────────────────────────
    
    def save_message(self, user_id: int, role: str, content: str, course_filter: str = None):
        """Save a chat message to history"""
        session = self.get_session()
        try:
            message = ChatMessage(
                user_id=user_id,
                role=role,
                content=content,
                course_filter=course_filter
            )
            session.add(message)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error saving message: {e}")
        finally:
            session.close()
    
    def get_user_messages(self, user_id: int, limit: int = 100):
        """
        Get chat history for a user
        Returns: list of dicts with {role, content, timestamp, course_filter}
        """
        session = self.get_session()
        try:
            messages = session.query(ChatMessage)\
                .filter_by(user_id=user_id)\
                .order_by(ChatMessage.timestamp.desc())\
                .limit(limit)\
                .all()
            
            return [{
                'role': msg.role,
                'content': msg.content,
                'timestamp': msg.timestamp,
                'course_filter': msg.course_filter
            } for msg in reversed(messages)]  # Reverse to chronological order
            
        finally:
            session.close()
    
    def clear_user_messages(self, user_id: int):
        """Clear all chat history for a user"""
        session = self.get_session()
        try:
            session.query(ChatMessage).filter_by(user_id=user_id).delete()
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error clearing messages: {e}")
        finally:
            session.close()
    
    def get_message_count(self, user_id: int) -> int:
        """Get total message count for a user"""
        session = self.get_session()
        try:
            return session.query(ChatMessage).filter_by(user_id=user_id).count()
        finally:
            session.close()
    
    # ─────────────────────────────────────────────
    # Admin/Stats Functions
    # ─────────────────────────────────────────────
    
    def get_user_count(self) -> int:
        """Get total number of registered users"""
        session = self.get_session()
        try:
            return session.query(User).count()
        finally:
            session.close()
    
    def get_total_messages(self) -> int:
        """Get total messages across all users"""
        session = self.get_session()
        try:
            return session.query(ChatMessage).count()
        finally:
            session.close()

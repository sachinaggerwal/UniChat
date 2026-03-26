"""
forum_db.py - Database models for community forum
Handles threads, comments, likes, and user interactions
"""
from datetime import datetime
from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

Base = declarative_base()


class ForumThread(Base):
    """Forum thread/post model"""
    __tablename__ = 'forum_threads'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False, index=True)
    username = Column(String(50), nullable=False)
    title = Column(String(200), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_pinned = Column(Boolean, default=False)
    
    # Relationships
    comments = relationship("ForumComment", back_populates="thread", cascade="all, delete-orphan")
    likes = relationship("ForumLike", back_populates="thread", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<ForumThread(id={self.id}, title='{self.title}', user='{self.username}')>"


class ForumComment(Base):
    """Comment/reply model"""
    __tablename__ = 'forum_comments'
    
    id = Column(Integer, primary_key=True)
    thread_id = Column(Integer, ForeignKey('forum_threads.id'), nullable=False, index=True)
    user_id = Column(Integer, nullable=False)
    username = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationship
    thread = relationship("ForumThread", back_populates="comments")
    
    def __repr__(self):
        return f"<ForumComment(id={self.id}, thread_id={self.thread_id}, user='{self.username}')>"


class ForumLike(Base):
    """Like model"""
    __tablename__ = 'forum_likes'
    
    id = Column(Integer, primary_key=True)
    thread_id = Column(Integer, ForeignKey('forum_threads.id'), nullable=False, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    thread = relationship("ForumThread", back_populates="likes")
    
    def __repr__(self):
        return f"<ForumLike(thread_id={self.thread_id}, user_id={self.user_id})>"


class ForumManager:
    """Manages forum database operations"""
    
    def __init__(self, db_path: str = "unichat_forum.db"):
        """Initialize forum database"""
        self.db_path = Path(db_path)
        self.engine = create_engine(f'sqlite:///{self.db_path}', echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def get_session(self):
        """Get a new database session"""
        return self.Session()
    
    # ═══════════════════════════════════════════════
    # Thread Operations
    # ═══════════════════════════════════════════════
    
    def create_thread(self, user_id: int, username: str, title: str, content: str) -> tuple[bool, str, int]:
        """
        Create a new thread
        Returns: (success: bool, message: str, thread_id: int)
        """
        session = self.get_session()
        try:
            thread = ForumThread(
                user_id=user_id,
                username=username,
                title=title,
                content=content
            )
            session.add(thread)
            session.commit()
            thread_id = thread.id
            return True, "Thread created successfully!", thread_id
        except Exception as e:
            session.rollback()
            return False, f"Error creating thread: {str(e)}", 0
        finally:
            session.close()
    
    def get_all_threads(self, limit: int = 50, sort_by: str = "recent"):
        """
        Get all threads
        sort_by: "recent", "popular", "oldest"
        Returns: list of thread dicts
        """
        session = self.get_session()
        try:
            query = session.query(ForumThread)
            
            if sort_by == "recent":
                query = query.order_by(ForumThread.created_at.desc())
            elif sort_by == "oldest":
                query = query.order_by(ForumThread.created_at.asc())
            elif sort_by == "popular":
                # Sort by number of likes
                query = query.outerjoin(ForumLike).group_by(ForumThread.id).order_by(
                    ForumThread.is_pinned.desc(),
                    session.query(ForumLike).filter(ForumLike.thread_id == ForumThread.id).count().desc()
                )
            
            threads = query.limit(limit).all()
            
            return [{
                'id': t.id,
                'user_id': t.user_id,
                'username': t.username,
                'title': t.title,
                'content': t.content,
                'created_at': t.created_at,
                'is_pinned': t.is_pinned,
                'comment_count': len(t.comments),
                'like_count': len(t.likes)
            } for t in threads]
        finally:
            session.close()
    
    def get_thread_by_id(self, thread_id: int):
        """Get a specific thread with all details"""
        session = self.get_session()
        try:
            thread = session.query(ForumThread).filter_by(id=thread_id).first()
            if not thread:
                return None
            
            return {
                'id': thread.id,
                'user_id': thread.user_id,
                'username': thread.username,
                'title': thread.title,
                'content': thread.content,
                'created_at': thread.created_at,
                'is_pinned': thread.is_pinned,
                'comment_count': len(thread.comments),
                'like_count': len(thread.likes)
            }
        finally:
            session.close()
    
    def delete_thread(self, thread_id: int, user_id: int) -> tuple[bool, str]:
        """Delete a thread (only by owner)"""
        session = self.get_session()
        try:
            thread = session.query(ForumThread).filter_by(id=thread_id).first()
            if not thread:
                return False, "Thread not found"
            if thread.user_id != user_id:
                return False, "You can only delete your own threads"
            
            session.delete(thread)
            session.commit()
            return True, "Thread deleted successfully"
        except Exception as e:
            session.rollback()
            return False, f"Error deleting thread: {str(e)}"
        finally:
            session.close()
    
    # ═══════════════════════════════════════════════
    # Comment Operations
    # ═══════════════════════════════════════════════
    
    def add_comment(self, thread_id: int, user_id: int, username: str, content: str) -> tuple[bool, str]:
        """Add a comment to a thread"""
        session = self.get_session()
        try:
            comment = ForumComment(
                thread_id=thread_id,
                user_id=user_id,
                username=username,
                content=content
            )
            session.add(comment)
            session.commit()
            return True, "Comment added successfully!"
        except Exception as e:
            session.rollback()
            return False, f"Error adding comment: {str(e)}"
        finally:
            session.close()
    
    def get_thread_comments(self, thread_id: int):
        """Get all comments for a thread"""
        session = self.get_session()
        try:
            comments = session.query(ForumComment)\
                .filter_by(thread_id=thread_id)\
                .order_by(ForumComment.created_at.asc())\
                .all()
            
            return [{
                'id': c.id,
                'user_id': c.user_id,
                'username': c.username,
                'content': c.content,
                'created_at': c.created_at
            } for c in comments]
        finally:
            session.close()
    
    def delete_comment(self, comment_id: int, user_id: int) -> tuple[bool, str]:
        """Delete a comment (only by owner)"""
        session = self.get_session()
        try:
            comment = session.query(ForumComment).filter_by(id=comment_id).first()
            if not comment:
                return False, "Comment not found"
            if comment.user_id != user_id:
                return False, "You can only delete your own comments"
            
            session.delete(comment)
            session.commit()
            return True, "Comment deleted"
        except Exception as e:
            session.rollback()
            return False, f"Error: {str(e)}"
        finally:
            session.close()
    
    # ═══════════════════════════════════════════════
    # Like Operations
    # ═══════════════════════════════════════════════
    
    def toggle_like(self, thread_id: int, user_id: int) -> tuple[bool, str, bool]:
        """
        Toggle like on a thread
        Returns: (success: bool, message: str, is_liked: bool)
        """
        session = self.get_session()
        try:
            existing_like = session.query(ForumLike)\
                .filter_by(thread_id=thread_id, user_id=user_id)\
                .first()
            
            if existing_like:
                # Unlike
                session.delete(existing_like)
                session.commit()
                return True, "Unliked", False
            else:
                # Like
                new_like = ForumLike(thread_id=thread_id, user_id=user_id)
                session.add(new_like)
                session.commit()
                return True, "Liked", True
        except Exception as e:
            session.rollback()
            return False, f"Error: {str(e)}", False
        finally:
            session.close()
    
    def user_has_liked(self, thread_id: int, user_id: int) -> bool:
        """Check if user has liked a thread"""
        session = self.get_session()
        try:
            like = session.query(ForumLike)\
                .filter_by(thread_id=thread_id, user_id=user_id)\
                .first()
            return like is not None
        finally:
            session.close()
    
    def get_thread_likes(self, thread_id: int) -> int:
        """Get like count for a thread"""
        session = self.get_session()
        try:
            return session.query(ForumLike).filter_by(thread_id=thread_id).count()
        finally:
            session.close()
    
    # ═══════════════════════════════════════════════
    # Statistics
    # ═══════════════════════════════════════════════
    
    def get_user_stats(self, user_id: int):
        """Get user's forum statistics"""
        session = self.get_session()
        try:
            thread_count = session.query(ForumThread).filter_by(user_id=user_id).count()
            comment_count = session.query(ForumComment).filter_by(user_id=user_id).count()
            
            return {
                'threads_created': thread_count,
                'comments_posted': comment_count,
                'total_activity': thread_count + comment_count
            }
        finally:
            session.close()
    
    def get_total_stats(self):
        """Get overall forum statistics"""
        session = self.get_session()
        try:
            return {
                'total_threads': session.query(ForumThread).count(),
                'total_comments': session.query(ForumComment).count(),
                'total_likes': session.query(ForumLike).count()
            }
        finally:
            session.close()

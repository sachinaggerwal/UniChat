"""
forum_ui.py - Community forum UI components  
Minimalistic, modern design with threads, comments, and likes
SIMPLE VERSION - No nested comments, just HTML fix
"""
import streamlit as st
from datetime import datetime
from forum_db import ForumManager
import html  # For HTML escaping


def render_forum_page(db_manager, user):
    """
    Render the complete community forum page
    
    Args:
        db_manager: Main database manager (for user info)
        user: Current logged-in user dict
    """
    
    # Initialize forum manager
    if "forum_manager" not in st.session_state:
        st.session_state.forum_manager = ForumManager()
    
    forum_mgr = st.session_state.forum_manager
    
    # Forum state management
    if "forum_view" not in st.session_state:
        st.session_state.forum_view = "list"
    if "selected_thread_id" not in st.session_state:
        st.session_state.selected_thread_id = None
    if "forum_sort" not in st.session_state:
        st.session_state.forum_sort = "recent"
    
    # Inject forum-specific CSS
    st.markdown(get_forum_css(), unsafe_allow_html=True)
    
    # Render based on view
    if st.session_state.forum_view == "list":
        render_thread_list(forum_mgr, user)
    elif st.session_state.forum_view == "thread":
        render_thread_detail(forum_mgr, user)


def get_forum_css():
    """Get forum-specific CSS"""
    return """
    <style>
    /* Forum Container */
    .forum-container {
        max-width: 900px;
        margin: 0 auto;
    }
    
    /* Forum Header */
    .forum-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 24px;
        padding-bottom: 16px;
        border-bottom: 2px solid var(--accent-red);
    }
    
    .forum-title {
        font-size: 28px;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
    }
    
    .forum-subtitle {
        font-size: 14px;
        color: var(--text-secondary);
        margin: 4px 0 0;
    }
    
    /* Thread Card */
    .thread-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        transition: all 0.2s ease;
        cursor: pointer;
    }
    
    .thread-card:hover {
        border-color: var(--accent-red);
        box-shadow: 0 4px 12px rgba(179,17,45,0.1);
        transform: translateY(-2px);
    }
    
    .thread-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 12px;
    }
    
    .thread-title {
        font-size: 18px;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0 0 4px;
        line-height: 1.4;
    }
    
    .thread-meta {
        font-size: 12px;
        color: var(--text-tertiary);
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .thread-author {
        font-weight: 600;
        color: var(--accent-red);
    }
    
    .thread-content {
        font-size: 14px;
        color: var(--text-secondary);
        line-height: 1.6;
        margin: 12px 0;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    
    .thread-stats {
        display: flex;
        gap: 16px;
        font-size: 13px;
        color: var(--text-tertiary);
        margin-top: 12px;
        padding-top: 12px;
        border-top: 1px solid var(--border-light);
    }
    
    .thread-stat {
        display: flex;
        align-items: center;
        gap: 4px;
    }
    
    /* Comment Card */
    .comment-card {
        background: var(--bg-tertiary);
        border-left: 3px solid var(--accent-red);
        border-radius: 8px;
        padding: 16px;
        margin: 12px 0;
    }
    
    .comment-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    
    .comment-author {
        font-weight: 600;
        font-size: 14px;
        color: var(--accent-red);
    }
    
    .comment-time {
        font-size: 12px;
        color: var(--text-tertiary);
    }
    
    .comment-content {
        font-size: 14px;
        color: var(--text-primary);
        line-height: 1.6;
    }
    
    /* Empty State */
    .empty-state {
        text-align: center;
        padding: 60px 20px;
        color: var(--text-tertiary);
    }
    
    .empty-state-icon {
        font-size: 48px;
        margin-bottom: 16px;
    }
    </style>
    """


def render_thread_list(forum_mgr, user):
    """Render the thread list view"""
    
    st.markdown('<div class="forum-container">', unsafe_allow_html=True)
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(
            '<div class="forum-header">'
            '<div>'
            '<h1 class="forum-title">💬 Community Forum</h1>'
            '<p class="forum-subtitle">Share ideas, ask questions, and connect with fellow students</p>'
            '</div></div>',
            unsafe_allow_html=True
        )
    
    # Controls row
    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 1])
    with ctrl1:
        if st.button("➕ New Thread", type="primary", use_container_width=True):
            st.session_state.show_new_thread = True
            st.rerun()
    
    with ctrl2:
        sort_option = st.selectbox(
            "Sort by",
            ["Recent", "Popular", "Oldest"],
            key="forum_sort_select"
        )
        st.session_state.forum_sort = sort_option.lower()
    
    with ctrl3:
        if st.button("🔄 Refresh"):
            st.rerun()
    
    # New thread modal
    if st.session_state.get("show_new_thread", False):
        render_new_thread_form(forum_mgr, user)
    
    st.markdown("---")
    
    # Get threads
    threads = forum_mgr.get_all_threads(limit=50, sort_by=st.session_state.forum_sort)
    
    if not threads:
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-state-icon">📭</div>'
            '<div class="empty-state-text">No threads yet</div>'
            '<p>Be the first to start a conversation!</p>'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        # Render threads
        for thread in threads:
            render_thread_card(thread, forum_mgr, user)
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_thread_card(thread, forum_mgr, user):
    """Render a single thread card"""
    
    time_ago = get_time_ago(thread['created_at'])
    
    # CRITICAL: Escape HTML to prevent code from showing
    safe_title = html.escape(thread['title'])
    safe_content = html.escape(thread['content'])
    safe_username = html.escape(thread['username'])
    
    card_html = f"""
    <div class="thread-card">
        <div class="thread-header">
            <div>
                <h3 class="thread-title">{safe_title}</h3>
                <div class="thread-meta">
                    <span class="thread-author">@{safe_username}</span>
                    <span>•</span>
                    <span>{time_ago}</span>
                </div>
            </div>
        </div>
        <div class="thread-content">{safe_content}</div>
        <div class="thread-stats">
            <div class="thread-stat">
                <span>❤️</span>
                <span>{thread['like_count']} likes</span>
            </div>
            <div class="thread-stat">
                <span>💬</span>
                <span>{thread['comment_count']} comments</span>
            </div>
        </div>
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)
    
    # Clickable button
    if st.button(f"Open thread", key=f"thread_{thread['id']}", use_container_width=True):
        st.session_state.forum_view = "thread"
        st.session_state.selected_thread_id = thread['id']
        st.rerun()


def render_new_thread_form(forum_mgr, user):
    """Render form to create a new thread"""
    
    st.markdown("### ✍️ Create New Thread")
    
    with st.form("new_thread_form", clear_on_submit=True):
        title = st.text_input(
            "Thread Title *",
            placeholder="What's on your mind?",
            max_chars=200
        )
        
        content = st.text_area(
            "Content *",
            placeholder="Share your thoughts, ask a question, or start a discussion...",
            height=150,
            max_chars=2000
        )
        
        col1, col2 = st.columns(2)
        with col1:
            submit = st.form_submit_button("📝 Post Thread", type="primary", use_container_width=True)
        with col2:
            cancel = st.form_submit_button("Cancel", use_container_width=True)
        
        if cancel:
            st.session_state.show_new_thread = False
            st.rerun()
        
        if submit:
            if not title or not content:
                st.error("Please fill in both title and content")
            else:
                success, message, thread_id = forum_mgr.create_thread(
                    user_id=user['id'],
                    username=user['username'],
                    title=title,
                    content=content
                )
                
                if success:
                    st.success(message)
                    st.session_state.show_new_thread = False
                    st.balloons()
                    st.rerun()
                else:
                    st.error(message)


def render_thread_detail(forum_mgr, user):
    """Render detailed thread view with comments"""
    
    thread_id = st.session_state.selected_thread_id
    thread = forum_mgr.get_thread_by_id(thread_id)
    
    if not thread:
        st.error("Thread not found")
        if st.button("← Back to Forum"):
            st.session_state.forum_view = "list"
            st.rerun()
        return
    
    st.markdown('<div class="forum-container">', unsafe_allow_html=True)
    
    # Back button
    if st.button("← Back to Forum", type="secondary"):
        st.session_state.forum_view = "list"
        st.rerun()
    
    st.markdown("---")
    
    # Thread header - ESCAPE HTML
    time_ago = get_time_ago(thread['created_at'])
    safe_title = html.escape(thread['title'])
    safe_username = html.escape(thread['username'])
    safe_content = html.escape(thread['content']).replace('\n', '<br>')
    
    st.markdown(
        f'<h1 class="forum-title">{safe_title}</h1>'
        f'<div class="thread-meta">'
        f'<span class="thread-author">@{safe_username}</span>'
        f'<span>•</span>'
        f'<span>{time_ago}</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    
    # Thread content
    st.markdown(f'<div class="comment-content">{safe_content}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Like button and stats
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        user_liked = forum_mgr.user_has_liked(thread_id, user['id'])
        like_label = "❤️ Liked" if user_liked else "🤍 Like"
        like_type = "primary" if user_liked else "secondary"
        
        if st.button(f"{like_label} ({thread['like_count']})", type=like_type, key="like_btn"):
            forum_mgr.toggle_like(thread_id, user['id'])
            st.rerun()
    
    with col2:
        st.metric("💬 Comments", thread['comment_count'])
    
    with col3:
        if thread['user_id'] == user['id']:
            if st.button("🗑️ Delete Thread", type="secondary"):
                success, message = forum_mgr.delete_thread(thread_id, user['id'])
                if success:
                    st.success(message)
                    st.session_state.forum_view = "list"
                    st.rerun()
                else:
                    st.error(message)
    
    st.markdown("---")
    
    # Comments section
    st.markdown("### 💬 Comments")
    
    # Add comment form
    with st.form("add_comment_form", clear_on_submit=True):
        comment_content = st.text_area(
            "Add a comment",
            placeholder="Share your thoughts...",
            height=100,
            max_chars=1000
        )
        
        submit_comment = st.form_submit_button("💬 Post Comment", type="primary")
        
        if submit_comment:
            if not comment_content:
                st.error("Comment cannot be empty")
            else:
                success, message = forum_mgr.add_comment(
                    thread_id=thread_id,
                    user_id=user['id'],
                    username=user['username'],
                    content=comment_content
                )
                
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
    
    # Display comments
    comments = forum_mgr.get_thread_comments(thread_id)
    
    if not comments:
        st.info("No comments yet. Be the first to comment!")
    else:
        for comment in comments:
            render_comment_card(comment, forum_mgr, user)
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_comment_card(comment, forum_mgr, user):
    """Render a single comment"""
    
    time_ago = get_time_ago(comment['created_at'])
    
    # ESCAPE HTML
    safe_username = html.escape(comment['username'])
    safe_content = html.escape(comment['content']).replace('\n', '<br>')
    
    comment_html = f"""
    <div class="comment-card">
        <div class="comment-header">
            <span class="comment-author">@{safe_username}</span>
            <span class="comment-time">{time_ago}</span>
        </div>
        <div class="comment-content">{safe_content}</div>
    </div>
    """
    
    st.markdown(comment_html, unsafe_allow_html=True)
    
    # Delete button (only for comment owner)
    if comment['user_id'] == user['id']:
        if st.button("🗑️ Delete", key=f"del_comment_{comment['id']}", type="secondary"):
            success, message = forum_mgr.delete_comment(comment['id'], user['id'])
            if success:
                st.rerun()
            else:
                st.error(message)


def get_time_ago(dt):
    """Convert datetime to 'time ago' string"""
    now = datetime.utcnow()
    diff = now - dt
    
    seconds = diff.total_seconds()
    
    if seconds < 60:
        return "just now"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes}m ago"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours}h ago"
    elif seconds < 604800:
        days = int(seconds / 86400)
        return f"{days}d ago"
    else:
        return dt.strftime("%b %d, %Y")

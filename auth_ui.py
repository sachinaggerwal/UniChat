"""
auth_ui.py - Streamlit authentication UI components
Provides login and signup forms with Carleton red/black styling
FIXED: Form button issue, session state management, per-user chat isolation
"""
import streamlit as st
import re


def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_password(password: str) -> tuple[bool, str]:
    """
    Validate password strength
    Returns: (is_valid: bool, message: str)
    """
    if len(password) < 6:
        return False, "Password must be at least 6 characters"
    if len(password) > 50:
        return False, "Password must be less than 50 characters"
    return True, ""


def validate_username(username: str) -> tuple[bool, str]:
    """
    Validate username
    Returns: (is_valid: bool, message: str)
    """
    if len(username) < 3:
        return False, "Username must be at least 3 characters"
    if len(username) > 20:
        return False, "Username must be less than 20 characters"
    if not re.match(r'^[a-zA-Z0-9_]+$', username):
        return False, "Username can only contain letters, numbers, and underscores"
    return True, ""


def render_login_page(db_manager):
    """Render the login page"""
    
    # Custom CSS for auth pages
    st.markdown("""
    <style>
    /* Auth page specific styles */
    .auth-container {
        max-width: 450px;
        margin: 60px auto;
        padding: 40px;
        background: white;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    .auth-title {
        text-align: center;
        color: #1a1a1a;
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .auth-subtitle {
        text-align: center;
        color: #666;
        font-size: 14px;
        margin-bottom: 30px;
    }
    .auth-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #B3112D, transparent);
        margin: 20px 0;
    }
    .auth-link {
        text-align: center;
        margin-top: 20px;
        font-size: 14px;
        color: #666;
    }
    .auth-link a {
        color: #B3112D;
        text-decoration: none;
        font-weight: 600;
    }
    .auth-link a:hover {
        text-decoration: underline;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Center container
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        st.markdown(
            '<h1 class="auth-title">Welcome to UniChat</h1>'
            '<p class="auth-subtitle">Sign in to access your personalized course assistant</p>',
            unsafe_allow_html=True
        )
        
        # Login form
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input(
                "Username",
                placeholder="Enter your username",
                help="Your UniChat username"
            )
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password"
            )
            
            col_a, col_b = st.columns(2)
            with col_a:
                remember_me = st.checkbox("Remember me")
            
            submit = st.form_submit_button("🔐 Sign In", use_container_width=True, type="primary")
            
            if submit:
                if not username or not password:
                    st.error("Please enter both username and password")
                else:
                    success, message, user_data = db_manager.authenticate_user(username, password)
                    
                    if success:
                        # CRITICAL: Clear all session state before setting new user
                        # This prevents chat history bleed between users
                        for key in ['messages', 'chat_loaded', 'pending_example']:
                            if key in st.session_state:
                                del st.session_state[key]
                        
                        # Set new user data
                        st.session_state.authenticated = True
                        st.session_state.user = user_data
                        st.session_state.active_page = "unichat"
                        st.session_state.chat_loaded = False  # Force reload for new user
                        st.session_state.messages = []  # Start with empty messages
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
        
        st.markdown('<div class="auth-divider"></div>', unsafe_allow_html=True)
        
        # Switch to signup
        st.markdown(
            '<div class="auth-link">'
            'Don\'t have an account? '
            '<a href="#" onclick="return false;">Create one now</a>'
            '</div>',
            unsafe_allow_html=True
        )
        
        if st.button("➕ Create New Account", use_container_width=True, type="secondary"):
            st.session_state.show_signup = True
            st.session_state.signup_success = False  # Reset success flag
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)


def render_signup_page(db_manager):
    """Render the signup page"""
    
    # Same CSS as login
    st.markdown("""
    <style>
    .auth-container {
        max-width: 500px;
        margin: 40px auto;
        padding: 40px;
        background: white;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    .auth-title {
        text-align: center;
        color: #1a1a1a;
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .auth-subtitle {
        text-align: center;
        color: #666;
        font-size: 14px;
        margin-bottom: 30px;
    }
    .auth-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #B3112D, transparent);
        margin: 20px 0;
    }
    .auth-link {
        text-align: center;
        margin-top: 20px;
        font-size: 14px;
        color: #666;
    }
    .auth-link a {
        color: #B3112D;
        text-decoration: none;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        st.markdown(
            '<h1 class="auth-title">Create Your Account</h1>'
            '<p class="auth-subtitle">Join UniChat and get personalized course assistance</p>',
            unsafe_allow_html=True
        )
        
        # Signup form
        with st.form("signup_form", clear_on_submit=True):
            full_name = st.text_input(
                "Full Name",
                placeholder="John Doe",
                help="Your full name (optional)"
            )
            
            username = st.text_input(
                "Username *",
                placeholder="Choose a unique username",
                max_chars=20,
                help="3-20 characters, letters, numbers, and underscores only"
            )
            
            email = st.text_input(
                "Email Address *",
                placeholder="you@carleton.ca",
                help="Your email address"
            )
            
            student_id = st.text_input(
                "Student ID",
                placeholder="101234567",
                help="Your Carleton student ID (optional)"
            )
            
            col_a, col_b = st.columns(2)
            with col_a:
                password = st.text_input(
                    "Password *",
                    type="password",
                    placeholder="Create a password",
                    help="At least 6 characters"
                )
            with col_b:
                confirm_password = st.text_input(
                    "Confirm Password *",
                    type="password",
                    placeholder="Re-enter password"
                )
            
            agree_terms = st.checkbox("I agree to the Terms of Service and Privacy Policy")
            
            submit = st.form_submit_button("🎓 Create Account", use_container_width=True, type="primary")
            
            if submit:
                # Validation
                errors = []
                
                if not username or not email or not password:
                    errors.append("Please fill in all required fields (marked with *)")
                
                if username:
                    valid, msg = validate_username(username)
                    if not valid:
                        errors.append(msg)
                
                if email and not validate_email(email):
                    errors.append("Please enter a valid email address")
                
                if password:
                    valid, msg = validate_password(password)
                    if not valid:
                        errors.append(msg)
                
                if password != confirm_password:
                    errors.append("Passwords do not match")
                
                if not agree_terms:
                    errors.append("You must agree to the Terms of Service")
                
                if errors:
                    for error in errors:
                        st.error(error)
                else:
                    # Create account
                    success, message = db_manager.create_user(
                        username=username,
                        email=email,
                        password=password,
                        full_name=full_name,
                        student_id=student_id
                    )
                    
                    if success:
                        st.success(message)
                        st.info("You can now sign in with your credentials")
                        st.balloons()
                        # Set flag to show "Go to Login" button OUTSIDE form
                        st.session_state.signup_success = True
                    else:
                        st.error(message)
        
        # Show "Go to Login" button OUTSIDE form if signup was successful
        if st.session_state.get("signup_success", False):
            st.success("✅ Account created successfully! Please log in with your credentials.")
            if st.button("➡️ Go to Login Now", use_container_width=True, type="primary", key="goto_login"):
                st.session_state.show_signup = False
                st.session_state.signup_success = False  # Reset flag
                st.rerun()
        
        st.markdown('<div class="auth-divider"></div>', unsafe_allow_html=True)
        
        # Switch to login
        st.markdown(
            '<div class="auth-link">'
            'Already have an account? '
            '<a href="#" onclick="return false;">Sign in</a>'
            '</div>',
            unsafe_allow_html=True
        )
        
        if st.button("🔙 Back to Login", use_container_width=True, type="secondary"):
            st.session_state.show_signup = False
            st.session_state.signup_success = False  # Also reset here
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)


def render_user_profile_sidebar(db_manager):
    """Render user profile info in sidebar"""
    if not st.session_state.get('user'):
        return
    
    user = st.session_state.user
    
    st.markdown("---")
    st.markdown("### 👤 User Profile")
    st.markdown(f"**{user.get('full_name') or user['username']}**")
    st.caption(f"@{user['username']}")
    
    if user.get('student_id'):
        st.caption(f"ID: {user['student_id']}")
    
    # Message count - get fresh count from database for THIS user
    user_id = user['id']
    msg_count = db_manager.get_message_count(user_id)
    st.metric("Messages Sent", msg_count, help=f"Total messages for user {user['username']}")
    
    st.markdown("---")
    
    # Logout button
    if st.button("🚪 Logout", use_container_width=True):
        # Clear ALL session state to prevent data bleed between users
        keys_to_clear = [
            'authenticated', 
            'user', 
            'messages', 
            'chat_loaded',  # CRITICAL - must clear this
            'active_page',
            'pending_example',
            'selected_course',
            'retrieval_k',
            'signup_success'
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        # Reset to defaults
        st.session_state.authenticated = False
        st.session_state.user = None
        st.session_state.messages = []
        st.session_state.chat_loaded = False
        st.session_state.active_page = "unichat"
        
        st.rerun()

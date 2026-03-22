"""
theme_manager.py - Dynamic theme management for UniChat
Provides light/dark mode toggle with CSS variable-based theming
FIXED: Proper color inversion, button text colors, icon-only toggle
"""
import streamlit as st


def get_theme_css(dark_mode: bool = True) -> str:
    """
    Generate theme-specific CSS using CSS variables
    
    Args:
        dark_mode: If True, use dark theme. If False, use light theme.
    
    Returns:
        CSS string with theme variables injected
    """
    
    if dark_mode:
        # DARK MODE (Carleton Theme - Default)
        # Both sidebar AND main area are DARK
        theme_vars = """
        /* Dark Theme Variables */
        :root {
            --bg-primary: #1a1a1a;              /* Main area - DARK */
            --bg-secondary: #2a2a2a;
            --bg-tertiary: #1f1f1f;
            
            --sidebar-bg: #1a1a1a;              /* Sidebar - DARK (same as main) */
            --sidebar-text: #d0d0d0;
            --sidebar-heading: #ffffff;
            
            --nav-bg: #1a1a1a;                  /* Nav bar - DARK */
            --nav-text: #c8c8c8;
            --nav-text-hover: #ffffff;
            --nav-active-bg: #B3112D;
            
            --text-primary: #ffffff;            /* Main text - LIGHT */
            --text-secondary: #d0d0d0;
            --text-tertiary: #b0b0b0;
            
            --border-color: #444;
            --border-light: #3a3a3a;
            
            --card-bg: #2a2a2a;
            --card-border: #3a3a3a;
            
            --pill-bg: #3a2a2d;
            --pill-border: #4a3a3d;
            --pill-text: #ff6b6b;
            
            --badge-bg: #2a2f3a;
            --badge-border: #3a3f4a;
            --badge-label: #9aa5b9;
            --badge-text: #ffffff;
            
            --accent-red: #B3112D;
            --accent-red-hover: #950f27;
            --accent-red-light: rgba(179,17,45,0.35);
            
            --chat-bg: #1f1f1f;
            --chat-border: #3a3a3a;
            
            --button-text: #ffffff;             /* Button text - WHITE */
        }
        """
    else:
        # LIGHT MODE
        # Both sidebar AND main area are LIGHT
        theme_vars = """
        /* Light Theme Variables */
        :root {
            --bg-primary: #f4f5f7;              /* Main area - LIGHT */
            --bg-secondary: #ffffff;
            --bg-tertiary: #fafafa;
            
            --sidebar-bg: #f4f5f7;              /* Sidebar - LIGHT (same as main) */
            --sidebar-text: #333333;
            --sidebar-heading: #1a1a1a;
            
            --nav-bg: #f4f5f7;                  /* Nav bar - LIGHT */
            --nav-text: #333333;
            --nav-text-hover: #1a1a1a;
            --nav-active-bg: #B3112D;
            
            --text-primary: #1a1a1a;            /* Main text - DARK */
            --text-secondary: #666666;
            --text-tertiary: #888888;
            
            --border-color: #e4e4e4;
            --border-light: #eeeeee;
            
            --card-bg: #ffffff;
            --card-border: #e4e4e4;
            
            --pill-bg: #fff0f2;
            --pill-border: #f0c0c8;
            --pill-text: #B3112D;
            
            --badge-bg: #f0f4ff;
            --badge-border: #d4dff5;
            --badge-label: #6b7a99;
            --badge-text: #1a1a1a;
            
            --accent-red: #B3112D;
            --accent-red-hover: #950f27;
            --accent-red-light: rgba(179,17,45,0.25);
            
            --chat-bg: #fafafa;
            --chat-border: #e8e8e8;
            
            --button-text: #ffffff;             /* Button text - WHITE */
        }
        """
    
    # Base CSS using variables (theme-independent)
    base_css = """
<style>
""" + theme_vars + """

/* ═══════════════════════════════════════════════
   BASE / RESET
   ═══════════════════════════════════════════════ */
.stApp {
    background-color: var(--bg-primary);
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    transition: background-color 0.3s ease;
}

/* ═══════════════════════════════════════════════
   LEFT PANE  –  Streamlit sidebar
   ═══════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background-color: var(--sidebar-bg) !important;
    border-right: 3px solid var(--accent-red);
    min-width: 240px;
    max-width: 300px;
    transition: background-color 0.3s ease;
}

[data-testid="stSidebarCollapseButton"] {
    background: transparent !important;
    border: none !important;
}

[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stText,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: var(--sidebar-text) !important;
    font-size: 13px !important;
}

[data-testid="stSidebar"] h3 {
    color: var(--sidebar-heading) !important;
    font-size: 14px !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 20px !important;
    margin-bottom: 8px !important;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border-light);
}

/* ─── COMPACT LOGO SECTION ─── */
.sidebar-logo-section {
  overflow: visible;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;

  /* slightly bigger badge */
  padding: 26px 16px;
  margin: 14px 10px 14px;
  width: calc(100% - 20px);

  background: #ffffff;
  border: 2px solid #B3112D;
  border-radius: 16px;
}
.sidebar-logo-section img {
  width: 100%;
  max-width: 285px;   /* increase for larger logo */
  max-height: 105px;   /* increase for larger logo */
  object-fit: contain;
  display: block;
  margin: 0 auto;;
  background: #fff;               /* fixes the “transparent checkerboard” look */
  padding: 8px 10px;
  border-radius: 12px;
}
.sidebar-divider {
    width: 100%;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-red), transparent);
    margin: 10px 0;
}

/* ─── SIDEBAR BUTTONS - Force white text ─── */
[data-testid="stSidebar"] .stButton button {
    background-color: var(--accent-red) !important;
    color: var(--button-text) !important;  /* WHITE TEXT */
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    padding: 6px 12px !important;
    transition: background-color 0.2s ease;
}

[data-testid="stSidebar"] .stButton button:hover {
    background-color: var(--accent-red-hover) !important;
    color: var(--button-text) !important;  /* Keep WHITE on hover */
}

/* ═══════════════════════════════════════════════
   RIGHT PANE - Main content area
   ═══════════════════════════════════════════════ */
.main .block-container {
    padding-top: 0 !important;
    padding-left: 24px !important;
    padding-right: 24px !important;
    max-width: 100% !important;
}

/* ─── Top Navigation Bar ─── */
.cu-topnav {
    display: flex;
    align-items: center;
    background-color: var(--nav-bg);
    padding: 0 20px;
    height: 52px;
    gap: 6px;
    margin: -8px -24px 0 -24px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.25);
    transition: background-color 0.3s ease;
}

.cu-nav-btn {
    display: inline-block;
    color: var(--nav-text);
    background: transparent;
    padding: 7px 18px;
    font-size: 14px;
    font-weight: 500;
    border-radius: 6px;
    white-space: nowrap;
    transition: all 0.2s ease;
}

.cu-nav-btn.active {
    background: var(--nav-active-bg);
    color: #fff !important;
    font-weight: 600;
}

.cu-red-rule {
    height: 3px;
    background: var(--accent-red);
    margin: 0 -24px;
}

/* ─── Page Title ─── */
.cu-page-title {
    font-size: 26px;
    font-weight: 700;
    color: var(--text-primary);
    margin: 20px 0 4px;
    letter-spacing: -0.3px;
}

.cu-page-subtitle {
    font-size: 13px;
    color: var(--text-tertiary);
    margin: 0 0 18px;
}

/* ─── UniChat logo row ─── */
.cu-unichat-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin: 18px 0 16px;
}

.cu-unichat-header img {
    width: 64px;
    height: 64px;
    object-fit: contain;
}

.cu-unichat-header .cu-uh-text h2 {
    margin: 0;
    font-size: 22px;
    color: var(--text-primary);
}

.cu-unichat-header .cu-uh-text p {
    margin: 2px 0 0;
    font-size: 13px;
    color: var(--text-secondary);
}

/* ─── Example question pills ─── */
.cu-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 12px 0 4px;
}

.cu-pill {
    background: var(--pill-bg);
    border: 1px solid var(--pill-border);
    color: var(--pill-text);
    border-radius: 20px;
    padding: 5px 13px;
    font-size: 12.5px;
    transition: all 0.2s ease;
}

/* ─── Metrics badges ─── */
.cu-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 14px;
}

.cu-badge {
    background: var(--badge-bg);
    border: 1px solid var(--badge-border);
    border-radius: 8px;
    padding: 8px 14px;
    min-width: 110px;
    text-align: center;
}

.cu-badge-label {
    font-size: 10.5px;
    color: var(--badge-label);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.cu-badge-value {
    font-size: 17px;
    font-weight: 700;
    color: var(--badge-text);
    margin-top: 2px;
}

/* ─── Cards (About, FAQ) ─── */
.cu-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 10px;
    padding: 22px 26px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}

.cu-card h3 {
    color: var(--accent-red);
    margin-top: 0;
    font-size: 16px;
    border-bottom: 1px solid var(--border-light);
    padding-bottom: 8px;
}

.cu-card p, .cu-card li {
    color: var(--text-secondary);
    font-size: 14px;
    line-height: 1.6;
}

/* ═══════════════════════════════════════════════
   STREAMLIT NATIVE ELEMENTS
   ═══════════════════════════════════════════════ */

/* Chat messages */
.stChatMessage {
    background-color: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
}

.stChatMessage p {
    color: var(--text-primary) !important;
}

/* Text inputs */
.stTextInput input, 
.stTextArea textarea,
.stSelectbox select {
    background-color: var(--card-bg) !important;
    color: var(--text-primary) !important;
    border-color: var(--border-color) !important;
}

/* Markdown content in main area */
.main .stMarkdown {
    color: var(--text-primary);
}

.main .stMarkdown p,
.main .stMarkdown li,
.main .stMarkdown span {
    color: var(--text-primary) !important;
}

/* Headers in main area */
.main h1, .main h2, .main h3, .main h4 {
    color: var(--text-primary) !important;
}

/* ═══════════════════════════════════════════════
   RESPONSIVE
   ═══════════════════════════════════════════════ */
@media (max-width: 768px) {
    .main .block-container { 
        padding-left: 12px !important; 
        padding-right: 12px !important; 
    }
}
</style>
"""
    
    return base_css


def render_theme_toggle():
    """
    Render a sleek icon-only theme toggle button in the sidebar
    Returns: True if theme was toggled, False otherwise
    """
    current_mode = st.session_state.get("dark_mode", True)
    
    # Icon ONLY based on what mode you'll switch TO
    if current_mode:
        # Currently dark mode → show sun (will switch to light)
        icon = "☀️"
        tooltip = "Switch to Light Mode"
    else:
        # Currently light mode → show moon (will switch to dark)
        icon = "🌙"
        tooltip = "Switch to Dark Mode"
    
    # Render toggle button with just the icon
    if st.button(
        icon,
        key="theme_toggle",
        use_container_width=True,
        help=tooltip
    ):
        # Toggle the theme
        st.session_state.dark_mode = not current_mode
        return True
    
    return False

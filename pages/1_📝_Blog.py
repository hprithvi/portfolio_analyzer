import streamlit as st
import re
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.theme import apply_theme

st.set_page_config(page_title="Blog", page_icon="📝", layout="wide")
apply_theme()

BLOGS_DIR = Path(__file__).resolve().parent.parent / "blogs"


def parse_front_matter(content):
    """Extract metadata from <!-- key: value --> comments at the top of a Markdown file."""
    metadata = {}
    for match in re.finditer(r'<!--\s*(\w+):\s*(.+?)\s*-->', content):
        metadata[match.group(1)] = match.group(2)
    return metadata


def load_posts():
    """Load all .md files from the blogs directory, sorted by date (newest first)."""
    posts = []
    if not BLOGS_DIR.exists():
        return posts
    for md_file in BLOGS_DIR.glob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        meta = parse_front_matter(content)
        posts.append({
            "filename": md_file.name,
            "title": meta.get("title", md_file.stem.replace("-", " ").title()),
            "date": meta.get("date", ""),
            "summary": meta.get("summary", ""),
            "content": content,
        })
    posts.sort(key=lambda p: p["date"], reverse=True)
    return posts


def show_listing(posts):
    """Render the blog listing view with cards for each post."""
    st.title("📝 Blog")
    st.markdown("Thoughts on investing, portfolio strategy, and market analysis.")
    st.markdown("---")

    if not posts:
        st.info("No blog posts yet. Add `.md` files to the `blogs/` folder to get started.")
        return

    for post in posts:
        with st.container():
            st.subheader(post["title"])
            if post["date"]:
                st.caption(post["date"])
            if post["summary"]:
                st.markdown(post["summary"])
            if st.button("Read →", key=f"read_{post['filename']}"):
                st.session_state.selected_post = post["filename"]
                st.rerun()
            st.markdown("---")


def show_post(posts, filename):
    """Render a single blog post."""
    post = next((p for p in posts if p["filename"] == filename), None)
    if post is None:
        st.error("Post not found.")
        return

    if st.button("← Back to all posts"):
        st.session_state.selected_post = None
        st.rerun()

    st.markdown("---")

    # Strip front-matter comments before rendering
    body = re.sub(r'<!--\s*\w+:\s*.+?\s*-->\n?', '', post["content"]).strip()
    st.markdown(body, unsafe_allow_html=True)


def main():
    if "selected_post" not in st.session_state:
        st.session_state.selected_post = None

    posts = load_posts()

    if st.session_state.selected_post:
        show_post(posts, st.session_state.selected_post)
    else:
        show_listing(posts)


main()

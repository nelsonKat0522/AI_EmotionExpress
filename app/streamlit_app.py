from pathlib import Path
import sys

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )


st.set_page_config(
    page_title="EmoExpress",
    page_icon="💙",
    layout="wide",
)


generate_page = st.Page(
    "pages/generate_page.py",
    title="Generate",
    icon="✨",
    default=True,
)

history_page = st.Page(
    "pages/history_page.py",
    title="History",
    icon="🕘",
)


navigation = st.navigation(
    [
        generate_page,
        history_page,
    ],
    position="top",
)

navigation.run()
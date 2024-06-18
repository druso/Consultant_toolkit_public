from src.setup import page_setup, page_footer
import streamlit as st
import pandas as pd
import os

page_config = {'page_title':"Consulting toolkit",
          'page_icon':"🛠️",}

page_setup(page_config)

with open("readme.md", "r", encoding="utf-8") as f:
    readme_content = f.read()

st.write(readme_content)

page_footer()
import urllib.parse
import streamlit as st

st.set_page_config(
    page_title="PRO-B GAN KG | Coming Soon",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DEFAULT_URL = "https://your-deploy-url.example.com"

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&display=swap');
    body {background: #f8fafc;}
    .title {font-family: 'Space Grotesk', sans-serif; font-size: 2.6rem; font-weight: 700; margin-bottom: 0.3rem;}
    .sub {font-family: 'Space Grotesk', sans-serif; font-size: 1.1rem; color: #334155;}
    .soon {font-family: 'Space Grotesk', sans-serif; font-size: 3.2rem; font-weight: 700; color: #0f172a; margin: 1.2rem 0 1.4rem 0;}
    .box {padding: 1.1rem 1.2rem; border-radius: 14px; background: #ffffff; border: 1px solid #e2e8f0; box-shadow: 3px 5px 0 #0ea5e9;}
    .label {font-size: 0.95rem; color: #475569;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="title">PRO-B GAN KG</div>
    <div class="sub">All-in-one knowledge graph with GAN-powered link predictions.</div>
    <div class="soon">COMING SOON</div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

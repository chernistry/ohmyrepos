"""Streamlit UI for Oh My Repos.

This module provides a web interface for searching GitHub starred repositories.
"""

import asyncio
import logging
import streamlit as st
import pandas as pd
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the project's root directory to the module search path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Now import modules from the project
from src.core.retriever import HybridRetriever
from src.core.reranker import JinaReranker
from src.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Oh My Repos",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global state for retriever and reranker
@st.cache_resource
def get_retriever():
    """Initialize and return the retriever."""
    retriever = HybridRetriever(
        bm25_variant=settings.BM25_VARIANT,
        bm25_weight=settings.BM25_WEIGHT,
        vector_weight=settings.VECTOR_WEIGHT,
    )
    
    # Инициализируем retriever синхронно
    # Используем nest_asyncio для решения проблемы с вложенными циклами событий
    import nest_asyncio
    nest_asyncio.apply()
    
    # Теперь можно безопасно запустить асинхронный код
    asyncio.run(retriever.initialize())
    return retriever

@st.cache_resource
def get_reranker():
    """Initialize and return the reranker."""
    return JinaReranker()

# Helper functions
def make_clickable(url, text):
    """Make a clickable link for Streamlit dataframe."""
    return f'<a href="{url}" target="_blank">{text}</a>'

def search_repos_sync(
    query: str,
    top_k: int = 25,
    filter_tags: Optional[List[str]] = None,
    filter_language: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Synchronous wrapper for the asynchronous search.

    Args:
        query: User search query.
        top_k: Number of final results requested.
        filter_tags: Optional tag filter for retriever.
        filter_language: Optional programming language filter (case-insensitive).
    """
    # Use nest_asyncio to solve nested event loop issues
    import nest_asyncio
    nest_asyncio.apply()

    # Run the asynchronous search
    return asyncio.run(_search_repos(query, top_k, filter_tags, filter_language))

async def _search_repos(
    query: str,
    top_k: int = 25,
    filter_tags: Optional[List[str]] = None,
    filter_language: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Async search with optional language filter.

    The language filter is applied before reranking to keep the reranker
    focused on relevant documents only.
    """
    retriever = get_retriever()
    reranker = get_reranker()

    # Perform hybrid search (over-retrieve for better rerank quality)
    results = await retriever.search(query, limit=top_k * 2, filter_tags=filter_tags)

    # Optional language filter (case-insensitive exact match)
    if filter_language:
        lang = filter_language.lower().strip()
        results = [r for r in results if r.get("language", "").lower() == lang]

    # Rerank if multiple results remain
    if len(results) > 1:
        results = await reranker.rerank(query, results, top_k=min(top_k, len(results)))

    return results

# UI Components
def render_header():
    """Render the application header."""
    st.title("🔍 Oh My Repos")
    st.markdown("""
    Search through your starred GitHub repositories using semantic search and hybrid retrieval.
    """)

def render_search_form():
    """Render the search form, including new filters."""
    with st.form("search_form"):
        query = st.text_input("Search Query", placeholder="Enter your search query here...")
        col1, col2 = st.columns([3, 1])
        with col1:
            top_k = st.slider("Number of results", min_value=5, max_value=50, value=25, step=5)
        with col2:
            search_button = st.form_submit_button("Search")

        # New controls
        language_filter = st.text_input("Filter by Language (optional)", placeholder="Python")
        sort_by_stars = st.checkbox("Sort by Stars (descending)", value=False)

    return query, top_k, language_filter, sort_by_stars, search_button

def render_results(results: List[Dict[str, Any]], sort_by_stars: bool = False):
    """Render search results with optional sorting."""
    if sort_by_stars:
        results = sorted(results, key=lambda x: x.get("stars", 0), reverse=True)

    if not results:
        st.info("No results found. Try a different search query.")
        return

    # Convert results to DataFrame for display
    df = pd.DataFrame(results)
    
    # Select and rename columns for display
    display_columns = [
        "repo_name", "repo_url", "summary", "language", "stars", 
        "score", "vector_score", "bm25_score"
    ]
    
    # Filter to only include columns that exist
    display_columns = [col for col in display_columns if col in df.columns]
    
    # Create a new DataFrame with selected columns
    display_df = df[display_columns].copy()
    
    # Format scores to 3 decimal places
    score_columns = ["score", "vector_score", "bm25_score", "rerank_score", "original_score"]
    for col in score_columns:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "")
    
    # Rename columns for display
    column_map = {
        "repo_name": "Repository",
        "repo_url": "URL",
        "summary": "Description",
        "language": "Language",
        "stars": "Stars",
        "score": "Score",
        "vector_score": "Vector Score",
        "bm25_score": "BM25 Score",
        "rerank_score": "Rerank Score",
        "original_score": "Original Score"
    }
    display_df.rename(columns={k: v for k, v in column_map.items() if k in display_df.columns}, inplace=True)
    
    # Make repository names clickable
    if "Repository" in display_df.columns and "URL" in display_df.columns:
        display_df["Repository"] = display_df.apply(
            lambda row: make_clickable(row["URL"], row["Repository"]), axis=1
        )
        # Remove URL column as it's now embedded in Repository
        display_df.drop(columns=["URL"], inplace=True)
    
    # Display results
    st.markdown("### Search Results")
    st.write(display_df.to_html(escape=False), unsafe_allow_html=True)
    
    # Show raw JSON for debugging if requested
    with st.expander("Show raw results data"):
        st.json(results)

def main():
    """Main application entry point."""
    render_header()
    query, top_k, language_filter, sort_by_stars, search_button = render_search_form()

    # Handle search
    if search_button and query:
        with st.spinner("Searching repositories..."):
            # Use synchronous wrapper
            results = search_repos_sync(query, top_k, filter_language=language_filter)
            render_results(results, sort_by_stars)
    elif search_button and not query:
        st.warning("Please enter a search query.")

if __name__ == "__main__":
    main()

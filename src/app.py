"""Enterprise-grade Streamlit UI for Oh My Repos with accessibility and performance.

WCAG 2.2 AA compliant web interface for semantic repository search with
comprehensive error handling and user experience optimization.
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from streamlit.runtime.caching import cache_data

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.core.logging import get_logger, log_exception, PerformanceLogger
from src.core.retriever import HybridRetriever
from src.core.reranker import JinaReranker
from src.core.monitoring import metrics

# Configure logging
logger = get_logger(__name__)

# WCAG 2.2 AA compliant page configuration
st.set_page_config(
    page_title="Oh My Repos - Semantic Repository Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/chernistry/ohmyrepos/issues",
        "Report a bug": "https://github.com/chernistry/ohmyrepos/issues/new",
        "About": "Enterprise-grade semantic search for GitHub repositories",
    },
)


# Application state management with proper error handling
@st.cache_resource(show_spinner="Initializing search engine...")
def get_retriever() -> Optional[HybridRetriever]:
    """Initialize and return the hybrid retriever with error handling."""
    try:
        with PerformanceLogger(logger, "retriever_initialization"):
            retriever = HybridRetriever(
                bm25_variant=settings.search.bm25_variant,
                bm25_weight=settings.search.bm25_weight,
                vector_weight=settings.search.vector_weight,
                merge_strategy="rrf",
            )

            # Safe async initialization
            import nest_asyncio
            nest_asyncio.apply()
            
            asyncio.run(retriever.initialize())
            
            logger.info("Retriever initialized successfully")
            metrics.record_api_request("retriever", "initialization", "success")
            return retriever
            
    except Exception as e:
        log_exception(logger, e, "Failed to initialize retriever")
        metrics.record_api_request("retriever", "initialization", "error")
        st.error(
            "❌ Failed to initialize search engine. Please check your configuration.",
            icon="🚨"
        )
        return None


@st.cache_resource(show_spinner="Initializing AI reranker...")
def get_reranker() -> Optional[JinaReranker]:
    """Initialize and return the reranker with error handling."""
    try:
        with PerformanceLogger(logger, "reranker_initialization"):
            if not settings.reranker:
                logger.warning("Reranker not configured, search quality may be reduced")
                return None
                
            # Initialize reranker but don't create aiohttp session yet
            # The session will be created when needed in an async context
            reranker = JinaReranker()
            logger.info("Reranker initialized successfully")
            metrics.record_api_request("reranker", "initialization", "success")
            return reranker
            
    except Exception as e:
        log_exception(logger, e, "Failed to initialize reranker")
        metrics.record_api_request("reranker", "initialization", "error")
        st.warning(
            "⚠️ AI reranker unavailable. Basic search results will be provided.",
            icon="⚠️"
        )
        return None


@cache_data(ttl=300, show_spinner="Checking system health...")
def get_system_health() -> Dict[str, Any]:
    """Get system health status with caching."""
    try:
        health = {
            "retriever": "healthy" if get_retriever() else "unhealthy",
            "reranker": "healthy" if get_reranker() else "degraded",
            "configuration": "healthy" if settings.qdrant and settings.embedding else "unhealthy",
        }
        
        overall_status = "healthy"
        if health["retriever"] == "unhealthy" or health["configuration"] == "unhealthy":
            overall_status = "unhealthy"
        elif health["reranker"] == "degraded":
            overall_status = "degraded"
            
        return {"status": overall_status, "components": health}
        
    except Exception as e:
        log_exception(logger, e, "Health check failed")
        return {"status": "unhealthy", "components": {}, "error": str(e)}


# Utility functions with security and accessibility improvements
def make_clickable(url: str, text: str) -> str:
    """Create accessible clickable link for Streamlit dataframe.
    
    Args:
        url: Target URL
        text: Link text
        
    Returns:
        HTML link with accessibility attributes
    """
    # Sanitize inputs to prevent XSS
    import html
    safe_url = html.escape(url)
    safe_text = html.escape(text)
    
    return (
        f'<a href="{safe_url}" target="_blank" rel="noopener noreferrer" '
        f'title="Open {safe_text} in new tab" '
        f'aria-label="Repository {safe_text}">{safe_text}</a>'
    )


def search_repos_sync(
    query: str,
    top_k: int = 25,
    filter_tags: Optional[List[str]] = None,
    filter_language: Optional[str] = None,
    enable_reranking: bool = True,
) -> Dict[str, Any]:
    """Synchronous wrapper for asynchronous search with comprehensive error handling.

    Args:
        query: User search query
        top_k: Number of final results requested
        filter_tags: Optional tag filter for retriever
        filter_language: Optional programming language filter
        enable_reranking: Whether to apply AI reranking

    Returns:
        Dictionary with results, metadata, and performance metrics
    """
    start_time = time.time()
    
    try:
        # Validate inputs
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")
            
        if top_k < 1 or top_k > 100:
            raise ValueError("Result limit must be between 1 and 100")
        
        # Use nest_asyncio for event loop compatibility
        import nest_asyncio
        nest_asyncio.apply()

        # Execute search
        with PerformanceLogger(logger, "search_operation", query=query):
            result = asyncio.run(_search_repos(
                query, top_k, filter_tags, filter_language, enable_reranking
            ))
            
        search_duration = (time.time() - start_time) * 1000
        
        # Record metrics
        metrics.record_api_request("search", "query", "success")
        metrics.record_request_duration("search", "query", search_duration / 1000)
        # Record search results count (no direct method available)
        
        return {
            **result,
            "search_duration_ms": search_duration,
            "timestamp": time.time(),
        }
        
    except Exception as e:
        log_exception(logger, e, "Search operation failed", query=query)
        metrics.record_api_request("search", "query", "error")
        
        return {
            "results": [],
            "error": str(e),
            "search_duration_ms": (time.time() - start_time) * 1000,
            "timestamp": time.time(),
        }


async def _search_repos(
    query: str,
    top_k: int = 25,
    filter_tags: Optional[List[str]] = None,
    filter_language: Optional[str] = None,
    enable_reranking: bool = True,
) -> Dict[str, Any]:
    """Execute asynchronous search with comprehensive processing.

    Args:
        query: Search query
        top_k: Number of results to return
        filter_tags: Optional tag filters
        filter_language: Optional language filter
        enable_reranking: Whether to apply AI reranking

    Returns:
        Dictionary with search results and metadata
    """
    retriever = get_retriever()
    reranker = get_reranker() if enable_reranking else None
    
    if not retriever:
        raise RuntimeError("Search engine not available")

    # Perform hybrid search with over-retrieval for better reranking
    search_limit = min(top_k * 3, 100) if reranker else top_k
    results = await retriever.search(
        query, 
        limit=search_limit, 
        filter_tags=filter_tags
    )

    # Apply language filter if specified
    if filter_language:
        lang = filter_language.lower().strip()
        original_count = len(results)
        results = [r for r in results if r.get("language", "").lower() == lang]
        logger.debug(f"Language filter reduced results from {original_count} to {len(results)}")

    # Apply AI reranking if available and beneficial
    rerank_applied = False
    if reranker and len(results) > 1 and enable_reranking:
        try:
            results = await reranker.rerank(
                query, results, top_k=min(top_k, len(results))
            )
            rerank_applied = True
            logger.debug(f"Reranking applied to {len(results)} results")
        except Exception as e:
            log_exception(logger, e, "Reranking failed, using original results")

    # Limit final results
    final_results = results[:top_k]
    
    return {
        "results": final_results,
        "total_found": len(results),
        "rerank_applied": rerank_applied,
        "filters_applied": {
            "tags": filter_tags,
            "language": filter_language,
        },
    }


# WCAG 2.2 AA compliant UI components
def render_header() -> None:
    """Render accessible application header with system status."""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(
            """
            # 🔍 Oh My Repos
            ### Enterprise Semantic Repository Search
            
            Discover GitHub repositories using AI-powered hybrid search combining 
            vector similarity and keyword matching for optimal relevance.
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        # System health indicator
        health = get_system_health()
        status = health["status"]
        
        if status == "healthy":
            st.success("🟢 System Healthy", icon="✅")
        elif status == "degraded":
            st.warning("🟡 System Degraded", icon="⚠️")
        else:
            st.error("🔴 System Issues", icon="🚨")
            if "error" in health:
                st.caption(f"Error: {health['error']}")


def render_search_form() -> tuple[str, int, Optional[str], bool, bool, bool]:
    """Render comprehensive search form with accessibility features.
    
    Returns:
        Tuple of (query, top_k, language_filter, sort_by_stars, enable_reranking, search_button)
    """
    with st.form("search_form", clear_on_submit=False):
        # Main search input with help text
        query = st.text_input(
            "🔍 Search Query",
            placeholder="e.g., machine learning python, web scraping, data visualization",
            help="Enter keywords to search repository names, descriptions, and tags. "
                 "Use natural language for best results.",
            max_chars=500,
            key="search_query"
        )
        
        # Advanced options in expander for cleaner UI
        with st.expander("🔧 Advanced Search Options", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                top_k = st.slider(
                    "Number of Results",
                    min_value=5,
                    max_value=50,
                    value=25,
                    step=5,
                    help="More results provide broader coverage but may include less relevant items"
                )
                
                language_filter = st.selectbox(
                    "Programming Language",
                    options=["Any", "Python", "JavaScript", "TypeScript", "Go", "Rust", 
                            "Java", "C++", "C#", "PHP", "Ruby", "Swift", "Kotlin"],
                    index=0,
                    help="Filter results by primary programming language"
                )
                
            with col2:
                sort_by_stars = st.checkbox(
                    "Sort by GitHub Stars",
                    value=False,
                    help="Prioritize repositories with more stars (popularity)"
                )
                
                enable_reranking = st.checkbox(
                    "AI-Powered Reranking",
                    value=True,
                    help="Use AI to improve result relevance (may increase search time)"
                )
        
        # Search button with keyboard shortcut hint
        search_button = st.form_submit_button(
            "🚀 Search Repositories",
            type="primary",
            help="Press Ctrl+Enter to search",
            use_container_width=True
        )
    
    # Convert language filter
    language_filter = None if language_filter == "Any" else language_filter
    
    return query, top_k, language_filter, sort_by_stars, enable_reranking, search_button


def render_results(
    search_result: Dict[str, Any], 
    sort_by_stars: bool = False
) -> None:
    """Render search results with modern, accessible design.
    
    Args:
        search_result: Complete search result including metadata
        sort_by_stars: Whether to sort by star count
    """
    results = search_result.get("results", [])
    metadata = {
        "total_found": search_result.get("total_found", len(results)),
        "search_duration_ms": search_result.get("search_duration_ms", 0),
        "rerank_applied": search_result.get("rerank_applied", False),
        "filters_applied": search_result.get("filters_applied", {}),
    }
    
    # Handle errors
    if "error" in search_result:
        st.error(f"❌ Search failed: {search_result['error']}", icon="🚨")
        return
    
    if not results:
        st.info(
            "🔍 No results found. Try:\n\n"
            "• Different keywords or synonyms\n"
            "• Broader search terms\n"
            "• Removing language filters\n"
            "• Checking spelling",
            icon="💡"
        )
        return

    # Results summary with performance metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Results Found", 
            len(results),
            delta=f"of {metadata['total_found']} total" if metadata['total_found'] > len(results) else None
        )
    with col2:
        st.metric(
            "Search Time", 
            f"{metadata['search_duration_ms']:.0f}ms",
            delta="AI Enhanced" if metadata['rerank_applied'] else "Standard"
        )
    with col3:
        avg_score = sum(r.get("score", 0) for r in results) / len(results) if results else 0
        st.metric("Avg Relevance", f"{avg_score:.2f}")

    # Optional sorting
    if sort_by_stars:
        results = sorted(results, key=lambda x: x.get("stars", 0), reverse=True)
        logger.debug("Results sorted by star count")

    # Render results as cards for better visual hierarchy
    st.markdown("---")
    
    for i, result in enumerate(results, 1):
        with st.container():
            col1, col2 = st.columns([4, 1])
            
            with col1:
                # Repository title with link
                repo_name = result.get("repo_name", "Unknown")
                repo_url = result.get("repo_url", "#")
                
                st.markdown(
                    f"### [{repo_name}]({repo_url})",
                    help=f"Open {repo_name} on GitHub"
                )
                
                # Description/summary
                description = result.get("summary") or result.get("description", "")
                if description:
                    st.markdown(f"*{description[:200]}{'...' if len(description) > 200 else ''}*")
                
                # Tags
                tags = result.get("tags", [])
                if tags:
                    tag_badges = " ".join([f"`{tag}`" for tag in tags[:8]])
                    st.markdown(f"**Tags:** {tag_badges}")
            
            with col2:
                # Metrics column
                stars = result.get("stars", 0)
                language = result.get("language", "")
                score = result.get("score", 0)
                
                if stars > 0:
                    st.metric("⭐ Stars", f"{stars:,}")
                
                if language:
                    st.markdown(f"**Language:** {language}")
                
                st.markdown(f"**Relevance:** {score:.3f}")
                
                # Score breakdown in expander
                with st.expander("Score Details"):
                    if result.get("vector_score"):
                        st.caption(f"Vector: {result['vector_score']:.3f}")
                    if result.get("bm25_score"):
                        st.caption(f"BM25: {result['bm25_score']:.3f}")
                    if result.get("rerank_score"):
                        st.caption(f"Rerank: {result['rerank_score']:.3f}")
        
        # Separator between results
        if i < len(results):
            st.markdown("---")

    # Export and debug options
    with st.expander("📊 Export & Debug Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Export options
            if st.button("📥 Export as CSV"):
                df = pd.DataFrame(results)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"search_results_{int(time.time())}.csv",
                    mime="text/csv"
                )
            
            if st.button("📄 Export as JSON"):
                import json
                json_data = json.dumps(search_result, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"search_results_{int(time.time())}.json",
                    mime="application/json"
                )
        
        with col2:
            # Debug information
            st.json({
                "search_metadata": metadata,
                "sample_result": results[0] if results else None
            })


def render_sidebar() -> None:
    """Render sidebar with application info and advanced features."""
    with st.sidebar:
        st.markdown("## ℹ️ About")
        st.markdown(
            """
            **Oh My Repos** uses advanced AI to search your GitHub starred 
            repositories with:
            
            - 🔍 **Semantic Search**: Understanding intent, not just keywords
            - 🤖 **AI Reranking**: Improved relevance with machine learning
            - ⚡ **Hybrid Retrieval**: Combines vector and keyword search
            - 🎯 **Smart Filtering**: Language and tag-based filtering
            """
        )
        
        st.markdown("---")
        
        # System metrics
        st.markdown("## 📊 System Status")
        health = get_system_health()
        
        for component, status in health.get("components", {}).items():
            icon = "🟢" if status == "healthy" else "🟡" if status == "degraded" else "🔴"
            st.markdown(f"{icon} **{component.title()}**: {status}")
        
        # Configuration info
        st.markdown("---")
        st.markdown("## ⚙️ Configuration")
        
        if settings.llm:
            st.markdown(f"**LLM Provider**: {settings.llm.provider.value}")
            st.markdown(f"**Model**: {settings.llm.model}")
        
        if settings.embedding:
            st.markdown(f"**Embeddings**: {settings.embedding.model}")
        
        if settings.qdrant:
            st.markdown(f"**Vector DB**: Qdrant")
            st.markdown(f"**Collection**: {settings.qdrant.collection_name}")
        
        # Performance tips
        st.markdown("---")
        st.markdown("## 💡 Search Tips")
        st.markdown(
            """
            - Use **natural language** queries
            - Combine **technical terms** with use cases
            - Try **synonyms** if no results found
            - Use **language filters** for specific ecosystems
            - Enable **AI reranking** for best results
            """
        )


def main() -> None:
    """Main application entry point with comprehensive error handling."""
    try:
        # Initialize logging
        from src.core.logging import setup_logging_from_config
        setup_logging_from_config()
        
        # Render UI components
        render_header()
        render_sidebar()
        
        # Get form inputs
        query, top_k, language_filter, sort_by_stars, enable_reranking, search_button = render_search_form()

        # Handle search execution
        if search_button:
            if not query or not query.strip():
                st.warning("⚠️ Please enter a search query to continue.", icon="⚠️")
                return
            
            # Show search progress
            with st.spinner("🔍 Searching repositories..."):
                # Record search event
                logger.info("Search initiated", query=query, limit=top_k, 
                           language_filter=language_filter, reranking=enable_reranking)
                
                # Execute search
                search_result = search_repos_sync(
                    query=query,
                    top_k=top_k,
                    filter_language=language_filter,
                    enable_reranking=enable_reranking
                )
                
                # Render results
                render_results(search_result, sort_by_stars=sort_by_stars)
                
        # Usage instructions for first-time users
        elif not st.session_state.get("search_performed", False):
            st.info(
                """
                👋 **Welcome to Oh My Repos!**
                
                Start by entering a search query above. You can search for:
                - Programming languages (e.g., "python machine learning")
                - Project types (e.g., "web scraping tools")
                - Specific technologies (e.g., "react components")
                - Use cases (e.g., "data visualization")
                
                Use the advanced options to fine-tune your search!
                """,
                icon="🚀"
            )
        
        # Mark that app has been used
        if search_button:
            st.session_state["search_performed"] = True
            
    except Exception as e:
        log_exception(logger, e, "Application error")
        st.error(
            "🚨 **Application Error**\n\n"
            "An unexpected error occurred. Please check the logs for details.",
            icon="🚨"
        )
        
        # Show error details in development
        if settings.is_development():
            st.exception(e)


if __name__ == "__main__":
    main()

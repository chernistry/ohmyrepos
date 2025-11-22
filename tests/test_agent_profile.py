import json
from pathlib import Path
from src.agent.profile import analyze_profile, InterestCluster

def test_analyze_profile_empty(tmp_path):
    repos_file = tmp_path / "repos.json"
    repos_file.write_text("[]", encoding="utf-8")
    clusters = analyze_profile(repos_file)
    assert clusters == []

def test_analyze_profile_basic(tmp_path):
    repos_file = tmp_path / "repos.json"
    data = [
        {"name": "repo1", "language": "Python", "topics": ["ai", "ml"]},
        {"name": "repo2", "language": "Python", "topics": ["web", "flask"]},
        {"name": "repo3", "language": "Rust", "topics": ["cli", "terminal"]},
        {"name": "repo4", "language": "Python", "topics": ["ai", "data"]},
    ]
    repos_file.write_text(json.dumps(data), encoding="utf-8")
    
    clusters = analyze_profile(repos_file)
    
    # Should have at least Python and Rust clusters
    assert len(clusters) >= 2
    
    python_cluster = next((c for c in clusters if "Python" in c.languages), None)
    assert python_cluster is not None
    assert "Python" in python_cluster.languages
    # "ai" appears twice, so it should be a keyword
    assert "ai" in python_cluster.keywords
    
    rust_cluster = next((c for c in clusters if "Rust" in c.languages), None)
    assert rust_cluster is not None
    assert "Rust" in rust_cluster.languages

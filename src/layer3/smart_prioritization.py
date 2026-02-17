"""
Layer 3: Smart Paper Prioritization

Multi-factor ranking for "what to read first" combining:
- Significance scores (must_read, changes_thinking, team_discussion)
- Front status (emerging > growing > stable > declining)
- Bridge score (cross-front connectors)
- Recency (last 7 days weighted higher)
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional


def _parse_date(date_str: str) -> datetime:
    """Parse YYYY-MM-DD date string."""
    try:
        return datetime.strptime(date_str[:10], '%Y-%m-%d')
    except (ValueError, TypeError):
        return datetime.now()


def _front_status_score(status: str) -> float:
    """Weight fronts by status (emerging=highest priority)."""
    weights = {
        'emerging': 1.0,
        'growing': 0.8,
        'stable': 0.5,
        'declining': 0.2,
        'merged': 0.3,
        'split': 0.6,
    }
    return weights.get(status, 0.5)


def _significance_score(sig: dict) -> float:
    """Compute significance score (0-3)."""
    score = 0.0
    if sig.get('must_read'):
        score += 1.5
    if sig.get('changes_thinking'):
        score += 1.0
    if sig.get('team_discussion'):
        score += 0.5
    return score


def _recency_score(published_date: str, days_window: int = 7) -> float:
    """Higher score for recent papers (1.0 = today, 0.5 = 7 days ago)."""
    pub_dt = _parse_date(published_date)
    now = datetime.now()
    days_ago = (now - pub_dt).days

    if days_ago < 0:
        return 1.0  # Future dates (errors) get max score
    if days_ago > days_window:
        return 0.0

    # Linear decay from 1.0 (today) to 0.0 (days_window ago)
    return 1.0 - (days_ago / days_window)


def compute_priority_score(paper: dict,
                           front_status: Optional[str] = None,
                           bridge_score: float = 0.0,
                           days_window: int = 7) -> float:
    """
    Compute composite priority score for a paper.

    Primary factors: M (Methodological), P (Problem), I (Inspirational) scores
    Secondary factors: Significance flags, front status, bridge score, recency

    Args:
        paper: Paper dict with relevance, significance, published_date
        front_status: Status of the front this paper belongs to
        bridge_score: Bridge score if paper is a bridge (0.0-1.0)
        days_window: Recency window in days

    Returns:
        Priority score (0-10 scale)

    Weight distribution:
        - M (Methodological): 30%  (0-10 scale)
        - P (Problem): 25%         (0-10 scale)
        - I (Inspirational): 15%   (0-10 scale)
        - Significance: 20%        (must_read, changes_thinking, etc.)
        - Front status: 5%         (emerging > growing > stable)
        - Bridge score: 3%         (cross-front connections)
        - Recency: 2%              (last 7 days boost)
    """
    # M, P, I scores (0-10 each)
    rel = paper.get('relevance', {})
    m_score = rel.get('methodological', 0)  # 0-10
    p_score = rel.get('problem', 0)          # 0-10
    i_score = rel.get('inspirational', 0)    # 0-10

    # Significance (0-3)
    sig = paper.get('significance', {})
    significance = _significance_score(sig)

    # Front status (0-1)
    status = _front_status_score(front_status) if front_status else 0.5

    # Bridge score (0-1)
    bridge = min(bridge_score, 1.0)

    # Recency (0-1)
    pub_date = paper.get('published_date', '')
    recency = _recency_score(pub_date, days_window)

    # Weighted combination (scale to 0-10)
    # Total weights: 3.0 + 2.5 + 1.5 + 0.67 + 0.5 + 0.3 + 0.2 = 8.67 (normalize to 10)
    score = (
        m_score * 0.30 +         # 30% Methodological relevance
        p_score * 0.25 +         # 25% Problem relevance
        i_score * 0.15 +         # 15% Inspirational relevance
        significance * 0.67 +    # 20% Significance (0-3 scale → ~0-2)
        status * 0.50 +          # 5% Front status
        bridge * 0.30 +          # 3% Bridge score
        recency * 0.20           # 2% Recency
    )

    # Normalize to 0-10 scale
    # Max possible: 10*0.3 + 10*0.25 + 10*0.15 + 3*0.67 + 1*0.5 + 1*0.3 + 1*0.2 = 10.01
    return min(score, 10.0)


def rank_papers(papers: List[dict],
                front_map: Optional[Dict[str, str]] = None,
                bridge_map: Optional[Dict[str, float]] = None,
                days_window: int = 7,
                top_n: int = 10) -> List[Dict]:
    """
    Rank papers by priority score.

    Args:
        papers: List of paper dicts
        front_map: Dict mapping paper_id -> front_status
        bridge_map: Dict mapping paper_id -> bridge_score
        days_window: Recency window in days
        top_n: Return top N papers

    Returns:
        List of dicts with keys: paper, score, rank
    """
    if front_map is None:
        front_map = {}
    if bridge_map is None:
        bridge_map = {}

    ranked = []
    for paper in papers:
        pid = paper.get('arxiv_id', '')
        front_status = front_map.get(pid)
        bridge_score = bridge_map.get(pid, 0.0)

        score = compute_priority_score(
            paper,
            front_status=front_status,
            bridge_score=bridge_score,
            days_window=days_window
        )

        ranked.append({
            'paper': paper,
            'score': score,
            'arxiv_id': pid,
        })

    # Sort by score descending
    ranked.sort(key=lambda x: x['score'], reverse=True)

    # Add rank numbers
    for i, item in enumerate(ranked, 1):
        item['rank'] = i

    return ranked[:top_n]


def get_priority_reasons(paper: dict,
                        front_status: Optional[str] = None,
                        bridge_score: float = 0.0) -> List[str]:
    """
    Get human-readable reasons for a paper's priority.

    Returns:
        List of reason strings (e.g., "MUST-READ", "High M score", "Emerging front")
    """
    reasons = []

    # Significance flags (highest priority)
    sig = paper.get('significance', {})
    if sig.get('must_read'):
        reasons.append('MUST-READ')
    if sig.get('changes_thinking'):
        reasons.append('Changes thinking')

    # M, P, I scores (show if any are high)
    rel = paper.get('relevance', {})
    m_score = rel.get('methodological', 0)
    p_score = rel.get('problem', 0)
    i_score = rel.get('inspirational', 0)

    # Highlight exceptional individual scores
    if m_score >= 8:
        reasons.append(f'Strong methodology (M={m_score})')
    if p_score >= 8:
        reasons.append(f'Key problem (P={p_score})')
    if i_score >= 8:
        reasons.append(f'Highly inspirational (I={i_score})')

    # Front status
    if front_status in ['emerging', 'growing']:
        reasons.append(f'{front_status.capitalize()} front')

    # Bridge papers
    if bridge_score >= 0.5:
        reasons.append('Bridge paper')

    return reasons

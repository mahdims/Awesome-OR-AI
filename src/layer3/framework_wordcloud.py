"""
Layer 3: Framework Word Cloud Generator

Generates word cloud visualizations for framework tracking where:
- Font size = number of papers in that framework
- Color intensity = proportion of must-read papers
- Active frameworks (last 30 days) highlighted
"""

import io
import base64
from typing import Dict, List, Optional
from pathlib import Path

try:
    from wordcloud import WordCloud
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False


def generate_framework_wordcloud(lineages: Dict[str, List[dict]],
                                 active_frameworks: List[tuple] = None,
                                 width: int = 800,
                                 height: int = 400,
                                 output_format: str = 'base64') -> Optional[str]:
    """
    Generate a word cloud from framework lineages.

    Args:
        lineages: Dict from framework_genealogy.extract_framework_lineages()
        active_frameworks: List of (framework, count) from get_active_frameworks()
        width: Image width in pixels
        height: Image height in pixels
        output_format: 'base64' for email embedding, 'file' for saving

    Returns:
        Base64-encoded PNG string (for HTML img src) or None if wordcloud unavailable
    """
    if not WORDCLOUD_AVAILABLE:
        print("[WARNING] wordcloud library not installed. Install with: pip install wordcloud")
        return None

    if not lineages:
        return None

    # Build frequency dict: framework -> paper_count
    frequencies = {}
    must_read_ratios = {}
    active_set = set(fw for fw, _ in (active_frameworks or []))

    for framework, papers in lineages.items():
        paper_count = len(papers)
        must_read_count = sum(1 for p in papers if p.get('must_read', False))
        must_read_ratio = must_read_count / paper_count if paper_count > 0 else 0

        frequencies[framework] = paper_count
        must_read_ratios[framework] = must_read_ratio

    # Color function: Green intensity based on must-read ratio, active frameworks in bold
    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        ratio = must_read_ratios.get(word, 0)
        is_active = word in active_set

        if is_active:
            # Active frameworks: Blue-green gradient
            # High must-read: Dark teal (#1A5F7A)
            # Low must-read: Light blue (#64B5CD)
            r = int(26 + (100 - 26) * (1 - ratio))
            g = int(95 + (181 - 95) * (1 - ratio))
            b = int(122 + (205 - 122) * (1 - ratio))
        else:
            # Inactive frameworks: Green gradient
            # High must-read: Dark green (#2E7D32)
            # Low must-read: Light green (#A5D6A7)
            r = int(46 + (165 - 46) * (1 - ratio))
            g = int(125 + (214 - 125) * (1 - ratio))
            b = int(50 + (167 - 50) * (1 - ratio))

        return f'rgb({r},{g},{b})'

    # Generate word cloud
    wc = WordCloud(
        width=width,
        height=height,
        background_color='white',
        color_func=color_func,
        relative_scaling=0.5,  # Mix of frequency and rank
        min_font_size=12,
        max_font_size=80,
        prefer_horizontal=0.7,
        collocations=False,
    )

    wc.generate_from_frequencies(frequencies)

    # Render to image
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout(pad=0)

    # Add legend
    from matplotlib.patches import Rectangle
    legend_elements = [
        Rectangle((0, 0), 1, 1, fc='#1A5F7A', label='Active (last 30d) + Must-read'),
        Rectangle((0, 0), 1, 1, fc='#64B5CD', label='Active (last 30d)'),
        Rectangle((0, 0), 1, 1, fc='#2E7D32', label='Inactive + Must-read'),
        Rectangle((0, 0), 1, 1, fc='#A5D6A7', label='Inactive'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8, framealpha=0.9)

    if output_format == 'base64':
        # Convert to base64 for email embedding
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64
    elif output_format == 'file':
        # Return figure for saving externally
        return fig
    else:
        plt.close(fig)
        return None


def generate_framework_wordcloud_html(lineages: Dict[str, List[dict]],
                                      active_frameworks: List[tuple] = None,
                                      width: int = 800,
                                      height: int = 400) -> str:
    """
    Generate HTML img tag with embedded word cloud.

    Returns:
        HTML string like: <img src="data:image/png;base64,...">
        Or fallback text if wordcloud unavailable
    """
    img_base64 = generate_framework_wordcloud(
        lineages, active_frameworks, width, height, output_format='base64'
    )

    if img_base64:
        return f'<img src="data:image/png;base64,{img_base64}" alt="Framework Word Cloud" style="max-width:100%; height:auto;" />'
    else:
        # Fallback: text list
        framework_list = sorted(lineages.items(), key=lambda x: len(x[1]), reverse=True)
        items = []
        for fw, papers in framework_list[:10]:
            must_read = sum(1 for p in papers if p.get('must_read', False))
            items.append(f'{fw} ({len(papers)} papers, {must_read} must-read)')
        return '<div style="font-size:12px;">' + ' • '.join(items) + '</div>'


def save_framework_wordcloud(lineages: Dict[str, List[dict]],
                             active_frameworks: List[tuple] = None,
                             output_path: str = 'framework_wordcloud.png',
                             width: int = 1200,
                             height: int = 600):
    """
    Save word cloud to file.

    Args:
        lineages: Framework lineages dict
        active_frameworks: Active frameworks list
        output_path: File path to save PNG
        width: Image width in pixels
        height: Image height in pixels
    """
    if not WORDCLOUD_AVAILABLE:
        print("[ERROR] wordcloud library not installed. Install with: pip install wordcloud")
        return

    fig = generate_framework_wordcloud(
        lineages, active_frameworks, width, height, output_format='file'
    )

    if fig:
        fig.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"[OK] Word cloud saved to: {output_path}")
    else:
        print("[ERROR] Failed to generate word cloud")


# Example usage:
if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from db.database import Database
    from layer3.framework_genealogy import extract_framework_lineages, get_active_frameworks

    db = Database()
    category = "LLMs for Algorithm Design"

    with db:
        lineages = extract_framework_lineages(db, category=category)
        active = get_active_frameworks(lineages, days=30)

    save_framework_wordcloud(lineages, active, 'framework_wordcloud.png', width=1200, height=600)
    print(f"Frameworks: {len(lineages)}, Active: {len(active)}")

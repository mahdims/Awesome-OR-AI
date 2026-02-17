#!/usr/bin/env python3
"""
Generate framework word cloud visualization.

Creates a visual representation where:
- Word size = number of papers in that framework
- Color intensity = proportion of must-read papers
- Active frameworks (last 30d) shown in blue-green
- Inactive frameworks shown in green

Usage:
    python src/scripts/generate_framework_wordcloud.py --category "LLMs for Algorithm Design"
    python src/scripts/generate_framework_wordcloud.py --category "LLMs for Algorithm Design" --output framework.png
    python src/scripts/generate_framework_wordcloud.py --all-categories
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.database import Database
from layer3.framework_genealogy import extract_framework_lineages, get_active_frameworks
from layer3.framework_wordcloud import save_framework_wordcloud


def main():
    parser = argparse.ArgumentParser(
        description='Generate framework word cloud visualization'
    )
    parser.add_argument('--category', help='Category to analyze (default: all)')
    parser.add_argument('--output', default='framework_wordcloud.png',
                        help='Output PNG file path')
    parser.add_argument('--width', type=int, default=1200,
                        help='Image width in pixels (default: 1200)')
    parser.add_argument('--height', type=int, default=600,
                        help='Image height in pixels (default: 600)')
    parser.add_argument('--active-days', type=int, default=30,
                        help='Days to consider "active" (default: 30)')
    parser.add_argument('--all-categories', action='store_true',
                        help='Analyze all categories combined')
    args = parser.parse_args()

    print(f"{'='*70}")
    print(f"FRAMEWORK WORD CLOUD GENERATOR")
    print(f"{'='*70}")
    print(f"  Category: {args.category or 'ALL'}")
    print(f"  Output: {args.output}")
    print(f"  Size: {args.width}x{args.height}")
    print()

    # Extract lineages
    print(f"[1/2] Extracting framework lineages...")
    db = Database()
    with db:
        if args.all_categories or not args.category:
            lineages = extract_framework_lineages(db, category=None)
        else:
            lineages = extract_framework_lineages(db, category=args.category)

    if not lineages:
        print(f"[ERROR] No frameworks found")
        return 1

    active = get_active_frameworks(lineages, days=args.active_days)

    total_papers = sum(len(papers) for papers in lineages.values())
    total_must_read = sum(
        sum(1 for p in papers if p.get('must_read', False))
        for papers in lineages.values()
    )

    print(f"  ✓ Frameworks: {len(lineages)}")
    print(f"  ✓ Active (last {args.active_days}d): {len(active)}")
    print(f"  ✓ Total papers: {total_papers} ({total_must_read} must-read)")
    print()

    # Show top frameworks
    print(f"  Top frameworks:")
    sorted_frameworks = sorted(lineages.items(), key=lambda x: len(x[1]), reverse=True)
    for i, (fw, papers) in enumerate(sorted_frameworks[:10], 1):
        must_read = sum(1 for p in papers if p.get('must_read', False))
        active_marker = " [ACTIVE]" if fw in [a[0] for a in active] else ""
        print(f"    {i}. {fw}: {len(papers)} papers ({must_read} must-read){active_marker}")
    print()

    # Generate word cloud
    print(f"[2/2] Generating word cloud...")
    save_framework_wordcloud(
        lineages,
        active,
        output_path=args.output,
        width=args.width,
        height=args.height
    )

    print()
    print(f"{'='*70}")
    print(f"DONE")
    print(f"{'='*70}")
    print(f"  Open {args.output} to view the word cloud!")

    return 0


if __name__ == '__main__':
    sys.exit(main())

import os
import re
import json
import time
import yaml
import logging
import argparse
import datetime
import requests
import xml.etree.ElementTree as ET

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from openai import OpenAI
    from pydantic import BaseModel
except ImportError:
    OpenAI = None
    BaseModel = None

try:
    from src.db.database import Database as _Database
except ImportError:
    try:
        from db.database import Database as _Database
    except ImportError:
        _Database = None

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

huggingface_papers_url = "https://huggingface.co/api/papers/"
github_url = "https://api.github.com/search/repositories"
arxiv_url = "http://arxiv.org/"
semantic_scholar_url = "https://api.semanticscholar.org/graph/v1/paper/"

MIN_YEAR = 2024
RELEVANCE_THRESHOLD = 6
SCORE_CACHE_PATH = './docs/relevance_cache.json'
ARXIV_API_URL = "http://export.arxiv.org/api/query"
ARXIV_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

def _parse_arxiv_entries(xml_text):
    """Parse arxiv API XML response into a list of paper dicts."""
    root = ET.fromstring(xml_text)
    results = []
    for entry in root.findall("atom:entry", ARXIV_NS):
        title_el = entry.find("atom:title", ARXIV_NS)
        if title_el is None or not title_el.text or title_el.text.strip() == "Error":
            continue

        raw_id = entry.find("atom:id", ARXIV_NS).text.strip().split("/abs/")[-1]

        authors = []
        for a in entry.findall("atom:author", ARXIV_NS):
            name_el = a.find("atom:name", ARXIV_NS)
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())

        summary_el = entry.find("atom:summary", ARXIV_NS)
        summary = summary_el.text.replace("\n", " ").strip() if summary_el is not None and summary_el.text else ""

        published_el = entry.find("atom:published", ARXIV_NS)
        updated_el = entry.find("atom:updated", ARXIV_NS)

        def parse_date(el):
            if el is not None and el.text:
                return datetime.date.fromisoformat(el.text.strip()[:10])
            return None

        primary_cat_el = entry.find("arxiv:primary_category", ARXIV_NS)
        primary_category = primary_cat_el.get("term", "") if primary_cat_el is not None else ""

        comment_el = entry.find("arxiv:comment", ARXIV_NS)
        comment = comment_el.text.strip() if comment_el is not None and comment_el.text else None

        journal_el = entry.find("arxiv:journal_ref", ARXIV_NS)
        journal_ref = journal_el.text.strip() if journal_el is not None and journal_el.text else None

        results.append({
            "id": raw_id,
            "title": title_el.text.strip(),
            "summary": summary,
            "authors": authors,
            "primary_category": primary_category,
            "published": parse_date(published_el),
            "updated": parse_date(updated_el),
            "comment": comment,
            "journal_ref": journal_ref,
        })
    return results

def _arxiv_api_call(params, timeout=30, max_retries=5):
    """Single arxiv API call with retry and 429 handling. Returns parsed entries or []."""
    query_short = (params.get("search_query") or params.get("id_list", ""))[:80]
    for attempt in range(max_retries):
        try:
            print(f"    [arxiv API] {query_short}... (attempt {attempt+1}) ", end="", flush=True)
            t0 = time.time()
            r = requests.get(ARXIV_API_URL, params=params, timeout=timeout)
            elapsed = time.time() - t0
            if r.status_code == 429:
                wait = min(10 * (attempt + 1), 60)
                print(f"RATE LIMITED ({elapsed:.1f}s), waiting {wait}s ...", flush=True)
                time.sleep(wait)
                continue
            r.raise_for_status()
            print(f"OK ({elapsed:.1f}s)", flush=True)
            return _parse_arxiv_entries(r.text)
        except requests.exceptions.Timeout:
            print(f"TIMEOUT ({timeout}s)", flush=True)
            if attempt < max_retries - 1:
                time.sleep(5)
        except Exception as e:
            print(f"ERROR: {e}", flush=True)
            if attempt < max_retries - 1:
                time.sleep(5)
    return []

# Max OR clauses per arxiv API call before splitting
ARXIV_MAX_QUERY_CLAUSES = 8

def arxiv_search(query=None, id_list=None, max_results=10, sort_by="submittedDate", timeout=30):
    """
    Search arxiv via the API using requests with a hard timeout.
    Automatically splits long OR queries into smaller chunks to avoid
    URL length issues and arxiv API errors.
    Returns a list of dicts with paper metadata (deduplicated by ID).
    """
    if id_list:
        params = {
            "max_results": max_results,
            "id_list": ",".join(id_list) if isinstance(id_list, list) else id_list,
        }
        return _arxiv_api_call(params, timeout=timeout)

    if not query:
        return []

    # Split long OR queries into chunks
    # Detect OR-separated clauses (handles both parenthesized and plain terms)
    clauses = [c.strip() for c in re.split(r'\s+OR\s+', query)]

    if len(clauses) <= ARXIV_MAX_QUERY_CLAUSES:
        # Short enough to send as one request
        params = {
            "max_results": max_results,
            "search_query": query,
            "sortBy": sort_by,
            "sortOrder": "descending",
        }
        return _arxiv_api_call(params, timeout=timeout)

    # Split into chunks and merge results
    print(f"    [arxiv] Query has {len(clauses)} clauses, splitting into chunks of {ARXIV_MAX_QUERY_CLAUSES} ...", flush=True)
    seen_ids = set()
    all_results = []
    for i in range(0, len(clauses), ARXIV_MAX_QUERY_CLAUSES):
        chunk = clauses[i:i + ARXIV_MAX_QUERY_CLAUSES]
        chunk_query = " OR ".join(chunk)
        print(f"    [arxiv] Chunk {i // ARXIV_MAX_QUERY_CLAUSES + 1}/{(len(clauses) + ARXIV_MAX_QUERY_CLAUSES - 1) // ARXIV_MAX_QUERY_CLAUSES}:", flush=True)
        params = {
            "max_results": max_results,
            "search_query": chunk_query,
            "sortBy": sort_by,
            "sortOrder": "descending",
        }
        chunk_results = _arxiv_api_call(params, timeout=timeout)
        for paper in chunk_results:
            if paper["id"] not in seen_ids:
                seen_ids.add(paper["id"])
                all_results.append(paper)
        # Respect rate limits between chunks
        time.sleep(3)

    print(f"    [arxiv API] Got {len(all_results)} unique results total", flush=True)
    return all_results

def load_config(config_file:str) -> dict:
    '''
    config_file: input config file path
    return: a dict of configuration
    '''
    # make filters pretty
    def pretty_filters(**config) -> dict:
        keywords = dict()
        ESCAPE = '\"'
        OR = ' OR '
        AND = ' AND '
        def make_term(t):
            """Wrap multi-word terms in quotes for exact phrase match."""
            t = t.strip()
            if len(t.split()) > 1:
                return ESCAPE + t + ESCAPE
            return t
        def parse_filters(filters:list):
            parts = []
            for f in filters:
                if isinstance(f, list):
                    # List of terms: AND them together, e.g. ["integer programming", "LLM"]
                    # becomes ("integer programming" AND LLM)
                    and_part = AND.join(make_term(t) for t in f)
                    parts.append('(' + and_part + ')')
                else:
                    parts.append(make_term(f))
            return OR.join(parts)
        for k,v in config['keywords'].items():
            if 'filters' in v:
                keywords[k] = parse_filters(v['filters'])
        return keywords
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config['kv'] = pretty_filters(**config)
        logging.info(f'config = {config}')
    return config

def get_authors(authors, first_author = False):
    output = str()
    if first_author == False:
        output = ", ".join(str(author) for author in authors)
    else:
        output = authors[0]
    return output

def sort_papers(papers):
    """Sort papers by date (newest first), parsing date from content string."""
    def extract_date(item):
        match = re.search(r'(\d{4}[-.]?\d{2}[-.]?\d{2})', str(item[1]))
        if match:
            date_str = match.group(1).replace('.', '-')
            return date_str
        return item[0]  # fallback to arxiv id
    sorted_items = sorted(papers.items(), key=extract_date, reverse=True)
    return dict(sorted_items)

def get_code_link(qword:str) -> str:
    """
    Search GitHub for code repositories matching the query.
    @param qword: query string, eg. arxiv ids and paper titles
    @return paper_code in github: string, if not found, return None
    """
    query = f"{qword}"
    params = {
        "q": query,
        "sort": "stars",
        "order": "desc"
    }
    headers = {}
    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"token {github_token}"
    try:
        r = requests.get(github_url, params=params, headers=headers, timeout=15)
        results = r.json()
        if results.get("total_count", 0) > 0:
            return results["items"][0]["html_url"]
    except Exception as e:
        logging.warning(f"GitHub code search failed for '{qword}': {e}")
    return None

def get_code_link_pwc(arxiv_id: str) -> str:
    """
    Search Papers with Code API for a code repository matching this arxiv paper.
    @param arxiv_id: ArXiv paper ID (e.g., "2401.12345")
    @return repo URL string, or None if not found
    """
    try:
        url = f"https://paperswithcode.com/api/v1/papers/?arxiv_id={arxiv_id}"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            results = data.get("results", [])
            if results:
                # Get repositories for the first matching paper
                paper_id = results[0].get("id")
                if paper_id:
                    repo_url = f"https://paperswithcode.com/api/v1/papers/{paper_id}/repositories/"
                    rr = requests.get(repo_url, timeout=10)
                    if rr.status_code == 200:
                        repos = rr.json().get("results", [])
                        if repos:
                            # Pick highest-starred repo
                            best = max(repos, key=lambda x: x.get("stars", 0))
                            return best.get("url")
    except Exception as e:
        logging.debug(f"Papers with Code lookup failed for '{arxiv_id}': {e}")
    return None


def get_code_link_s2(s2_meta: dict) -> str:
    """
    Extract code/repo URL from Semantic Scholar paper metadata.
    Checks externalIds for GitHub URLs and openAccessPdf.
    @param s2_meta: dict from S2 batch API response
    @return repo URL string, or None
    """
    if not s2_meta:
        return None
    # Check if S2 has a linked GitHub repo via externalIds
    ext_ids = s2_meta.get("externalIds") or {}
    github_id = ext_ids.get("GitHub")
    if github_id:
        return f"https://github.com/{github_id}"
    return None


def find_code_link_waterfall(arxiv_id: str, paper_title: str, s2_meta: dict = None) -> str:
    """
    Try multiple sources to find code for a paper, stopping at first hit.
    Order: Papers with Code → HuggingFace → Semantic Scholar → GitHub search.
    """
    # 1. Papers with Code (most reliable for ML papers)
    url = get_code_link_pwc(arxiv_id)
    if url:
        logging.info(f"  Code found via PwC: {arxiv_id}")
        return url

    # 2. HuggingFace API
    try:
        r = requests.get(huggingface_papers_url + arxiv_id, timeout=10)
        if r.status_code == 200:
            hf = r.json()
            if hf.get("githubRepo"):
                logging.info(f"  Code found via HuggingFace: {arxiv_id}")
                return hf["githubRepo"]
    except Exception:
        pass

    # 3. Semantic Scholar externalIds
    url = get_code_link_s2(s2_meta)
    if url:
        logging.info(f"  Code found via S2: {arxiv_id}")
        return url

    # 4. GitHub search (least reliable, last resort)
    title_clean = re.sub(r'\*+', '', paper_title).strip() if paper_title else arxiv_id
    url = get_code_link(title_clean)
    if url:
        logging.info(f"  Code found via GitHub search: {arxiv_id}")
        return url

    return None


def get_paper_metadata_s2(arxiv_ids: list) -> dict:
    """
    Batch-fetch paper metadata from Semantic Scholar for enrichment.
    Returns dict keyed by arxiv_id with venue, affiliations, and code info.

    @param arxiv_ids: list of ArXiv IDs (e.g., ["2401.12345", "2402.67890"])
    @return dict: {arxiv_id: {venue, affiliation, code_url, ...}}
    """
    if not arxiv_ids:
        return {}

    s2_api_key = os.environ.get("S2_API_KEY")
    headers = {"Content-Type": "application/json"}
    if s2_api_key:
        headers["x-api-key"] = s2_api_key

    result = {}
    # S2 batch API accepts up to 500 IDs per request
    batch_size = 500
    fields = "venue,externalIds,authors,authors.affiliations,openAccessPdf"

    for i in range(0, len(arxiv_ids), batch_size):
        batch = arxiv_ids[i:i + batch_size]
        paper_ids = [f"ArXiv:{aid}" for aid in batch]

        try:
            url = f"https://api.semanticscholar.org/graph/v1/paper/batch?fields={fields}"
            r = requests.post(url, json={"ids": paper_ids}, headers=headers, timeout=30)

            if r.status_code == 429:
                logging.warning("S2 rate limited, waiting 5s...")
                time.sleep(5)
                r = requests.post(url, json={"ids": paper_ids}, headers=headers, timeout=30)

            if r.status_code == 200:
                papers = r.json()
                for aid, paper in zip(batch, papers):
                    if paper is None:
                        continue
                    # Extract first author affiliation
                    affiliation = ""
                    authors = paper.get("authors") or []
                    if authors:
                        affiliations = authors[0].get("affiliations") or []
                        if affiliations:
                            affiliation = affiliations[0]

                    # Extract venue
                    venue = paper.get("venue") or ""

                    # Extract code link from externalIds
                    code_url = get_code_link_s2(paper)

                    result[aid] = {
                        "venue": venue,
                        "affiliation": affiliation,
                        "code_url": code_url,
                        "raw": paper,
                    }
            else:
                logging.warning(f"S2 batch API returned {r.status_code}")

        except Exception as e:
            logging.warning(f"S2 batch API failed: {e}")

        if i + batch_size < len(arxiv_ids):
            time.sleep(1)  # rate limit between batches

    logging.info(f"S2 metadata fetched for {len(result)}/{len(arxiv_ids)} papers")
    return result


def load_score_cache(cache_path=SCORE_CACHE_PATH):
    """Load the persistent relevance score cache from disk."""
    try:
        with open(cache_path, "r", encoding='utf-8') as f:
            content = f.read()
            if content:
                return json.loads(content)
    except FileNotFoundError:
        pass
    except Exception as e:
        logging.warning(f"Failed to load score cache: {e}")
    return {}

def save_score_cache(cache, cache_path=SCORE_CACHE_PATH):
    """Save the relevance score cache to disk."""
    try:
        with open(cache_path, "w", encoding='utf-8') as f:
            json.dump(cache, f)
    except Exception as e:
        logging.warning(f"Failed to save score cache: {e}")

class RelevanceScore(BaseModel):
    score: int

# === Database-backed score + abstract storage ===

def _init_score_db():
    """Initialize DB connection and ensure rescore_cache table exists."""
    if _Database is None:
        return None
    try:
        db = _Database()
        db.connect()
        db.execute("""
            CREATE TABLE IF NOT EXISTS rescore_cache (
                arxiv_id        TEXT NOT NULL,
                category        TEXT NOT NULL,
                title           TEXT,
                abstract        TEXT,
                score           INTEGER,
                score_date      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (arxiv_id, category)
            )
        """)
        db.execute("""
            CREATE INDEX IF NOT EXISTS idx_rescore_category
            ON rescore_cache(category)
        """)
        db.commit()
        return db
    except Exception as e:
        logging.warning(f"Failed to init score DB: {e}")
        return None

_SCORE_DB = _init_score_db()

def _store_score_to_db(paper_id, category, title, abstract, score):
    """Persist score and abstract to the database."""
    if _SCORE_DB is None:
        return
    try:
        _SCORE_DB.execute(
            """INSERT OR REPLACE INTO rescore_cache
               (arxiv_id, category, title, abstract, score, score_date)
               VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
            (paper_id, category, title, abstract, score)
        )
        _SCORE_DB.commit()
    except Exception as e:
        logging.warning(f"Failed to store score to DB: {e}")


def _load_score_from_db(paper_id, category):
    """Load a previously cached score from DB, or None if missing."""
    if _SCORE_DB is None:
        return None
    try:
        row = _SCORE_DB.fetchone(
            "SELECT score FROM rescore_cache WHERE arxiv_id = ? AND category = ?",
            (paper_id, category)
        )
        if row is None:
            return None
        score = row["score"]
        if score is None:
            return None
        return int(score)
    except Exception as e:
        logging.warning(f"Failed to load score from DB: {e}")
        return None

RESEARCHER_PROFILE_PATH = os.path.join(os.path.dirname(__file__), 'src', 'layer1', 'prompts', 'researcher_profile.md')

def _load_researcher_profile_sections():
    """
    Load researcher_profile.md and extract per-category sections.
    Sections are delimited by ### [Category Name] headings.
    Returns dict mapping category name -> section text.
    """
    try:
        with open(RESEARCHER_PROFILE_PATH, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        logging.warning(f"Researcher profile not found at {RESEARCHER_PROFILE_PATH}")
        return {}

    sections = {}
    current_category = None
    current_lines = []

    for line in content.split('\n'):
        # Match ### [Category Name] with optional suffixes
        if line.startswith('### ['):
            # Save previous section
            if current_category:
                sections[current_category] = '\n'.join(current_lines).strip()
            # Extract category name from ### [Category Name]
            bracket_end = line.index(']')
            current_category = line[5:bracket_end]
            current_lines = [line]
        elif current_category is not None:
            # Stop collecting when we hit a non-category ## heading
            if line.startswith('## ') and not line.startswith('### '):
                sections[current_category] = '\n'.join(current_lines).strip()
                current_category = None
                current_lines = []
            else:
                current_lines.append(line)

    # Save last section
    if current_category:
        sections[current_category] = '\n'.join(current_lines).strip()

    return sections

# Load once at module level
_PROFILE_SECTIONS = _load_researcher_profile_sections()

def get_relevance_score(paper_id, title, abstract, category, category_description="", client=None, score_cache=None):
    """
    Use lightweight LLM to score paper relevance to category (0-10).
    Checks cache first (keyed by paper_id + category) to avoid repeat API calls.
    Returns 10 (include) if no client available.
    """
    # Build a cache key: paper_id + category ensures same paper scored once per category
    cache_key = f"{paper_id}::{category}"

    # Check cache first
    if score_cache is not None and cache_key in score_cache:
        cached = score_cache[cache_key]
        logging.info(f"Cache hit for {paper_id} in '{category}': score={cached}")
        return cached

    # Fallback to DB cache if JSON cache misses (e.g., file lost/corrupted).
    db_cached = _load_score_from_db(paper_id, category)
    if db_cached is not None:
        logging.info(f"DB cache hit for {paper_id} in '{category}': score={db_cached}")
        if score_cache is not None:
            score_cache[cache_key] = db_cached
        return db_cached

    if client is None:
        return 10

    desc = category_description if category_description else category
    researcher_context = _PROFILE_SECTIONS.get(category, "")

    prompt = f"""Rate how relevant this paper is to the research category "{category}" on a scale of 0 to 10.

Category description: {desc}

Researcher's active work and interests in this category:
{researcher_context if researcher_context else "No specific researcher context available."}

Scoring guide:
- 8-10: Directly addresses the researcher's active projects, methods, or stated priorities
- 5-7: Related to the category and potentially useful to the researcher's work
- 2-4: Tangentially related, different focus or methods
- 0-1: Completely unrelated

Title: {title}
Abstract: {abstract}"""

    try:
        response = client.chat.completions.parse(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": prompt}],
            response_format=RelevanceScore
        )
        result = response.choices[0].message.parsed
        if result is None:
            logging.warning(f"Empty LLM response for '{title}', defaulting to 10")
            score = 10
        else:
            score = min(max(result.score, 0), 10)
    except Exception as e:
        logging.warning(f"Relevance scoring failed for '{title}': {e}")
        score = 10  # include paper if scoring fails

    # Store in JSON cache
    if score_cache is not None:
        score_cache[cache_key] = score

    # Store score + abstract in database
    _store_score_to_db(paper_id, category, title, abstract, score)

    return score

def extract_venue(journal_ref=None, comment=None):
    """Extract publication venue from arxiv metadata."""
    if journal_ref:
        return journal_ref
    if comment:
        venue_patterns = [
            r'(?:accepted|published|appear(?:ing|s)?)\s+(?:at|in|by)\s+([^,.]+)',
            r'(NeurIPS|ICML|ICLR|AAAI|IJCAI|ACL|EMNLP|CVPR|ICCV|ECCV|KDD|WWW|SIGIR|NAACL|COLING|AISTATS|UAI|COLT|SODA|FOCS|STOC|ICRA|IROS|RSS|CoRL)\s*\d{4}',
        ]
        for pattern in venue_patterns:
            match = re.search(pattern, comment, re.IGNORECASE)
            if match:
                return match.group(1).strip() if match.lastindex else match.group(0).strip()
    return ""

def get_daily_papers(topic, query="slam", max_results=2, llm_client=None, category_description="", score_cache=None, existing_ids=None):
    """
    @param topic: str
    @param query: str
    @param max_results: int
    @param llm_client: OpenAI client for relevance filtering (or None)
    @param category_description: str for LLM relevance scoring
    @param score_cache: dict - persistent cache of relevance scores
    @param existing_ids: set of paper IDs already in database (skip these)
    @return paper_with_code: dict
    """
    content = dict()
    content_to_web = dict()
    if existing_ids is None:
        existing_ids = set()

    results = arxiv_search(query=query, max_results=max_results)

    skipped_existing = 0
    for idx, result in enumerate(results, 1):
        paper_id            = result["id"]
        paper_title         = result["title"]
        paper_abstract      = result["summary"]
        paper_authors       = ", ".join(result["authors"])
        paper_first_author  = result["authors"][0] if result["authors"] else "Unknown"
        primary_category    = result["primary_category"]
        publish_time        = result["published"]
        update_time         = result["updated"]
        comments            = result["comment"]
        journal_ref         = result["journal_ref"]

        # Year filter: skip papers before MIN_YEAR
        if publish_time and publish_time.year < MIN_YEAR:
            continue

        # eg: 2108.09112v1 -> 2108.09112
        ver_pos = paper_id.find('v')
        if ver_pos == -1:
            paper_key = paper_id
        else:
            paper_key = paper_id[0:ver_pos]

        # Skip papers already in database
        if paper_key in existing_ids:
            skipped_existing += 1
            continue

        paper_url = arxiv_url + 'abs/' + paper_key

        # LLM relevance filtering (with cache lookup)
        print(f"    [{idx}/{len(results)}] Scoring {paper_key} ...", end=" ", flush=True)
        t0 = time.time()
        score = get_relevance_score(paper_key, paper_title, paper_abstract, topic, category_description, llm_client, score_cache)
        elapsed = time.time() - t0
        if score < RELEVANCE_THRESHOLD:
            print(f"FILTERED (score={score}, {elapsed:.1f}s): {paper_title[:50]}", flush=True)
            continue
        print(f"PASS (score={score}, {elapsed:.1f}s): {paper_title[:50]}", flush=True)

        # Extract venue
        venue = extract_venue(journal_ref, comments)

        # Source code link via HuggingFace API (replaces broken PwC endpoint)
        repo_url = None
        try:
            r = requests.get(huggingface_papers_url + paper_key, timeout=10)
            if r.status_code == 200:
                hf = r.json()
                if hf.get("githubRepo"):
                    repo_url = hf["githubRepo"]
        except Exception:
            pass

        # Fallback: search GitHub
        if repo_url is None:
            repo_url = get_code_link(paper_title)

        code_str = f"**[link]({repo_url})**" if repo_url else "null"
        entry = format_paper_string(
            f"**{update_time}**", f"**{paper_title}**", f"{paper_first_author} et.al.",
            "", venue, f"[{paper_key}]({paper_url})", code_str)
        content[paper_key] = entry
        content_to_web[paper_key] = entry

    if skipped_existing:
        print(f"    Skipped {skipped_existing} papers already in database", flush=True)
    data = {topic:content}
    data_web = {topic:content_to_web}
    return data,data_web

def get_citation_papers(topic, seed_papers, llm_client=None, category_description="", score_cache=None, existing_ids=None):
    """
    Get papers that cite the given seed papers using Semantic Scholar API.
    @param topic: str - category name
    @param seed_papers: list of arxiv ids (e.g. ["2405.17743", "2310.06116"])
    @param llm_client: OpenAI client for relevance filtering (or None)
    @param category_description: str for LLM relevance scoring
    @param score_cache: dict - persistent cache of relevance scores
    @param existing_ids: set of paper IDs already in database (skip these)
    @return: (data, data_web) same format as get_daily_papers
    """
    content = dict()
    content_to_web = dict()
    if existing_ids is None:
        existing_ids = set()

    for si, seed_id in enumerate(seed_papers, 1):
        print(f"    [Seed {si}/{len(seed_papers)}] Fetching citations for {seed_id} ...", flush=True)
        offset = 0
        page = 0
        while True:
            api_url = f"{semantic_scholar_url}ArXiv:{seed_id}/citations"
            params = {
                "fields": "title,externalIds,year,authors,publicationDate,abstract,venue",
                "offset": offset,
                "limit": 500
            }
            page += 1
            try:
                print(f"      [S2 API] page {page}, offset={offset} ...", end=" ", flush=True)
                t0 = time.time()
                r = requests.get(api_url, params=params, timeout=30)
                elapsed = time.time() - t0
                if r.status_code == 429:
                    print(f"RATE LIMITED ({elapsed:.1f}s), waiting 5s ...", flush=True)
                    time.sleep(5)
                    continue
                r.raise_for_status()
                result = r.json()
                print(f"OK ({elapsed:.1f}s)", flush=True)
            except requests.exceptions.Timeout:
                print(f"TIMEOUT (30s)", flush=True)
                break
            except Exception as e:
                print(f"ERROR: {e}", flush=True)
                break

            citations = result.get("data", [])
            if not citations:
                print(f"      No more citations", flush=True)
                break

            print(f"      Processing {len(citations)} citations ...", flush=True)
            scored_count = 0
            for item in citations:
                paper = item.get("citingPaper", {})
                ext_ids = paper.get("externalIds", {})
                arxiv_id = ext_ids.get("ArXiv")

                if not arxiv_id:
                    continue
                if arxiv_id in content:
                    continue
                if arxiv_id in existing_ids:
                    continue

                paper_title = paper.get("title", "Unknown")
                authors = paper.get("authors", [])
                first_author = authors[0]["name"] if authors else "Unknown"
                pub_date = paper.get("publicationDate", "")
                paper_abstract = paper.get("abstract", "") or ""
                venue = paper.get("venue", "") or ""
                paper_year = paper.get("year")
                paper_url = arxiv_url + "abs/" + arxiv_id
                paper_key = arxiv_id

                # Year filter: skip papers before MIN_YEAR
                if paper_year and paper_year < MIN_YEAR:
                    continue
                # Also check pub_date if year not available
                if not paper_year and pub_date:
                    try:
                        if int(pub_date[:4]) < MIN_YEAR:
                            continue
                    except (ValueError, IndexError):
                        pass

                # LLM relevance filtering (with cache lookup)
                scored_count += 1
                print(f"        Scoring citation {paper_key} ...", end=" ", flush=True)
                t0 = time.time()
                score = get_relevance_score(paper_key, paper_title, paper_abstract, topic, category_description, llm_client, score_cache)
                elapsed = time.time() - t0
                if score < RELEVANCE_THRESHOLD:
                    print(f"FILTERED (score={score}, {elapsed:.1f}s): {paper_title[:50]}", flush=True)
                    continue
                print(f"PASS (score={score}, {elapsed:.1f}s): {paper_title[:50]}", flush=True)

                # Try to get code link from HuggingFace API
                repo_url = None
                try:
                    cr = requests.get(huggingface_papers_url + arxiv_id, timeout=10)
                    if cr.status_code == 200:
                        hf = cr.json()
                        if hf.get("githubRepo"):
                            repo_url = hf["githubRepo"]
                except Exception:
                    pass

                # Fallback: search GitHub
                if repo_url is None:
                    repo_url = get_code_link(paper_title)

                code_str = f"**[link]({repo_url})**" if repo_url else "null"
                entry = format_paper_string(
                    f"**{pub_date}**", f"**{paper_title}**", f"{first_author} et.al.",
                    "", venue, f"[{paper_key}]({paper_url})", code_str)
                content[paper_key] = entry
                content_to_web[paper_key] = entry

            print(f"      Scored {scored_count} papers from this page, {len(content)} total kept so far", flush=True)

            # Pagination
            if "next" in result:
                offset = result["next"]
                time.sleep(1)  # respect rate limits
            else:
                break

        time.sleep(2)  # pause between seed papers

    print(f"    Found {len(content)} unique citing papers for topic: {topic}", flush=True)
    data = {topic: content}
    data_web = {topic: content_to_web}
    return data, data_web

def clean_json_data(filename):
    """
    Remove pre-2024 papers from JSON file and migrate old format entries
    to include venue column.
    """
    with open(filename, "r", encoding='utf-8') as f:
        content = f.read()
        if not content:
            return
        data = json.loads(content)

    new_data = {}
    removed_count = 0
    migrated_count = 0

    for topic, papers in data.items():
        new_papers = {}
        for paper_id, entry in papers.items():
            entry = str(entry)
            # Extract year from date in content
            date_match = re.search(r'(\d{4})-(\d{2})-(\d{2})', entry)
            if date_match:
                year = int(date_match.group(1))
                if year < MIN_YEAR:
                    removed_count += 1
                    continue

            # Check if entry needs venue column migration (old format has 5 data fields = 6 pipes)
            pipe_count = entry.count('|')
            if pipe_count <= 6 and '|' in entry:
                # Old format: |date|title|author|id|code|
                # New format: |date|title|author|venue|id|code|
                parts = entry.split('|')
                # parts: ['', date, title, author, id, code, '\n'] (7 elements for old)
                if len(parts) == 7:
                    parts.insert(4, '')  # insert empty venue
                    entry = '|'.join(parts)
                    migrated_count += 1

            new_papers[paper_id] = entry
        new_data[topic] = new_papers

    with open(filename, "w", encoding='utf-8') as f:
        json.dump(new_data, f)

    logging.info(f"Cleaned {filename}: removed {removed_count} pre-{MIN_YEAR} papers, migrated {migrated_count} entries")

def parse_arxiv_string(s):
    """Parse pipe-delimited paper string, handling 3 format generations."""
    parts = s.split("|")
    if len(parts) >= 9:
        # Newest: |date|title|authors|affiliation|venue|arxiv_id|code|
        date = parts[1].strip()
        title = parts[2].strip()
        authors = parts[3].strip()
        affiliation = parts[4].strip()
        venue = parts[5].strip()
        arxiv_id = parts[6].strip()
        code = parts[7].strip()
    elif len(parts) >= 8:
        # Current: |date|title|authors|venue|arxiv_id|code|
        date = parts[1].strip()
        title = parts[2].strip()
        authors = parts[3].strip()
        affiliation = ''
        venue = parts[4].strip()
        arxiv_id = parts[5].strip()
        code = parts[6].strip()
    else:
        # Legacy: |date|title|authors|arxiv_id|code|
        date = parts[1].strip()
        title = parts[2].strip()
        authors = parts[3].strip()
        affiliation = ''
        venue = ''
        arxiv_id = parts[4].strip()
        code = parts[5].strip()
    arxiv_id = re.sub(r'v\d+', '', arxiv_id)
    return date, title, authors, affiliation, venue, arxiv_id, code


def format_paper_string(date, title, authors, affiliation, venue, arxiv_id, code):
    """Build the 8-field pipe-delimited paper string."""
    return "|{}|{}|{}|{}|{}|{}|{}|\n".format(
        date, title, authors, affiliation, venue, arxiv_id, code)


def update_paper_links(filename):
    """
    Weekly enrichment: update code links, affiliations, and venues in JSON file.

    For each paper:
      1. AFFILIATION: if empty → use Semantic Scholar first-author affiliation
      2. VENUE: if empty → re-check ArXiv metadata + Semantic Scholar
      3. CODE: if null → waterfall: PwC → HuggingFace → S2 → GitHub search
    """
    with open(filename, "r", encoding='utf-8') as f:
        content = f.read()
        if not content:
            m = {}
        else:
            m = json.loads(content)

    json_data = m.copy()

    # Collect all paper IDs for batch S2 lookup
    all_paper_ids = []
    for keywords, v in json_data.items():
        all_paper_ids.extend(v.keys())

    # Batch-fetch metadata from Semantic Scholar (one API call for all papers)
    logging.info(f"Fetching S2 metadata for {len(all_paper_ids)} papers...")
    s2_metadata = get_paper_metadata_s2(all_paper_ids)

    # Collect IDs that need venue re-check from ArXiv
    needs_venue_arxiv = []
    for keywords, v in json_data.items():
        for paper_id, contents in v.items():
            _, _, _, _, venue, _, _ = parse_arxiv_string(str(contents))
            if not venue:
                needs_venue_arxiv.append(paper_id)

    # Batch-fetch ArXiv metadata for papers missing venue
    arxiv_venue_map = {}
    if needs_venue_arxiv:
        logging.info(f"Re-checking ArXiv metadata for {len(needs_venue_arxiv)} papers missing venue...")
        # ArXiv id_list lookup in batches of 50
        for i in range(0, len(needs_venue_arxiv), 50):
            batch = needs_venue_arxiv[i:i + 50]
            id_list = ",".join(batch)
            entries = _arxiv_api_call({"id_list": id_list, "max_results": len(batch)})
            for entry in entries:
                raw_id = re.sub(r'v\d+$', '', entry["id"])
                venue = extract_venue(entry.get("journal_ref"), entry.get("comment"))
                if venue:
                    arxiv_venue_map[raw_id] = venue
            time.sleep(1)

    # Enrich each paper
    stats = {"affiliation_updated": 0, "venue_updated": 0, "code_updated": 0}

    for keywords, v in json_data.items():
        logging.info(f'Enriching category: {keywords}')
        for paper_id, contents in v.items():
            contents = str(contents)
            date, title, authors, affiliation, venue, arxiv_link, code = parse_arxiv_string(contents)

            s2 = s2_metadata.get(paper_id, {})
            changed = False

            # 1. AFFILIATION: fill if empty
            if not affiliation and s2.get("affiliation"):
                affiliation = s2["affiliation"]
                stats["affiliation_updated"] += 1
                changed = True

            # 2. VENUE: fill if empty
            if not venue:
                # Try ArXiv metadata first (more authoritative for journal refs)
                if paper_id in arxiv_venue_map:
                    venue = arxiv_venue_map[paper_id]
                    stats["venue_updated"] += 1
                    changed = True
                elif s2.get("venue"):
                    venue = s2["venue"]
                    stats["venue_updated"] += 1
                    changed = True

            # 3. CODE: fill if null using waterfall
            has_code = code and code != 'null' and 'null' not in code.lower()
            if not has_code:
                try:
                    repo_url = find_code_link_waterfall(paper_id, title, s2.get("raw"))
                    if repo_url:
                        code = f"**[link]({repo_url})**"
                        stats["code_updated"] += 1
                        changed = True
                except Exception as e:
                    logging.error(f"Code link error for {paper_id}: {e}")

            # Reconstruct in 8-field format (always upgrade)
            json_data[keywords][paper_id] = format_paper_string(
                date, title, authors, affiliation, venue, arxiv_link, code)

    logging.info(f"Enrichment complete: {stats}")

    # Update database with enriched metadata (if DB module available)
    if _Database is not None:
        try:
            db = _Database()
            with db:
                db_updates = 0
                for keywords, v in json_data.items():
                    for paper_id, contents in v.items():
                        _, _, _, affiliation, venue, _, code = parse_arxiv_string(str(contents))
                        has_code = code and code != 'null' and 'null' not in code.lower()
                        repo_url = None
                        if has_code:
                            url_match = re.search(r'\[link\]\((.+?)\)', code)
                            if url_match:
                                repo_url = url_match.group(1)
                        if affiliation or venue or repo_url:
                            db.update_paper_metadata(paper_id, affiliations=affiliation,
                                                     venue=venue, code_url=repo_url)
                            db_updates += 1
                logging.info(f"Database updated: {db_updates} papers enriched")
        except Exception as e:
            logging.warning(f"Database update skipped: {e}")

    # Dump updated JSON
    with open(filename, "w", encoding='utf-8') as f:
        json.dump(json_data, f)

def update_json_file(filename,data_dict):
    '''
    daily update json file using data_dict
    '''
    with open(filename, "r", encoding='utf-8') as f:
        content = f.read()
        if not content:
            m = {}
        else:
            m = json.loads(content)

    json_data = m.copy()

    # update papers in each keywords
    for data in data_dict:
        for keyword in data.keys():
            papers = data[keyword]

            if keyword in json_data.keys():
                json_data[keyword].update(papers)
            else:
                json_data[keyword] = papers

    with open(filename, "w", encoding='utf-8') as f:
        json.dump(json_data, f)

def json_to_md(filename,md_filename,
               task = '',
               to_web = False,
               use_title = True,
               use_tc = True,
               use_b2t = True):
    """
    @param filename: str
    @param md_filename: str
    @return None
    """
    def pretty_math(s:str) -> str:
        ret = ''
        match = re.search(r"\$.*\$", s)
        if match == None:
            return s
        math_start,math_end = match.span()
        space_trail = space_leading = ''
        if s[:math_start][-1] != ' ' and '*' != s[:math_start][-1]: space_trail = ' '
        if s[math_end:][0] != ' ' and '*' != s[math_end:][0]: space_leading = ' '
        ret += s[:math_start]
        ret += f'{space_trail}${match.group()[1:-1].strip()}${space_leading}'
        ret += s[math_end:]
        return ret

    DateNow = datetime.date.today()
    DateNow = str(DateNow)
    DateNow = DateNow.replace('-','.')

    with open(filename, "r", encoding='utf-8') as f:
        content = f.read()
        if not content:
            data = {}
        else:
            data = json.loads(content)

    # clean README.md if daily already exist else create it
    with open(md_filename, "w+", encoding='utf-8') as f:
        pass

    # write data into README.md
    with open(md_filename, "a+", encoding='utf-8') as f:

        if (use_title == True) and (to_web == True):
            f.write("---\n" + "layout: default\n" + "---\n\n")

        if use_title == True:
            f.write("## Updated on " + DateNow + "\n")
        else:
            f.write("> Updated on " + DateNow + "\n")

        #Add: table of contents
        if use_tc == True:
            f.write("<details>\n")
            f.write("  <summary>Table of Contents</summary>\n")
            f.write("  <ol>\n")
            for keyword in data.keys():
                day_content = data[keyword]
                if not day_content:
                    continue
                kw = keyword.replace(' ','-')
                f.write(f"    <li><a href=#{kw.lower()}>{keyword}</a></li>\n")
            f.write("  </ol>\n")
            f.write("</details>\n\n")

        for keyword in data.keys():
            day_content = data[keyword]
            if not day_content:
                continue
            # the head of each part
            f.write(f"## {keyword}\n\n")

            if use_title == True :
                if to_web == False:
                    f.write("|Publish Date|Title|Authors|Affiliation|Venue|PDF|Code|\n" + "|---|---|---|---|---|---|---|\n")
                else:
                    f.write("| Publish Date | Title | Authors | Affiliation | Venue | PDF | Code |\n")
                    f.write("|:---------|:-----------------------|:---------|:---------|:------|:------|:------|\n")

            # sort papers by date (newest first)
            day_content = sort_papers(day_content)

            for _,v in day_content.items():
                if v is not None:
                    f.write(pretty_math(v)) # make latex pretty

            f.write(f"\n")

            #Add: back to top
            if use_b2t:
                top_info = f"#Updated on {DateNow}"
                top_info = top_info.replace(' ','-').replace('.','')
                f.write(f"<p align=right>(<a href={top_info.lower()}>back to top</a>)</p>\n\n")

    logging.info(f"{task} finished")

def _load_existing_ids(json_path):
    """Load all paper IDs already in the JSON database."""
    ids = set()
    try:
        with open(json_path, "r", encoding='utf-8') as f:
            content = f.read()
            if content:
                data = json.loads(content)
                for topic_papers in data.values():
                    ids.update(topic_papers.keys())
    except FileNotFoundError:
        pass
    return ids

def demo(**config):
    start_time = time.time()
    data_collector = []
    data_collector_web= []

    keywords = config['kv']
    max_results = config['max_results']
    publish_readme = config['publish_readme']
    publish_gitpage = config['publish_gitpage']

    # Optional single-category filter (--category flag)
    filter_category = config.get('filter_category')
    if filter_category:
        keywords = {k: v for k, v in keywords.items() if k == filter_category}
        print(f"[--category] Restricting fetch to: '{filter_category}'", flush=True)

    # Initialize OpenAI client for relevance filtering
    llm_client = None
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key and OpenAI is not None:
        llm_client = OpenAI(api_key=api_key)
        print("[1/6] OpenAI client initialized (LLM filtering ON)", flush=True)
    else:
        print("[1/6] No OPENAI_API_KEY set - LLM filtering OFF (all papers pass through)", flush=True)

    # Load persistent relevance score cache
    score_cache = load_score_cache()
    print(f"[2/6] Loaded score cache: {len(score_cache)} previously scored papers", flush=True)

    # Build category descriptions lookup
    category_descriptions = {}
    for topic, cfg in config['keywords'].items():
        category_descriptions[topic] = cfg.get('description', topic)

    b_update = config['update_paper_links']

    # Clean old papers from JSON files first
    if publish_readme:
        clean_json_data(config['json_readme_path'])
    if publish_gitpage:
        clean_json_data(config['json_gitpage_path'])

    if config['update_paper_links'] == False:
        # Load existing paper IDs to skip re-processing
        existing_ids = _load_existing_ids(config['json_readme_path'])
        print(f"[3/6] Starting paper retrieval (max {max_results} per query, {len(existing_ids)} existing papers to skip)...", flush=True)
        total_found = 0
        cache_before = len(score_cache)

        # keyword-based search
        for i, (topic, keyword) in enumerate(keywords.items(), 1):
            cat_max = config['keywords'].get(topic, {}).get('max_results', max_results)
            print(f"\n  [{i}/{len(keywords)}] Keyword search: '{topic}' (max {cat_max}) ...", flush=True)
            t0 = time.time()
            data, data_web = get_daily_papers(topic, query=keyword,
                                            max_results=cat_max,
                                            llm_client=llm_client,
                                            category_description=category_descriptions.get(topic, topic),
                                            score_cache=score_cache,
                                            existing_ids=existing_ids)
            count = len(data.get(topic, {}))
            total_found += count
            elapsed = time.time() - t0
            print(f"  [{i}/{len(keywords)}] '{topic}' done: {count} papers kept ({elapsed:.1f}s)", flush=True)
            data_collector.append(data)
            data_collector_web.append(data_web)

        # citation-based search
        citation_topics = [(t, c) for t, c in config['keywords'].items()
                           if 'seed_papers' in c and (not filter_category or t == filter_category)]
        for i, (topic, cfg) in enumerate(citation_topics, 1):
            seed_count = len(cfg['seed_papers'])
            print(f"\n  [Citation {i}/{len(citation_topics)}] '{topic}' ({seed_count} seed papers) ...", flush=True)
            t0 = time.time()
            data, data_web = get_citation_papers(topic, cfg['seed_papers'],
                                                 llm_client=llm_client,
                                                 category_description=category_descriptions.get(topic, topic),
                                                 score_cache=score_cache,
                                                 existing_ids=existing_ids)
            count = len(data.get(topic, {}))
            total_found += count
            elapsed = time.time() - t0
            print(f"  [Citation {i}/{len(citation_topics)}] '{topic}' done: {count} papers kept ({elapsed:.1f}s)", flush=True)
            data_collector.append(data)
            data_collector_web.append(data_web)

        new_scores = len(score_cache) - cache_before
        print(f"\n[4/6] Retrieval done: {total_found} total papers, {new_scores} new LLM scores (cached: {cache_before})", flush=True)
    else:
        print("[3/6] Update mode: re-checking code links for existing papers...", flush=True)

    # Save the updated score cache
    save_score_cache(score_cache)
    print(f"[5/6] Score cache saved ({len(score_cache)} entries)", flush=True)

    # 1. update JSON files (README generated separately by src/scripts/generate_readme.py)
    if publish_readme:
        json_file = config['json_readme_path']
        if config['update_paper_links']:
            print("  Enriching paper links (readme JSON) ...", end=" ", flush=True)
            update_paper_links(json_file)
        else:
            update_json_file(json_file, data_collector)
        print("done", flush=True)

    # 2. update docs JSON file (to gitpage)
    if publish_gitpage:
        json_file = config['json_gitpage_path']
        if config['update_paper_links']:
            print("  Enriching paper links (gitpage JSON) ...", end=" ", flush=True)
            update_paper_links(json_file)
        else:
            update_json_file(json_file, data_collector_web)
        print("done", flush=True)

    elapsed = time.time() - start_time
    # Count total papers in output
    total_papers = 0
    try:
        with open(config['json_readme_path'], 'r', encoding='utf-8') as f:
            content = f.read()
            if content:
                all_data = json.loads(content)
                total_papers = sum(len(p) for p in all_data.values())
    except Exception:
        pass
    print(f"[6/6] Done! {total_papers} papers in database. Elapsed: {elapsed:.1f}s", flush=True)

def clear_category_from_json(filename, category):
    """Remove all papers for a given category from a JSON file."""
    try:
        with open(filename, "r", encoding='utf-8') as f:
            content = f.read()
            data = json.loads(content) if content else {}
    except FileNotFoundError:
        return 0
    removed = len(data.get(category, {}))
    data[category] = {}
    with open(filename, "w", encoding='utf-8') as f:
        json.dump(data, f)
    return removed

def clear_category_from_cache(category, cache_path=SCORE_CACHE_PATH):
    """Remove all score cache entries for a given category (JSON + DB)."""
    cache = load_score_cache(cache_path)
    suffix = f"::{category}"
    keys_to_remove = [k for k in cache if k.endswith(suffix)]
    for k in keys_to_remove:
        del cache[k]
    save_score_cache(cache, cache_path)

    removed_db = 0
    if _SCORE_DB is not None:
        try:
            cur = _SCORE_DB.execute(
                "DELETE FROM rescore_cache WHERE category = ?",
                (category,)
            )
            _SCORE_DB.commit()
            if cur is not None and getattr(cur, "rowcount", None) is not None and cur.rowcount > 0:
                removed_db = cur.rowcount
        except Exception as e:
            logging.warning(f"Failed to clear DB score cache for '{category}': {e}")

    return len(keys_to_remove) + removed_db

def redo_category(category_name, **config):
    """Redo a single category from scratch: clear its data, re-fetch, and regenerate."""
    start_time = time.time()

    # Validate category exists
    all_categories = list(config['keywords'].keys())
    if category_name not in all_categories:
        print(f"ERROR: Category '{category_name}' not found.", flush=True)
        print(f"Available categories: {all_categories}", flush=True)
        return

    print(f"{'='*60}", flush=True)
    print(f"REDO CATEGORY: '{category_name}'", flush=True)
    print(f"{'='*60}", flush=True)

    keywords = config['kv']
    max_results = config['max_results']
    publish_readme = config['publish_readme']
    publish_gitpage = config['publish_gitpage']
    cfg = config['keywords'][category_name]
    category_description = cfg.get('description', category_name)
    cat_max = cfg.get('max_results', max_results)

    # Step 1: Clear old data for this category
    print(f"\n[1/5] Clearing old data for '{category_name}' ...", flush=True)
    if publish_readme:
        n = clear_category_from_json(config['json_readme_path'], category_name)
        print(f"  Removed {n} papers from {config['json_readme_path']}", flush=True)
    if publish_gitpage:
        n = clear_category_from_json(config['json_gitpage_path'], category_name)
        print(f"  Removed {n} papers from {config['json_gitpage_path']}", flush=True)
    n = clear_category_from_cache(category_name)
    print(f"  Removed {n} entries from score cache", flush=True)

    # Step 2: Init LLM client
    llm_client = None
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key and OpenAI is not None:
        llm_client = OpenAI(api_key=api_key)
        print("[2/5] OpenAI client initialized (LLM filtering ON)", flush=True)
    else:
        print("[2/5] No OPENAI_API_KEY set - LLM filtering OFF", flush=True)

    score_cache = load_score_cache()

    # Step 3: Re-fetch this category
    print(f"\n[3/5] Re-fetching '{category_name}' from scratch ...", flush=True)
    data_collector = []
    data_collector_web = []

    # Keyword-based search (if this category has filters)
    if category_name in keywords:
        keyword = keywords[category_name]
        print(f"  Keyword search (max {cat_max} results) ...", flush=True)
        t0 = time.time()
        data, data_web = get_daily_papers(category_name, query=keyword,
                                          max_results=cat_max,
                                          llm_client=llm_client,
                                          category_description=category_description,
                                          score_cache=score_cache)
        count = len(data.get(category_name, {}))
        elapsed = time.time() - t0
        print(f"  Keyword search done: {count} papers kept ({elapsed:.1f}s)", flush=True)
        data_collector.append(data)
        data_collector_web.append(data_web)

    # Citation-based search (if this category has seed_papers)
    if 'seed_papers' in cfg:
        seed_count = len(cfg['seed_papers'])
        print(f"  Citation search ({seed_count} seed papers) ...", flush=True)
        t0 = time.time()
        data, data_web = get_citation_papers(category_name, cfg['seed_papers'],
                                              llm_client=llm_client,
                                              category_description=category_description,
                                              score_cache=score_cache)
        count = len(data.get(category_name, {}))
        elapsed = time.time() - t0
        print(f"  Citation search done: {count} papers kept ({elapsed:.1f}s)", flush=True)
        data_collector.append(data)
        data_collector_web.append(data_web)

    # Step 4: Save results (merge into existing JSON, replacing only this category)
    # README is generated separately by src/scripts/generate_readme.py
    print(f"\n[4/5] Saving results ...", flush=True)
    if publish_readme:
        update_json_file(config['json_readme_path'], data_collector)
        print(f"  JSON (readme) updated", flush=True)
    if publish_gitpage:
        update_json_file(config['json_gitpage_path'], data_collector_web)
        print(f"  JSON (gitpage) updated", flush=True)

    save_score_cache(score_cache)
    print(f"  Score cache saved ({len(score_cache)} entries)", flush=True)

    elapsed = time.time() - start_time
    total_papers = 0
    try:
        with open(config['json_readme_path'], 'r', encoding='utf-8') as f:
            content = f.read()
            if content:
                all_data = json.loads(content)
                total_papers = sum(len(p) for p in all_data.values())
    except Exception:
        pass
    print(f"\n[5/5] Done! '{category_name}' rebuilt. {total_papers} total papers in database. Elapsed: {elapsed:.1f}s", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',type=str, default='config.yaml',
                            help='configuration file path')
    parser.add_argument('--update_paper_links', default=False,
                        action="store_true",help='whether to update paper links etc.')
    parser.add_argument('--redo_category', type=str, default=None,
                        help='Redo a single category from scratch (clear and re-fetch). Other categories are kept as-is.')
    parser.add_argument('--category', type=str, default=None,
                        help='Fetch only this category (leave others unchanged).')
    args = parser.parse_args()
    config = load_config(args.config_path)
    config = {**config, 'update_paper_links':args.update_paper_links,
              'filter_category': args.category}

    if args.redo_category:
        redo_category(args.redo_category, **config)
    else:
        demo(**config)

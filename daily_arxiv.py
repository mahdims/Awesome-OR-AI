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

try:
    from openai import OpenAI
    from pydantic import BaseModel
except ImportError:
    OpenAI = None
    BaseModel = None

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
    with open(config_file,'r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
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

def load_score_cache(cache_path=SCORE_CACHE_PATH):
    """Load the persistent relevance score cache from disk."""
    try:
        with open(cache_path, "r") as f:
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
        with open(cache_path, "w") as f:
            json.dump(cache, f)
    except Exception as e:
        logging.warning(f"Failed to save score cache: {e}")

class RelevanceScore(BaseModel):
    score: int

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

    if client is None:
        return 10

    desc = category_description if category_description else category
    prompt = f"""Rate how relevant this paper is to the research category "{category}" on a scale of 0 to 10.

Category description: {desc}

0 = completely unrelated, 10 = perfectly on-topic.

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

    # Store in cache
    if score_cache is not None:
        score_cache[cache_key] = score

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

def get_daily_papers(topic, query="slam", max_results=2, llm_client=None, category_description="", score_cache=None):
    """
    @param topic: str
    @param query: str
    @param max_results: int
    @param llm_client: OpenAI client for relevance filtering (or None)
    @param category_description: str for LLM relevance scoring
    @param score_cache: dict - persistent cache of relevance scores
    @return paper_with_code: dict
    """
    content = dict()
    content_to_web = dict()

    results = arxiv_search(query=query, max_results=max_results)

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
            print(f"    [{idx}/{len(results)}] Skipping old paper ({publish_time.year}): {paper_title[:60]}", flush=True)
            continue

        # eg: 2108.09112v1 -> 2108.09112
        ver_pos = paper_id.find('v')
        if ver_pos == -1:
            paper_key = paper_id
        else:
            paper_key = paper_id[0:ver_pos]
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

        if repo_url is not None:
            content[paper_key] = "|**{}**|**{}**|{} et.al.|{}|[{}]({})|**[link]({})**|\n".format(
                   update_time,paper_title,paper_first_author,venue,paper_key,paper_url,repo_url)
            content_to_web[paper_key] = "|**{}**|**{}**|{} et.al.|{}|[{}]({})|**[link]({})**|\n".format(
                   update_time,paper_title,paper_first_author,venue,paper_key,paper_url,repo_url)
        else:
            content[paper_key] = "|**{}**|**{}**|{} et.al.|{}|[{}]({})|null|\n".format(
                   update_time,paper_title,paper_first_author,venue,paper_key,paper_url)
            content_to_web[paper_key] = "|**{}**|**{}**|{} et.al.|{}|[{}]({})|null|\n".format(
                   update_time,paper_title,paper_first_author,venue,paper_key,paper_url)

    data = {topic:content}
    data_web = {topic:content_to_web}
    return data,data_web

def get_citation_papers(topic, seed_papers, llm_client=None, category_description="", score_cache=None):
    """
    Get papers that cite the given seed papers using Semantic Scholar API.
    @param topic: str - category name
    @param seed_papers: list of arxiv ids (e.g. ["2405.17743", "2310.06116"])
    @param llm_client: OpenAI client for relevance filtering (or None)
    @param category_description: str for LLM relevance scoring
    @param score_cache: dict - persistent cache of relevance scores
    @return: (data, data_web) same format as get_daily_papers
    """
    content = dict()
    content_to_web = dict()

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

                if repo_url is not None:
                    content[paper_key] = "|**{}**|**{}**|{} et.al.|{}|[{}]({})|**[link]({})**|\n".format(
                        pub_date, paper_title, first_author, venue, paper_key, paper_url, repo_url)
                    content_to_web[paper_key] = "|**{}**|**{}**|{} et.al.|{}|[{}]({})|**[link]({})**|\n".format(
                        pub_date, paper_title, first_author, venue, paper_key, paper_url, repo_url)
                else:
                    content[paper_key] = "|**{}**|**{}**|{} et.al.|{}|[{}]({})|null|\n".format(
                        pub_date, paper_title, first_author, venue, paper_key, paper_url)
                    content_to_web[paper_key] = "|**{}**|**{}**|{} et.al.|{}|[{}]({})|null|\n".format(
                        pub_date, paper_title, first_author, venue, paper_key, paper_url)

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
    with open(filename, "r") as f:
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

    with open(filename, "w") as f:
        json.dump(new_data, f)

    logging.info(f"Cleaned {filename}: removed {removed_count} pre-{MIN_YEAR} papers, migrated {migrated_count} entries")

def update_paper_links(filename):
    '''
    weekly update paper links in json file
    '''
    def parse_arxiv_string(s):
        parts = s.split("|")
        # Handle both old (6 fields) and new (7 fields) format
        if len(parts) >= 8:
            # New format: |date|title|authors|venue|arxiv_id|code|
            date = parts[1].strip()
            title = parts[2].strip()
            authors = parts[3].strip()
            venue = parts[4].strip()
            arxiv_id = parts[5].strip()
            code = parts[6].strip()
        else:
            # Old format: |date|title|authors|arxiv_id|code|
            date = parts[1].strip()
            title = parts[2].strip()
            authors = parts[3].strip()
            venue = ''
            arxiv_id = parts[4].strip()
            code = parts[5].strip()
        arxiv_id = re.sub(r'v\d+', '', arxiv_id)
        return date,title,authors,venue,arxiv_id,code

    with open(filename,"r") as f:
        content = f.read()
        if not content:
            m = {}
        else:
            m = json.loads(content)

    json_data = m.copy()

    for keywords,v in json_data.items():
        logging.info(f'keywords = {keywords}')
        for paper_id,contents in v.items():
            contents = str(contents)

            update_time, paper_title, paper_first_author, venue, paper_url, code_url = parse_arxiv_string(contents)

            contents = "|{}|{}|{}|{}|{}|{}|\n".format(update_time,paper_title,paper_first_author,venue,paper_url,code_url)
            json_data[keywords][paper_id] = str(contents)
            logging.info(f'paper_id = {paper_id}, contents = {contents}')

            valid_link = False if '|null|' in contents else True
            if valid_link:
                continue
            try:
                repo_url = None
                # Try HuggingFace API first
                cr = requests.get(huggingface_papers_url + paper_id, timeout=10)
                if cr.status_code == 200:
                    hf = cr.json()
                    if hf.get("githubRepo"):
                        repo_url = hf["githubRepo"]

                # Fallback: search GitHub
                if repo_url is None:
                    title_match = re.search(r'\*\*(.+?)\*\*', paper_title)
                    search_term = title_match.group(1) if title_match else paper_id
                    repo_url = get_code_link(search_term)

                if repo_url is not None:
                    new_cont = contents.replace('|null|',f'|**[link]({repo_url})**|')
                    logging.info(f'ID = {paper_id}, contents = {new_cont}')
                    json_data[keywords][paper_id] = str(new_cont)

            except Exception as e:
                logging.error(f"exception: {e} with id: {paper_id}")
    # dump to json file
    with open(filename,"w") as f:
        json.dump(json_data,f)

def update_json_file(filename,data_dict):
    '''
    daily update json file using data_dict
    '''
    with open(filename,"r") as f:
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

    with open(filename,"w") as f:
        json.dump(json_data,f)

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

    with open(filename,"r") as f:
        content = f.read()
        if not content:
            data = {}
        else:
            data = json.loads(content)

    # clean README.md if daily already exist else create it
    with open(md_filename,"w+") as f:
        pass

    # write data into README.md
    with open(md_filename,"a+") as f:

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
                    f.write("|Publish Date|Title|Authors|Venue|PDF|Code|\n" + "|---|---|---|---|---|---|\n")
                else:
                    f.write("| Publish Date | Title | Authors | Venue | PDF | Code |\n")
                    f.write("|:---------|:-----------------------|:---------|:------|:------|:------|\n")

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

def demo(**config):
    start_time = time.time()
    data_collector = []
    data_collector_web= []

    keywords = config['kv']
    max_results = config['max_results']
    publish_readme = config['publish_readme']
    publish_gitpage = config['publish_gitpage']

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
        print(f"[3/6] Starting paper retrieval (max {max_results} per query)...", flush=True)
        total_found = 0
        cache_before = len(score_cache)

        # keyword-based search
        for i, (topic, keyword) in enumerate(keywords.items(), 1):
            print(f"\n  [{i}/{len(keywords)}] Keyword search: '{topic}' ...", flush=True)
            t0 = time.time()
            data, data_web = get_daily_papers(topic, query=keyword,
                                            max_results=max_results,
                                            llm_client=llm_client,
                                            category_description=category_descriptions.get(topic, topic),
                                            score_cache=score_cache)
            count = len(data.get(topic, {}))
            total_found += count
            elapsed = time.time() - t0
            print(f"  [{i}/{len(keywords)}] '{topic}' done: {count} papers kept ({elapsed:.1f}s)", flush=True)
            data_collector.append(data)
            data_collector_web.append(data_web)

        # citation-based search
        citation_topics = [(t, c) for t, c in config['keywords'].items() if 'seed_papers' in c]
        for i, (topic, cfg) in enumerate(citation_topics, 1):
            seed_count = len(cfg['seed_papers'])
            print(f"\n  [Citation {i}/{len(citation_topics)}] '{topic}' ({seed_count} seed papers) ...", flush=True)
            t0 = time.time()
            data, data_web = get_citation_papers(topic, cfg['seed_papers'],
                                                 llm_client=llm_client,
                                                 category_description=category_descriptions.get(topic, topic),
                                                 score_cache=score_cache)
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

    # 1. update README.md file
    if publish_readme:
        print("  Updating README ...", end=" ", flush=True)
        json_file = config['json_readme_path']
        md_file   = config['md_readme_path']
        if config['update_paper_links']:
            update_paper_links(json_file)
        else:
            update_json_file(json_file,data_collector)
        json_to_md(json_file,md_file, task ='Update Readme')
        print("done", flush=True)

    # 2. update docs/index.md file (to gitpage)
    if publish_gitpage:
        print("  Updating GitPage ...", end=" ", flush=True)
        json_file = config['json_gitpage_path']
        md_file   = config['md_gitpage_path']
        if config['update_paper_links']:
            update_paper_links(json_file)
        else:
            update_json_file(json_file,data_collector_web)
        json_to_md(json_file, md_file, task ='Update GitPage', \
            to_web = True, use_tc=False, use_b2t=False)
        print("done", flush=True)

    elapsed = time.time() - start_time
    # Count total papers in output
    total_papers = 0
    try:
        with open(config['json_readme_path'], 'r') as f:
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
        with open(filename, "r") as f:
            content = f.read()
            data = json.loads(content) if content else {}
    except FileNotFoundError:
        return 0
    removed = len(data.get(category, {}))
    data[category] = {}
    with open(filename, "w") as f:
        json.dump(data, f)
    return removed

def clear_category_from_cache(category, cache_path=SCORE_CACHE_PATH):
    """Remove all score cache entries for a given category."""
    cache = load_score_cache(cache_path)
    suffix = f"::{category}"
    keys_to_remove = [k for k in cache if k.endswith(suffix)]
    for k in keys_to_remove:
        del cache[k]
    save_score_cache(cache, cache_path)
    return len(keys_to_remove)

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
        print(f"  Keyword search (max {max_results} results) ...", flush=True)
        t0 = time.time()
        data, data_web = get_daily_papers(category_name, query=keyword,
                                          max_results=max_results,
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
    print(f"\n[4/5] Saving results ...", flush=True)
    if publish_readme:
        update_json_file(config['json_readme_path'], data_collector)
        json_to_md(config['json_readme_path'], config['md_readme_path'], task='Update Readme')
        print(f"  README updated", flush=True)
    if publish_gitpage:
        update_json_file(config['json_gitpage_path'], data_collector_web)
        json_to_md(config['json_gitpage_path'], config['md_gitpage_path'],
                   task='Update GitPage', to_web=True, use_tc=False, use_b2t=False)
        print(f"  GitPage updated", flush=True)

    save_score_cache(score_cache)
    print(f"  Score cache saved ({len(score_cache)} entries)", flush=True)

    elapsed = time.time() - start_time
    total_papers = 0
    try:
        with open(config['json_readme_path'], 'r') as f:
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
    args = parser.parse_args()
    config = load_config(args.config_path)
    config = {**config, 'update_paper_links':args.update_paper_links}

    if args.redo_category:
        redo_category(args.redo_category, **config)
    else:
        demo(**config)

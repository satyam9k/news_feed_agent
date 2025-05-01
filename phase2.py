#phase2
import os
import pickle
import time
import random
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from collections import Counter, defaultdict
from urllib.parse import urlparse

# --- scraping & parsing ---
import requests
from bs4 import BeautifulSoup
from newspaper import Article, Config

# --- sentence transformers ---
from sentence_transformers import SentenceTransformer

# --- google search API ---
from googleapiclient.discovery import build

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder 

# --- configuration ---

# phase 1 output directory
PHASE1_OUTPUT_DIR = './phase1_output_20k_all_mpnet/'

# paths to load artifacts
BASE_EMBEDDINGS_PATH = os.path.join(PHASE1_OUTPUT_DIR, 'base_article_embeddings.pkl')
CF_MODEL_PATH = os.path.join(PHASE1_OUTPUT_DIR, 'cf_model.pkl')
USER_PROFILES_PATH = os.path.join(PHASE1_OUTPUT_DIR, 'user_profiles.pkl')
PROCESSED_DATA_PATH = os.path.join(PHASE1_OUTPUT_DIR, 'processed_articles.csv')

# base model
BASE_MODEL_NAME = 'all-mpnet-base-v2'

# google search API credentials
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
    GOOGLE_CSE_ID = user_secrets.get_secret("GOOGLE_CSE_ID")
    print("Loaded Google API credentials from Kaggle secrets.")
except Exception as e:
    print(f"Kaggle secrets not found or error: {e}. Attempting environment variables...")
    try:
        from dotenv import load_dotenv
        load_dotenv()
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
        if GOOGLE_API_KEY and GOOGLE_CSE_ID: print("Loaded Google API credentials from environment variables.")
        else: print("Google API credentials not found in environment variables."); GOOGLE_API_KEY = None; GOOGLE_CSE_ID = None
    except ImportError: print(".env handling not available."); GOOGLE_API_KEY = None; GOOGLE_CSE_ID = None

# recommendation parameters
NUM_RECOMMENDATIONS = 10
NUM_SEARCH_RESULTS_TOTAL = 20
DEFAULT_NEWS_QUERY = "top breaking news"

# scraping configuration
REQUESTS_TIMEOUT = 10
REQUESTS_HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
NP3K_CONFIG = Config()
NP3K_CONFIG.browser_user_agent = REQUESTS_HEADERS['User-Agent']
NP3K_CONFIG.request_timeout = REQUESTS_TIMEOUT
NP3K_CONFIG.fetch_images = False
NP3K_CONFIG.memoize_articles = False
NP3K_CONFIG.verbose = False

#to load phase1 artifacts such as emneddings, simulated user profiles etc.
def load_phase1_artifacts(p1_dir, base_emb_file, cf_file, profiles_file):
    print("Loading Phase 1 artifacts...")
    artifacts = {'base_embeddings': None, 'cf_model': None, 'user_profiles': None}
    load_success = True
    base_emb_path = os.path.join(p1_dir, base_emb_file); cf_path = os.path.join(p1_dir, cf_file); profiles_path = os.path.join(p1_dir, profiles_file)
    if os.path.exists(base_emb_path):
        try:
            with open(base_emb_path, 'rb') as f: artifacts['base_embeddings'] = pickle.load(f)
            print(f"  - Loaded base embeddings ({len(artifacts['base_embeddings'])} articles).")
        except Exception as e: print(f"  - ERROR loading base embeddings: {e}"); load_success = False
    else: print(f"  - ERROR: Base embeddings file not found: {base_emb_path}"); load_success = False
    if os.path.exists(cf_path):
        try:
            with open(cf_path, 'rb') as f: artifacts['cf_model'] = pickle.load(f)
            print("  - Loaded CF model.")
        except Exception as e: print(f"  - WARNING: Error loading CF model: {e}")
    else: print(f"  - WARNING: CF model file not found: {cf_path}")
    if os.path.exists(profiles_path):
        try:
            with open(profiles_path, 'rb') as f: artifacts['user_profiles'] = pickle.load(f)
            print(f"  - Loaded user profiles ({len(artifacts['user_profiles'])} users).")
        except Exception as e: print(f"  - ERROR loading user profiles: {e}"); load_success = False
    else: print(f"  - ERROR: User profiles file not found: {profiles_path}"); load_success = False
    return artifacts, load_success
#to get google search results
def get_google_search_results(query, api_key, cse_id, num_results):
    if not api_key or not cse_id: print("ERROR: Google API Key or CSE ID not configured."); return []
    results = []; num_fetched = 0; start_index = 1; max_per_request = 10
    print(f"    Executing search for '{query}' (target: {num_results})...")
    try:
        service = build("customsearch", "v1", developerKey=api_key)
        while num_fetched < num_results:
            num_to_fetch = min(max_per_request, num_results - num_fetched);
            if num_to_fetch <= 0: break
            res = service.cse().list(q=query, cx=cse_id, num=num_to_fetch, start=start_index).execute()
            items = res.get("items", []); results.extend(items); num_fetched += len(items)
            next_page_info = res.get("queries", {}).get("nextPage")
            if next_page_info and num_fetched < num_results:
                start_index = next_page_info[0].get("startIndex");
                if not start_index: break
            else: break
        print(f"    ...fetched {len(results)} results.")
        return results
    except Exception as e: print(f"ERROR during Google Search API call for '{query}': {e}"); return results
#scrapping articles
def scrape_article_details(url):
    """
    Tries to scrape meta description using BeautifulSoup and full text using Newspaper3k.
    Returns a dictionary {'description': str|None, 'full_text': str|None}.
    """
    scraped_data = {'description': None, 'full_text': None}
    html_content = None

    # 1. Try fetching with Requests first for BS4
    try:
        response = requests.get(url, headers=REQUESTS_HEADERS, timeout=REQUESTS_TIMEOUT)
        response.raise_for_status() 
        html_content = response.text
    except requests.exceptions.RequestException as e:
        pass # Proceed to newspaper3k attempt

    # 2. Try parsing meta description with BeautifulSoup if HTML was fetched
    if html_content:
        try:
            soup = BeautifulSoup(html_content, 'lxml') # 'lxml' is generally faster
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc and meta_desc.get('content'):
                scraped_data['description'] = meta_desc['content'].strip()
            else:
                first_p = soup.find('p')
                if first_p and first_p.get_text():
                     scraped_data['description'] = first_p.get_text().strip()[:300] + "..." # Limit length

        except Exception as e:
            pass # Continue to newspaper3k

    # 3. Try fetching full text (and description) with Newspaper3k
    try:
        article = Article(url, config=NP3K_CONFIG)
        # Pass fetched HTML if available, otherwise newspaper downloads again
        if html_content and not article.is_downloaded:
            article.download(input_html=html_content)
        else:
             article.download()

        if article.is_downloaded:
            article.parse()
            scraped_data['full_text'] = article.text
            # Use newspaper's meta description if BS4 failed and newspaper found one
            if not scraped_data['description'] and article.meta_description:
                 scraped_data['description'] = article.meta_description.strip()
              
    except Exception as e:
        pass 
    return scraped_data
#embeddings
def get_embeddings(texts, model, device, batch_size=32):

    model.to(device)
    try:
        embeddings = model.encode(texts, show_progress_bar=False, batch_size=batch_size, device=device, normalize_embeddings=True)
        return embeddings
    except Exception as e: print(f"Error during embedding generation: {e}"); return None

def recalculate_profile_embedding(user_profile, base_embeddings, embedding_dim):
    liked_items = user_profile.get('liked_item_ids', []); profile_embeddings = []
    found_count = 0
    for item_id in liked_items:
        emb = base_embeddings.get(str(item_id));
        if emb is not None: profile_embeddings.append(emb); found_count += 1
    if profile_embeddings:
        avg_embedding = np.mean(np.array(profile_embeddings), axis=0); norm = np.linalg.norm(avg_embedding)
        if norm > 0: avg_embedding = avg_embedding / norm
        return avg_embedding
    else: return np.zeros(embedding_dim)
#user top categories
def get_user_top_categories(user_profile, id_to_category_map, num_topics=3):
    liked_ids = user_profile.get('liked_item_ids', []);
    liked_categories = [id_to_category_map.get(str(id)) for id in liked_ids if str(id) in id_to_category_map]
    liked_categories = [cat for cat in liked_categories if cat and isinstance(cat, str)]
    if not liked_categories: return []
    category_counts = Counter(liked_categories); common_generic = {'news', 'general', ''}
    filtered_counts = {cat: count for cat, count in category_counts.items() if cat not in common_generic}
    target_counts = filtered_counts if filtered_counts else category_counts
    sorted_categories = sorted(target_counts.items(), key=lambda item: (-item[1], item[0]))
    top_categories = [cat for cat, count in sorted_categories[:num_topics]]
    return top_categories
#article url
def is_likely_article_url(url):
    try:
        parsed_url = urlparse(url);
        if not parsed_url.path or parsed_url.path == '/': return False
        path_segments = [seg for seg in parsed_url.path.split('/') if seg]
        if len(path_segments) == 0: return False
        if len(path_segments) > 1: return True
        if len(path_segments) == 1 and (any(c.isdigit() for c in path_segments[0]) or '-' in path_segments[0] or len(path_segments[0]) > 15): return True
        if path_segments[0].lower() in ['video', 'videos', 'gallery', 'galleries', 'live']: return False
        return False
    except Exception: return False

# --- main recommendation function ---

def get_recommendations(user_id, artifacts, embedding_model, device, id_to_category_map, num_recs=NUM_RECOMMENDATIONS):
    print(f"\n--- Generating recommendations for user: {user_id} ---")

    # 1. Get User Profile & Recalculate Embedding
    user_profiles = artifacts.get('user_profiles')
    base_embeddings = artifacts.get('base_embeddings')
    if not user_profiles or user_id not in user_profiles or not base_embeddings: print(f"ERROR: User profile/base embeddings missing for user {user_id}."); return []
    user_profile = user_profiles[user_id]
    embedding_dim = embedding_model.get_sentence_embedding_dimension()
    print("Recalculating user profile embedding using base embeddings...")
    user_profile_embedding = recalculate_profile_embedding(user_profile, base_embeddings, embedding_dim)
    if np.all(user_profile_embedding == 0): print("Warning: User profile embedding is zero.")

    # 2. Determine User Topics & Generate Queries
    print("Determining user topics...")
    top_categories = get_user_top_categories(user_profile, id_to_category_map, num_topics=3)
    search_queries = []
    if top_categories: print(f"  - Top categories: {top_categories}"); search_queries = [f"latest news {cat}" for cat in top_categories]
    else: print(f"  - No top categories, using default query: '{DEFAULT_NEWS_QUERY}'"); search_queries = [DEFAULT_NEWS_QUERY]
    results_per_query = {}; remaining_results = NUM_SEARCH_RESULTS_TOTAL
    for i, q in enumerate(search_queries): count = max(1, remaining_results // (len(search_queries) - i)); results_per_query[q] = count; remaining_results -= count
    print(f"Using search queries: {list(results_per_query.keys())}")

    # 3. Fetch Live News Candidates
    print(f"Fetching up to {NUM_SEARCH_RESULTS_TOTAL} live news candidates...")
    all_search_results = []; urls_fetched = set()
    for query, num_to_fetch in results_per_query.items():
        results = get_google_search_results(query, GOOGLE_API_KEY, GOOGLE_CSE_ID, num_to_fetch)
        for item in results:
            url = item.get('link');
            if url and url not in urls_fetched: all_search_results.append(item); urls_fetched.add(url)
    if not all_search_results: print("No live news candidates found from searches."); return []
    print(f"Fetched {len(all_search_results)} unique results total.")

    # 4. Filter for Likely Articles & Process (Scrape Details)
    print(f"Filtering and processing {len(all_search_results)} candidates (scraping & embedding)...")
    candidate_articles = []
    texts_to_embed = []
    urls_processed = set()

    for result in tqdm(all_search_results, desc="Processing Candidates"):
        url = result.get('link'); title = result.get('title'); snippet = result.get('snippet')
        if not url or not title or url in urls_processed: continue
        if not is_likely_article_url(url): continue # Skip non-article URLs

        urls_processed.add(url)

        scraped_details = scrape_article_details(url)
        full_text = scraped_details.get('full_text')
        meta_description = scraped_details.get('description')

        text_for_embedding = full_text if full_text and len(full_text) > len(snippet or "") * 1.5 else snippet
        display_description = meta_description if meta_description else (full_text[:250]+"..." if full_text else snippet)

        if not text_for_embedding or len(text_for_embedding) < 30 : continue # Skip if no usable text

        candidate_articles.append({
            'title': title, 'url': url, 'snippet': snippet,
            'display_description': display_description, 
            'text_for_embedding': text_for_embedding[:1000] # Limit length for embedding
        })
        texts_to_embed.append(text_for_embedding[:1000])

    if not candidate_articles: print("No candidates remaining after filtering/processing."); return []
    print(f"Processing {len(candidate_articles)} filtered candidates.")

    # 5. Generate Embeddings
    print(f"Generating embeddings for {len(candidate_articles)} filtered candidates...")
    candidate_embeddings = get_embeddings(texts_to_embed, embedding_model, device, batch_size=ENCODE_BATCH_SIZE) 
    if candidate_embeddings is None or len(candidate_embeddings) != len(candidate_articles): print("ERROR: Failed to generate embeddings."); return []
    for i, article in enumerate(candidate_articles): article['embedding'] = candidate_embeddings[i]

    # 6. Calculate Content Similarity Score
    print("Calculating content similarity scores...")
    profile_emb_2d = user_profile_embedding.reshape(1, -1)
    content_scores = cosine_similarity(candidate_embeddings, profile_emb_2d).flatten()
    for i, article in enumerate(candidate_articles): article['content_score'] = content_scores[i]

    # 7. Rank Candidates
    print("Ranking candidates...")
    ranked_articles = sorted(candidate_articles, key=lambda x: x['content_score'], reverse=True)

    # 8. Format Output
    recommendations = []
    for i, article in enumerate(ranked_articles[:num_recs]):
        recommendations.append({
            'rank': i + 1, 'title': article['title'], 'url': article['url'],
             'snippet': article['snippet'], 
            'description': article.get('display_description', article.get('snippet')), 
            'content_score': float(article['content_score'])
        })
    print(f"Generated {len(recommendations)} recommendations.")
    return recommendations

# --- main execution ---
if __name__ == "__main__":
    overall_start_time = time.time()

    # --- Device Setup --- GPU
    if torch.cuda.is_available(): eval_device = torch.device("cuda")
    else: eval_device = torch.device("cpu")
    print(f"Using device: {eval_device}")
    # --------------------

    # --- Load Phase 1 Artifacts ---
    artifacts, load_ok = load_phase1_artifacts(
        PHASE1_OUTPUT_DIR, os.path.basename(BASE_EMBEDDINGS_PATH),
        os.path.basename(CF_MODEL_PATH), os.path.basename(USER_PROFILES_PATH)
    )
    if not load_ok: print("\nERROR: Failed to load essential Phase 1 artifacts. Exiting."); exit()

    # --- Load Processed Data for Category Lookup ---
    try:
        df_processed = pd.read_csv(PROCESSED_DATA_PATH)
        article_id_to_category = pd.Series(df_processed[CATEGORY_COL].values, index=df_processed['article_id'].astype(str)).to_dict() 
        print(f"Loaded processed data for category lookup ({len(df_processed)} articles).")


    # --- Load Embedding Model ---
    print(f"\nLoading base embedding model: {BASE_MODEL_NAME}...")
    try:
        embedding_model = SentenceTransformer(BASE_MODEL_NAME, device=eval_device)
        print("Embedding model loaded successfully.")
    except Exception as e: print(f"ERROR: Failed to load embedding model '{BASE_MODEL_NAME}': {e}"); exit()

    # --- Select User and Get Recommendations ---
    if artifacts.get('user_profiles'):
        available_users = list(artifacts['user_profiles'].keys())
        if not available_users: print("ERROR: No users found in loaded profiles."); exit()

        target_user_id = random.choice(available_users)
        print(f"\nTarget User ID: {target_user_id}")
        if target_user_id in artifacts['user_profiles']: print(f"  User's Phase 1 Preferences (from simulation): {artifacts['user_profiles'][target_user_id].get('preferences', 'N/A')}")

        recommendations = get_recommendations(
            target_user_id, artifacts, embedding_model,
            eval_device, article_id_to_category, num_recs=NUM_RECOMMENDATIONS
        )

        # --- Print Recommendations ---
        print("\n--- Top Recommendations ---")
        if recommendations:
            for rec in recommendations:
                print(f"Rank: {rec['rank']}")
                print(f"  Title: {rec['title']}")
                print(f"  URL: {rec['url']}")
                print(f"  Description: {rec.get('description', 'N/A')}") 
                print(f"  Score: {rec['content_score']:.4f}")
                print("-" * 20)
        else: print("No recommendations generated for this user.")

    else: print("ERROR: User profiles artifact not loaded.")

    overall_end_time = time.time()
    print(f"\nPhase 2 execution finished in {overall_end_time - overall_start_time:.2f} seconds.")

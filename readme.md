# Personalized News Feed Agent

## 1. Objective

This project implements an AI agent designed to recommend news articles tailored to individual user preferences. The goal is to combat information overload and enhance user engagement by delivering relevant content using a combination of content analysis (embeddings) and collaborative behavior patterns.

## 2. Approach Overview

The project utilizes a two-phase approach:

1.  **Phase 1: Offline Training & Evaluation:**
    *   **Data:** Uses a historical news dataset (derived from NewsAPI via Kaggle) containing article metadata and content.
    *   **Embeddings:** Explores generating sentence embeddings using pre-trained transformer models (`sentence-transformers`). Specifically, `all-mpnet-base-v2` was chosen after evaluation showed it outperformed fine-tuning attempts (SimCSE) on clustering metrics for this dataset. Embeddings for the historical articles are generated using this base model.
    *   **User Simulation:** Simulates user profiles and their interactions ('likes') based on matching predefined category preferences with article categories.
    *   **Collaborative Filtering:** Trains an SVD (Singular Value Decomposition) model using the `scikit-surprise` library on the simulated user-item interactions to capture collaborative patterns. Offline evaluation (Precision/Recall) is performed.
    *   **User Profiles:** Creates initial user profiles storing liked article IDs and an average profile embedding calculated from the **base model embeddings** of liked items.
    *   **Evaluation:** Includes clustering metrics (Silhouette, ARI, NMI) to compare base vs. fine-tuned embedding performance and offline CF metrics.

2.  **Phase 2: Online Recommendation:**
    *   **Candidate Generation:** Given a target user, identifies their top interest categories from their profile. Uses the Google Custom Search API to fetch recent news article candidates based on dynamic search queries related to these topics.
    *   **Filtering & Scraping:** Filters search results to prioritize likely article URLs (over homepages). Uses `requests`, `BeautifulSoup`, and `newspaper3k` to scrape article details, prioritizing meta descriptions and falling back to snippets or body text.
    *   **Embedding & Scoring:** Generates embeddings for the filtered, scraped candidates using the loaded **base `all-mpnet-base-v2` model**. Calculates content relevance score based on the cosine similarity between candidate embeddings and the target user's profile embedding.
    *   **Ranking & Output:** Ranks candidates primarily by content similarity score and outputs the top N recommendations, including title, URL, description, and score.

## 3. Setup Instructions

1.  **Clone Repository:**
    ```bash
    git clone https://github.com/satyam9k/news_feed_agent.git
    cd news_feed_agent
    ```
2.  **Create Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download Data:**
    *   Obtain the news dataset CSV file (https://www.kaggle.com/datasets/everydaycodings/global-news-dataset?select=data.csv).
    *   Update the `DATA_FILE` variable in the Phase 1 script (`phase1.py`) to point to the correct path of the CSV file.
5.  **Setup Credentials:**
    *   Obtain a Google API Key (enabled for Custom Search API) and a Custom Search Engine ID.
    *   **Local:** Create a file named `.env` in the project root and add:
        ```
        GOOGLE_API_KEY=YOUR_ACTUAL_API_KEY
        GOOGLE_CSE_ID=YOUR_ACTUAL_CSE_ID
        ```
    *   **Kaggle:** Add `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` as secrets to your notebook.

## 5. Running Instructions

1.  **Execute Phase 1 (Offline Training & Evaluation):**
    *   Run the Phase 1 script from your terminal (ensure virtual environment is active):
        ```bash
        python phase1.py
        ```
    *   This script will process data, generate embeddings (base first, then attempt fine-tuning, then generate final embeddings using the chosen model - likely the base model based on evaluation), evaluate models, simulate interactions, train CF, create profiles, and save all artifacts to the directory specified by `OUTPUT_DIR`.
    *   Monitor the console output for progress and check the `evaluation_results.txt` file in the output directory.

2.  **Execute Phase 2 (Online Recommendation):**
    *   Run the Phase 2 script:
        ```bash
        python phase2.py
        ```
    *   **Important:** Ensure the `PHASE1_OUTPUT_DIR` variable inside `phase2.py` matches the output directory created by Phase 1.
    *   The script will load Phase 1 artifacts, load the base embedding model, select a random user, determine their topics, fetch live news via Google Search, scrape/process candidates, and print the top recommended articles (Title, URL, Description, Score) to the console.


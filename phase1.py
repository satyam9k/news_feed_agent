import os
os.environ['WANDB_DISABLED'] = 'true'

import pandas as pd
import numpy as np
import random
import pickle
import time
import torch
from tqdm import tqdm
from collections import defaultdict 

# --- sentence transformers ---
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# --- collaborative filtering ---
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split # For CF evaluation

# --- evaluation metrics ---
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder 

# --- configuration ---
DATA_FILE = '/kaggle/input/news-csv/data.csv'
NUM_ARTICLES_TO_USE = 20000 

TEXT_COLS = ['title', 'description', 'content']
ID_COL = 'article_id'
CATEGORY_COL = 'category'
DATE_COL = 'published_at'

# model/output paths
OUTPUT_DIR = './phase1_output_20k_all_mpnet' 
os.makedirs(OUTPUT_DIR, exist_ok=True)
BASE_EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR, 'base_article_embeddings.pkl')
EMBEDDING_MODEL_PATH = os.path.join(OUTPUT_DIR, 'fine_tuned_embedding_model_simcse')
ARTICLE_EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR, 'article_embeddings.pkl')
CF_MODEL_PATH = os.path.join(OUTPUT_DIR, 'cf_model.pkl')
USER_PROFILES_PATH = os.path.join(OUTPUT_DIR, 'user_profiles.pkl')
INTERACTION_DATA_PATH = os.path.join(OUTPUT_DIR, 'simulated_interactions.csv')
PROCESSED_DATA_PATH = os.path.join(OUTPUT_DIR, 'processed_articles.csv')
EVALUATION_RESULTS_PATH = os.path.join(OUTPUT_DIR, 'evaluation_results.txt')

# model parameters
SENTENCE_TRANSFORMER_MODEL_NAME = 'all-mpnet-base-v2'
FINETUNE_EPOCHS = 2
FINETUNE_BATCH_SIZE = 8 # reduced batch size for preventation of computation overload
LEARNING_RATE = 2e-5
ENCODE_BATCH_SIZE = 24 # reduced encode batch size for preventation of computation overload

# evaluation parameters
CLUSTER_EVAL_SAMPLE_SIZE = 3000
KMEANS_N_INIT = 5 #  5 to potentially reduce memory slightly during eval
PRECISION_RECALL_K = 10

#--clean text--
def clean_text(text):
    if isinstance(text, str): return text.lower().strip()
    return ""
#--combine--text--fields--
def combine_text_fields(row, fields):
    combined = []
    for field in fields:
        text = row.get(field, None)
        if pd.notna(text):
            text_str = str(text).strip()
            if text_str.lower() not in ['nan', 'none', '']: combined.append(text_str)
    return ". ".join(combined)
#calculate---clustering---metrics---
def calculate_clustering_metrics(embeddings, labels, n_clusters):
    if len(embeddings) < n_clusters: return {'silhouette': np.nan, 'ari': np.nan, 'nmi': np.nan}
    #reduce n_init if memory is tight
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=KMEANS_N_INIT, verbose=0)
    try:
        cluster_labels = kmeans.fit_predict(embeddings)
        silhouette = silhouette_score(embeddings, cluster_labels) if len(set(cluster_labels)) > 1 else np.nan
        ari = adjusted_rand_score(labels, cluster_labels)
        nmi = normalized_mutual_info_score(labels, cluster_labels)
        return {'silhouette': silhouette, 'ari': ari, 'nmi': nmi}
    except MemoryError:
        print("MemoryError during KMeans clustering evaluation.")
        return {'silhouette': np.nan, 'ari': np.nan, 'nmi': np.nan}
#--calculate--precision--recall
def calculate_precision_recall_at_k(predictions, k=10, threshold=1.0):
    user_est_true = defaultdict(list); [user_est_true[uid].append((est, true_r)) for uid, _, true_r, est, _ in predictions]
    precisions, recalls = {}, {}
    if not user_est_true: return 0, 0
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold)) for (est, true_r) in user_ratings[:k])
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0
    avg_precision = sum(prec for prec in precisions.values()) / len(precisions) if precisions else 0
    avg_recall = sum(rec for rec in recalls.values()) / len(recalls) if recalls else 0
    return avg_precision, avg_recall

# --- main--script ---
if __name__ == "__main__":
    start_overall_time = time.time()
    evaluation_log = []
    print(f"--- Phase 1: Offline Training & Evaluation ({NUM_ARTICLES_TO_USE} data, {SENTENCE_TRANSFORMER_MODEL_NAME}) ---") 

    # === Device Setup === To run on GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA available! Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")
    evaluation_log.append(f"Using device: {device}")
    evaluation_log.append(f"Using base model: {SENTENCE_TRANSFORMER_MODEL_NAME}")
    evaluation_log.append(f"Target article count: {NUM_ARTICLES_TO_USE}")
    evaluation_log.append(f"Fine-tuning Batch Size: {FINETUNE_BATCH_SIZE}")
    evaluation_log.append(f"Encoding Batch Size: {ENCODE_BATCH_SIZE}")
    # ====================

    #------simulation--parameters--definition---
    NUM_USERS = 50
    USER_PREFERENCES = {
        f"user_{i}": random.sample(['business', 'technology', 'entertainment', 'health', 'science', 'sports', 'general'], k=random.randint(1, 3))
        for i in range(NUM_USERS)
    }
    INTERACTION_PROBABILITY = 0.7
    evaluation_log.append(f"Simulation: {NUM_USERS} users, prob={INTERACTION_PROBABILITY}") # log simulation params


    # 1. load and prepare sata-------
    print(f"\n[1/8] Loading and preparing data from {DATA_FILE}...")
    try:
        df = pd.read_csv(DATA_FILE)
        print(f"Original dataset shape: {df.shape}")
        if len(df) > NUM_ARTICLES_TO_USE:
            df = df.sample(n=NUM_ARTICLES_TO_USE, random_state=42).copy()
            print(f"Using random sample of {len(df)} articles.")

        # --- data cleaning ---
        if not ID_COL in df.columns: df[ID_COL] = df.index
        df[ID_COL] = df[ID_COL].astype(str)
        df.drop_duplicates(subset=[ID_COL], keep='first', inplace=True)
        print("Combining text fields...")
        df['combined_text'] = df.apply(lambda row: combine_text_fields(row, TEXT_COLS), axis=1)
        df.dropna(subset=[CATEGORY_COL], inplace=True)
        df = df[df['combined_text'].str.strip().astype(bool)]
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')
        df.dropna(subset=[DATE_COL], inplace=True)

        # --- prepare labels for clustering ---
        df[CATEGORY_COL] = df[CATEGORY_COL].astype(str).str.lower().str.strip()
        label_encoder = LabelEncoder()
        df['category_encoded'] = label_encoder.fit_transform(df[CATEGORY_COL])
        n_clusters = df['category_encoded'].nunique()
        print(f"Found {n_clusters} unique categories for clustering.")

        df_processed = df[[ID_COL, CATEGORY_COL, 'category_encoded', DATE_COL, 'combined_text']].copy()
        df_processed.to_csv(PROCESSED_DATA_PATH, index=False)
        processed_count = len(df_processed)
        print(f"Processed data saved ({processed_count} articles).")
        evaluation_log.append(f"Number of processed articles: {processed_count}")
        if processed_count == 0: raise ValueError("No articles left after processing.")

    except Exception as e:
        print(f"Error during data loading/preparation: {e}")
        exit()

    # 2. generate base model embeddings (Before Fine-Tuning)
    print(f"\n[2/8] Generating Base Model Embeddings ({SENTENCE_TRANSFORMER_MODEL_NAME})...")
    base_embeddings_np = None # initialize
    try:
        base_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL_NAME, device=device)
        article_texts = df_processed['combined_text'].tolist()
        article_ids = df_processed[ID_COL].tolist()
        print(f"Using encode batch size: {ENCODE_BATCH_SIZE}")
        base_embeddings_np = base_model.encode(article_texts, show_progress_bar=True, batch_size=ENCODE_BATCH_SIZE, device=device)
        base_article_embeddings = {id: emb for id, emb in zip(article_ids, base_embeddings_np)}
        with open(BASE_EMBEDDINGS_PATH, 'wb') as f: pickle.dump(base_article_embeddings, f)
        print(f"Generated and saved {len(base_article_embeddings)} base embeddings.")
        del base_model # free memory
        if torch.cuda.is_available(): torch.cuda.empty_cache() # clear cache
    except Exception as e:
        print(f"Error generating base embeddings: {e}")
        evaluation_log.append(f"ERROR generating base embeddings: {e}")

    # 3. fine-tune embedding model using SimCSE approach
    print(f"\n[3/8] Fine-tuning embedding model ({SENTENCE_TRANSFORMER_MODEL_NAME}) using SimCSE approach...")
    if processed_count < FINETUNE_BATCH_SIZE:
        print(f"Warning: Not enough processed articles ({processed_count}) for the fine-tuning batch size ({FINETUNE_BATCH_SIZE}). Skipping fine-tuning.")
        evaluation_log.append("\nSkipped fine-tuning (insufficient data for batch size)")
        # Use base model path if skipping fine-tuning-----------------------
        EMBEDDING_MODEL_PATH = SENTENCE_TRANSFORMER_MODEL_NAME
    else:
        try:
            # load model again for fine-tuning
            model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL_NAME)
            model.to(device)
            print(f"Model loaded to device: {model.device}")

            train_examples = [InputExample(texts=[text, text]) for text in article_texts] # use full texts list
            print(f"Created {len(train_examples)} SimCSE training examples.")

            train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=FINETUNE_BATCH_SIZE)
            train_loss = losses.MultipleNegativesRankingLoss(model=model, scale=20.0)
            num_training_steps = len(train_dataloader) * FINETUNE_EPOCHS
            warmup_steps = int(num_training_steps * 0.1)
            print(f"Warmup steps: {warmup_steps}")
            print(f"NOTE: Fine-tuning {len(train_examples)} examples for {FINETUNE_EPOCHS} epochs with BATCH SIZE {FINETUNE_BATCH_SIZE} will take longer.")

            print(f"[{time.strftime('%H:%M:%S')}] Starting model.fit...")
            start_fit_time = time.time()
            save_steps = max(1, len(train_dataloader) // 2)
            model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=FINETUNE_EPOCHS,
                      optimizer_params={'lr': LEARNING_RATE}, warmup_steps=warmup_steps, show_progress_bar=True,
                      output_path=EMBEDDING_MODEL_PATH + '_checkpoints', checkpoint_save_steps=save_steps,
                      checkpoint_path=EMBEDDING_MODEL_PATH + '_checkpoints')
            end_fit_time = time.time()
            fit_duration = end_fit_time - start_fit_time
            print(f"[{time.strftime('%H:%M:%S')}] model.fit finished in {fit_duration:.2f} seconds.")
            evaluation_log.append(f"Fine-tuning duration ({FINETUNE_EPOCHS} epochs): {fit_duration:.2f}s")

            model.save(EMBEDDING_MODEL_PATH)
            print(f"Fine-tuned embedding model saved to {EMBEDDING_MODEL_PATH}")

        except Exception as e:
            print(f"Error during model fine-tuning: {e}")
            evaluation_log.append(f"Fine-tuning FAILED: {e}")
            try:
                SentenceTransformer(SENTENCE_TRANSFORMER_MODEL_NAME)
                EMBEDDING_MODEL_PATH = SENTENCE_TRANSFORMER_MODEL_NAME
                print(f"Using base model '{EMBEDDING_MODEL_PATH}' as fallback path.")
            except Exception as e_fallback:
                print(f"ERROR: Fine-tuning failed and base model unavailable: {e_fallback}. Cannot proceed.")
                exit()


    # 4. generate final embeddings and evaluate clustering
    print("\n[4/8] Generating Final Embeddings and Evaluating Clustering...")
    tuned_embeddings_np = None #initialize
    try:
        print(f"Loading model for final embedding generation from: {EMBEDDING_MODEL_PATH}")
        # load the final model (fine-tuned or base fallback)
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH, device=device)

        print(f"Using encode batch size: {ENCODE_BATCH_SIZE}")
        tuned_embeddings_np = embedding_model.encode(article_texts, show_progress_bar=True, batch_size=ENCODE_BATCH_SIZE, device=device)
        article_embeddings = {id: emb for id, emb in zip(article_ids, tuned_embeddings_np)}
        with open(ARTICLE_EMBEDDINGS_PATH, 'wb') as f: pickle.dump(article_embeddings, f)
        print(f"Generated and saved {len(article_embeddings)} final embeddings.")

        # --- clustering evaluation ---
        print("\nPerforming Clustering Evaluation...")
        if tuned_embeddings_np is not None and processed_count > 1:
            true_labels = df_processed['category_encoded'].values
            eval_indices = np.arange(processed_count)
            if processed_count > CLUSTER_EVAL_SAMPLE_SIZE:
                print(f"Evaluating clustering on a random sample of {CLUSTER_EVAL_SAMPLE_SIZE} articles.")
                sample_k = min(CLUSTER_EVAL_SAMPLE_SIZE, processed_count)
                eval_indices = np.random.choice(eval_indices, sample_k, replace=False)

            eval_labels = true_labels[eval_indices]
            eval_tuned_embeddings = tuned_embeddings_np[eval_indices]

            print("Evaluating Final Model Clustering...")
            tuned_metrics = calculate_clustering_metrics(eval_tuned_embeddings, eval_labels, n_clusters)
            print(f"  - Silhouette: {tuned_metrics['silhouette']:.4f}")
            print(f"  - ARI: {tuned_metrics['ari']:.4f}")
            print(f"  - NMI: {tuned_metrics['nmi']:.4f}")
            evaluation_log.append("\n--- Embedding Clustering (Final Model) ---")
            evaluation_log.append(f"Silhouette: {tuned_metrics['silhouette']:.4f}")
            evaluation_log.append(f"ARI: {tuned_metrics['ari']:.4f}")
            evaluation_log.append(f"NMI: {tuned_metrics['nmi']:.4f}")

            if base_embeddings_np is not None:
                print("Evaluating Base Model Clustering...")
                if len(base_embeddings_np) == processed_count:
                    eval_base_embeddings = base_embeddings_np[eval_indices]
                    base_metrics = calculate_clustering_metrics(eval_base_embeddings, eval_labels, n_clusters)
                    print(f"  - Silhouette: {base_metrics['silhouette']:.4f}")
                    print(f"  - ARI: {base_metrics['ari']:.4f}")
                    print(f"  - NMI: {base_metrics['nmi']:.4f}")
                    evaluation_log.append("\n--- Embedding Clustering (Base Model) ---")
                    evaluation_log.append(f"Silhouette: {base_metrics['silhouette']:.4f}")
                    evaluation_log.append(f"ARI: {base_metrics['ari']:.4f}")
                    evaluation_log.append(f"NMI: {base_metrics['nmi']:.4f}")
                else:
                    print("Skipping base model clustering evaluation (dimension mismatch).")
                    evaluation_log.append("\n--- Embedding Clustering (Base Model) --- Skipped (Dimension Mismatch)")
            else:
                 print("Skipping base model clustering evaluation (base embeddings failed).")
                 evaluation_log.append("\n--- Embedding Clustering (Base Model) --- Skipped (Base embedding generation failed)")
        else:
             print("Skipping clustering evaluation (no embeddings generated or insufficient data).")
             evaluation_log.append("\nSkipped Clustering Evaluation (No Embeddings or Data)")

    except Exception as e:
        print(f"Error generating/evaluating final embeddings: {e}")
        evaluation_log.append(f"\nError during Final Embedding Generation/Evaluation: {e}")
        if 'article_embeddings' not in locals(): article_embeddings = {}


    # 5. simulate user interactions
    print("\n[5/8] Simulating user interactions...")
    interactions = [] 
    if 'df_processed' not in locals(): raise ValueError("df_processed not found.")
    for _, row in tqdm(df_processed.iterrows(), total=len(df_processed), desc="Simulating"):
        article_id = row[ID_COL]; category = row[CATEGORY_COL]; pub_date = row[DATE_COL]
        for user_id, prefs in USER_PREFERENCES.items():
            if isinstance(category, str) and category.lower() in [p.lower() for p in prefs]:
                if random.random() < INTERACTION_PROBABILITY:
                    interactions.append({'user_id': user_id,'item_id': article_id,'rating': 1.0, 'timestamp': pub_date})
    if not interactions: interactions_df = pd.DataFrame(columns=['user_id', 'item_id', 'rating', 'timestamp'])
    else: interactions_df = pd.DataFrame(interactions)
    interactions_df.to_csv(INTERACTION_DATA_PATH, index=False)
    num_interactions = len(interactions_df)
    print(f"Simulated {num_interactions} interactions. Saved.")
    evaluation_log.append(f"\nNumber of simulated interactions: {num_interactions}")


    # 6. evaluate and train collaborative filtering model
    print("\n[6/8] Evaluating and Training Collaborative Filtering model...")
    if num_interactions > 20:
        try:
            reader = Reader(rating_scale=(0, 1))
            data = Dataset.load_from_df(interactions_df[['user_id', 'item_id', 'rating']], reader)
            print("Splitting interactions for CF evaluation...")
            trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
            print("Training CF model on training split...")
            algo_eval = SVD(n_factors=50, n_epochs=20, random_state=42, verbose=False)
            algo_eval.fit(trainset)
            print("Making predictions on test split...")
            predictions = algo_eval.test(testset)
            avg_precision, avg_recall = calculate_precision_recall_at_k(predictions, k=PRECISION_RECALL_K)
            print(f"\n--- CF Evaluation (on test split) ---")
            print(f"Precision@{PRECISION_RECALL_K}: {avg_precision:.4f}")
            print(f"Recall@{PRECISION_RECALL_K}:    {avg_recall:.4f}")
            evaluation_log.append(f"\n--- CF Evaluation (test split) ---")
            evaluation_log.append(f"Precision@{PRECISION_RECALL_K}: {avg_precision:.4f}")
            evaluation_log.append(f"Recall@{PRECISION_RECALL_K}:    {avg_recall:.4f}")

            print("\nRetraining CF model on FULL interaction dataset...")
            full_trainset = data.build_full_trainset()
            algo = SVD(n_factors=50, n_epochs=20, random_state=42, verbose=False)
            start_cf_time = time.time(); algo.fit(full_trainset); end_cf_time = time.time()
            cf_duration = end_cf_time - start_cf_time
            print(f"Final CF model training completed in {cf_duration:.2f}s.")
            with open(CF_MODEL_PATH, 'wb') as f: pickle.dump(algo, f)
            print(f"Final CF model saved to {CF_MODEL_PATH}")
            evaluation_log.append(f"Final CF model training duration: {cf_duration:.2f}s")
        except Exception as e:
            print(f"Error during CF model training/evaluation: {e}")
            evaluation_log.append(f"\nError during CF training/evaluation: {e}")
            if os.path.exists(CF_MODEL_PATH): os.remove(CF_MODEL_PATH)
    else:
        print("Skipping CF model training/evaluation (insufficient simulated interactions).")
        evaluation_log.append("\nSkipped CF training/evaluation (insufficient interactions)")


    # 7. create initial user profiles
    print("\n[7/8] Creating initial user profiles...")
    user_profiles = {}
    if not interactions_df.empty and article_embeddings: 
        user_interactions = interactions_df.groupby('user_id')['item_id'].apply(list).to_dict()
        try:
            if 'embedding_model' in locals() and isinstance(embedding_model, SentenceTransformer):
                emb_dim = embedding_model.get_sentence_embedding_dimension()
            else: # fallback if model loading failed
                emb_dim = 768 # MPNet dimension
                print(f"Warning: Using fallback embedding dimension {emb_dim}")
        except Exception:
            emb_dim = 768 # fallback for MPNet dimension
            print(f"Warning: Using fallback embedding dimension {emb_dim}")

        num_users_with_profiles = 0
        for user_id, liked_items in tqdm(user_interactions.items(), desc="Creating Profiles"):
            embs = [article_embeddings.get(item_id) for item_id in liked_items if item_id in article_embeddings]
            valid_liked_items = [item_id for item_id in liked_items if item_id in article_embeddings]
            if embs:
                 avg_embedding = np.mean(np.array(embs), axis=0)
                 user_profiles[user_id] = {'liked_item_ids': valid_liked_items,'profile_embedding': avg_embedding, 'preferences': USER_PREFERENCES.get(user_id, [])}
                 num_users_with_profiles += 1
        print(f"Created profiles for {num_users_with_profiles} users with valid liked items.")
        evaluation_log.append(f"\nNumber of users with profiles created: {num_users_with_profiles}")

    else:
        print("Skipping profile creation (no interactions or embeddings).")
        evaluation_log.append("\nSkipped user profile creation.")
    with open(USER_PROFILES_PATH, 'wb') as f: pickle.dump(user_profiles, f)
    print(f"User profiles saved.")


    # 8. save evaluation results & completion summary
    print("\n[8/8] Saving Evaluation Results & Final Summary ---")
    try:
        with open(EVALUATION_RESULTS_PATH, 'w') as f:
            f.write(f"--- Phase 1 Evaluation Report ({time.strftime('%Y-%m-%d %H:%M:%S')}) ---\n")
            f.write(f"Model: {SENTENCE_TRANSFORMER_MODEL_NAME}\n")
            f.write(f"Data Sample Size: {NUM_ARTICLES_TO_USE}\n")
            f.write(f"Processed Articles: {processed_count}\n")
            f.write("-" * 30 + "\n\n")
            for line in evaluation_log: f.write(line + "\n")
        print(f"Evaluation results saved to: {EVALUATION_RESULTS_PATH}")
    except Exception as e: print(f"Error saving evaluation results: {e}")

    end_overall_time = time.time()
    total_duration = end_overall_time - start_overall_time
    print(f"\n--- Phase 1 Complete ---")
    print(f"Total execution time: {total_duration:.2f} seconds.")
    # append final time outside the main log list
    try:
        with open(EVALUATION_RESULTS_PATH, 'a') as f:
             f.write(f"\nTotal execution time: {total_duration:.2f}s\n")
    except: pass
    print(f"Artifacts saved in: {OUTPUT_DIR}")
    print(f"- Processed Data: {PROCESSED_DATA_PATH}")
    print(f"- Base Embeddings: {BASE_EMBEDDINGS_PATH}")
    print(f"- Final Embedding Model Used: {EMBEDDING_MODEL_PATH}/")
    print(f"- Final Article Embeddings: {ARTICLE_EMBEDDINGS_PATH}")
    print(f"- Simulated Interactions: {INTERACTION_DATA_PATH}")
    if os.path.exists(CF_MODEL_PATH):
        print(f"- CF Model: {CF_MODEL_PATH}")
    else:
        print("- CF Model: Not generated")
    print(f"- User Profiles: {USER_PROFILES_PATH}")
    print(f"- Evaluation Results: {EVALUATION_RESULTS_PATH}")

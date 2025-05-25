import argparse
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer


def similarity(x, z):
    cosine_sim = (x @ z.T)[0, 0] / (np.linalg.norm(x) * np.linalg.norm(z))
    # Transform the cosine similarity to [0, 1]
    return (1 + cosine_sim) / 2


def diversity_score(x, z, selected, alpha):
    term_1 = similarity(x, z)

    if selected:
        term_2 = max([similarity(z, w) for w in selected])
    else:
        term_2 = 0

    score = alpha * term_1 - (1 - alpha) * term_2
    return score


def compute_diversity_based_top_k(x, candidates, alpha, k):
    selected = []
    selected_ids = []

    for selection in range(k):
        diversity_scores = []

        # Compute diversity score for non-selected candidates
        for i, (idx, curr_candidate) in enumerate(candidates):
            if idx in selected_ids:
                continue

            score = diversity_score(x, curr_candidate, selected, alpha)
            diversity_scores.append((idx, score))

        # Get the candidate with the highest diversity score
        max_item = max(diversity_scores, key=lambda item: item[1])

        # Add maximum score candidate to selected
        max_idx = max_item[0]
        selected.append(candidates[max_idx][1])
        selected_ids.append(max_idx)

    return selected, selected_ids


def extract_text(sample, dataset_name):
    """
    Extracts the text field(s) based on dataset structure.
    """
    if dataset_name.lower() == 'sst2':
        return sample.get('sentence', '')
    elif dataset_name.lower() == 'rte':
        s1 = sample.get('sentence1', '')
        s2 = sample.get('sentence2', '')
        return f"{s1} {s2}".strip()
    elif dataset_name.lower() == 'trec':
        return sample.get('text', '')
    elif dataset_name.lower() == 'cola':
        return sample.get('sentence', '')
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def main(args):
    # Initialize components based on the method
    if args.method == 'SBERT':
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
    elif args.method == 'TFIDF':
        vectorizer = TfidfVectorizer()
    else:
        raise ValueError("Unsupported method. Choose either 'SBERT' or 'TFIDF'.")

    # Load datasets
    if args.method == 'TFIDF':
        with open(args.train_file, 'r') as f:
            train_data = [json.loads(x) for x in f]
        train_sentences = [extract_text(s, args.dataset) for s in train_data]
        vectorizer.fit(train_sentences)

    with open(args.test_file, 'r') as f:
        test_data = json.load(f)

    print("Loaded datasets")

    reranked_data = []
    count = 1

    for sample in test_data:
        # Extract and encode the query
        query_text = extract_text(sample, args.dataset)

        if args.method == 'SBERT':
            x = embedder.encode([query_text], convert_to_tensor=False)
        elif args.method == 'TFIDF':
            x = vectorizer.transform([query_text])

        # Encode the candidates
        candidates = [
            (i, embedder.encode([extract_text(candidate, args.dataset)], convert_to_tensor=False))
            if args.method == 'SBERT'
            else (i, vectorizer.transform([extract_text(candidate, args.dataset)]))
            for i, candidate in enumerate(sample['top_k'])
        ]

        # Re-rank the candidates
        reranked_candidates, selected_ids = compute_diversity_based_top_k(x, candidates, args.alpha, args.top_k)
        ranked_candidates = [sample['top_k'][i] for i in selected_ids]

        sample.pop('top_k')
        sample['alpha'] = args.alpha
        sample['reranked_top_k'] = ranked_candidates
        reranked_data.append(sample)

        if count % 50 == 0:
            print(f"Processed {count} samples")

        count += 1

    # Save the re-ranked data
    with open(args.output_file, 'w') as f:
        json.dump(reranked_data, f, indent=2)

    print(f"Saved re-ranked data to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-rank candidates based on diversity.")
    parser.add_argument("--train_file", type=str, help="Path to the training dataset file (required for TFIDF).")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test dataset file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output re-ranked file.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., SST2, RTE, TREC, COLA).")
    parser.add_argument("--method", type=str, required=True, choices=['SBERT', 'TFIDF'], help="Method to use: SBERT or TFIDF.")
    parser.add_argument("--alpha", type=float, default=0.7, help="Diversity parameter alpha.")
    parser.add_argument("--top_k", type=int, default=15, help="Number of top candidates to select.")

    args = parser.parse_args()
    main(args)

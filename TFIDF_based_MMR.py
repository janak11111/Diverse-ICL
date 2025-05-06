import argparse
import numpy as np
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer


def similarity(x, z):
    cosine_sim = (x @ z.T)[0, 0] / (np.linalg.norm(x.toarray()) * np.linalg.norm(z.toarray()))
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
        for (idx, curr_candidate) in candidates:
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


def main(args):
    # Load the train dataset
    with open(args.train_file, 'r') as f:
        train_data = [json.loads(x) for x in f]

    # Load the test dataset
    with open(args.test_file, 'r') as f:
        test_data = json.load(f)

    print("Loaded datasets")

    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    train_question = [s['sentence'] for s in train_data]
    vectorizer.fit(train_question)

    # Re-rank candidates based on diversity
    reranked_data = []
    count = 1

    for sample in test_data:
        # Transform the test sample
        x = vectorizer.transform([sample['sentence']])

        # Transform candidates
        candidates = [(i, vectorizer.transform([candidate['sentence']])) for i, candidate in enumerate(sample['top_k'])]

        # Re-rank the candidates
        reranked_candidates, selected_ids = compute_diversity_based_top_k(x, candidates, args.alpha, args.top_k)

        # Extract the re-ranked candidates
        ranked_candidates = [sample['top_k'][i] for i in selected_ids]

        sample.pop('top_k')
        sample['alpha'] = args.alpha
        sample['reranked_top_k'] = ranked_candidates
        reranked_data.append(sample)

        if count % 50 == 0:
            print(f"Processed {count} samples")

        count += 1

    # Write the re-ranked data to output file
    with open(args.output_file, 'w') as f:
        json.dump(reranked_data, f, indent=2)

    print(f"Re-ranked data saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-rank candidates based on diversity.")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training dataset file.")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test dataset file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output re-ranked file.")
    parser.add_argument("--alpha", type=float, default=0.7, help="Diversity parameter alpha.")
    parser.add_argument("--top_k", type=int, default=15, help="Number of top candidates to select.")

    args = parser.parse_args()
    main(args)

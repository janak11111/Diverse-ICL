import argparse
import json
import numpy as np
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


def main(args):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Load the test dataset
    with open(args.test_file, 'r') as f:
        test_data = json.load(f)

    print("Loaded test dataset")

    reranked_data = []
    count = 1

    for sample in test_data:
        # Encode the query (sentence) for the sample
        x = embedder.encode([sample['sentence']], convert_to_tensor=False)

        # Encode the candidates (top-k similar samples from the train set)
        candidates = [(i, embedder.encode([candidate['sentence']], convert_to_tensor=False))
                      for i, candidate in enumerate(sample['top_k'])]

        reranked_candidates, selected_ids = compute_diversity_based_top_k(x, candidates, args.alpha, args.top_k)
        ranked_candidates = [sample['top_k'][i] for i in selected_ids]

        sample.pop('top_k')
        sample['alpha'] = args.alpha
        sample['reranked_top_k'] = ranked_candidates
        reranked_data.append(sample)

        count += 1
        if count % 100 == 0:
            print(f"Processed {count} samples")

    # Save the re-ranked data to a file
    with open(args.output_file, 'w') as f:
        json.dump(reranked_data, f, indent=2)

    print(f"Saved re-ranked data to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-rank candidates based on diversity.")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test dataset file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output re-ranked file.")
    parser.add_argument("--alpha", type=float, default=0.7, help="Diversity parameter alpha.")
    parser.add_argument("--top_k", type=int, default=15, help="Number of top candidates to select.")

    args = parser.parse_args()
    main(args)



# python SBERT_based_MMR.py --test_file <path_to_test_file> --output_file <path_to_output_file> --alpha 0.5 --top_k 15

import argparse
import torch
import json
from sentence_transformers import SentenceTransformer, util

def main(train_file, test_file, output_file, model_name, top_k):
    # Load the SentenceTransformer model
    embedder = SentenceTransformer(model_name)
    print(f"Loaded model: {model_name}")

    # Load the training data
    with open(train_file, 'r') as f:
        train_data = [json.loads(line) for line in f]
    print("Training data loaded")

    # Load the test data
    with open(test_file, 'r') as f:
        test_data = [json.loads(line) for line in f]
    print("Test data loaded")

    # Extract sentences
    corpus = [sample['sentence'] for sample in train_data]
    queries = [sample['sentence'] for sample in test_data]

    # Encode sentences
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
    query_embeddings = embedder.encode(queries, convert_to_tensor=True)

    # Perform semantic search
    hits = util.semantic_search(query_embeddings, corpus_embeddings, top_k=top_k)

    # Prepare the output data
    new_data = []
    for sample, hit in zip(test_data, hits):
        top_k_samples = [train_data[h['corpus_id']] for h in hit]

        # Include scores with the top-k samples
        top_k_with_scores = []
        for i, temp in enumerate(top_k_samples):
            temp_copy = temp.copy()
            temp_copy['score'] = hit[i]['score']
            top_k_with_scores.append(temp_copy)

        # Add top-k similar samples to the original sample
        new_sample = sample.copy()
        new_sample['top_k'] = top_k_with_scores
        new_data.append(new_sample)

    # Save the re-ranked data
    with open(output_file, 'w') as f:
        json.dump(new_data, f, indent=2)
    print(f"Output saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic search and re-ranking based on SBERT.")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training dataset file.")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test dataset file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output file.")
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2", help="Name of the SBERT model to use.")
    parser.add_argument("--top_k", type=int, default=45, help="Number of top candidates to select.")
    args = parser.parse_args()

    main(args.train_file, args.test_file, args.output_file, args.model_name, args.top_k)




# python script_name.py  --train_file /path/to/train.jsonl  --test_file /path/to/test.jsonl --output_file /path/to/output.json  --model_name all-MiniLM-L6-v2   --top_k 45
import argparse
import json
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

def main(train_file, test_file, output_file, top_k):
    # Load train and test data
    with open(train_file, 'r') as train_f:
        train_data = [json.loads(line) for line in train_f]
    
    with open(test_file, 'r') as test_f:
        test_data = [json.loads(line) for line in test_f]
    
    print("Train and test data loaded.")

    # Extract sentences from train and test sets
    train_sentences = [sample['sentence'] for sample in train_data]
    test_sentences = [sample['sentence'] for sample in test_data]

    print(f"Number of train samples: {len(train_sentences)}")
    print(f"Number of test samples: {len(test_sentences)}")

    # Vectorize sentences using TF-IDF
    vectorizer = TfidfVectorizer()
    train_tfidf_matrix = vectorizer.fit_transform(train_sentences)
    test_tfidf_matrix = vectorizer.transform(test_sentences)

    # Configure NearestNeighbors
    neighbors = NearestNeighbors(n_neighbors=top_k)
    neighbors.fit(train_tfidf_matrix)

    # Find top-k neighbors for each test sample
    distances, indices = neighbors.kneighbors(test_tfidf_matrix)

    # Store results in a new list
    new_data = []
    for i, test_sample in enumerate(test_data):
        tfidf_top_k = []
        for j in range(top_k):
            neighbor_index = indices[i][j]
            neighbor_distance = distances[i][j]
            
            tfidf_top_k.append({
                'idx': train_data[neighbor_index]['idx'],
                'sentence': train_data[neighbor_index]['sentence'],
                'label': train_data[neighbor_index]['label'],
                'distance': neighbor_distance
            })

        test_sample['top_k'] = tfidf_top_k
        new_data.append(test_sample)

    # Save the results to a JSON file
    with open(output_file, 'w') as output_f:
        json.dump(new_data, output_f, indent=2)

    print(f"Top-k neighbors for test samples saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find top-k TF-IDF neighbors for test samples.")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training dataset file.")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test dataset file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output file.")
    parser.add_argument("--top_k", type=int, default=45, help="Number of top neighbors to find.")
    args = parser.parse_args()

    main(args.train_file, args.test_file, args.output_file, args.top_k)


# python script_name.py  --train_file /path/to/train.jsonl --test_file /path/to/test.jsonl  --output_file /path/to/output.json  --top_k 45
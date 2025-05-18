import argparse
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer, util


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


def retrieve_top_k_tfidf(train_texts, query_texts, k):
    vectorizer = TfidfVectorizer()
    train_matrix = vectorizer.fit_transform(train_texts)
    query_matrix = vectorizer.transform(query_texts)

    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(train_matrix)
    distances, indices = neighbors.kneighbors(query_matrix)
    return distances, indices


def retrieve_top_k_sbert(train_embeddings, query_embeddings, k):
    hits = util.semantic_search(query_embeddings, train_embeddings, top_k=k)
    # hits: list of lists with dicts {corpus_id, score}
    return hits


def load_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]


def save_jsonl(data, path):
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item, indent=2) + '\n')


def main(args):
    # Load datasets
    train_data = load_jsonl(args.train_file)
    val_data = load_jsonl(args.val_file) if args.val_file else []
    test_data = load_jsonl(args.test_file)

    # Extract texts
    train_texts = [extract_text(s, args.dataset_name) for s in train_data]
    val_texts = [extract_text(s, args.dataset_name) for s in val_data]
    test_texts = [extract_text(s, args.dataset_name) for s in test_data]

    # Prepare SBERT embeddings if needed
    sbert_model = None
    train_embeds = query_embeds = None
    if args.feature_type.upper() == 'SBERT':
        sbert_model = SentenceTransformer(args.model_name)
        train_embeds = sbert_model.encode(train_texts, convert_to_tensor=True)
        if val_texts:
            query_embeds_val = sbert_model.encode(val_texts, convert_to_tensor=True)
        query_embeds_test = sbert_model.encode(test_texts, convert_to_tensor=True)

    # Retrieve and append top-k
    os.makedirs(args.output_path, exist_ok=True)

    def process_split(data, texts, split_name):
        new_split = []
        if args.feature_type.upper() == 'TFIDF':
            distances, indices = retrieve_top_k_tfidf(train_texts, texts, args.k)
            for i, sample in enumerate(data):
                top_k_list = []
                for rank, idx in enumerate(indices[i]):
                    top_k_list.append({
                        **train_data[idx],
                        'distance': float(distances[i][rank])
                    })
                sample['top_k'] = top_k_list
                new_split.append(sample)
        else:
            # SBERT
            embeds = query_embeds_val if split_name == 'val' and val_texts else query_embeds_test
            hits = retrieve_top_k_sbert(train_embeds, embeds, args.k)
            for sample, hit in zip(data, hits):
                top_k_list = []
                for h in hit:
                    item = dict(train_data[h['corpus_id']])
                    item['score'] = float(h['score'])
                    top_k_list.append(item)
                sample['top_k'] = top_k_list
                new_split.append(sample)
        save_jsonl(new_split, os.path.join(args.output_path, f"{split_name}_top_{args.k}.jsonl"))
        print(f"Saved {split_name} top-{args.k} to {args.output_path}")

    # Process validation and test splits
    if val_data:
        process_split(val_data, val_texts, 'val')
    process_split(test_data, test_texts, 'test')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retrieve top-k examples for validation and test sets across multiple datasets')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='One of: SST2, RTE, TREC, COLA')
    parser.add_argument('--train_file', type=str, required=True, help='Path to training JSONL file')
    parser.add_argument('--val_file', type=str, default=None, help='Path to validation JSONL file')
    parser.add_argument('--test_file', type=str, required=False, help='Path to test JSONL file')
    parser.add_argument('--k', type=int, required=True, help='Number of top-k candidates')
    parser.add_argument('--feature_type', type=str, required=True, choices=['TFIDF', 'SBERT'],
                        help='Retrieval feature: TFIDF or SBERT')
    parser.add_argument('--model_name', type=str, default='all-MiniLM-L6-v2',
                        help='SBERT model name for embeddings')
    parser.add_argument('--output_path', type=str, required=True, help='Directory to save output files')
    args = parser.parse_args()
    main(args)

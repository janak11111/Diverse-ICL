# 🚀 Exploring the Role of Diversity in Example Selection for In-Context Learning


## Methodology

![Diverse_ICL_Workflow](diverse_ICL_workflow.jpg)



## Installation  

1. Clone the repository:  
   ```bash
   git clone https://github.com/janak11111/Diverse-ICL
   cd Diverse-ICL
   ```

2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  

3. Create and activate the conda environment:
   ```
   conda create -n Diverse_ICL python=3.10 -y
   conda activate Diverse_ICL
   ```
   
---

## Top-k Candidate Retrieval with TFIDF

Run the script to retrieve top-K similar examples with sparse features (TFIDF):  

```bash
# Run in Python environment
python TFIDF_Top_K_retrieval.py  --train_file /path/to/train.jsonl --test_file /path/to/test.jsonl  --output_file /path/to/output.json  --top_k 45
```  

## Top-k Candidate Retrieval with SBERT

Run the script to retrieve top-K similar examples with dense features (SBERT):    

```bash
# Run in Python environment
python SBERT_Top_K_retrieval.py --train_file /path/to/train.jsonl  --test_file /path/to/test.jsonl --output_file /path/to/output.json  --model_name all-MiniLM-L6-v2   --top_k 45
```

---

## Rerank candidates with TFIDF-MMR

Run the script to Rerank top-K similar examples using MMR with sparse features (TFIDF):   

```bash
# Run in Python environment
python TFIDF_based_MMR.py --test_file <path_to_test_file> --output_file <path_to_output_file> --alpha 0.5 --top_k 15
```

---

## Rerank candidates with SBERT-MMR

Run the script to Rerank top-K similar examples using MMR with dense features (SBERT):   

```bash
# Run in Python environment
python SBERT_based_MMR.py --test_file <path_to_test_file> --output_file <path_to_output_file> --alpha 0.5 --top_k 15
```

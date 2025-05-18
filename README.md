## 🚀 Exploring the Role of Diversity in Example Selection for In-Context Learning


**Published:** SIGIR 2025 Conference on Research and Development in Information Retrieval, Padua, Italy

**Authors**: *Janak Kapuriya*, *Manit Kaushik*, *Debasis Ganguly*, *Sumit Bhatia*

---

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
python Top_K_Candidate_Retrieval.py  --dataset_name SST2 --train_file /path/to/train.jsonl --test_file /path/to/test.jsonl --top_k 45 --feature_type TFIDF --output_file /path/to/output.json  
```  

## Top-k Candidate Retrieval with SBERT

Run the script to retrieve top-K similar examples with dense features (SBERT):    

```bash
# Run in Python environment
python Top_K_Candidate_Retrieval.py --dataset_name COLA --train_file /path/to/train.jsonl  --test_file /path/to/test.jsonl --top_k 45  --feature_type SBERT --model_name all-MiniLM-L6-v2 --output_file /path/to/output.json     
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

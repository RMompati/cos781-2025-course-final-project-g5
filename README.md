# COS781-2025 Project: Replicating Association Rules and CF on Sparse Amazon E-Commerce Data

## Project Overview
This project replicates the methodology from the assigned paper ["Association Rules and Collaborative Filtering on Sparse Data of a Leading Online Retailer"](https://ieeexplore.ieee.org/document/8290000) using the Amazon Books dataset (sparse user-item ratings as proxy for views/purchases). We address sparsity in recommender systems by testing two models: Apriori-based association rules and memory-based collaborative filtering (user-based UCFA and item-based ICFA). Key insight: Restrict to popular items for better performance, as ~80% revenue comes from them.

**Research Question**: How can traditional recommenders (association rules & CF) handle extreme sparsity (~99%) in e-commerce data, and does focusing on popular items improve accuracy without losing utility?

**Why Interesting?**: E-commerce sparsity causes poor recommendations; solutions like popular-item filtering are simple, scalable, and revenue-focused—relevant for African retail (e.g., via Lacuna Fund extensions).

## Dataset
- **Source**: Amazon Books, Appliances, or All Beauty or Electronics subset from https://gist.github.com/entaroadun/1653794 or RUCAIBox/RecSysDatasets (GitHub).
- **Size**: ~8M ratings from 200K+ users on 60K+ items (5-core filtered for sparsity).
- **Attributes**: UserID, ItemID (ASIN), Rating (1-5; map >3 to "views," 5 to "purchases"), Timestamp.
- **Sparsity**: ~99.9% (calculated as 1 - (interactions / (users * items))).
- **Prep Note**: Download CSV; ensure ready by proposal deadline. Alternative: Fall back to Instacart if needed.

## Approach/Methods
1. **Exploratory Data Analysis (EDA)**:
   - Compute sparsity metric.
   - Visualize: Histograms (ratings dist), Degree dist (users/items per interaction), Heatmap (top users/items).
   - Identify popular items: Top 50 by total ratings (proxy for views/purchases).

2. **Data Preprocessing**:
   - Map ratings to compound index: Rating * p + (if rating==5) * q; p = total_ratings / total_purchases (~300), q=1.
   - Split: 80/20 train/test.
   - Handle sparsity: No imputation (preserve); filter to popular items for variant.

3. **Data Mining Methods**:
   - **Model 1: Association Rules (Apriori)**:
     - Use mlxtend.frequent_patterns.apriori on transaction matrix (user baskets as itemsets).
     - Thresholds: Min support=0.001, confidence=0.1.
     - Output: Frequent itemsets (e.g., 20-50 rules); expect few strong rules (80% conf >0.6) due to sparsity.
   - **Model 2: Collaborative Filtering (Memory-Based)**:
     - Build user-item matrix with Surprise library.
     - Similarities: Pearson (paper) + Cosine (our alt).
     - Neighbors: Best-k (k=20-50; better than threshold for sparsity).
     - Predictions: UCFA (user neighbors), ICFA (item neighbors).
     - Variants: Full data vs. popular items only (top 30-50).
   - **Implementation**: Python notebook (Jupyter); scikit-surprise for CF, mlxtend for Apriori.

4. **Evaluation**:
   - Metrics: MRPE (Mean Relative % Error: avg(|actual-pred|/actual *100)), MAE (Mean Abs Error).
   - Baselines: Paper's results (MRPE~50% full, <25% popular); random predictor.
   - Compare: ICFA vs. UCFA (expect ICFA better, faster); full vs. popular (expect 2x error reduction).
   - Success: Reproduce paper (high error on full, low on popular); our alt (cosine) improves MAE by 5-10%.

## Expected Outputs
- **By End of Semester**: 
  - Replicated results: Charts (MRPE/MAE vs. data size/top items), 50+ rules, predictions on test set.
  - Insights: CF outperforms rules on sparse data; popular filtering as practical fix.
  - Artifacts: 4-page KDD report, 7-slide deck, GitHub repo with notebook/docs/code.
- **Risks/Mitigations**: High compute—subsample to 100K records; if sparsity too low, add noise/mask.

## Timeline & Tasks
| Milestone | Date | Tasks | Owner |
|-----------|------|--------|-------|
| Proposal/Abstract | 25 Sep 2025 | Answer 5 questions; submit ClickUp | All |
| EDA & Preprocessing | 15 Oct 2025 | Notebook w/ visuals; compute sparsity | Lead |
| Implement Models | 31 Oct 2025 | Code Apriori + CF; run experiments | All |
| Report Draft | 7 Nov 2025 | KDD format; results tables/charts | Lead |
| Presentation & Files | 21 Nov 2025 | 7 slides; doc code/repo | All |

## Setup Instructions
1. Clone repo: `git clone <your-repo>`.
2. Install: `pip install pandas numpy scikit-surprise mlxtend matplotlib seaborn` (use course env).
3. Run: `jupyter notebook main.ipynb`.
4. Data: Place `amazon_books.csv` in `/data/`.

## Generative AI Statement
No AI used for core analysis/code; only for planning (e.g., README structure). All results from manual replication.

## References
- Paper: Wu et al. (2017). IEEE IEEM.
- Dataset: McAuley et al. (Amazon Reviews).
- Libs: Surprise docs, mlxtend Apriori guide.

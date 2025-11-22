# COS781-2025 Project: Replicating Association Rules and CF on Sparse Amazon E-Commerce Data

## Project Overview
This project replicates and extends the methodology from the assigned paper ["Association Rules and Collaborative Filtering on Sparse Data of a Leading Online Retailer"](https://ieeexplore.ieee.org/document/8290000). We utilise the Amazon Appliances dataset to address the challenge of extreme data sparsity in recommender systems. We evaluate three distinct modelling approaches: Apriori-based association rules, memory-based collaborative filtering (UCFA & ICFA), and model-based Matrix Factorization (SVD).

**Research Question**: How do traditional (Association Rules, Neighbor-based CF) and modern (SVD) recommenders handle extreme sparsity (~99%) in e-commerce data? Does restricting data to popular items consistently improve accuracy, or is there a limit to this approach?

**Key Insight**: While the paper suggests that restricting to popular items improves performance, our extension reveals that filtering improves accuracy only up to a specific point (e.g., the Top 90 items), after which information loss causes performance to degrade.

## Dataset
- **Source**: Amazon Appliances subset from HuggingFace (`McAuley-Lab/Amazon-Reviews-2023`) or standard repositories.
- **Size**: ~2.1M interactions reduced to a "core" set of ~122K interactions via iterative filtering.
- **Attributes**: UserID, ParentASIN, Rating (1-5), Timestamp.
- **Sparsity**: **99.99%** raw; reduced to **~99.95%** after preprocessing.

## Approach/Methods
1. **Exploratory Data Analysis (EDA)**:
   - Compute sparsity metrics before and after filtering.
   - Visualize: "Long tail" distributions (users/items), Rating frequency comparison.
   - Identify popular items vs. niche items to understand information loss.

2. **Data Preprocessing**:
   - **Iterative Filtering**: Recursively filter users (min 3 reviews) and items (min 5 reviews) until dataset stability is reached.
   - **Split**: Standard cross-validation (k-fold) for robust error estimation.
   - **Subsets**: Create variants of the dataset containing only the "Top N" items (N=100 down to 30) to test the popularity hypothesis.

3. **Data Mining Methods**:
   - **Model 1: Association Rules (Apriori)**:
     - Library: `mlxtend`.
     - Method: Transaction-based mining on user history.
     - thresholds: Low `min_support` (0.001) to catch rare pairs; `lift` metric to identify complementary bundles (e.g., machine + specific pods).
   - **Model 2: Memory-Based Collaborative Filtering**:
     - Library: `scikit-surprise`.
     - Algorithms: **UCFA** (User-Based) and **ICFA** (Item-Based).
     - Metrics: Cosine similarity.
   - **Model 3: Model-Based Collaborative Filtering (SVD)**:
     - Library: `scikit-surprise`.
     - Algorithm: Singular Value Decomposition (SVD) using Matrix Factorization.
     - Purpose: To capture latent factors (hidden features) between users and items, offering a robust alternative to neighbor-based methods in sparse environments.

4. **Evaluation**:
   - **Metrics**: MAE (Mean Absolute Error).
   - **Comparisons**:
     - **Algorithm Superiority**: Compare UCFA vs. ICFA vs. SVD on the core baseline.
     - **Sparsity Hypothesis**: Compare performance trend lines as the dataset is filtered from Top 100 down to Top 30 items.
   - **Success Criteria**: Reproduce paper's finding (ICFA > UCFA) and demonstrate SVD's superior handling of sparsity (SVD > ICFA).

## Expected Outputs
- **By End of Semester**:
  - **Replicated Results**: Confirmation that ICFA outperforms UCFA.
  - **New Findings**: SVD achieves the lowest baseline error (MAE ~0.709).
  - **Visualizations**: The curve showing the trade-off between data density and item diversity.
  - **Artifacts**: 4-page KDD report, 7-slide deck, GitHub repo with notebook/docs/code.

## Timeline & Tasks
| Milestone | Date | Tasks | Owner |
|-----------|------|--------|-------|
| Proposal/Abstract | 25 Sep 2025 | Answer 5 questions; submit ClickUp | All |
| EDA & Preprocessing | 15 Oct 2025 | Notebook w/ visuals; iterative filtering logic | Lead |
| Implement Models | 31 Oct 2025 | Code Apriori, UCFA, ICFA, **and SVD** | All |
| Report Draft | 7 Nov 2025 | KDD format; include "Goldilocks" analysis | Lead |
| Presentation & Files | 22 Nov 2025 | 7 slides; doc code/repo | All |

## Setup Instructions
1. Clone repo: `git clone <your-repo>`.
2. Install: `pip install pandas numpy scikit-surprise mlxtend matplotlib seaborn datasets` (use course env).
3. Run: `jupyter notebook cos781-A3-sub.ipynb`.
4. Data: Ensure internet access to load HuggingFace dataset or place JSONL in `/data/`.

## Generative AI Statement
No AI used for core analysis/code; only for planning (e.g., README structure). All results from manual replication.

## References
- Paper: Wu et al. (2017). IEEE IEEM.
- Dataset: McAuley et al. (Amazon Reviews 2023).
- Libs: Surprise SVD docs, mlxtend Apriori guide.

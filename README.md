# LexiLaw_NLP_PIPELINE
Official executable pipeline for diachronic semantic analysis of AI governance instruments (EU, US, Brazil, 2021–2025), accompanying the manuscript submitted to Machine Learning and Knowledge Extraction (MAKE, MDPI).
> **A Computational Pipeline for Diachronic Semantic Analysis of AI Governance Instruments (EU, US, Brazil, 2021-2025)**

Official code repository and supplementary data associated with the manuscript submitted to **Machine Learning and Knowledge Extraction (MAKE)**, MDPI. This pipeline fully reproduces all tables, figures, co-occurrence networks, and similarity matrices reported in the study.

---

## 📂 Repository Structure

The pipeline is structured modularly within the primary Jupyter Notebook (`LexiLaw_NLP_PIPELINE_clean.ipynb`):

| Stage | Purpose | Key Outputs |
| :---: | :--- | :--- |
| **0** | Corpus manifest and raw whitespace token counts per document | `prominencia_lexical_dirigida.csv` |
| **1** | Shared preprocessing definitions (CONCEPT_MAP, stoplists, lemmatization) | — |
| **2** | Topic-count selection sweep ($K = 2..10$): Perplexity vs. Coherence | `lda_metrics_combined.png` |
| **3** | Bootstrap stability validation of the $K = 5$ solution (30 iterations) | `resultado_estabilidade_lda.txt` |
| **4** | Final LDA model ($K = 5$): topic table, wordclouds, and Sankey diagram | `wordclouds_lda.png`, `sankey_diagram.html` |
| **5** | Co-occurrence networks, dynamic threshold ($T_{dynamic}$), and Louvain communities | `network_complexity_metrics.csv`, `radar_chart_complexity.png` |
| **6** | Triangulated similarity metrics (Jaccard, Dice, TF-IDF Cosine, Sentence-BERT) | `matriz_*_6x6.csv`, `heatmap_*.png` |
| **7** | Package all output artifacts into a single archive | `outputs.zip` |

---

## ⚙️ Requirements & Execution

The pipeline is designed to run end-to-end in **Google Colab** or local Python 3.10+ environments.

### 1. Dependencies
The notebook automatically installs all required third-party packages (`gensim`, `sentence-transformers`, `networkx`, `scikit-learn`, `plotly`, etc.) in its initial setup cell.

### 2. Execution Steps
1. Ensure the pre-cleaned corpus files (`*_LIMPOS.txt`) are placed in the designated input directory (`/content/corpus`).
2. Run the notebook cells **strictly in order** within a single uninterrupted session to preserve variable states and ensure all artifacts are correctly written to the `outputs/` folder.

---

## 📊 Analyzed Corpus
The harmonized corpus comprises 6 key regulatory frameworks:
* **European Union:** EU_2021 (Initial Draft) and EU_2024 (Enacted AI Act).
* **Brazil:** BR_2023 (Initial PL 2338/2023) and BR_2025 (Chamber Substitute Draft).
* **United States:** US_2023 (Executive Order 14110) and US_2025 (Consolidated Macro-Corpus featuring EO 14148, EO 14179, and OSTP plan).

---

## 📜 License
This material is distributed under the MIT License for scientific transparency and academic reproducibility.

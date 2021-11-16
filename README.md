# Artificial Text Detection via Examining the Topology of Attention Maps (EMNLP 2021)

This repository contains code base for the [paper](https://aclanthology.org/2021.emnlp-main.50.pdf) which introduces a novel method for artificial text detection (ATD) based on Topological Data Analysis (TDA) which has been understudied in the field of NLP.

Despite the prominent performance of existing ATD method, they still lack interpretability and robustness towards unseen text generation models. To this end, we propose three types of interpretable TDA features for this task, and empirically show that the features derived from the BERT model outperform count- and neural-based baselines up to 10% on three common datasets, and tend to be the most robust towards unseen GPT-style generation models as opposed to existing methods. The probing analysis of the features reveals their sensitivity to the surface and syntactic properties. The results demonstrate that TDA is a promising line with respect to NLP tasks, specifically the ones that incorporate surface and structural information.

We briefly list the features below, and refer the reader to the paper for more details:
* Topological features (Betti numbers, the number of edges, the number of strong connected components, etc.);
* Features derived from barcodes (the sum of lengths of bars, the variance of lengths of bars, the time of birth/death of the longest bar, etc.);
* Features based on distance to patterns (attention to previous token, attention to punctuation marks, attention to CLS-token, etc.).

# Dependencies
The code base requires:
* python 3.8.3
* matplotlib 3.3.1
* networkx 2.5.1
* numpy 1.19.1
* pandas 1.1.1
* ripserplusplus 1.1.2
* scipy 1.5.2
* sklearn 0.23.2
* tqdm 4.46.0
* transformers 4.3.0


# Usage
* For calculating topological invariants by thresholds (Section 4.1), use `features_calculation_by_thresholds.ipynb`.
* For calculating barcodes (Section 4.2) and template (Section 4.3) features, use `features_calculation_barcodes_and_templates.ipynb`.
* For making predictions with the logistic regression upon calculated features, use `features_prediction_gpt_web.ipynb`.
* The head-wise probing analysis by [Jo and Myaeng (2020)](https://aclanthology.org/2020.acl-main.311/) is conducted using [an open-source implementation](https://github.com/heartcored98/transformer_anatomy). 


# Cite us
Our paper is accepted to the EMNLP 2021 main conference and is selected for oral session. 

```
@inproceedings{kushnareva-etal-2021-artificial,
    title = "Artificial Text Detection via Examining the Topology of Attention Maps",
    author = "Kushnareva, Laida  and
      Cherniavskii, Daniil  and
      Mikhailov, Vladislav  and
      Artemova, Ekaterina  and
      Barannikov, Serguei  and
      Bernstein, Alexander  and
      Piontkovskaya, Irina  and
      Piontkovski, Dmitri  and
      Burnaev, Evgeny",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.50",
    pages = "635--649",
    abstract = "The impressive capabilities of recent generative models to create texts that are challenging to distinguish from the human-written ones can be misused for generating fake news, product reviews, and even abusive content. Despite the prominent performance of existing methods for artificial text detection, they still lack interpretability and robustness towards unseen models. To this end, we propose three novel types of interpretable topological features for this task based on Topological Data Analysis (TDA) which is currently understudied in the field of NLP. We empirically show that the features derived from the BERT model outperform count- and neural-based baselines up to 10{\%} on three common datasets, and tend to be the most robust towards unseen GPT-style generation models as opposed to existing methods. The probing analysis of the features reveals their sensitivity to the surface and syntactic properties. The results demonstrate that TDA is a promising line with respect to NLP tasks, specifically the ones that incorporate surface and structural information.",
}
```

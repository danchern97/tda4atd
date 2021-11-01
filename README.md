This repo provides the source code of our paper: 
[Artificial Text Detection via Examining the Topology of Attention Maps](https://arxiv.org/abs/2109.04825)

@article{EMNLP2021,
  title={Artificial Text Detection via Examining the Topology of Attention Maps},
  author={Kushnareva, Laida and Cherniavskii, Daniil and Mikhailov, Vladislav and Artemova, Ekaterina  and Barannikov, Serguei and  Bernstein, Alexander and Piontkovskaya, Irina and  Piontkovski, Dmitri and Burnaev, Evgeny},
  journal={Proc. of EMNLP conf., oral talk},
  year={2021},
archivePrefix = "arXiv",
   eprint = {2109.04825}
}

# Overview

The repository contains tools for topological analysis of attention graphs of the transformer models.

# Dependencies

Scripts from this repository require the following to run:

* Python 3.8.3
* Matplotlib 3.3.1
* Networkx 2.5.1
* Numpy 1.19.1
* Pandas 1.1.1
* Ripserplusplus 1.1.2
* Scipy 1.5.2
* Sklearn 0.23.2
* Tqdm 4.46.0
* Transformers 4.3.0

# Using

* For calculating topological invariants by thresholds (p.4.1 in paper), use `features_calculation_by_thresholds.ipynb`.
* For calculating barcodes (p.4.2) and template (p.4.3) features, use `features_calculation_barcodes_and_templates.ipynb`.
* For making predictions with the logistic regression upon calculated features, as we did in our paper, use `features_prediction_gpt_web.ipynb`.

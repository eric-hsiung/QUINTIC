# Learning Quantitative Automata Modulo Theories

The project website can be found at [https://eric-hsiung.github.io/quintic](https://eric-hsiung.github.io/quintic).

This is the experimental code repository for the paper **[Learning Quantitative Automata Modulo Theories](https://eric-hsiung.github.io/quintic/static/pdfs/quintic_full_ijcai_2026.pdf)** (IJCAI 2026).

It contains an implementation of the QUINTIC algorithm, code for running experiments, and plotting scripts.

To cite this work:
```
@inproceedings{hsiung2026quintic,
  title     = {Learning Quantitative Automata Modulo Theories},
  author    = {Hsiung, Eric and Tsoi, Nathan and Chaudhuri, Swarat and Biswas, Joydeep},
  booktitle = {Proceedings of the Thirty-Fifth International Joint Conference on
               Artificial Intelligence, {IJCAI-26}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Diego Calvanese},
  pages     = {2247--2255},
  year      = {2026},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2026/250},
  url       = {https://doi.org/10.24963/ijcai.2026/250},
}
```

Requirements:

python==3.8.13
z3-solver==4.13.0.0

Requirements for Plot Generation:

scipy==1.8.1
scikit-image==0.21.0
scikit-learn==1.1.1
matplotlib==3.5.2
numpy==1.23.1
pandas==1.5.0

# Mini-Kinetics-200
#### Mini-Kinetics-200 training/validation splits, used in paper [*Rethinking Spatiotemporal Feature Learning For Video Understanding*](https://arxiv.org/abs/1712.04851) by Saining Xie, Chen Sun, Jonathan Huang, Zhuowen Tu and Kevin Murphy.

This **subset** of [Kinetics dataset](https://deepmind.com/research/open-source/open-source-datasets/kinetics/) consists of the
200 categories with most training examples; for each category, we randomly sample 400 examples from the training
set, and 25 examples from the validation set, resulting in 80K training examples and 5K validation examples in total.

Each line in the training/validation split file provided here is a **youtube_id**. For detailed information and experimental results, please refer to our ArXiv paper.

# Guided-NMT

---

This repository contains code used during my Master's thesis, and refers to an implementations heavily based on the work ["Guiding Neural Machine Translation with Retrieved Translation Pieces"](https://arxiv.org/abs/1804.02559) by J. Zhang et al.

The `retrieve\_faiss.py` leverages Faiss to find similar sentences, and then creates translation pieces. The process is described in Chapter 5 of the master thesis' document. Then, it is possible to use leverage those translation pieces in any NMT framework. In this work it is used OpenNMT-py.

#### Prerequisites

- Faiss (https://github.com/facebookresearch/faiss)
- OpenNMT-py fork (`dev_extra` branch at https://github.com/Unbabel/OpenNMT-py/tree/dev_extra)

#### Necessary Changes/Files

- Manually edit the paths in lines 275 to 302.
- Add the stopwords file if necessary.
- Download the correct fastText pre-trained embeddings .bin [Source](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md)
- For the source language: extra, dev and test with/without bpe.
- For the source language: concatenation of all available without bpe.
- For the target language: extra with bpe.
- Alignments (as in [fast\_align](https://github.com/clab/fast_align)) between the source and target extra files with bpe.

#### How to run

> python retrieve\_faiss.py -k 5 -n\_max 4 -simi\_th 0.2 -dev

> python retrieve\_faiss.py -k 5 -n\_max 4 -simi\_th 0.2

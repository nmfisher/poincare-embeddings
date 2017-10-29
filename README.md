# poincare-embedding

This is a fork of https://github.com/TatsuyaShirakawa/poincare-embedding, which was designed to train word embeddings from WordNet pairs.

This implementation allows training word embeddings on arbitrary text files using either the [continuous-bag-of-words or skipgram models](https://cs224d.stanford.edu/lecture_notes/notes1.pdf). 

I eventually plan to combine this with [Facebook's fastText](https://github.com/facebookresearch/fastText) (e.g. char-ngrams), but so far there have only been a few minor additions. 

Gitter link: https://gitter.im/poincare-embeddings/Lobby

## Build

* gcc-4.6.3 / CMake 3 or newer.

```shell
cd poincare-embedding
mkdir work & cd work
cmake ..
make
```

## Run

```
poincare-embedding/work/poincare-embedding [input_file] [output_file] -arg1 -arg2.....
```

The executable accepts any text file consisting of words separated by a space (i.e. sentences). For each word in the text file, the Poincare model is then trained on a sliding window to the left and right of that word. The model will train across sentence boundaries (e.g. full-stops) but not across newlines.

## Arguments

```data_file                 : string    input txt file
result_embeddng_file      : string    result file into which resulting embeddings are written
-s, --seed                : int >= 0   random seed
-t, --num_threads         : int > 0   number of threads
-m, --model               : string    model name ("cbow" (default) or "skipgram")
-n, --neg_size            : int > 0   negativ sample size
-e, --max_epoch           : int > 0   maximum training epochs
-d, --dim                 : int > 0   dimension of embeddings
-u, --uniform_range       : float > 0 embedding uniform initializer range
-l, --learning_rate_init  : float > 0 initial learning rate
-L, --learning_rate_final : float > 0 final learning rate
-v, --verbose             : int 0,1 verbosity
-w, --window_size        : int >= 0 window size for CBOW```


## References

[Poincaré Embeddings for Learning Hierarchical Representations](https://arxiv.org/abs/1705.08039)
[Neural Embeddings of Graphs in Hyperbolic Space](https://arxiv.org/abs/1705.10359)

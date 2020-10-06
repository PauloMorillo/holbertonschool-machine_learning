# Transformer Applications

This directory has the exercises to apply Transformers
## Before to code

Topics to learn:

How to use Transformers for Machine Translation
How to write a custom train/test loop in Keras
How to use Tensorflow Datasets

## Requirements or used modules
Python(3.5), Numpy(1.16), tensorflow(1.15) and pycodestyle modules


```bash
pip install --user numpy==1.16
pip install --user tensorflow==1.15
pip install --user pycodestyle==2.5
pip install --user tensorflow-datasets

```

Testing datasets library

```bash
#!/usr/bin/env python3
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

tf.compat.v1.enable_eager_execution()
pt2en_train = tfds.load('ted_hrlr_translate/pt_to_en', split='train', as_supervised=True)
for pt, en in pt2en_train.take(1):
  print(pt.numpy().decode('utf-8'))
  print(en.numpy().decode('utf-8'))
$ ./load_dataset.py
e quando melhoramos a procura , tiramos a única vantagem da impressão , que é a serendipidade .
and when you improve searchability , you actually take away the one advantage of print , which is serendipity .
```
##Running main files
```bash
./0-main.py

```


## Author
[Paulo Morillo](https://www.linkedin.com/in/paulo-morillo-mu%C3%B1oz-191745143/)

## License
[MIT](https://choosealicense.com/licenses/mit/)
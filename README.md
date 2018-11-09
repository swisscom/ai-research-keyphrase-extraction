This is the implementation of the following paper: https://arxiv.org/abs/1801.04470

# Installation

## Local Installation

1. Download full Stanford Tagger version 3.8.0
https://nlp.stanford.edu/software/tagger.shtml

2. Install sent2vec from 
https://github.com/epfml/sent2vec
    * Clone/Download the directory
    * go to sent2vec directory
    * git checkout f827d014a473aa22b2fef28d9e29211d50808d48
    * make
    * pip install cython
    * inside the src folder 
        * ``python setup.py build_ext``
        * ``pip install . ``
        * (In OSX) If the setup.py throws an **error** (ignore warnings), open setup.py and add '-stdlib=libc++' in the compile_opts list.        
    * Download a pre-trained model (see readme of Sent2Vec repo) , for example wiki_bigrams.bin
     
3. Install requirements
    
    After cloning this repository go to the root directory and
    ``pip install -r requirements.txt``

4. Download NLTK data
```
import nltk 
nltk.download('punkt')
```

5. Set the paths in config.ini.template
    
    * For [STANFORDTAGGER] :
        * set jar_path to your_stanford_path/stanford-postagger.jar
        * set model_directory_path to your_stanford_path/models
    * For [SENT2VEC]:
        * set your model_path to the pretrained model
        your_path_to_model/wiki_bigrams.bin (if you choosed wiki_bigrams.bin)
    * rename config.ini.template to config.ini

## Docker

Probably the easiest way to get started is by using the provided Docker image.
From the project's root directory, the image can be built like so:
```
$ docker build . -t keyphrase-extraction
```
This can take a few minutes to finish.
Also, keep in mind that pre-trained sent2vec models will not be downloaded since each model is several GBs in size and don't forget to allocate enough memory to your docker container (models are loaded in RAM).

To extract key phrases from raw text, simply run
```
$ docker run -v {path to wiki_bigrams.bin}:/sent2vec/pretrained_model.bin keyphrase-extraction launch.py -raw_text 'the quick brown fox jumps over the lazy dog' -N 2
```

To launch the model in an interactive mode, in order to use your own code, run
```
$ docker run -v {path to wiki_bigrams.bin}:/sent2vec/pretrained_model.bin -it keyphrase-extraction
>>> import launch
```
In both cases, you have to specify the path to your sent2vec model using the `-v` argument.
If, for example, you should choose not to use the *wiki_bigrams.bin* model, adjust your path accordingly (and of course, remember to remove the curly brackets).

# Usage

You can launch the script to extract keyphrases from one document either by giving the raw text directly using
-raw_text

python launch.py -raw_text 'the quick brown fox jumps over the lazy dog' -N 2

or you can specify a the path of a text file using -text_file argument :

python launch.py -text_file 'path/to/your/textfile' -N 10

If you have several documents in which you want to extract keyphrases the previous approach will be very slow because
it will load the embedding model and the part of speech tagger each time. If you have several documents it is better to
load the embedding model and the part of speech tagger once :

```
import launch

embedding_distributor = launch.load_local_embedding_distributor('en')
pos_tagger = launch.load_local_pos_tagger('en')

kp1 = launch.extract_keyphrases(embedding_distributor, pos_tagger, raw_text, 10, 'en')  #extract 10 keyphrases
kp2 = launch.extract_keyphrases(embedding_distributor, pos_tagger, raw_text2, 10, 'en')
...
```

This return for each text a tuple containing three lists:
1) The top N candidates (string) i.e keyphrases
2) For each keyphrase the associated relevance score
3) For each keyphrase a list of alias (other candidates very similar to the one selected
as keyphrase)

# Method

This is the implementation of the following paper:
https://arxiv.org/abs/1801.04470

![embedrank](embedrank.gif)

By using sentence embeddings , EmbedRank embeds both the document and candidate phrases into the same embedding space.

N candidates are selected as keyphrases by using Maximal Margin Relevance using the cosine similarity between the candidates and the
document in order to model the informativness and the cosine
similarity between the candidates is used to model the diversity.

An hyperparameter, beta (default=0.55), controls the importance given to 
informativness and diversity when extracting keyphrases.
(beta = 1 only informativness , beta = 0 only diversity)
You can change the beta hyperparameter value when calling extract_keyphrases:

```
kp1 = launch.extract_keyphrases(embedding_distributor, pos_tagger, raw_text, 10, 'en', beta=0.8)  #extract 10 keyphrases with beta=0.8

```


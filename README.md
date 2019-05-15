This is the implementation of the following paper: https://arxiv.org/abs/1801.04470

# Installation

## Local Installation

1. Download full Stanford CoreNLP Tagger version 3.8.0
http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip

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

5. Launch Stanford Core NLP tagger
    * Open a new terminal
    * Go to the stanford-core-nlp-full directory
    * Run the server `java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos -status_port 9000 -port 9000 -timeout 15000 & `


6. Set the paths in config.ini.template
    * You can leave [STANFORDTAGGER] parameters empty
    * For [STANFORDCORENLPTAGGER] :
        * set host to localhost
        * set port to 9000
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

To launch the model in an interactive mode, in order to use your own code, run
```
$ docker run -v {path to wiki_bigrams.bin}:/sent2vec/pretrained_model.bin -it keyphrase-extraction
# Run the corenlp server
/app # cd /stanford-corenlp
/stanford-corenlp # nohup java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos -status_port 9000 -port 9000 -timeout 15000 &
# Press enter to get stdin back
/stanford-corenlp # cd /app
/app # python
>>> import launch
```
You have to specify the path to your sent2vec model using the `-v` argument.
If, for example, you should choose not to use the *wiki_bigrams.bin* model, adjust your path accordingly (and of course, remember to remove the curly brackets).

# Usage

Once the CoreNLP server is running

```
import launch

embedding_distributor = launch.load_local_embedding_distributor()
pos_tagger = launch.load_local_corenlp_pos_tagger()

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

If you want to replicate the results of the paper you have to set beta to 1 or 0.5 and turn off the alias feature by specifiying alias_threshold=1 to extract_keyphrases method.


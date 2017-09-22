## Installation

1) Download full Stanford Tagger version 3.8.0
https://nlp.stanford.edu/software/tagger.shtml

2) Install requirements
pip -r requirements.txt

3) Download NLTK data
```
import nltk 
nltk.download('punkt')
```

4) Set the paths in config.ini.template and rename to config.ini [TODO : Add more detail]

# Usage

You can launch the script to extract keyphrases from one document either by giving the raw text directly using
-raw_text

python launch.py -raw_text 'this is the text i want to extract keyphrases from' -N 10

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
kp3 = launch.extract_keyphrases(embedding_distributor, pos_tagger, raw_text3, 10, 'en')

```




# Assistant Intent Classification

## Setting up

Install the requirements via
`pip3 install -r requirements.txt` and run the jupyter notebook

Download Glove for word embeddings
```sh
mkdir GloVe
curl -Lo GloVe/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip GloVe/glove.840B.300d.zip -d GloVe/
```

and 
Download infersent for sentence embeddings
```sh
mkdir encoder
curl -Lo encoder/infersent1.pkl https://dl.fbaipublicfiles.com/infersent/infersent1.pkl
```

Run the classifier notebook classifier.ipynb to load and view the data.

Current Accuracy:

Download deepspeech-0.7.3-models from deepspeech directory. The files that should be included are 
├── deepspeech-0.7.3-models.pbmm
├── lm.binary
├── output_graph.pb
├── output_graph.pbmm
├── output_graph.tflite
└── trie

Download ffc_keras_model from the notebook 
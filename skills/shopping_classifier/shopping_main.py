import os
import pickle
import sys
from fuzzywuzzy import process
import numpy as np
import pandas as pd
import spacy
import tensorflow as tf
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    Concatenate,
    Flatten,
    Conv2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from models import InferSent

np.random.seed(3252)
PATH = (os.path.dirname(os.path.abspath(__file__))) + "/"


def train_model():
    df = pd.read_json(PATH + "data_processed_42_intents.json", orient="records")
    intent_counts = (
        df.groupby(["intents"]).size().to_frame("count")
    )  # df.groupby(['intents']).count().to_csv('intent_counts.csv')
    intent_counts = intent_counts.sort_values("count", ascending=False)

    total_count = intent_counts["count"].sum()
    intent_counts["prop"] = intent_counts["count"] / total_count * 100
    intent_counts.to_csv(PATH + "intent_counts.csv")

    df_train, df_test = train_test_split(df, test_size=0.1)
    df_train, df_val = train_test_split(df_train, test_size=0.1)

    y_train = np.load(PATH + "y_train.npy")
    y_val = np.load(PATH + "y_val.npy")
    y_test = np.load(PATH + "y_test.npy")

    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y_train_encoded = label_encoder.transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    y_test_encoded = label_encoder.transform(y_test)

    number_items = max(y_train_encoded) - min(y_train_encoded) + 1

    def get_one_hot(arr, number_items):
        return tf.one_hot(arr, number_items)

    def get_labels_decoded(arr):
        return label_encoder.inverse_transform(arr)

    y_train_one_hot = get_one_hot(y_train_encoded, number_items)
    y_val_one_hot = get_one_hot(y_val_encoded, number_items)
    y_test_one_hot = get_one_hot(y_test_encoded, number_items)

    pickle.dump(label_encoder, open(PATH + "label_encoding.pkl", "wb"))

    # Load model
    def generate_Infersent_model():
        # Load model
        model_version = 1
        MODEL_PATH = "encoder/infersent%s.pkl" % model_version
        params_model = {
            "bsize": 64,
            "word_emb_dim": 300,
            "enc_lstm_dim": 2048,
            "pool_type": "max",
            "dpout_model": 0.0,
            "version": model_version,
        }
        model = InferSent(params_model)
        model.load_state_dict(torch.load(MODEL_PATH))

        W2V_PATH = PATH + "GloVe/glove.840B.300d.txt"
        model.set_w2v_path(W2V_PATH)

        model.build_vocab_k_words(K=100000)

        return model

    def generate_basic_fully_connected():
        input_1 = Input((4096,), dtype=tf.float32)
        x = Dense(2000, activation="relu", kernel_regularizer=regularizers.l2(0.0001))(
            input_1
        )

        x = Dense(500, activation="relu", kernel_regularizer=regularizers.l2(0.0001))(x)
        x = Dropout(rate=0.5)(x)

        x = Dense(450, activation="relu", kernel_regularizer=regularizers.l2(0.0001))(x)
        x = Dropout(rate=0.5)(x)

        x = Dense(250, activation="relu", kernel_regularizer=regularizers.l2(0.0001))(x)
        out = Dense(2, activation="softmax")(x)
        dual_model = Model(inputs=input_1, outputs=out)

        adamOpti = Adam()
        dual_model.compile(
            optimizer=adamOpti, loss="categorical_crossentropy", metrics=["acc"]
        )

        dual_model.summary()
        return dual_model

    def get_cnn_model(num_output=140):
        # https://github.com/ajinkyaT/CNN_Intent_Classification/blob/master/Intent_Classification_Keras_Glove.ipynb
        MAX_SEQUENCE_LENGTH = 10  # Maximum number of words in a sentence
        MAX_NB_WORDS = 100000  # Vocabulary size
        EMBEDDING_DIM = 1000  # Dimensions of Glove word vectors
        VALIDATION_SPLIT = 0.10

        filter_sizes = [2, 3, 5]
        num_filters = 512
        drop = 0.5

        inputs = Input(shape=(4096,), dtype="int32")
        # embedding = Embedding(input_dim=len(word_index) + 1, output_dim=EMBEDDING_DIM, weights=[embedding_matrix],
        #                       input_length=MAX_SEQUENCE_LENGTH, trainable=False)(inputs)
        # reshape = Reshape((MAX_SEQUENCE_LENGTH,EMBEDDING_DIM,1))(inputs)

        conv_0 = Conv2D(
            num_filters,
            kernel_size=(filter_sizes[0], 4096),
            padding="valid",
            kernel_initializer="normal",
            activation="relu",
        )(inputs)
        conv_1 = Conv2D(
            num_filters,
            kernel_size=(filter_sizes[1], 4096),
            padding="valid",
            kernel_initializer="normal",
            activation="relu",
        )(inputs)
        conv_2 = Conv2D(
            num_filters,
            kernel_size=(filter_sizes[2], 4096),
            padding="valid",
            kernel_initializer="normal",
            activation="relu",
        )(inputs)

        maxpool_0 = MaxPool2D(
            pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[0] + 1, 1),
            strides=(1, 1),
            padding="valid",
        )(conv_0)
        maxpool_1 = MaxPool2D(
            pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[1] + 1, 1),
            strides=(1, 1),
            padding="valid",
        )(conv_1)
        maxpool_2 = MaxPool2D(
            pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[2] + 1, 1),
            strides=(1, 1),
            padding="valid",
        )(conv_2)

        concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
        flatten = Flatten()(concatenated_tensor)
        dropout = Dropout(drop)(flatten)
        preds = Dense(num_output, activation="softmax")(dropout)
        adamOpti = Adam()
        preds.compile(
            optimizer=adamOpti, loss="categorical_crossentropy", metrics=["acc"]
        )

        preds.summary()
        return preds

    model = generate_Infersent_model()
    torch.save(model, "infraset_model.torch")

    # print("loading X Train vec")
    X_train_vec = get_doc2vec(X_train, model)
    # X_train_vec = np.load(PATH + "X_train_vec.npy")

    # X_train_vec
    # print("loading X Val vec")
    X_val_vec = get_doc2vec(X_val, model)
    # X_val_vec = np.load(PATH + "X_val_vec.npy")

    # print("loading X Test vec")
    # X_test_vec = np.load(PATH + "X_test_vec.npy")

    X_test_vec = get_doc2vec(X_test, model)

    # Save the data using np.savez('X_train_val_test.npz', X_train_vec=X_train_vec, X_val_vec=X_val_vec, X_test_vec=X_test_vec)
    # ffcc = generate_basic_fully_connected()
    # ffcc.fit(X_train_vec, y_train_one_hot,
    #          epochs=100, batch_size=64,
    #          validation_data=(X_val_vec, y_val_one_hot)
    #          , callbacks = [EarlyStopping(monitor='val_loss', patience=12)])
    def calculate_accuracy(
        model,
        X_vec,
        y_labels,
        nueral_net=True,
        with_oos=True,
        ignore_correct=True,
        cutoff=0.7,
    ):
        predictions = model.predict(X_vec)
        predicted_labels = get_labels_decoded(
            np.argmax(predictions, axis=1)
        )  # each integer is [0, 0, 0,1]
        posterior_prob = np.max(predictions, axis=1)
        # posterior_prob =
        # TODO posterior prob not defined from some functions
        correct = 0
        #     classification = {key:defaultdict(int) for key in y_labels}
        classification = []
        num_oos = 0
        for pred, y_true, prob in zip(predicted_labels, y_labels, posterior_prob):
            # if prob < cutoff:
            #     pred = 'oos'
            if y_true == "oos" and not with_oos:
                num_oos += 1
                continue
            if pred == y_true:
                correct += 1
                if ignore_correct:
                    continue
            classification.append({"y_true": y_true, "pred": pred, "prob": prob})

        classification = pd.DataFrame(classification)
        classification = (
            classification.groupby(["y_true", "pred"])
            .agg({"prob": [("prob", "mean")], "pred": [("count", "count")]})
            .reset_index()
        )
        classification.columns = ["y_true", "pred", "prob", "count"]
        # print(classification)

        accuracy = correct / (len(predictions) - num_oos)

        return accuracy, classification

    def analyze_missclassifications(classifications):
        dic = {}
        for index, row in classifications.iterrows():
            y_true = row["y_true"]
            pred = row["pred"]
            posterior_prob = row["prob"]
            count = row["count"]
            if y_true not in dic:
                dic[y_true] = {"count": 0}
            dic[y_true]["count"] += count

            if y_true == pred:
                dic[y_true]["true count"] = count
                dic[y_true]["confidence"] = posterior_prob
        l = []
        for key, val in dic.items():
            if "true count" not in val:
                dic[key]["true count"] = 0
                dic[key]["confidence"] = 0.0
            dic[key]["TP"] = dic[key]["true count"] / dic[key]["count"]
            l.append(
                {
                    "y_true": key,
                    "confidence": dic[key]["confidence"],
                    "count": dic[key]["count"],
                    "TP": dic[key]["TP"],
                }
            )
        res = pd.DataFrame(l).sort_values(
            ["confidence", "TP", "count"], ascending=False
        )
        return res

    ffcc = tf.keras.models.load_model(PATH + "ffcc_keras_model")
    train_accuracy, train_classifications = calculate_accuracy(
        ffcc, X_train_vec, y_train, ignore_correct=False
    )
    # print("Train Accuracy With OOS", train_accuracy)


def get_doc2vec(text, model, verbose=False):
    emb = model.encode(text, verbose=verbose)
    return emb


def find_shopping_item(text):
    # TODO rewrite code to just find shopping item
    # TODO code is also difficult to read
    model = torch.load(PATH + "infraset_model.torch")
    model.eval()
    label_encoder = pickle.load(open(PATH + "label_encoding.pkl", "rb"))

    def get_labels_decoded(arr):
        return label_encoder.inverse_transform(arr)
    def cosine(u, v):
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    en_nlp = spacy.load("en_core_web_sm")
    doc = en_nlp(text)
    sentence = next(doc.sents)
    predicted_item = set()
    productlisting = list()
    with open(PATH + "food.txt") as my_file:
        for line in my_file:
            productlisting.append(str(line).strip())
    print(sentence)
    for word in sentence:
        print(word)
        if "obj" in word.dep_ or "conj" in word.dep_:
            if cosine( model.encode(["shopping market or grocery store"])[0],
                model.encode([str(word)])[0],
            ) < cosine(
                model.encode(["food or item"])[0], model.encode([str(word)])[0]
            ):
                # print(word)
                temp = word
                if prev[1] == "amod":
                    # This is used for compound words
                    word = prev[0]+ " " + word
                # checks cosine similarity
                cosSim = cosine(
                    model.encode(["grocery item"])[0],
                    model.encode([str(word)])[0],
                )
                print(cosSim) 
                curwordvec = en_nlp(str(temp))
                if cosSim > 0.35:
                    predicted_item.add(word)
                else:
                    for product in productlisting:
                        if en_nlp(product).similarity(curwordvec) > 0.8:
                            predicted_item.add(word)
                            break
        prev = (word, word.dep_)
    return list(predicted_item)


if __name__ == "__main__":
    # print(
    #     "Please enter the phrase/sentence to test the passive listening add to shopping list feature"
    # )
    # print("Press control+C to exit the program")
    # print("Check the txt folder for current shopping list")
    # import codecs
    # UTF8Reader = codecs.getreader('utf8')
    # sys.stdin = UTF8Reader(sys.stdin)

    res = find_shopping_item("We ran out of juice and soda")
    print(res)
    # for line in sys.stdin:
    # print(line.strip())
    # # print(everything("I ran out of broccoli"))
    # theStr = find_shopping_item(line.strip())
    # print("---------------------------")
    # print("---------------------------")
    # print("---------------------------")
    # print("---------------------------")
    # print("---------------------------")
    # print("---------------------------")
    # print("Classification and confidence level of the phrase's classification:")
    # print(theStr)

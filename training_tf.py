from __future__ import print_function

import json

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from model import create_dense_models
from sentence_transformers import SentenceTransformer

from os import listdir
from os import path

from tensorflow.keras.applications.resnet50 import ResNet50

import numpy as np
from pathlib import Path
from gensim.models.keyedvectors import KeyedVectors

from common import load, extract_features_file

# globals
dataset_path = 'dataset_small'
dataset_cut = 100 # 8000: 7300, -791:
dataset_sup = 114
output_path = 'output_en'
embedding_path = 'embeddings'

def load_embeddings(word_index, path_to_file):
    path_obj = Path(path_to_file)
    embeddings_index = None
    embeddings_dim = 0
    if path_obj.suffix == '.txt':
        embeddings_index = {}
        with open(path_to_file,errors='ignore') as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                embeddings_index[word] = coefs
                if embeddings_dim == 0:
                    embeddings_dim = coefs.shape[0]
        nb_vectors = len(embeddings_index)

    elif path_obj.suffix == '.bin':
        embeddings_index = KeyedVectors.load_word2vec_format(path_to_file, binary=True)
        embeddings_dim = embeddings_index.vector_size
        nb_vectors = len(embeddings_index.vocab)
    else:
        raise Exception('unknown format for embeddings')

    # print("Found %s word vectors." % nb_vectors)

    num_tokens = len(word_index)
    hits = 0
    misses = 0

    # Prepare embedding matrix
    embeddings_matrix = np.zeros((num_tokens, embeddings_dim))
    for word, i in word_index.items():
        if word in embeddings_index:
            embeddings_vector = embeddings_index[word]
            embeddings_matrix[i] = embeddings_vector
            hits += 1
        else:
            # Words not found in embedding index will be all-zeros.
            embeddings_matrix[i] = np.zeros(embeddings_dim)
            misses += 1

    # print("Converted %d words (%d misses)" % (hits, misses))
    return embeddings_matrix

# extract features from each photo in the directory
def extract_features_dir(model, directory):
   # extract features from each photo
   all_features = []
   for name in listdir(directory):
      # load an image from file
      filename = path.join(directory, name)
      features = extract_features_file(model, filename)
      all_features.append(features)
      print('>%s' % name)
   return all_features

# -*- coding: utf-8 -*-

if __name__ == '__main__':

    NAME = "TF_BOARD"
    tensorboard = TensorBoard(log_dir="./logs/{}".format(NAME))
    resnet_model = ResNet50(weights='imagenet')     # , include_top=False)

    # extract features from all images
    directory = dataset_path + '/Flickr8k_image/'
    features = extract_features_dir(resnet_model, directory)
    print('Extracted Features for %d images' % len(features))

    # save to file
    np_features = np.array(features)
    np.save(output_path + '/flicker_img_features.npy', np_features)

    # t = np.load(output_path + '/flicker_img_features.npy')
    # print(t.shape)

    text_file = dataset_path + '/Flickr8k_text/Flickr8k.token.en.txt'
    features_file = output_path + '/flicker_img_features.npy'
    features, images, texts = load(text_file, output_path + '/flicker_img_features.npy')

    model_name_or_path = 'distilbert-base-nli-mean-tokens'
    model = SentenceTransformer(model_name_or_path)
    captions = model.encode(texts, convert_to_numpy=True)

    main_model, image_model, caption_model = create_dense_models(caption_dim=1000, input_dim=768)

    # preparation des jeux de données
    noise = np.copy(captions)
    fake_labels = np.zeros((len(features), 1))
    ind_train = 32000
    max_valid = 40000
    X_train = [features[:ind_train], captions[:ind_train], noise[:ind_train]]
    Y_train = fake_labels[:ind_train]

    X_valid = [features[ind_train:max_valid], captions[ind_train:max_valid], noise[ind_train:max_valid]]
    Y_valid = fake_labels[ind_train:max_valid]

    # actual training
    print('x : %d' % len(X_train))
    print('y : %d' % len(Y_train))

    # premier entrainement avec les négatifs trié de façon aléatoire, puis on fait 50% aléatoire, 50% semi-hard
    np.random.shuffle(noise)
    for epoch in range(10):
        main_model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), epochs=5, batch_size=64,
                       callbacks=[tensorboard], shuffle=True)
        noise = compute_noise(noise, captions, features, image_model, caption_model)

    # save models
    image_model.save(output_path + '/model.image')
    caption_model.save(output_path + '/model.caption')

    # save representations
    captions_representations = caption_model.predict(captions)
    print(captions_representations.shape)
    np.save(output_path + '/caption_representations', captions_representations)
    image_representations = image_model.predict(features)
    np.save(output_path + '/image_representations', image_representations)
    print(image_representations.shape)





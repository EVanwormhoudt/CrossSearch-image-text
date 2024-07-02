

from __future__ import print_function

import json
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.resnet50 import ResNet50
from common import extract_features_file, load, preprocess_texts

# globals
dataset_path = 'dataset_big'
output_path = 'output_en'
embedding_path = 'embeddings'


def search_captions(image_filename, model, image_model, caption_representations, texts, n=10):
    # generate image representation for new image
    features = extract_features_file(model, image_filename)
    image_representation = image_model.predict(np.array([features]))
    # compute score of all captions in the dataset
    scores = np.dot(caption_representations,image_representation.T).flatten()
    # compute indices of n best captions
    indices = np.argpartition(scores, -n)[-n:]
    indices = indices[np.argsort(scores[indices])]
    # display them
    for i in [int(x) for x in reversed(indices)]:
        print(scores[i], texts[i])

def search_images(vocab, caption, caption_model, image_representations, images, n=10):
    caption_representation = caption_model.predict(preprocess_texts([caption], vocab))
    scores = np.dot(image_representations,caption_representation.T).flatten()
    indices = np.argpartition(scores, -n)[-n:]
    indices = indices[np.argsort(scores[indices])]
    for i in [int(x) for x in reversed(indices)]:
        print(scores[i], images[i])

if __name__ == '__main__':
    resnet_model = ResNet50(weights='imagenet')     # , include_top=False)

    text_file = dataset_path + '/Flickr8k_text/Flickr8k.token.en.txt'
    features_file = output_path + '/flicker_img_features'

    features, images, texts = load(text_file, output_path + '/flicker_img_features.npy')

    image_model = load_model(output_path + '/model.image')
    caption_model = load_model(output_path + '/model.caption')

    # load representations (you could as well recompute them)
    caption_representations = np.load(output_path + '/caption_representations.npy')
    image_representations = np.load(output_path + '/image_representations.npy')

    # generate a caption for an image
    print('1: generate captions for an image')
    image = dataset_path + '/Flickr8k_image/3730011219_588cdc7972.jpg'
    search_captions(
        image,
        resnet_model,
        image_model,
        caption_representations,
        texts
    )

    # search the image corresponding to this caption
    vocab = json.loads(open(output_path + '/vocab.json').read())
    print('2: search the image corresponding to this caption')
    search_images(
        vocab,
        'high mountains',
        caption_model,
        image_representations,
        images
    )
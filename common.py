
import numpy as np

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load(captions_filename, features_filename):
    features = np.load(features_filename)
    images = []
    texts = []
    with open(captions_filename) as fp:
        for line in fp:
          if (len(line)==1) :
            continue
          else :
            tokens = line.strip().split()
            images.append(tokens[0])
            texts.append(' '.join(tokens[1:]))
    return features, images, texts

def extract_features_file(model, img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return np.squeeze(features)

def preprocess_texts(texts, vocab):
    output = []
    for text in texts:
        output.append([vocab[word] if word in vocab else 0 for word in text.split()])
    return pad_sequences(output, maxlen=16)
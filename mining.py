import numpy as np

from tensorflow.python.keras.layers.merge import Dot
from tensorflow.python.keras.layers.merge import Dot
from tensorflow.keras import backend as K

def custom_loss_noise(positive, negative):
    return K.sum(K.maximum(0., 1. - positive + negative))

#mise à jour de la liste noise à chaque itération
def compute_noise(noise, captions, features, image_model, caption_model):
    # premier shuffle afin d'avoir 50% aléatoire
    np.random.shuffle(noise)
    n = len(captions)

    # pour la moitié du vecteur
    for i in range(n // 2, n):
        x = np.array([captions[i]])
        im = image_model.predict(np.array([features[i]]))
        x = caption_model.predict(x)
        found = False
        j = 0
        # On calcul le Loss du triplet (image, x (texte positif) et y (texte négatif)) jusqu'à avoir un y tq le triplet est semi-hard
        while not found:
            y = np.array([captions[j]])
            y = caption_model.predict(y)
            positive_pair = Dot(axes=1)([im, x])
            negative_pair = Dot(axes=1)([im, y])
            loss = custom_loss_noise(positive_pair, negative_pair)
            if loss < 1 and loss > 1e-6:
                noise[i] = captions[j]
                found = True
            else:
                j += 1
    return (noise)


from __future__ import print_function

import numpy as np
import pandas as pd

# globals
directory = 'data'
dataset_path = directory + '/Flicker8k_Dataset'
output_path = 'output'
embedding_path = 'embeddings'


def save_tsv_file(path, data_frame):
    np.savetxt(path, data_frame, delimiter='\t')


if __name__ == '__main__':
    caption_representations = np.load(output_path + '/caption_representations.npy')
    texts = np.load(output_path + '/texts_extend.npy')

    save_tsv_file(output_path + '/visualisation/vect_caption_representations.tsv')
    save_tsv_file(output_path + '/visualisation/meta_caption_representations.tsv')

### Pour générer la matrice de taille 48546 lignes avec les 40455 lignes de texte et les 8091 lignes d'image :
### Permet de projetter les images et les textes dans le même espace vectoriel 


output_path = 'output_en'
texts = np.load(output_path + '/texts_extended.npy')
caption_representations = np.load(output_path + '/caption_representations.npy')
image_representations = np.load(output_path + '/image_representations.npy')

# vecteur des 8091 images encodées
image_representations_reduced = []
# vecteur d'une des 5 captions des images
captions_reduced = []
for i in range(len(image_representations) // 5):
    image_representations_reduced.append(image_representations[i * 5])
    captions_reduced.append(texts[i * 5])

image_representations_reduced = np.array(image_representations_reduced)

# matrice de 48546 lignes
img_caption_reprs = np.concatenate((caption_representations, image_representations_reduced), axis=0)
# vecteur de 48546 captions
meta_captions = np.concatenate((texts, captions_reduced), axis=0)
# vecteur de 48546 labels (40455*'caption', 8091*'image')
label = []

for i in range(5 * 8091):
    label.append('caption')
for i in range(8091):
    label.append('image')

d = np.column_stack((meta_captions, label))
df = pd.DataFrame(d, columns=['captions', 'labels'])

df.to_csv('visualisation/meta_captions_image_representations.tsv', header=True, sep='\t')
np.savetxt('visualisation/vec_captions_image_representations.tsv', img_caption_reprs, delimiter='\t')
df.to_csv('df.csv', index=True, header=True, sep=' ')

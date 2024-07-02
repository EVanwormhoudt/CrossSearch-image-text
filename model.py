
from tensorflow.keras.layers import Input, Dense, Embedding, GRU
from tensorflow.python.keras.layers.merge import Dot, Concatenate
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers.merge import Dot, Concatenate
from tensorflow.keras import backend as K

def custom_loss(y_true, y_pred):
    positive = y_pred[:,0]
    negative = y_pred[:,1]
    return K.sum(K.maximum(0., 1. - positive + negative))

def accuracy(y_true, y_pred):
    positive = y_pred[:,0]
    negative = y_pred[:,1]
    return K.mean(positive > negative)

def create_dense_models(image_dim, caption_dim):
    image_input = Input(shape=(image_dim,))
    caption_input = Input(shape=(caption_dim,))
    noise_input = Input(shape=(caption_dim,))
    image_dense = Dense(256, activation='tanh')
    caption_dense = Dense(256, activation='tanh')
    caption_pipeline = caption_dense(caption_input)
    noise_pipeline = caption_dense(noise_input)
    image_pipeline = image_dense(image_input)
    positive_pair = Dot(axes=1)([image_pipeline, caption_pipeline])
    negative_pair = Dot(axes=1)([image_pipeline, noise_pipeline])
    output = Concatenate()([positive_pair, negative_pair])
    main_model = Model(inputs=[image_input, caption_input, noise_input], outputs=output)
    image_model = Model(inputs=image_input, outputs=image_pipeline)
    caption_model = Model(inputs=caption_input, outputs=caption_pipeline)
    main_model.compile(loss=custom_loss, optimizer='adam', metrics=[accuracy])
    return main_model, image_model, caption_model

def create_gru_models(image_dim, input_length, embedding_weights, vocab):
    image_input = Input(shape=(image_dim,))
    caption_input = Input(shape=(input_length,))
    noise_input = Input(shape=(input_length,))
    embedding_dim = embedding_weights.shape[1]
    caption_embedding = Embedding(len(vocab), embedding_dim, input_length=input_length, weights=[embedding_weights])
    caption_rnn = GRU(256)
    image_dense = Dense(256, activation='tanh')
    image_pipeline = image_dense(image_input)
    caption_pipeline = caption_rnn(caption_embedding(caption_input))
    noise_pipeline = caption_rnn(caption_embedding(noise_input))
    positive_pair = Dot(axes=1)([image_pipeline, caption_pipeline])
    negative_pair = Dot(axes=1)([image_pipeline, noise_pipeline])
    output = Concatenate()([positive_pair, negative_pair])
    main_model = Model(inputs=[image_input, caption_input, noise_input], outputs=output)
    image_model = Model(inputs=image_input, outputs=image_pipeline)
    caption_model = Model(inputs=caption_input, outputs=caption_pipeline)
    main_model.compile(loss=custom_loss, optimizer='adam', metrics=[accuracy])
    return main_model, image_model, caption_model





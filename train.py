import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow_core.python.keras.distribute.distribute_strategy_test import get_dataset

from get_dataset import download_annotation, download_images
from models.CNN import CNN_Encoder
from models.RNN import RNN_Decoder
from process_image import get_image_captions, get_feature_extraction_model, load_image, cache_image_features
import time

top_k = 10000
attention_features_shape = 64
embedding_dim = 256
units = 768
vocab_size = top_k + 1
BATCH_SIZE = 64
BUFFER_SIZE = 1000

EPOCHS = 30

def calc_max_length(tensor):
    return max(len(t) for t in tensor)


def map_func(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8') + '.npy')
    return img_tensor, cap


def get_tokenizer(train_captions, top_k=5000):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                      oov_token="<unk>",
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(train_captions)
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    return tokenizer


def get_cap_vector(train_captions):
    tokenizer = get_tokenizer(train_captions, top_k)
    train_seqs = tokenizer.texts_to_sequences(train_captions)
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
    return cap_vector


def train():
    annotation_file = 'annotations/captions_train2014.json'
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    train_captions, img_name_vector = get_image_captions(annotations)
    cap_vector = get_cap_vector(train_captions)
    img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,
                                                                        cap_vector,
                                                                        test_size=0.2,
                                                                        random_state=0)

    num_steps = len(img_name_train) // BATCH_SIZE

    dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

    # Use map to load the numpy files in prallel
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(
        map_func, [item1, item2], [tf.float32, tf.int32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Shuffle and batch
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    checkpoint_path = "./checkpoints/train"
    ckpt = tf.train.Checkpoint(encoder=encoder,
                               decoder=decoder,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    start_epoch = 0
    if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
        # restoring the latest checkpoint in checkpoint_path
        ckpt.restore(ckpt_manager.latest_checkpoint)

    tokenizer = get_tokenizer(train_captions, top_k)

    loss_plot = []

    @tf.function
    def train_step(img_tensor, target):
        loss = 0

        # initializing the hidden state for each batch
        # because the captions are not related from image to image
        hidden = decoder.reset_state(batch_size=target.shape[0])

        dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

        with tf.GradientTape() as tape:
            features = encoder(img_tensor)

            for i in range(1, target.shape[1]):
                # passing the features through the decoder
                predictions, hidden, _ = decoder(dec_input, features, hidden)

                loss += loss_function(target[:, i], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(target[:, i], 1)

        total_loss = (loss / int(target.shape[1]))

        trainable_variables = encoder.trainable_variables + decoder.trainable_variables

        gradients = tape.gradient(loss, trainable_variables)

        optimizer.apply_gradients(zip(gradients, trainable_variables))

        return loss, total_loss



    for epoch in range(start_epoch, EPOCHS):
        start = time.time()
        total_loss = 0

        for (batch, (img_tensor, target)) in enumerate(dataset):
            batch_loss, t_loss = train_step(img_tensor, target)
            total_loss += t_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(
                    epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
        # storing the epoch end loss value to plot later
        loss_plot.append(total_loss / num_steps)

        if epoch % 5 == 0:
            ckpt_manager.save()

        print('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                            total_loss / num_steps))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


def evaluate(image):
    annotation_file = 'annotations/captions_train2014.json'
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    train_captions, img_name_vector = get_image_captions(annotations)

    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)

    tokenizer = get_tokenizer(train_captions, top_k)
    train_seqs = tokenizer.texts_to_sequences(train_captions)
    max_length = calc_max_length(train_seqs)

    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    image_features_extract_model = get_feature_extraction_model()
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', help='evaluate with an image')
    args = parser.parse_args()
    if args.eval:
        image = args.eval
        res, _ = evaluate(image)
        print('Prediction Caption:', ' '.join(res))
    else:
        if not (os.path.exists('annotations') and os.path.exists('train2014')):
            download_annotation()
            download_images()
            annotation_file = 'annotations/captions_train2014.json'
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)

            _, img_name_vector = get_image_captions(annotations)
            cache_image_features(img_name_vector)
        train()

from unittest import TestCase, main

import tensorflow as tf
import json

from process_image import get_image_captions, load_image


class ImageModelTest(TestCase):

    def setUp(self):
        annotation_file = 'annotations/captions_train2014.json'
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        _, self.img_name_vector = get_image_captions(annotations)

    def test_dataset(self):
        encode_train = sorted(set(self.img_name_vector))
        image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)

        for img in image_dataset.take(1):
            self.assertEqual(img, 'train2014/COCO_train2014_000000000009.jpg', 'image_dataset test')
            # print(img.numpy())

    def test_load_image(self):
        img, path = load_image('train2014/COCO_train2014_000000000009.jpg')
        self.assertEqual(img.shape, (224, 224, 3), 'load_image test shapes match')
        # print(img.shape)

    def test_inception_model(self):
        image_model = tf.keras.applications.VGG19(weights='imagenet')
        images = ['test/test.jpg', 'test/test4.jpg','test/test2.jpg','test/test3.jpg']
        image_dataset = tf.data.Dataset.from_tensor_slices(images)
        image_dataset = image_dataset.map(
            load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(2)
        # print(image_dataset)
        # img,_ = load_image()
        prediction = image_model.predict(image_dataset)
        res = tf.keras.applications.vgg19.decode_predictions(prediction)
        print(res)


if __name__ == '__main__':
    main()

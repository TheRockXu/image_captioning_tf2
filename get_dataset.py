import os
import tensorflow as tf

def download_annotation():
    annotation_zip = tf.keras.utils.get_file('captions.zip',
                                             cache_subdir=os.path.abspath('.'),
                                             origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                             extract=True)
    annotation_file = os.path.dirname(annotation_zip) + '/annotations/captions_train2014.json'
    os.remove(annotation_zip)
    return annotation_file


def download_images():

    image_zip = tf.keras.utils.get_file('train2014.zip',
                                        cache_subdir=os.path.abspath('.'),
                                        origin='http://images.cocodataset.org/zips/train2014.zip',
                                        extract=True)
    PATH = os.path.dirname(image_zip)
    os.remove(image_zip)


    return PATH


if __name__ == '__main__':
    download_annotation()
    download_images()
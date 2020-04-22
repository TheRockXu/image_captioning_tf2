import os
import tensorflow as tf

def download_annotation(folder):
    if not os.path.exists(os.path.abspath('.') + folder):
        annotation_zip = tf.keras.utils.get_file('captions.zip',
                                                 cache_subdir=os.path.abspath('.'),
                                                 origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                                 extract=True)
        annotation_file = os.path.dirname(annotation_zip) + '/annotations/captions_train2014.json'
        os.remove(annotation_zip)
        print(annotation_file)
    else:
        annotation_file = os.path.abspath('.')+folder+'/annotations/captions_train2014.json'

    return annotation_file

def download_images(folder):
    if not os.path.exists(os.path.abspath('.') + folder):
        image_zip = tf.keras.utils.get_file('train2014.zip',
                                            cache_subdir=os.path.abspath('.'),
                                            origin='http://images.cocodataset.org/zips/train2014.zip',
                                            extract=True)
        PATH = os.path.dirname(image_zip) + folder
        os.remove(image_zip)
    else:
        PATH = os.path.abspath('.') + folder

    print(PATH)
    return PATH


if __name__ == '__main__':
    download_images('data/')
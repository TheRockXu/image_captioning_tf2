# Image Captioning with Tensorflow 2
Image Captioning with CNN, LSTM, attention model by Tenorflow 2+.
This is mostly copied from [Tensorfow Tutorial](https://www.tensorflow.org/tutorials/text/image_captioning).

The model architecture is similar to [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/pdf/1502.03044.pdf).
## Dependencies
> `Tensorflow`\
`pandas`\
`numpy`\
`scikit-learn`\
`Pillow`

## Training

When run `python train.py` for the first time, it will trigger a download of  the MS-COCO dataset, preprocesses and caches a subset of images using Inception V3, trains an encoder-decoder model, and generates captions on new images using the trained model.

**Caution**: It's a large dataset. You'll use the training set, which is a 13GB file.

## Evaluation

After the training is done, you can evaluate with a single image, like this - \
`python train.py --eval test/test.png`


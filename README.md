  # Multimodal Machine Translation

  Implementation of MSc Thesis "Image Informed Neural Machine Translation with Transformers". The Transformer takes as input text and image features extracted from a ResNet-50. The provided code is for the first case of the input image features mentioned in Image_Informed_Neural_Machine_Translation.pdf
  
  You can find Multi-30K dataset here: http://www.statmt.org/wmt16/multimodal-task.html#task1
   
  
  # Training
  `python train_mm.py -data /path/to/text/data -train_image_feat /path/to/train/image/features -val_image_feat /path/to/validation/image/features`
 
  # Translate
  `python translate_mm.py -model /path/to/model/chkpt -src /path/to/source/sentences -vocab /path/to/source/vocabulary -test_image_feat /path/to/test/image/features`

# Image Verification for Authenticity

Bachelor's final project for Saint Petersburg State University, Faculty of Applied Mathematics and Control Processes on the topic 'Image Verification for Authenticity'. The program consists of a CNN created with TensorFlow that plays the role of a feature extractor for an image patches and a classifier that uses a fused feature to tell if an image has been forged.

## Getting Started

### Prerequisites

Use pip or what you prefer to get these modules:

```
OpenCV
Scikit-learn
TensorFlow
```

### Creating Dataset
For this work CASIA v2 was used. gn.py and gnt.py can be used to extract authentic and tampered patches from it.

### Training
Copy all the files to your machine. In train.py you can specify path to your training and testing data. 'authentic' and 'tampered' directories with corresponding 64x64 image patches must be there. Run command

```
python train.py x
```

to train CNN for x epochs. When you're satisfied with verification accuracy, specifiy paths to directories with whole pictures in desc.py. Run

```
python desc.py
```

to create files with picture descriptors. If you want to use grid search uncomment the block with it in classifier.py. Then run

```
python classifier.py
```

to train the classifier.

### Using Pretrained Model
Just copy tampered-authentic-model, classifier.joblib.pkl and verify.py to your machine. 

## Usage

To verify some image run 

```
python verify.py pic.jpg
```

where pic.jpg is the path to your picture.

## Authors

* **Elena Fedchenko** - *Initial work* - [carnwyr](https://github.com/carnwyr)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Yuan Rao, Jiangqun Ni - A deep learning approach to detection of splicing and copy-move forgeries in images (CNN architecture)
* TensorFlow tutorials
* Credits for the use of the CASIA Image Tampering Detection Evaluation Database (CAISA TIDE) V2.0 are given to the National Laboratory of Pattern Recognition, Institute of Automation, Chinese Academy of Science, Corel Image Database and the photographers. http://forensics.idealtest.org


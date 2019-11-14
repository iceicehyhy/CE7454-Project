### Prerequisites

- Python 3
- NVIDIA GPU + CUDA 9.0 + cuDNN
- PyTorch 1.3.1
- Tensorflow 1.7.0
- Git-lfs

### Packages Requirments
- numpy ~= 1.14.3
- scipy ~= 1.0.1
- future ~= 0.16.0
- matplotlib ~= 2.2.2
- pillow >= 6.2.0
- opencv-python ~= 3.4.0
- scikit-image ~= 0.14.0
- pyaml
- glob

# Model Dataset
## 1) Images
We use our own dataset from Google Image and PictureSG

To run own your dataset, run scripts/flist.py to generate train, test and validation set file lists. For example, to generate the training set file list on your dataset run:
```bash
mkdir data
python ./scripts/flist.py --path PATH_TO_YOUR_DATA --output ./data/flist/XXX.flist
```
## 2) Masks
The model is trained on randomly generated irregular mask 

# Model Training
Our model is trained in four stages: 
1) training the edge model
2) training the coarse inpaint model 
3) training the fine inpaint model 
4) training the joint model

To train the model:

```bash
python EIWA.py --mode 1 --model XXX
```

#Model Testing
To test the model:

```bash
python EIWA.py --mode 2
```
 
# Other Models
 1) Context Encoder
 2) GLCIC
 3) Contextual Attention

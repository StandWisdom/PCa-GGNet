# PCa-ISUPNet
A radiologist-like agent based on reinforcement learning to predict case-level ISUP grade in prostate cancer for restriction of upgrading and downgrading risk of pathological evaluation.<br>

The goal of this repository is:
- to help researchers to reproduce the PCa-ISUPNet framework and expand for other prostate research or relevant research.
- to help researchers to build a generator-net alone for predicting ISUP grade of radical prostatectomy in case-level (multi-category),
- to help researchers to build an action-net alone for attentional slice searching.

## Installation

1. [python3 with anaconda](https://www.continuum.io/downloads)
2. [pytorch with/out CUDA](http://pytorch.org)
3. `pip install pretrainedmodels`
4. `pip install pandas`

## Prepare your data for generator-net:
The training list was essential for the Framework.

For generator-net, you should split your 3D MRI into 2D slice, and record the data-ID, label, flag of tumor slice, and path of every slice as .csv or .xlsx. The names of columns were 'ID','label','Z', and 'path' in our project. More details can be found in DataSet.py.

For action-net, you should first extract the CNN features of each slice (not only tumor slice) by generator-net. Then, you should record the data-ID, label, flag of tumor slice, predicted result of the slice with probability, and extracted features as .csv or .xlsx. 
More details can be found in action-net.py and actionNet_trainlist_example.csv.
 
## Training for generator-net
`python generator-net.py`

## Features extraction by generator-net
`python generator-net-forward.py`

## Training for action-net and results of case-level prediction
`python action-net.py`

## Acknowledgement

Thanks to the https://github.com/Cadene/pretrained-models.pytorch for pretrained ConvNets with a unique interface/API inspired by torchvision.<br>
Thanks to the https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch. for implemention of basical DQN framwork in pytorch.<br>
Thanks to Peking University Third Hospital (PUTH) and Peking University People Hospital (PUPH) for data support.<br>
Thanks to Key laboratory of molecular imaging, Insistute of automation, Chinese Sciences of Academy for support of platform.<br>
Thanks to LIST, Key Laboratory of Computer Network and Information Integration, Southest University for technical support.<br>

# Process of our method
![orig](https://github.com/StandWisdom/PCa-ISUPNet/blob/master/ABSTRACT-gif.gif)<br>


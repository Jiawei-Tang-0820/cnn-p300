# Using CNN to Detect P300
COGS 189 Final Project - Jiawei Tang, Yujie Xu

## Dataset

The dataset is aquired from [here](https://www.kaggle.com/rramele/p300samplingdataset). It was produced using the standard 6x6 Donchin and Farewell P300 Speller Matrix, with an ISI of 125ms. In total, there were8 subjects in the experiment.

## Environment

The model is implemented using pytorch 1.3.1

## Preprocessing

The original data is stored in matlab format. We used `data/preprocessing.ipynb` to preprocess the data.

## Training

To train the model, simply use `python train_subject.py <subject-number>`. This will tell the script to train/test on a specific subject.

## Results

Presention containing the details and results of the model can be found [here](https://docs.google.com/presentation/d/1yibohU11Lvgp0MyxUL5aQJuGrIfr7labSf89mA4V1Do/edit?usp=sharing) (in Google slides)

## References

C. Guger, S. Daban, E. Sellers, C. Holzner, G. Krausz, R. Carabalona, F. Gramatica, and G. Edlinger, “How many people are able to control a P300-based brain-computer interface (BCI)?,” Neurosci. Lett., vol. 462, no. 1, pp. 94–98, Oct. 2009

H. Cecotti and A. Graser, "Convolutional Neural Networks for P300 Detection with Application to Brain-Computer Interfaces," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 33, no. 3, pp. 433-445, March 2011

S. Kundu, S. Ari, A Deep Learning Architecture for P300 Detection with Brain-Computer Interface Application, IRBM, Volume 41, Issue 1, 2020, Pages 31-38, ISSN 1959-0318, https://doi.org/10.1016/j.irbm.2019.08.001
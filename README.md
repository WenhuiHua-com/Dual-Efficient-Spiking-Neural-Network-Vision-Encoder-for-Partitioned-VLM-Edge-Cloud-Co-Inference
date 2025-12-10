# Dual-Efficient-Spiking-Neural-Network-Vision-Encoder-for-Partitioned-VLM-Edge-Cloud-Co-Inference
Test code for articles submitted to IEEE Transactions on Circuits and Systems for Video Technology

## Requirements
Dependencies can be found in requirements.txt

## Download the pre-trained codebook or model from the link below and place them in the designated directory: 
Pre-trained BLIP weights and BERT need to be downloaded before running. [BLIP weights used in this paper]: (https://github.com/salesforce/BLIP) #In this paper, The BERT weights should be placed in the BERT folder.
In order to run the feature coding for machine algorithm proposed in this paper, it is also necessary to download the codebook：
[codebook]:(https://drive.google.com/file/d/1jv3pt70uSgXHUaRnpFujRQKNYIpAdNC0/view?usp=drive_link)

## Data preparation
Please place your dataset files as follows before running any scripts:

flick/ ├── annotation/ │ └── <annotation files> └── flickr30k-images/ └── flickr30k-images/ └── <image files>

- **annotation/**: Contains all annotation files.
- **flickr30k-images/**: The main folder for the Flickr30k dataset images.
  - **flickr30k-images/**: Subfolder with the actual image files.

Make sure to update the dataset paths in your configuration or script parameters accordingly. Also be careful to modify the path in retrieval_flickr.yaml.
## Test hybrid progressive token compression

To generate a caption for an image with SNN encoder:

<pre> python test_spike_caption.py  </pre> 

After running, some results will be output as follows:

Loading codebook sets
Creating model
loading spikeformer successfully
flickr30k-images/1921102799.jpg 100 ['a young boy playing soccer in a field']
flickr30k-images/2504007911.jpg 200 ['a man riding a bike in front of a building']
flickr30k-images/2900560501.jpg 300 ['a group of people in a room']
flickr30k-images/327955368.jpg 400 ['a group of birds in a park']
flickr30k-images/3671851846.jpg 500 ['a woman standing in a field']
flickr30k-images/428979011.jpg 600 ['a man in a yellow shirt and a yellow hat']
flickr30k-images/4700788144.jpg 700 ['a man riding a bike on a trail']
flickr30k-images/4915716087.jpg 800 ['a man driving a horse drawn carriage']
flickr30k-images/6278649113.jpg 900 ['a person doing a trick on a skateboard']
flickr30k-images/97234558.jpg 1000 ['a person standing on the beach']

## follow-up work
After the review finished, the training code will be updated.

## Some renderings based on the language-guided visual token compression proposed in this paper
 [Visual interpretation]
---
![](importance-visual.png)

 [Visual interpretation1]
---
![](images/visual.png)

##The t-sne result
 [Language-based]
---
![](images/aTSNE_analysis_Text_3D.png)

 [Norm-based]
---
![](images/aTSNE_analysis_Max_3D.png)


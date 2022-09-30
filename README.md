# Few Shot Classification GUI-based Training
- This is a [Streamlit](https://streamlit.io/) based tool to train and develop a [Few Shot Classification](https://neptune.ai/blog/understanding-few-shot-learning-in-computer-vision) ML model very rapidly. 
- It uses [Prototypical Networks](https://towardsdatascience.com/few-shot-learning-with-prototypical-networks-87949de03ccd) with [Resnet18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) as the backbone network.
- During training, model parameters are also saved on disk.


## Requirements (installable via pip)
- torch
- torchvision
- easyfsl
- matplotlib
- streamlit

## What I used?
1. [Omniglot Dataset](https://github.com/brendenlake/omniglot) - for Demo - It is designed for developing more human-like learning algorithms. It contains 1623 different handwritten characters from 50 different alphabets. Each of the 1623 characters was drawn online via Amazon's Mechanical Turk by 20 different people.
2. [Streamlit](https://streamlit.io/) - for GUI - Streamlit is an open-source app framework for Machine Learning and Data Science teams.
3. [Matplotlib](https://matplotlib.org/) - for plotting loss function - Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python. Matplotlib makes easy things easy and hard things possible.
4. [PyTorch](https://pytorch.org/) - An open source machine learning framework that accelerates the path from research prototyping to production deployment.
	- [Resnet18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) CNN
	- Loading datasets
	- Image transformations

## How to run the program?
1. **Download this GitHub repository**
	- Either Clone the repository
	```
	git clone https://github.com/Kunal-Attri/Few-Shot-Classification-GUI-based-Training.git
	```
	- Or download and extract the zip archive of the repository.

2. **Download & Install requirements**
	- Ensure that you have Python 3 installed.
	- Open terminal in the Repository folder on your local machine.
	- Run the following command to install requirements.
	```
	pip3 install -r requirements.txt
 	```
3. **Run the Program**
```
streamlit run train.py
```




# Phishing Email Detection by Machine Learning Techniques

## Objective
A phishing email is a common social engineering method that mimics trustworthy email addresses and content to deceive recipients.The objective of this project is to train machine learning models on a curated dataset to predict phishing emails. Both phishing and legitimate emails are collected to form the dataset, and from them, relevant features based on email headers, content, and metadata are extracted. The performance of each model is measured and compared to evaluate their effectiveness in detecting phishing emails.

## Data Collection
The set of phishing emails are collected from opensource service called **kaggle**. This service provide a set of phishing URLs in multiple formats like csv, json etc. that gets updated hourly. To download the data: (https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset/data). From this dataset, 5000 random phishing emails and their links are collected to train the ML models.

The set of ligitimate emails are also obtained fron same opensource service.Some of them are also obtained from our(the team) personal emails.


The above mentioned datasets are uploaded to the '[DataFiles](  HYA DATSET HALNE GITHUB MA PUSH GAREKO )' folder of this repository.

## Members
Sushovan Bikram Shahi
[Github](https://github.com/sushovanbikramshahi) / [LinkedIn](https://www.linkedin.com/in/sushovan-bikram-shahi-767202312)


Prabhat Amagain
[Github](https://github.com/Prabhat-amgain) / [LinkedIn](https://www.linkedin.com/in/prabhat-amgain-909363277)


Puskar Shrestha
[Github](https://github.com/Puskar-Shrestha) /[LinkedIn](https://www.linkedin.com/in/puskar-shrestha-6a112336a/)




## Feature Extraction
The below mentioned category of features are extracted from the  dataset:

1. URL
2. Sender name
3. Sender domain
4. Text(Subject+Body)
5. label
6. Hour
7. Day of weeks

## Models & Training

Before stating the ML model training, the data is split into 80-20 i.e., 8000 training samples & 2000 testing samples. From the dataset, it is clear that this is a supervised machine learning task. There are two major types of supervised machine learning problems, called classification and regression.

This data set comes under classification problem, as the input email is classified as phishing (1) or legitimate (0). The supervised machine learning models (classification) considered to train the dataset in this project are:

* Multi-Layer Perceptron(MLP)
* Random Forest
* Logistic Regression
* Naïve Bayes
* Support Vector Machines (SVM)




All these models are trained on the dataset and evaluation of the model is done with the test dataset. The elaborate details of the models & its training are mentioned in 
https://github.com/Prabhat-amgain/PhishShield/blob/main/Resources/Model_Evaluation.pdf

## Presentation


The presentation is given in (HYA PRESENTATION KO LINK HALNE)

## End Results
From the obtained results of the above models,Support Vector Machines(SVM) has highest model performance. So the result analysis is given 
### Next Steps

This project can be further applied in phishing detection in Websites and their URls.We (the team) have furthur plans to implement this model in web detectioon and many more to make people safe from phishing.
We want to be the shield that protects common people from all types of phisphing.

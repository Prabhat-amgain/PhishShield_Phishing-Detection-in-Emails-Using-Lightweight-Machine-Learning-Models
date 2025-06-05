# Phishing Email Detection by Machine Learning Techniques

## Objective
A phishing email is a common social engineering method that mimics trustful uniform emails and their URLs(Uniform Resource Locater). The objective of this project is to train machine learning models and deep neural nets on the dataset created to predict phishing emails. Both phishing and benign content of emails are gathered to form a dataset and from them required URL and email content-based features are extracted. The performance level of each model is measures and compared.

## Data Collection
The set of phishing emails are collected from opensource service called **kaggle**. This service provide a set of phishing URLs in multiple formats like csv, json etc. that gets updated hourly. To download the data: (https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset/data). From this dataset, 5000 random phishing emails and their links are collected to train the ML models.

The set of ligitimate emails are also obtained fron same opensource service.Some of them are also obtained from our(the team) personal emails.


The above mentioned datasets are uploaded to the '[DataFiles](  HYA DATSET HALNE GITHUB MA PUSH GAREKO )' folder of this repository.

## Feature Extraction
The below mentioned category of features are extracted from the  dataset:

1. Label
2. Email body
3. URL links
4. Sender email adress.
5. Date of email send. 

## Models & Training

Before stating the ML model training, the data is split into 80-20 i.e., 8000 training samples & 2000 testing samples. From the dataset, it is clear that this is a supervised machine learning task. There are two major types of supervised machine learning problems, called classification and regression.

This data set comes under classification problem, as the input email is classified as phishing (1) or legitimate (0). The supervised machine learning models (classification) considered to train the dataset in this project are:

* Decision Tree
* Random Forest
* Logistic Regression
* Naïve Bayes
* Support Vector Machines (SVM)




All these models are trained on the dataset and evaluation of the model is done with the test dataset. The elaborate details of the models & its training are mentioned in 
(HYA GITHUB ko TRAINED MODEL KO LINK HALNE)

## Presentation


The presentation is given in (HYA PRESENTATION KO LINK HALNE)

## End Results
From the obtained results of the above models, (              THA XAINA          )has highest model performance of (         ). So the model is saved to the file ( MODEL KO LINK)
### Next Steps

This project can be further applied in phishing detection in Websites and their URls.We (the team) have furthur plans to implement this model in web detectioon and many more to make people safe from phishing.
We want to be the shield that protects common people from all types of phisphing.

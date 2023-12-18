# Sentiment Analysis Project Report

## By: Anas Zaheer

## Introduction

Sentiment analysis, a subset of natural language processing, plays a pivotal role in understanding the opinions and emotions expressed in textual data. In this project, we embarked on a comprehensive exploration of sentiment analysis using the IMDb movie review dataset. The primary objective was to build, train, and evaluate machine learning models, including Logistic Regression, Support Vector Machines (SVM), Naive Bayes, K-Nearest Neighbors (KNN), and Random Forest, to discern their performance in classifying movie reviews as positive or negative.

## Step 1: Data Import and Exploration

The objective of this project was to conduct sentiment analysis on a dataset of 50,000 IMDb movie reviews. The dataset was explored, preprocessed, and utilized to train various machine learning models, including Logistic Regression, Support Vector Machines (SVM), Naive Bayes, K-Nearest Neighbors (KNN), and Random Forest. The goal was to evaluate their performance in classifying reviews as either positive or negative.

## Step 2: Model Training and Evaluation

### Logistic Regression

Models were trained using both Bag of Words (BoW) and TF-IDF features. The CountVectorizer transformed the text into a high-dimensional sparse matrix for the BoW model, while the TfidfVectorizer was employed for the TF-IDF model. Model evaluation encompassed accuracy scores, precision, recall, and F1-scores, providing a comprehensive assessment of the models' performance.

### Support Vector Machine (SVM)

SVM models, utilizing the stochastic gradient descent (SGD) classifier, were trained on BoW and TF-IDF features. The BoW-based SVM exhibited commendable performance, while the TF-IDF-based SVM faced challenges, particularly in accuracy and F1-score.

## Model Comparison

The analysis expanded to include Naive Bayes, KNN, and Random Forest models, presenting a holistic comparison. A bar graph visually shows the accuracy scores of all models, with Logistic Regression consistently outperforming others.

## Precision, Recall, and F1-Score Analysis

A granular analysis of precision, recall, and F1-scores for each model provided nuanced insights into their strengths and weaknesses. Logistic Regression maintained a balance between precision and recall, making it a robust choice for sentiment analysis.

## Conclusion

This sentiment analysis project underscored the significance of thoughtful model selection, feature representation, and thorough evaluation. Logistic Regression emerged as a consistent performer, showcasing a nuanced understanding of the delicate balance between precision and recall. The exploration of various models and metrics lays the groundwork for future optimizations, potentially involving hyperparameter tuning, ensemble methods, or deep learning approaches. In conclusion, this project serves as a testament to the nuanced nature of sentiment analysis and the importance of comprehensive evaluation methodologies.

Feel free to explore the project and use the insights for your own sentiment analysis tasks. Contributions and feedback are welcome!

Happy analyzing!

# Youtube_Dashboard

## Goals: 
1. Perform sentiment analysis on video comments to better convey audience reaction to content 
2. Create a functional dashboard to provide more usable and visual results

## Technologies
**Packages:** Pandas, NumPy, NLTK, Plotly's Dash, and Scikit-learn <br>
**Models:** Naïve Bayes, Logistic Regression, Support Vector Machine, K-NN, Neural Network, and Random Forest <br>
**Languages:** Python, Javascript, HTML, possibly some SQL <br>
**APIs:** Google Development's Youtube API <br>

## Dashboard (In Progress): 
![dashboard screenshot](https://github.com/UCSB-dataScience-ProjectGroup/youtube/blob/Andies-Branch/images/Dashboard_Screenshot.png)

### Table of Contents: 
1. [Youtube API Call](https://github.com/adonovan7/YoutubeAnalysis/blob/master/apiCall.py)
	* pulls comments for a specified video from the API
2. [Machine Learning Jupyter Notebook](https://github.com/adonovan7/YoutubeAnalysis/blob/master/Youtube_Analysis.ipynb)
	* conducts natural language processing to parse comments
	* performs sentiment analysis through machine learning algorithms to classify data as positive, negative, or neutral 
3. [Dashboard Script](https://github.com/adonovan7/YoutubeAnalysis/blob/master/dash/Dashboard.py)
	* provides an interactive platform for visually analyzing results
	* written in Plotly's Dash, which is built on the Plotly.js, React, and Flask platforms

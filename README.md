# Youtube Analysis

* The first iteration of this project was created for the UCSB Data Science Club with Matthew Peterschmidt, Shon Inouye, and Conor O'Brien (see original repo [here](https://github.com/UCSB-dataScience-ProjectGroup/youtube))

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

### Preliminary Instructions: 
1. Set up an Google Dev account and get a personal API key for accessing the Youtube API
2. Download the repo and open it with a text editor
3. Create a file named `config.js` in the main directory of the repo. Edit the script and type in your key with the following format (replacing the text within the single quotes with your key): 
    * The secret key should be about 39 characters in length and consists of upper and lower case letters as well as numbers
    
```java
var config = {
	SECRET_KEY : 'AIzaSyCaMqxx3_RQXgVhS6QTmfpnFYRhoAKfvVA' 
}
```

3. In your terminal, cd into the master file and run the `user_prompt.py` script by typing the following: 

	`Python3 user_prompt.py` 
	* The script will ask you for some user input, follow the instructions provided in the terminal
4. Check that the comments were correctly outputted in the data/data.csv file
5. Run the machine learning and sentiment analysis code in a notebook by typing `jupyter notebook YoutubeAnalysis.ipynb` into your terminal. 
	* Click shift + enter to run each box
	* Make sure to change the os.chdir() path to match your current directory
	* change the pd.read_csv('...') to pd.read_csv('data.csv') to work with the comments from the video you selected
	* **NOTE:** still in progress of cleaning this part up
6. For the Dashboard capabilites, type `python -W ignore dash/Dashboard.py >/dev/null` to run the plotly Dash application 
	* for Windows, the slashes are reversed: `Python3 dash\Dashboard.py`
	* the `-W ignore` component of that command is to ignore the warnings to be printed in bash and the `>/dev/null` supressed any other output and downloading messages. **This is generally not good practice**, but for now it helps with readability of prior terminal commands, so I left it in place. 
7. Open up the link [http://127.0.0.1:8050/ ](http://127.0.0.1:8050/) in your browser to view the dashboard locally


### TO DO
- [X] Create a more integrated/ cohesive workflow between scripts
- [X] Run the ApiCall with user inputted video
- [ ] Finish Dashboard
- [ ] Improve API call output format (fix encoding issues)
- [ ] Add cross validation, neural nets, and grid search to Machine Learning notebook
- [ ] Use AWS to improve model accuracy
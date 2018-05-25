#pip install vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import time
analyzer = SentimentIntensityAnalyzer()

tot_count = 0
pos_count = 0
pos_correct = 0
neg_count = 0
neg_correct = 0
neu_count = 0
neu_correct = 0

#change "positive.txt" to our data set, mostlikely it will be the f.read line
#we will prob have to change around these numbers to make sure we get the most accurate results
#with open("positive.txt","r") as f:
	#im talking about here, might need to be something different cause it's a csv
    #for line in f.read().split('\n'):

text = ["wow this is awesome", "I hate this song", "how lovely"]

    for line in text:
        vs = analyzer.polarity_scores(line)
        #for the line below im not sure what the range of vs['neg'] is as well as the ranges for 'pos' and 'neu'
        if not vs['neg'] > 0.1:
        	#prob gunna have to change the line below here too, not confident in the 0.1 value
            if vs['pos']-vs['neg'] >= 0.1:
                pos_correct += 1
            #pos_count +=1
            tot_count += 1
        elif not vs['pos'] > 0.1:
        	#also not confident in the 0.1 value
            if vs['neg']-vs['pos'] >= 0.1:
                neg_correct += 1
            #neg_count +=1
            tot_count += 1
        else:
        	if vs['neu'] - vs['neg'] >= 0 && vs['neu'] - vs['pos'] >= 0:
       			neu_correct += 1
       		tot_count += 1

#Don't need all of this code under here, combine the two
#neg_count = 0
#neg_correct = 0

#with open("negative.txt","r") as f:
#    for line in f.read().split('\n'):
#        vs = analyzer.polarity_scores(line)
#        if not vs['pos'] > 0.1:
#            if vs['pos']-vs['neg'] <= 0:
#                neg_correct += 1
#            neg_count +=1

print("Positive percentage = {}% via {} samples".format(pos_correct/tot_count*100.0, tot_count))
print("Negative percentage = {}% via {} samples".format(neg_correct/tot_count*100.0, tot_count))
print("Neutral percentage = {}% via {} samples".format(neu_correct/tot_count*100.0, tot_count))

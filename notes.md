# Personal Notes for Project

free text query: terms of the query are typed freeform into the search interface, without any connecting search operators
* views queries simply as a set of words
* assign each term in the document a weight by number of occurances: term freq = tf _ t,d `($\mbox{tf}_{t,d}$)` 

stop words - words that we do not index at all; do not contribute in any way to retrieval and scoring


TF-IDF (Term Frequency-Inverse Document Frequency) is a text mining technique used to categorize documents
sentiment analysis aims to classify documents into opinions such as 'positive' and 'negative'
On the other hand, TF-IDF classifies documents into categories inside the documents themselves



 Functions: 
 text.concordance - shows occurences of word + context
 text.similar - finds words with similar contexts
 text.dispersion_plot(["word 1", "word 2", ...]) - plots use of the word by places in the whole document
 len - number of tokens (sequence of characters) in a text 
 sorted(set(text)) - gives every unique word in alphbetical order

 len(set(text)) / len(text) - lexical richness; % of distinct words

 text = sequence of words in an ordered list
 concatenation - combines lists together
 text.append - add an element to a list
 text[i] - indexes the i+1th element of the list
 slicing - accessing sublists 
 	ex: text[2:4] - prints elements 3, 4, 5
 	ex: text[2:] - prints all elements to end after 3rd

 sparse data = high dimensional but with few instances


****
 Pyplot is an interactive to matplotlib, mostly for use in notebooks like jupyter. You generally use it like this: import matplotlib.pyplot as plt.

Pylab is the same thing as pyplot, but with extra features (its use is currently discouraged).

Roughly, pylab = pyplot + numpy
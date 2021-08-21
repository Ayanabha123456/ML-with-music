# ML-with-music
## Data gathering
'BeautifulSoup' package in Python is used to scrap song lyrics from a website and stored in a dataframe

## Finding plagiarized songs
### Model 1 - kmeans with knn
K-means clustering is used to segregate the songs using the TFIDF matrix into an optimal number of clusters. Since the songs of the same cluster have higher intra-cluster similarity they can be inferred to be similar in terms of lyrical content and might be plagiarized. Going one step further, once each song has been assigned a cluster label, it can be trained along with the TFIDF matrix using a KNN classifier, which returns the distance and indices of the top closest songs to a particular target song and can be said to be the most similar (or plagiarized).

### Model 2 - Random Forest
Since each song is a TFIDF vector, cosine similarity is employed to check the similarity of the target song with the rest of the songs in the dataframe. Once the cosine measures are obtained, they are sorted in descending order and the songs with the highest similarity values are the ones most plagiarized with the target song. Going one step further, labels can be added to the songs based on a threshold similarity calculated as the mean of the cosine measures. The TFIDF vectors and the labels serve as input to a random forest ensemble for classification training. A test song is now fed to the model as a TFIDF vector and it returns whether it is plagiarized to the target song or not.

## Generating song lyrics
### Model 1 - ngram model
A n-gram model computes the probability of the next word in a sentence given the previous words that occur in it. By using this concept of conditional probability, the song lyrics can be parsed iteratively to keep count of the next word in a sentence, given the previous words in a python dictionary. These frequency counts can be normalized by the total count to obtain the said probabilities. Once the probabilities are obtained, the n-gram model is given a starting lyric input for which the word corresponding to highest probability is extracted from the dictionary and appended to the sentence. By repeating this procedure iteratively, the lyrics can be generated for a fixed number of words, considering the previous words in the sentence at each iteration as the conditional sample space.

### Model 2 - Long Short Term Memory Network(LSTM)
LSTM is a recurrent neural network capable of learning information for long periods of time in order to provide output for the current iteration. In this example, when the LSTM is given a set of character patterns from the lyrics it is capable of storing this information across iterations in order to predict the next characters for a given random seed of characters. This is the core idea by which it can generate lyrics. Each cell state of a LSTM decides which information to retain and which information to modify. It is a major factor in deciding the accuracy of prediction, which is why when the information is feed-forwarded through successive training cycles or epochs, the LSTM transforms from predicting gibberish characters in the beginning to more organized character prediction after each training cycle.

import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

def create_vocab(words):
    print("Creating vocabulary...")
    stop_words = set(stopwords.words('english') + [".", "!", "&", "@", "#",\
    "''", "``", ";", ",", "https", "amp", "n't", "?", "'s", "'", ":", "“",\
    ")", "(", "%", "...", "-", "http", "$", "'m", '”', "'ll", "1", '—', '‘',\
    "'gt", '--', "'ve", "==", "+", "*", "4", '’', "'re"])
    all_tweet_vocab = []
    for word in words:
        if word.lower().strip() not in stop_words:
            all_tweet_vocab.append(word.lower().strip())
    vocab_dict = {}
    vocab = []
    for word in all_tweet_vocab:
        if word not in vocab_dict:
            vocab_dict[word] = 0
        else:
            vocab_dict[word] = vocab_dict[word] + 1
    for word in vocab_dict:
        vocab.append([word, vocab_dict[word]])
    vocab = sorted(vocab, key=lambda x:x[1])
    vocab = vocab[-500:]
    return vocab

def create_matrix(tweets, vocab):
    print("Creating matrix...")
    arrays = []
    class_array = []
    for tweet in tweets:
        bag_vector = create_vector(tweet[0], vocab)
        arrays.append(bag_vector)
        class_array.append(tweet[1])
    matrix = np.matrix(arrays)
    return matrix, class_array

def create_vector(tweet, vocab):
    bag_vector = np.zeros(len(vocab))
    for word in tweet:
        for i, w in enumerate(vocab):
            if w[0] == word.strip("'"):
                bag_vector[i] += 1
    return bag_vector

def read_in_new():
    words = []
    tweets = []
    for tweet_file in os.listdir("C:\\tweets"):
        print("Reading file {}...".format(tweet_file))
        with open(os.path.join("C:\\tweets", tweet_file)) as tf:
            tweet_data = [json.loads(line) for line in tf]
        for tweet in tweet_data:
            words += (word_tokenize(tweet["content"]))
            tweets.append([word_tokenize(tweet["content"]), tweet_file[-5:-4]])
    with open("C:\\Users/acheson/words.txt", "w", encoding="utf-8") as w:
        for line in words:
            w.write(line + "\n")
    with open("C:\\Users/acheson/tweets.txt", "w", encoding="utf-8") as t:
        for line in tweets:
            t.write(str(",".join(line[0])) + "\t" + str(line[1]) + "\n")
    return words, tweets

def read_in_existing():
    with open("C:\\Users/acheson/words.txt", encoding="utf-8") as w:
        words = w.readlines()
    with open("C:\\Users/acheson/tweets.txt", encoding="utf-8") as t:
        tweet_lines = t.readlines()
    tweets = []
    for line in tweet_lines:
        tweets.append([line.split("\t")[0].split(", "), line.split("\t")[1]])
    return words, tweets


def enter_a_tweet(vocab, classifier):
    tweet = input("Tweet text: ")
    bag_vector = create_vector(tweet, vocab)
    pred = classifier.predict(bag_vector.reshape(1, -1))
    print("I guess this tweet is from a {}".format(pred))

def main():
    #words, tweets = read_in_existing()
    #words, tweets = read_in_new()

    # vocab = create_vocab(words)
    # matrix, class_array = create_matrix(tweets, vocab)
    # np.savetxt("C:\\Users/acheson/matrix.txt", matrix)
    # with open("C:\\Users/acheson/class_array.txt", "w") as c:
    #     for classif in class_array:
    #         c.write(classif)
    matrix = np.loadtxt("C:\\Users/acheson/matrix.txt")
    class_array = []
    with open("C:\\Users/acheson/class_array.txt") as c:
        for letter in c.read():
            class_array.append(letter)

    print("Training model...")
    train_data, test_data, train_truth, test_truth = train_test_split(matrix, class_array)
    # train_data = matrix
    # train_truth = class_array
    classifier = GaussianNB()
    classifier.fit(train_data, train_truth)
    # enter_a_tweet(vocab, classifier)
    pred = classifier.predict(test_data)
    accuracy = accuracy_score(test_truth, pred)
    print("Accuracy of test data:")
    print(accuracy)


main()

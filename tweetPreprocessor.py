# This is a simple tool for pre-processing tweets in large tweet databases.
import math
import re
import string
from sklearn.neighbors import LSHForest
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.feature_extraction import text

class singleTweet:
    """Allows basic operations to be performed on a single tweet. Must import
    regex for this to work properly.
    
                ------------------------------------------------
    Methods:
    
    .tweet               Returns the tweet as a string.
    .strip_non_ascii()   Strips non-ascii characters from the tweet string.
    .strip_punctuation() Strips punctuation. If strip_hashtags is set to False,
                         then it does NOT strip hashtags. Otherwise, strips them.
    .utf8()              Converts the tweet to utf-8 if possible.
    .urls                The regex used to match urls. Modify as needed.
    .strip_links()       Strips links by using regex to sub out any words that
                         contain 'http' or any of a number of suffixes contained
                         in self.urls
    .lowercase()         Converts tweet to all lowercase.
    .strip_newlines()    Strips newline characters from the tweet.
    .strip_and_lower()   Performs all stripping functions (including stripping 
                         non-ascii chars) and lowercases the tweet. Does NOT 
                         convert it to utf-8.
    """
    
    def __init__(self, tweet):
        self.tweet = tweet
        pass

    def strip_non_ascii(self):
        """Replaces all non-ascii characters in the tweet with a space. Returns
        tweet."""

        return ''.join([i if ord(i) < 128 else ' ' for i in self.tweet])

    punctuation = re.compile('[%s]' % re.escape(string.punctuation))
    punctuation_sans_hashtags = re.compile('[%s]' % 
                                           re.escape(string.punctuation.replace('#', '')))

    def strip_punctuation(self, strip_hashtags = True):
        """Removes punctuation. If strip_hashtags = True, then also removes 
        hashtags. Returns tweet."""

        if strip_hashtags:
            self.tweet = self.punctuation.sub('', self.tweet)
        else:
            self.tweet = self.punctuation_sans_hashtags.sub('', self.tweet)

    def utf8(self):
        """Converts tweet to utf-8 encoding, if possible."""

        self.tweet = unicode(self.tweet, "utf-8")

    urls = re.compile(r"""(http\S+ ?)|(\S+\.com?\S+ ?)|(\S+\.ly?\S+ ?)|(\S+\.net?\S+ ?)|(\S+\.gov?\S+ ?)|(\S+\.edu?\S+ ?)|(\S+\.org?\S+ ?)""")

    def strip_links(self):
        """Uses regex to remove anything that looks like a link in the tweet.
        Identifies text that looks like a tweet using URL suffixes."""

        self.tweet = self.urls.sub('', self.tweet)

    def lowercase(self):
        """Converts tweet to all lowercase characters."""

        self.tweet = self.tweet.lower()

    def strip_newlines(self):
        """Strips newline characters from tweet."""

        self.tweet = self.tweet.replace('\n', '')

    def strip_and_lower(self):
        """Performs all stripping functions (including stripping non-ascii chars)
        and lowercases the tweet. Does NOT convert it to utf-8."""

        self.strip_non_ascii()
        self.strip_links()
        self.lowercase()
        self.strip_punctuation()
        self.strip_newlines()


class tweetDatabase:
    """Takes a list of tweets as input and does operations on the entire list.
    Inside, it's doing the operations on batches of the tweets. 

    When initializing, you can set the batch_size. The larger the batch_size, 
    the slower the spam detection will run, but the more spam it will capture. 
    Batch sizes larger than 30,000 tweets don't lead to significantly more spam 
    detected. Batch sizes smaller than 5,000 detect less than 1/3 of all possible 
    spam.
                ------------------------------------------------
    Useful methods (not all methods, bc some are for internal use):
    
    .tweets                 The original tweets that it took as input
    .batch_size             How many tweets the spam detector looks through to 
                            discover spam tweets.
    .strip_and_lower()      If apply_on_copy is set to 1 (default) then it returns 
                            stripped and lowercased tweets. Uses the singleTweet 
                            class's strip_and_lower function. If apply_on_copy is 
                            set to 0, then .tweets_modified becomes the stripped 
                            and lowercased tweets.
    .tweets_modified        Where to find the stripped and lowercased tweets if you 
                            ran .strip_and_lower with apply_on_copy = 0
    .identify_spam()        Run this function to identify spam.
    .spam_tweets            This is the spam tweets discovered by .identify_spam()
    .spam_indices           This is the indices of the spam tweets in the original
                            dataset that you fed into the class.
    .strip_and_lower_spam() Strips and lowercases spam, and sorts it so that 
                            similar-looking spam tweets (sort of similar... just
                            sorted) are side-by-side, so the structure is easier
                            to see. Without this, it's easy to confuse computer-
                            generated spam tweets for human ones.
    .spam_tweets_stripped_and_lowered     The stripped and lowered spam tweets.
                                          To get these, first run strip_and_lower_spam()
    """

    # Initialize some variables that need to be empty but will be updated by
    # functions later on:
    spam_tweets = []
    spam_tweets_stripped_and_lowered = []
    spam_indices = []
    tweets_modified = []

    common_twitter_handles = ['katyperry', 'justinbieber', 'barackobama', \
    'taylorswift13', 'youtube', 'ladygaga', 'rihanna', 'jtimberlake', 'theellenshow', \
    'britneyspears', 'instagram', 'twitter', 'cristiano', 'jlo', 'kimkardashian', \
    'shakira', 'arianagrande', 'selenagomez', 'ddlovato', 'oprah', 'cnnbrk', 'pink', \
    'jimmyfallon', 'harrystyles', 'onedirection', 'liltunechi', 'kaka', 'drake', \
    'officialadele', 'niallofficial', 'aliciakeys', 'billgates', 'brunomars', \
    'pitbull', 'realliampayne', 'kingjames', 'wizkhalifa', 'louistomlinson', \
    'mileycyrus', 'eminem', 'nickiminaj', 'avrillavigne', 'espn', 'neymarjr', \
    'emwatson', 'kevinhart4real', 'cnn', 'davidguetta', 'danieltosh', 'aplusk', \
    'sportscenter', 'nytimes', 'conanobrien', 'actuallynph', 'mariahcarey', 'realmadrid', \
    'xtina', 'zaynmalik', 'srbachchan', 'coldplay', 'fcbarcelona', 'kourtneykardash', \
    'twitteres', 'nba', 'chrisbrown', 'vine', 'beyonce', 'jimcarrey', 'bbcbreaking', \
    'khloekardashian', 'facebook', 'edsheeran', 'iamsrk', 'parishilton', 'ryanseacrest', \
    'iamwill', 'ashleytisdale', 'agnezmo', 'narendramodi', 'leodicaprio', 'ivetesangalo', \
    'tyrabanks', 'alejandrosanz', 'ubersoc', 'mtv', 'blakeshelton', 'snoopdogg', \
    'aamirkhan', 'rickymartin', 'simoncowell', 'kanyewest', 'mohamadalarefe', \
    'beingsalmankhan', '10ronaldinho', 'charliesheen', 'google', 'nfl', 'waynerooney', \
    'claudialeitte', 'dalailam']

    def __init__(self, tweets, batch_size = 50000):
        self.tweets = tweets
        self.batch_size = batch_size

    def strip_and_lower(self, tweets = None, apply_on_copy = 1):
        if apply_on_copy == 0:
            for tweet in range(len(self.tweets)):
                t = singleTweet(self.tweets[tweet])
                t.strip_and_lower()
                self.tweets[tweet] = t.tweet
        else:
            tweets_to_return = []
            for tweet in range(len(tweets)):
                t = singleTweet(tweets[tweet])
                t.strip_and_lower()
                tweets_to_return.append(t.tweet)
            return tweets_to_return

    def tweets_batch_maker(self, all_tweets):
        """Returns a list of tuples. Each tuple is a batch of tweets (length = batch_size)
        and those same tweets, stripped and lowercased. Each batch of tweets is a list."""

        batches = int(math.ceil(len(all_tweets)/float(self.batch_size)))

        list_of_batches = []
        start = 0
        for batch in range(batches):
            end = start + self.batch_size
            if end < len(all_tweets):
                tweet_batch = all_tweets[start:end]
                
            else: 
                tweet_batch = all_tweets[start:]
                
            # Clean them up (lowercase everything, strip links, etc):
            stripped_lowered_tweets = self.strip_and_lower(tweets = tweet_batch, apply_on_copy = 1)
            
            list_of_batches.append((tweet_batch, stripped_lowered_tweets))

            start += self.batch_size

        return list_of_batches

    def single_batch(self, tweets):
        """Performs an approximate nearest neighbors search on tweets in the database
        passed to it. The database must be a list of tweets (text of the tweets only).
        
        Returns the indices of tweets with nearby neighbors (i.e. spam tweets).
        These indices correspond to indices within the batch of tweets fed to
        this function."""

        # Vectorize and fit tree:
        vect2 = CountVectorizer(stop_words = self.common_twitter_handles)
        X2 = vect2.fit_transform(tweets)
        tree2 = LSHForest()
        tree2.fit(X2)

        # Build tree:
        n_neighbors = []
        neighbors_indices = []
        working_batch_size = len(tweets)
        for x in vect2.transform(tweets):
            if len(n_neighbors) % 100 == 0: print "%r tweets analyzed out of %r for this batch" % (len(n_neighbors), working_batch_size)
            neighbors = tree2.radius_neighbors(x, radius = .3)[1]
            n_neighbors.append(len(neighbors[0]))
            neighbors_indices.append(neighbors)

        neighbors_indices = [x for x in range(len(neighbors_indices)) if len(neighbors_indices[x][0]) > 2]

        return neighbors_indices

    def identify_spam(self):
        
        """Operates on the tweets in the database (self.tweets). First, it uses
        the tweets_batch_maker to make batches of tweets. The batch size is set
        when the class is initiated (self.batch_size). Once it has batches of
        tweets, it passes each batch to the single_batch function. It then sets
        the class-level self.spam_indices to the indices of the spam tweets in
        the self.tweets database."""

        batches = self.tweets_batch_maker(self.tweets)
        batch_num = 0
        for batch in batches:
            print "NOW WORKING ON BATCH %r out of %r" % (batch_num + 1, len(batches))
            neighbors_indices = self.single_batch(batch[1])
            self.spam_tweets.extend([self.tweets[t + batch_num*self.batch_size] for t in neighbors_indices])
            self.spam_indices.extend([x + batch_num*self.batch_size for x in neighbors_indices])
            batch_num += 1

    def strip_and_lower_spam(self):
        """Applies teh strip and lower function to the spam tweets and puts the
        stripped and lowered spam tweets into the .spam_tweets_stripped_and_lowered
        object. If sort isn't turned to 0, it'll also sort them for easy viewing."""
        
        for tweet in range(len(self.spam_tweets)):
                t = singleTweet(self.spam_tweets[tweet])
                t.strip_and_lower()
                self.spam_tweets_stripped_and_lowered.append(t.tweet)

        self.spam_tweets_stripped_and_lowered = sorted(self.spam_tweets_stripped_and_lowered)


# Pull some tweets from my mongo database. Note: tweets that are being pulled are all from the same 1-week period.
from pymongo import MongoClient
client = MongoClient()
db = client.tweets
collect = db.test_collection #change this be the right collection!

found_tweets = collect.find()
tweets = []
for found in found_tweets:
    tweets.append(found)
tweets = tweets[200000:350000]
tweets = [x['text'] for x in tweets]

def strip_non_ascii(text):
    """Replaces all non-ascii characters in the tweet with a space. Returns
    tweet."""

    return ''.join([i if ord(i) < 128 else ' ' for i in text])

# batch_sizes_to_try = [50000]
# seconds_per_tweet = []
percent_spam_tweets = []
# for b in batch_sizes_to_try:
    # tweet_db_main = []
    # ind = 0
    # for t in tweets:
    #     ind +=1
    #     if ind == b: break
    #     tweet_db_main.append(t['text'])

import time
start = time.time()
test = tweetDatabase(tweets = tweets, batch_size = 50000)
test.identify_spam()
total_time = time.time() - start
# seconds_per_tweet.append(total_time/float(b))

percent_spam = len(test.spam_tweets)/float(len(tweets))
percent_spam_tweets.append(percent_spam)

test.strip_and_lower_spam()
spam_count = len(test.spam_tweets_stripped_and_lowered)
with open('overnight_test_3_total_spam_in_200k_tweets_with_50k_batches.txt', 'w') as outfile:
    outfile.write('The total amount of spam in 200k tweets with 50k batches was %r\n' %spam_count)
    for x in test.spam_tweets_stripped_and_lowered:
        outfile.write(strip_non_ascii(x) + '\n')

from matplotlib import pyplot as plt

# plt.plot(batch_sizes_to_try, seconds_per_tweet)
# plt.title("Batch sizes vs seconds per tweet", fontsize = 16)
# plt.xlabel("Batch sizes", fontsize = 15)
# plt.ylabel("Seconds per tweet", fontsize = 15)
# plt.savefig("fig1_overnight_2.png")

# plt.plot(batch_sizes_to_try, percent_spam_tweets)
# plt.title("Batch sizes vs percent spam tweets", fontsize = 16)
# plt.xlabel("Batch sizes", fontsize = 15)
# plt.ylabel("Percent spam tweets", fontsize = 15)
# plt.savefig("fig2_overnight_2.png")


# with open('testy.txt', 'w') as outfile:
#     for x in test.spam_tweets_stripped_and_lowered:
#         outfile.write(strip_non_ascii(x) + '\n')

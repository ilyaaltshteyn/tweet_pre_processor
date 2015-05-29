# This is a simple tool for pre-processing tweets in large tweet databases.

class singleTweet:
    """Allows basic operations to be performed on a single tweet. Must import
    regex for this to work properly.
    
                ------------------------------------------------
    Methods:
    
    .tweet               Returns the tweet as a string.
    .strip_non_ascii()   Strips non-ascii characters from the tweet string.
    .strip_punctuation   Strips punctuation. If strip_hashtags is set to False,
                         then it does NOT strip hashtags. Otherwise, strips them.
    .utf8                Converts the tweet to utf-8 if possible.
    .urls                The regex used to match urls. Modify as needed.
    .strip_links()       Strips links by using regex to sub out any words that
                         contain 'http' or any of a number of suffixes contained
                         in self.urls
    .lowercase           Converts tweet to all lowercase."""
    
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

        self.tweet = urls.sub('', self.tweet)

    def lowercase(self):
        """Converts tweet to all lowercase characters."""

        self.tweet = self.tweet.lower()

    def strip_and_lower(self):
        """Performs all stripping functions (including stripping non-ascii chars)
        and lowercases the tweet. Does NOT convert it to utf-8."""

        self.strip_non_ascii()
        self.strip_links()
        self.lowercase()
        self.strip_punctuation()


class tweetDatabase:
    """Takes a list of tweets as input and does operations on the entire list
    all at once."""

    def __init__(self, tweets):
        self.tweets = tweets

    def strip_and_lower(self):


#    def remove_spam

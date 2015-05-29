# tweet_pre_processor
Pre-process single tweets and entire tweet databases, including removing spam.

Provides two classes for manipulating tweets:
singleTweet     Useful for doing basic operations on a single tweet.

tweetDatabase   Useful for applying all of the basic singleTweet operations to
                and entire database of tweets at a time, and also for detecting
                and removing spam tweets from your database. Spam tweets are
                operationalized as tweets that are extremely similar to each
                other in the database, so the spam filter can be applied only
                to the entire database at a time, and not to individual tweets.

Specific methods for each class:
singleTweet
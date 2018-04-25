"""
-*- coding: utf-8 -*-
========================

"""

import json
import sys
from urllib import *
import argparse
from urllib.parse import urlparse, urlencode, parse_qs
from urllib.request import  urlopen
import secret_key

from nltk.tokenize import sent_tokenize, word_tokenize

corpus = []

YOUTUBE_COMMENT_URL = 'https://www.googleapis.com/youtube/v3/commentThreads'
YOUTUBE_SEARCH_URL = 'https://www.googleapis.com/youtube/v3/search'


class YouTubeApi():

    def get_video_comment(self):

        def load_comments(self):
            for item in mat["items"]:
                comment = item["snippet"]["topLevelComment"]
                author = comment["snippet"]["authorDisplayName"]
                text = comment["snippet"]["textDisplay"]
                #print("Comment by {}: {}".format(author, text))
                #print(text)
                corpus.append(text)

               # .sentiment(text)

                if 'replies' in item.keys():
                    for reply in item['replies']['comments']:
                        rauthor = reply['snippet']['authorDisplayName']
                        rtext = reply["snippet"]["textDisplay"]

                    #print("\n\tReply by {}: {}".format(rauthor, rtext), "\n")
                    #print(rtext)
                    #corpus.append(rtext)
                    corpus.append(rtext)

        parser = argparse.ArgumentParser()
        mxRes = 20
        vid = str()
        parser.add_argument("--c", help="calls comment function by keyword function", action='store_true')
        parser.add_argument("--max", help="number of comments to return")
        parser.add_argument("--videourl", help="Required URL for which comments to return")
        parser.add_argument("--key", help="Required API key")

        args = parser.parse_args()

        if not args.max:
            args.max = mxRes

        if not args.videourl:
            exit("Please specify video URL using the --videourl=parameter.")

        if not args.key:
            args.key= secret_key.SECRET_KEY

        try:
            video_id = urlparse(str(args.videourl))
            q = parse_qs(video_id.query)
            vid = q["v"][0]

        except:
            print("Invalid YouTube URL")

        parms = {
                    'part': 'snippet,replies',
                    'maxResults': args.max,
                    'videoId': vid,
                    'textFormat': 'plainText',
                    'key': args.key
                }

        try:

            matches = self.openURL(YOUTUBE_COMMENT_URL, parms)
            i = 2
            mat = json.loads(matches)
            nextPageToken = mat.get("nextPageToken")
            #print("\nPage : 1")
            #print("------------------------------------------------------------------")
            load_comments(self)

            while nextPageToken:
                parms.update({'pageToken': nextPageToken})
                matches = self.openURL(YOUTUBE_COMMENT_URL, parms)
                mat = json.loads(matches)
                nextPageToken = mat.get("nextPageToken")
                #print("\nPage : ", i)
                #print("------------------------------------------------------------------")

                load_comments(self)

                i += 1
        except KeyboardInterrupt:
            print("User Aborted the Operation")

        except:
            print("Cannot Open URL or Fetch comments at a moment")

    def search_keyword(self):

        def load_search_res(self):
            for search_result in search_response.get("items", []):
                if search_result["id"]["kind"] == "youtube#video":
                  videos.append("{} ({})".format(search_result["snippet"]["title"],
                                             search_result["id"]["videoId"]))
                elif search_result["id"]["kind"] == "youtube#channel":
                  channels.append("{} ({})".format(search_result["snippet"]["title"],
                                               search_result["id"]["channelId"]))
                elif search_result["id"]["kind"] == "youtube#playlist":
                  playlists.append("{} ({})".format(search_result["snippet"]["title"],
                                    search_result["id"]["playlistId"]))

            print("Videos:\n", "\n".join(videos), "\n")
            print("Channels:\n", "\n".join(channels), "\n")
            print("Playlists:\n", "\n".join(playlists), "\n")

        parser = argparse.ArgumentParser()
        mxRes = 20
        parser.add_argument("--s", help="calls the search by keyword function", action='store_true')
        parser.add_argument("--r", help="define country code for search results for specific country", default="IN")
        parser.add_argument("--search", help="Search Term", default="Srce Cde")
        parser.add_argument("--max", help="number of results to return")
        parser.add_argument("--key", help="Required API key")

        args = parser.parse_args()

        if not args.max:
            args.max = mxRes

        if not args.key:
            exit("Please specify API key using the --key= parameter.")

        parms = {
                    'q': args.search,
                    'part': 'id,snippet',
                    'maxResults': args.max,
                    'regionCode': args.r,
                    'key': args.key
                }

        try:
            matches = self.openURL(YOUTUBE_SEARCH_URL, parms)

            search_response = json.loads(matches)
            i = 2

            nextPageToken = search_response.get("nextPageToken")

            videos = []
            channels = []
            playlists = []
            print("\nPage : 1 --- Region : {}".format(args.r))
            print("------------------------------------------------------------------")
            load_search_res(self)

            while nextPageToken:
                parms.update({'pageToken': nextPageToken})
                matches = self.openURL(YOUTUBE_SEARCH_URL, parms)

                search_response = json.loads(matches)
                nextPageToken = search_response.get("nextPageToken")
                #print("Page : {} --- Region : {}".format(i, args.r))
                #print("------------------------------------------------------------------")

                load_search_res(self)

                i += 1

        except KeyboardInterrupt:
            print("User Aborted the Operation")

        except:
            print("Cannot Open URL or Fetch comments at a moment")

    def channel_videos(self):

        def load_channel_vid(self):

            for search_result in search_response.get("items", []):
                if search_result["id"]["kind"] == "youtube#video":
                    videos.append("{} ({})".format(search_result["snippet"]["title"],
                                             search_result["id"]["videoId"]))

            print("###Videos:###\n", "\n".join(videos), "\n")

        parser = argparse.ArgumentParser()
        mxRes = 20
        parser.add_argument("--sc", help="calls the search by channel by keyword function", action='store_true')
        parser.add_argument("--channelid", help="Search Term", default="Srce Cde")
        parser.add_argument("--max", help="number of results to return")
        parser.add_argument("--key", help="Required API key")

        args = parser.parse_args()

        if not args.max:
            args.max = mxRes

        if not args.channelid:
            exit("Please specify channelid using the --channelid= parameter.")

        if not args.key:
            exit("Please specify API key using the --key= parameter.")

        parms = {
                   'part': 'id,snippet',
                   'channelId': args.channelid,
                   'maxResults': args.max,
                   'key': args.key
               }

        try:
            matches = self.openURL(YOUTUBE_SEARCH_URL, parms)

            search_response = json.loads(matches)

            videos = []
            i = 2

            nextPageToken = search_response.get("nextPageToken")
            #print("\nPage : 1")
            #print("------------------------------------------------------------------")

            load_channel_vid(self)

            while nextPageToken:
                    parms.update({'pageToken': nextPageToken})
                    matches = self.openURL(YOUTUBE_SEARCH_URL, parms)

                    search_response = json.loads(matches)
                    nextPageToken = search_response.get("nextPageToken")
                    #print("Page : ", i)
                    #print("------------------------------------------------------------------")

                    load_channel_vid(self)

                    i += 1

        except KeyboardInterrupt:
            print("User Aborted the Operation")

        except:
            print("Cannot Open URL or Fetch comments at a moment")

    def openURL(self, url, parms):
            f = urlopen(url + '?' + urlencode(parms))
            data = f.read()
            f.close()
            matches = data.decode("utf-8")
            return matches

def sentiment(comment):
	words = word_tokenize(comment)
	filtered_comment = [w for w in words if not w in stop_words]
	stemmedComment = ps.filteredComment(w)


"""
def NLP():
	all_words = []
	tokenized_sents = [word_tokenize(i) for i in corpus]
	for word in tokenized_sents:
		all_words.append(words.lower())
		print(word)
	all_words = nltk.FreqDist(all_words)
	cropSize = 3000
	trainSize = cropSize * 0.7
	testSize = cropSize * 0.3
	word_Features = list(all_words.keys())[:cropSize]
	import pdb
	pdb.set_trace()
	def find_features(document):
		words = set(document)
		features = {}
		for w in word_Features:
			features[w] = (w in words)
		return features
	featureSets = [(find_features(rev), category) for (rec, category) in documents]
	trainSet = featureSet[:trainSize]
	testSet = featureSet[testSize:]
	classifier = nltk.NaiveBayesClassifier.train(trainSet)
	print ("Accuracy :", (nltk.classify.accuracy(classifier, testSet)))
	classifier.show_most_informative_features(15)
"""

def main():
    y = YouTubeApi()

    if str(sys.argv[1]) == "--s":
        y.search_keyword()
    elif str(sys.argv[1]) == "--c":
        y.get_video_comment()
    elif str(sys.argv[1]) == "--sc":
        y.channel_videos()
    else:
        print("Invalid Arguments\nAdd --s for searching video by keyword after the filename\nAdd --c to list comments after the filename\nAdd --sc to list vidoes based on channel id")

#    NLP()

if __name__ == '__main__':
    main()

#script accepts bytes from the outside world, but processing should be done in unicode. Output in bytes again.
#In the 1980s, almost all personal computers were 8-bit, meaning that bytes could hold values ranging from 0 to 255. 
#ASCII codes only went up to 127, so some machines assigned values between 128 and 255 to accented characters.
# https://docs.python.org/2.7/howto/unicode.html
def safeStr(obj):
    try: return str(obj)
    except UnicodeEncodeError:
        return obj.encode('ascii', 'ignore').decode('ascii')
    except: return ""

'''  
for i in corpus:
    #i = i.encode('utf-8', errors='ignore')
    i = i.encode('ascii', 'ignore').decode('ascii')
    #i = unicode(i, errors='ignore')
    print(i, ", ")
'''
for i in corpus:
    a = safeStr(i)
    print(a, ",")


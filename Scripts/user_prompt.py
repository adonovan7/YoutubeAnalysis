'''
-*- coding: utf-8 -*-
~~~ Andie Donovan~~~~
'''

import subprocess
import sys
import os 

path = '/Users/andiedonovan/myProjects/Youtube_Python_Project/AndiesBranch/'

def mySubprocess(vidName, vidLink):
		print("\nURL for %s video: " %vidName, vidLink)
		args = 'Python3 apiCall.py --c --videourl=' + vidLink + ' >> ' + path + 'data/data.csv'
		print('Runing URL through API Call.')
		print('Hint: Press ^C to quit after a few minutes (wait longer if you would like more comments). \n')
		subprocess.run(args, shell=True)
		sys.exit(1)


def VideoURLfinder():
	print("Please enter either a YouTube video URL or one of the pre-selected videos for analysis.")
	print("To select a tutorial video, please type 'OKGO', 'Trump', or 'Soccer'")
	print("Otherwise, to input your own URL, please type 'URL and press enter'")
	next = input("> ")
	next = next.lower()
	if next == "url":
		print("Please paste URL here:")
		next = input("> ")
		url_link = next
		mySubprocess("personal", url_link)
	elif next == "okgo":
		okgo = "https://www.youtube.com/watch?v=LgmxMuW6Fsc"
		mySubprocess(next, okgo)
	elif next == "trump":
		trump = "https://www.youtube.com/watch?v=Nieiu8tmLIM"
		mySubprocess(next, trump)
	elif next == "soccer":
		soccer = "https://www.youtube.com/watch?v=-sfRVyDHT30"
		mySubprocess(next, soccer)
	else:
		print("")
		print("Error finding video URL, please try again...")
		print("")
		VideoURLfinder()


VideoURLfinder()

if __name__ == '__main__':
    sys.stdout.write("%s\n", VideoURLfinder())

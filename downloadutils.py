#!/usr/bin/env python
import sys
import os
import imagedownloader
import pref_utils
import urllib

# f = open('map_clsloc.txt', 'r')
# contents = f.read()
# contents = contents.split()
# wnid = [contents[i] for i in range(0, 3000, 3)]
# wnid = random.sample(wnid, 200)
# with open('ImageNet_classes_used.txt', 'w') as f:
#     for item in wnid:
#         f.write("%s\n" % item)

f = open('ImageNet_classes_used.txt', 'r')
contents = f.read()
wnid = contents.split()
#
# wnid = ['n02710324', 'n13003061']
downloadImages = True
downloadBoundingBox = False
wnid = [wnid[i] for i in range(127, 200)]

if wnid is None:
    print('No wnid')
    sys.exit()

downloader = imagedownloader.ImageNetDownloader()
i=0
if downloadImages is True:
    for id in wnid:
        print(i+1)
        i = i + 1
        list = downloader.getImageURLsOfWnid(id)
        downloader.downloadImagesByURLs(id, list)

if downloadBoundingBox is True:
    for id in wnid:
        # Download annotation files
        downloader.downloadBBox(id)
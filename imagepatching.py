import operator
from PIL import Image
import glob
from random import *
import re
import os


# NOTE: You will not be able to run this unless you have downloaded the lines directory from IAM
# (the one that has lines stored in .png format) and have that lines folder in the same directory as
# this python file. Once you do that, you should just be able to run this and it will create a
# new folder called data where it will save the image patches.


# get_images is where the bulk of what we have so far is. I have a few other methods to work on associating
# authors and their forms and just the work with directories to actually associate those things with the lines,
# but they aren't completely finished. get_images works fine for the example that I did on the first form.


# This requires you to have the program in the outer directory, next to the folders of the authors which contain
# The folders of each document

# saves cropped images given form and writer serial number as strings
def get_images(form, writer):
    # sets up path for saving cropped images
    # if path does not exist, it makes the path
    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists("data/"+writer):
        os.mkdir("data/" + writer)
    write_path = "data/" + writer + '/'

    #standard new height
    new_height = 113

    #number of patches taken from each line
    patches_per_line = 16

    num = 0
    dir1 = re.search('^.*?(?=-)', form).group()
    for filepath in glob.iglob('lines/' + dir1 + '/' + form + '/*.png'):
        im = Image.open(filepath)

        width, height = im.size
        scale = (float(new_height))/height
        new_width = int(width*scale)

        im = im.resize((new_width, new_height), Image.ANTIALIAS)
        # box = (rand, 0, rand+height, height)
        # region = im.crop(box)

        # crop 113x113 image patches
        for i in range(patches_per_line):
            left = randint(0, new_width-113)
            box = (left, 0, left+113, 113)
            region = im.crop(box)

            # saves patch with an ID that tells the form, the line (from num), and its order of creation (i)
            region.save(write_path + form + '-' + str(num) + '-' + str(i) + '.png', "PNG")
        num +=1
        # # This if statement is temporary: only for testing purposes
        # if num <= 10:
        #     # Instead of .show(), switch it to saving
        #     im.show()


# creates list of tuples (writer, list_of_forms)
# ranks writers by number of forms
def rank_writers():
    global writers
    forms = open("ascii/forms.txt", 'r').readlines()

    writers = dict()

    for line in forms:
        if line[0] != '#':
            line = line.split(' ')
            if line[1] not in writers:
                writers[line[1]] = set()
            writers[line[1]].add(line[0])

    writers = sorted(writers.items(), key=lambda kv: (len(kv[1]), kv[0]))
    writers.reverse()


# def writer_forms():
#     writers = dict()
#     with open("ascii/forms.txt", 'r') as infile:
#         for line in infile:
#             line = line.strip()
#             if line[0] != '#':
#                 line = line.split(' ')
#                 if line[1] in writers:
#                     writers[line[1]].add(line[0])
#                 else:
#                     writers[line[1]] = set()
#     return writers


# def form_to_line(form):
#     lines = set()
#     path = "lines/" + form[0:3] + "/" + form
#
#     return path


def main():
    # rank_writers()
    # for i in range(0, 20): # top 20 writers
    #     for form in writers[i][1]:
    #         get_images(form, writers[i][0])
    #     print(i)
    get_images('a01-000u', 'test')


if __name__ == '__main__':
    main()

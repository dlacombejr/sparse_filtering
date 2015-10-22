__author__ = 'dan'


import boto
import os
from boto.s3.key import Key

os.system("touch test_a.txt")

s3 = boto.connect_s3()
my_bucket = 'danbuckettest'
bucket = s3.get_bucket(my_bucket)

k = Key(bucket)

k.key = '/saved/test_a.txt'
k.set_contents_from_filename('test_a.txt')

os.system("touch test_b.txt")
k.key = '/saved/test_b.txt'
k.set_contents_from_filename('test_b.txt')
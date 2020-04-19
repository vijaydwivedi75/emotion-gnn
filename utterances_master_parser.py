# IEOMCAP Parser file (BETA)

# Improved Script to parse through mutiple files to generate the master file:

# Place this script within the Dialogue transcript folder of the IEOMCAP dataset

# Change and play around with the file IDs of the list to assimilate all the corpus

# Look For the 'utterances_combined.txt' file here

# Author: Suprojit Nandy
import random 
import os
import sys
import csv
import math

def main():

    # Concatenate all the files here::::

    corpus = []
    file_list = ['Ses01F_impro01.txt', 'Ses01F_impro02.txt', 'Ses01F_impro03.txt', 'Ses01F_impro04.txt', 'Ses01F_impro05.txt', 'Ses01F_impro06.txt', 'Ses01F_impro07.txt', 'Ses01F_script01_1.txt', 'Ses01F_script01_2.txt', 'Ses01F_script01_2.txt', 'Ses01F_script01_3.txt']

    list_uttances_final = []

    for file_element in file_list:

        with open (file_element, 'r', newline='') as source:

            for line in csv.reader(source):

                corpus.append(line[0])



    with open('utterances_combined.txt', 'w') as target_file:
        for line in corpus:
            target_file.write(line + '\n')

    print('Look for "Utterances_combined.txt" in the same host directory \n ')

main()

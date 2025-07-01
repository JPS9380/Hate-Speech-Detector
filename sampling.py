#This script is solely created for peforming a random sampling on the "HateSpeechDataset.csv",
#to get a new dataset
import pandas as pd
import numpy as np

data = pd.read_csv("Datasets/HateSpeechDataset.csv")

random_sample = data.sample(n = 3000, random_state=0)

print(random_sample.value_counts())

#storing the dataset to a csv file
random_sample.to_csv("Datasets/Random_Sample.csv")
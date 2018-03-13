import os
import sys
import sklearn

os.chdir("D:\\Project\\Python\\text_classifier")

from sklearn import datasets
cases = sklearn.datasets.load_files("D:/Project/Python/text_classifier/data/train_1", encoding='GBK')

print(len(cases.target_names),len(cases.data),len(cases.filenames))
print("\n".join(cases.data[0].split("\n")[:3]))
print(cases.target_names[cases.target[0]])
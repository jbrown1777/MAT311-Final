import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import re
import numpy as np
from sklearn.model_selection import train_test_split # make training & testing data
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score

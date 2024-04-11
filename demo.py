from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

#load data
column_names = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'Class']
data = pd.read_csv('./magic-telescope-data/magic04.data', names=column_names)

#data cleaning
telescope_shuffle = data.iloc[np.random.permutation(len(data))]
tele=telescope_shuffle.reset_index(drop=True)

#store 2 classes
tele['Class']=tele['Class'].map({'g':0, 'h':1})
tele_class = tele['Class'].values

#Split training, testing, and validation data
training_indices, validation_indices = training_indices, testing_indices = train_test_split(tele.index,
                            stratify= tele_class, train_size=0.75, test_size =0.25)

#Allow for genetic programming to find the best ML model and hyperparameters
tpot = TPOTClassifier(generations=5, verbosity=2)
tpot.fit(tele.drop('Class', axis=1).loc[training_indices].values,
	tele.loc[training_indices, 'Class'].values)

#Score the accuracy
tpot.score(tele.drop('Class', axis=1).loc[validation_indices].values,
	tele.loc[validation_indices, 'Class'].values)

#Export the generated code
tpot.export('pipeline.py')



# Margaret Li
# 1/28/23
# This program uses TensorFlow Decision Forests for 
# protein classification with an input data file

# import necessary libraries
import tensorflow_decision_forests as tfdf
import numpy as np
import pandas as pd
import tensorflow as tf
import io
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

### run the model just once
# read into pd dataframe
proteins = pd.read_csv("light_and_dark_protein_rf.tsv", sep='\t')

# split data into test and train sets
X_train, X_test = train_test_split(proteins, test_size=0.3, 
                                   random_state=42)


## training
# get train and test tensorflow dataframes
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(X_train, label="binary_status")
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(X_test, label="binary_status")

# Specify the model
model = tfdf.keras.RandomForestModel(verbose=2)

# Train the model
model.fit(train_ds)

# summary of model
model.summary()

# testing model and [print out accuracy]
model.compile(metrics=["accuracy"])
print(model.evaluate(test_ds))


## show a decision tree
tfdf.model_plotter.plot_tree(model, tree_idx=0, max_depth=3)


## Plotting feature importance
inspector = model.make_inspector()

# create figure
plt.figure(figsize=(12, 4))

# use SUM_SCORE as metric for feature importance
variable_importance_metric = "SUM_SCORE"

# extract the feature name and importance value
variable_importances = inspector.variable_importances()[variable_importance_metric]
feature_names = [vi[0].name for vi in variable_importances]
feature_importances = [vi[1] for vi in variable_importances]
feature_ranks = range(len(feature_names))

# scale importance values so that they are in the form 0.XX
for i in range(0, len(feature_importances)):
  feature_importances[i] = round(feature_importances[i] / 1000000, 2)

# generate bar graph
bar = plt.barh(feature_ranks, feature_importances, label=[str(x) for x in feature_ranks])
plt.yticks(feature_ranks, feature_names)
plt.gca().invert_yaxis()

# label each bar with values
for importance, patch in zip(feature_importances, bar.patches):
  plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{importance:.2f}", va="top")

# edit graph title and label and show graph
plt.xlabel(variable_importance_metric + " (1x10^6)")
plt.title("Feature Importance by SUM_SCORE (Top 2)")
plt.tight_layout()
plt.show()


### Creating a graph of 6 ROC curves 
###(5 from distinct features and 1 from all features)
# start the graphic for ROC curve. For loop below will add curves to it
plt.figure()
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC plot for canonical predictions")
plt.grid(True)

# list of features for display on ROC curve
features = ["molecular_weight", "gravy", "pI", "rna_detected", "highest_tpm"]

# loop through each feature and add ROC curve to graph
for i in range (0, 6):
  if i < 5:
    # determine which feature will be displayed on ROC curve
    feature = features[i]

    # get subset of data specific to this feature
    subset = proteins.iloc[:,[i, 5]]

    # split data into test and train
    X_train, X_test = train_test_split(subset, test_size=0.3, 
                                    random_state=42)
  
  else:
    # specify all features are used
    feature = "all"

    # split data into test and train
    X_train, X_test = train_test_split(proteins, test_size=0.3, 
                                    random_state=42)
  
  # get the answers. Will be used for ROC curve 
  y_test = X_test["binary_status"]
  
  # turn test and train sets to be tensorflow dfs
  train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(X_train, label="binary_status")
  test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(X_test, label="binary_status")

  # Specify the model
  model_sub = tfdf.keras.RandomForestModel(verbose=2)

  # Train the model
  model_sub.fit(train_ds)

  # compile model and get list of predictions 
  model_sub.compile(metrics=["accuracy"])
  print(model_sub.evaluate(test_ds))
  predictions = list((model_sub.predict(test_ds, verbose=0)))

  # sketch ROC curve for this feature
  fpr, tpr, thresholds = roc_curve(np.ravel(list(y_test)), np.ravel(predictions))
  roc_auc = auc(fpr, tpr)
  plt.plot(fpr,tpr,lw=2,label=feature+" (AUC %0.2f)" % roc_auc)
  plt.legend(loc="lower right")

# display plot
plt.show()


### Creating a graph of 10 ROC curves resulting from models 
### trained by 10 different splits of the input dataset
# start the graphic for ROC curves. For loop below will add curves to it
plt.figure()
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC plot for canonical predictions")
plt.grid(True)


# split the data in ten different ways and generate ROC curves for the 
# resulting test sets. Plot the ROC curves on the same figure
for i in range (0, 10):
  # split data into test and train
  X_train, X_test = train_test_split(proteins, test_size=0.3)
  
  # get the answers. Will be used for ROC curve 
  y_test = X_test["binary_status"]
  
  # turn test and train sets to be tensorflow dfs
  train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(X_train, label="binary_status")
  test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(X_test, label="binary_status")

  # Specify the model.
  model = tfdf.keras.RandomForestModel(verbose=2)

  # Train the model.
  model.fit(train_ds)
  
  # display accuracy of the model and get predictions
  model.compile(metrics=["accuracy"])
  print(model.evaluate(test_ds))
  predictions = list((model.predict(test_ds, verbose=0)))

  # sketch ROC curve for this iteration
  fpr, tpr, thresholds = roc_curve(np.ravel(list(y_test)), np.ravel(predictions))
  roc_auc = auc(fpr, tpr)
  plt.plot(fpr,tpr,lw=2,label="(AUC %0.2f)" % roc_auc)
  plt.legend(loc="lower right")

# display plot
plt.show()


### Get a list of predictions for all entries and export
# turn all data into a tensorflow dataset
all_data = tfdf.keras.pd_dataframe_to_tf_dataset(proteins, label="binary_status")

# get a list of predictions from this full set
predictions = list((model.predict(all_data, verbose=0)))

# remove [] from predicted values
for i in range(0, len(predictions)):
  predictions[i] = str(predictions[i]).replace("[", "")
  predictions[i] = str(predictions[i]).replace("]", "")

# add this new list of predictions to the dataset
proteins["predicted_prob"] = predictions

# check if all seems right
proteins.head()

# write to output file
with pd.ExcelWriter('Light_dark_tfdf_predicted.xlsx') as writer:
    proteins.to_excel(writer)
print('Dataframes written to excel!')
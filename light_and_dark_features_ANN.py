# Margaret Li
# 1/26/23
# This program uses ANN to plot five ROC curves split from the 
# same test derived from the light/dark dataset for each 
# of the five features on the same graph

# import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# read data in
protein_info = pd.read_csv("light_and_dark_protein_rf.tsv", sep='\t')

### plotting feature importance from ANN ###
features=['molecular_weight', 'gravy', 'pI', 'rna_detected_percent', 'highest_tpm']

## define classification parameters 
X = protein_info[features].values
y = protein_info['binary_status'].values


## scaling and splitting 
scaler = StandardScaler()
X = scaler.fit_transform(X)
# random state is specified so each split will be the same
# not totally necessary since split happens before the for loop
X_train_large, X_test_large, y_train, y_test = train_test_split(X, y,
                                                    train_size = 0.75, 
                                                    test_size = 0.25,
                                                    random_state = 42)


## start the graphic for ROC curve. For loop below will add curves to it
plt.figure()
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC plot for canonical predictions")
plt.grid(True)


## ROC Curve
# go through each feature, train, make predictions, and plot
# ROC curve for this feature. Afterwards, go through the 
# save process for all features
for i in range(0, 6):
    # in the last iteration, all features considered
    if i == 5:
        X_train = X_train_large
        X_test = X_test_large
        in_dim = 5
        # for curve legend
        feature = "all"
    # for all other iterations, only consider 1 feature
    else:
        # only look at the ith feature
        X_train = X_train_large[:,[i]]
        X_test = X_test_large[:,[i]]
        in_dim = 1
        # for curve legend
        feature = features[i]

    # check if input out sizes look reasonable
    print(X_train.shape)
    print(y_train.shape)


    ## building the network 
    classifier = Sequential()

    # First layer
    classifier.add(Dense(units=10, input_dim = in_dim, 
                kernel_initializer='uniform', activation='relu'))
    
    # Second layer
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

    # Output layer
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

    # add weights
    classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])

    classifier.summary()

    print("")
    print("Fitting model...")

    # fitting the Neural Network on the training data
    classifier.fit(X_train, y_train, batch_size = 20, epochs = 7)
    
    ##################################################################################
    ## predictions
    # get a list with a prediction for each protein entry
    Predictions=classifier.predict(X_test)
    
    ## sketch ROC curve for this feature or all features
    probabilities_list = list(Predictions)
    fpr, tpr, thresholds = roc_curve(np.ravel(list(y_test)), np.ravel(probabilities_list))
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr,tpr,lw=2,label=feature+" (AUC %0.2f)" % roc_auc)
    plt.legend(loc="lower right")

    # on the iteration where all features are used for model,
    # run predictions on all data and output resulting list
    # of predictions to the dataframe. Export dataframe to
    # an output file
    if i == 5:
        # Predictions on all data
        Predictions=classifier.predict(X)
        # add new column with the identifiers + result of predictions
        protein_info['predicted_prob'] = Predictions
    
        ## write to output file
        with pd.ExcelWriter('Light_dark_ann_predicted.xlsx') as writer:
            protein_info.to_excel(writer)
        print('Dataframes written to excel!')
    
# display plot
plt.show()

# banking-dataset-imbalanced-learn-comparison

## Category:

- Banking

- Finance

- Financial Institute

## Language:

- Python 

## Dataset: 

https://www.kaggle.com/prakharrathi25/banking-dataset-marketing-targets

test.csv & train.csv:

| age | job | marital | education | default | balance | housing | loan | contact | day | month | duration | campaign | pdays | previous | poutcome | y |
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|

## Overview:

“Term deposits are a major source of income for a bank. A term deposit is a cash investment held at a financial institution. Your money is invested for an agreed rate of interest over a fixed amount of time, or term. The bank has various outreach plans to sell term deposits to their customers such as email marketing, advertisements, telephonic marketing, and digital marketing. Telephonic marketing campaigns still remain one of the most effective way to reach out to people. However, they require huge investment as large call centers are hired to actually execute these campaigns. Hence, it is crucial to identify the customers most likely to convert beforehand so that they can be specifically targeted via call.”

## Courtesy:

This project was conducted by me during my last AI Practitioner class at the Center of Talent in AI (CoTAI), Sai Gon, Vietnam right before moving to Toronto, Canada. Hereby, I thank Mr. Hung Ngo - Founder of CoTAI and Mr. Huy Le - my supervisor there who had provided support, vision, guidance, training through this final project for the class. The ideas were completely came up by me. The origin of this dataset can be found on Kaggle. I had applied what I learned from this AI Pracitioner class and other reference/resources to conduct this analysis and upload it to my e-portoflio as a proof of project experience. This project can also be upgraded, expanded as per my noted Room for Improvement below.

## Scope:

Use different classification methods to predict more accurately how many customers can be targeted for the next marketing campaign of banking term deposit sales while avoiding overfitting.

## How to proceed:

Use different models such as Single Decision Tree, Ensemble Classifiers (Bagging, Balanced Bagging, Random Forest, Balanced Random Forest, Easy Ensemble Classifier, RUS Boost), XGBoost, Deep Neural Network to evaluate their performances on both imbalanced Train and Test set while avoid fitting.

Different metrics such as Accuracy (I do not rely on this one), Balanced Accuracy, Geometric Mean, Precision, Recall, F1 score, Confusion Matrix will be calculated.

Find the most important features of this dataset which can be used in policies to predict number of ‘Not Subscribed’ or ‘Subscribed’ customers after applying those new changes.

## Problem statement:

Imbalanced dataset could cause overfitting. We can not rely on Accuracy as a metric for imbalanced dataset (will be usually high and misleading) so we would use confusion matrix, balanced accuracy, geometric mean, F1 score instead. 

Overfitting is when ML models are overly complicated they tend to memorize the training data, resulting in low generalization capability. Techniques to prevent overfitting:

- Feature selection removes unimportant features, hence reducing model complexity.

- Regularization adds a penality term for model complexity to the loss function to optimize, forcing the search algorithm to balance between training accuracy and simplicity of the solution model.

- Early stopping uses validation set to track generalization capability of ML models and stops training further when the solution model starts overfitting. 

- Dropout randomly switches off weights/parameters during training (but uses *all* after training, with [activation scaling](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout)) which effectively adds a regularization term in the training loss.

- Batchnorm *standardizes* the input batch to each layer of a network, making the optimization landscape significantly smoother and allowing for faster training. See [NeurIPS'18](https://papers.nips.cc/paper/7515-how-does-batch-normalization-help-optimization.pdf).

![Imbalanced](https://user-images.githubusercontent.com/70437668/139254440-cd762722-bd78-4342-85d2-1cb474452d6c.jpg)

## Target statement:

Selecting the best classification method which also can avoid overfitting.

## Target achievement:

- RUS Boost is the best classifier with the second-highest Balanced Accuracy=0.84 on both sets, highest Geometric Mean=0.84, F1 score=0.57, Confusion Matrix [[4721 828][132 648]], Feature Importance on ‘duration’ variable > 0.30, ROC AUC=0.92 among Machine, Deep learning classifiers while the 2nd-ranked classifier XGBoost had the highest Balanced Accuracy=0.90 on both sets, Geometric Mean=0.49, low F1 score=0.26, Confusion Matrix [[5477 72][592 188]], Feature Importance on ‘duration’ reaching nearly 0.20, ROC AUC=0.88.

- 443 customers will be targeted for the term deposit campaign instead of 0 with duration = 150 (≤ 162.5) when changing the duration (the most influential feature by RUS Boost Classifier policies to predict potential ‘Subscribed’ customers) to 200 for example (as long as the gini >= 0.472). After modifying the duration to more than 800, 5,445 customers will be targeted.

## Room for improvement:

- Use one or two AutoML tools such as H20.ai to create models (tree, forest) and generate the metrics much faster than the Sckit-Learn, Keras or Tensorflow.

--> I personally like the H20.ai as it consume less time to provide an output of more (all) necessary metrics while the other libraries takes hours to run all classifiers and DNN  although I use mainly CPU on Google Colab Notebook. So, I will run and publish another notebook with H20.ai soon.

Reference: https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science.html

- Apply the Tensorflow's Classification method on the Credit Card Fraud Detection case for this project: https://drive.google.com/file/d/1TrKRiNGxrxkr3i1ugc7DFF7XrfHZ9tmk/view?usp=sharing

- Apply Label Encoding instead of One Hot Encoding to fit Ensemble Classifiers better

- Use Graphviz to draw tree as the roots can be too broad 

--> Graphiz can distribute higher definition for the output picture.

- Try out other methods: Oversampling, SMOTE, etc to avoid Overfitting in different ways

--> This can be found on Kaggle's top voted notebooks

## Dependencies:

imbalanced-learn offers a number of re-sampling techniques commonly used in strong between-class imbalanced datasets. This Python package is also compatible with scikit-learn.

In my case, I got the same error "dot: graph is too large for cairo-renderer bitmaps. scaling by 0.0347552 to fit" all of the time when running the Balanced Random Forest on old version of Colab Notebooks. Here are dependencies of imbalanced-learn:

- Python 3.6+

- scipy(>=0.19.1)

- numpy(>=1.13.3)

- scikit-learn(>=0.23)

- joblib(>=0.11)

- keras 2 (optional)

- tensorflow (optional)

- matplotlib(>=2.0.0)

- pandas(>=0.22)

## Installation:

I had to install imbalanced-learn on the PyPi's repository via pip from the begging and restart the runtime, then start my work:
```pip install -U imbalanced-learn```

### Anaconda Cloud platform: 

```conda install -c conda-forge imbalanced-learn```

Here are Classification methods which I would create and evaluate in my file:

**Single Decision Tree** 

Ensemble classifier using samplers internally:

**Easy Ensemble classifier** [1]

**Random Forest** (This model has 27MB so I will not upload it here.)

**Balanced Random Forest** [2] (This model has 32MB so I will not upload it here.)

**Bagging** 

**Balanced Bagging** [3]

**Easy Ensemble** [4]

**RUSBoost** [5]

**Mini-batch resampling for Keras and Tensorflow (Deep Neural Network - MLP)** [6]


### Table of Contents:

**Comparison of ensembling classifiers internally using sampling**

Ensembling classifiers have shown to improve classification performance compare to single learner. However, they will be affected by class imbalance. This example shows the benefit of balancing the training set before to learn learners. I had created a comparison with non-balanced ensemble methods, XGBoost Classifier and Deep Neural Network Model.

I made a comparison using the balanced accuracy and geometric mean which are metrics widely used in the literature to evaluate models learned on imbalanced set.

**A. Data Engineering:**

A.1. Load libraries

A.2. Load an imbalanced dataset

In the train and test data, features that belong to similar groupings are tagged as such in the feature names (e.g., ind, reg, car, calc). In addition, feature names include the postfix bin to indicate binary features and cat to indicate categorical features. Features without these designations are either continuous or ordinal. Values of -1 indicate that the feature was missing from the observation. The target columns signifies whether or not a claim was filed for that policy holder.

A.3. Data Exploration

A.4. Check Missing or Nan

A.5. Create X, y

A.6. One hot encoding [7] (One hot encoding is not ideally fit for Ensemble Classifiers so next time I will try to use Label Encoding for these kinds of imbalanced dataset instead.)

A.7. Split data

A.8. Unique values of each features

A.9. Draw Pairplot

![Pairplot](https://user-images.githubusercontent.com/70437668/139254519-d6f6f7ea-e090-4571-aecb-cab8b527177a.jpg)

A.10. Confusion Matrix Function

**B. Comparison of Classiers**

- Confusion Matrix

#### Decision Tree 
I used the training of Single Decision Tree classifier as a baseline to compare with other classifiers on this imbalanced dataset.

Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. A tree can be seen as a piecewise constant approximation.

Here are the main differences between nonlinear models MLPs and Decision Trees
- 1) Decision boundary: arbitrary curves for MLPs but *segments* parallel with axes for Decision Trees. For regression surface: arbitrary for MLPs but *block-wise* for Decision Trees; 
- 2) Explainability 

Talking about the Decision Trees' predicted output, I should look at how outputs are calculated and how to measure its quality:
- For regression: average of data subset at leaf nodes; smaller spreading (standard deviation) is better. 
- For classification: majority label of data subset at leaf nodes; smaller entropy is better. 

Balanced accuracy and geometric mean are reported followingly as they are metrics widely used in the literature to validate model trained on imbalanced set.

![DT Classifier](https://user-images.githubusercontent.com/70437668/139254549-6279da64-7e38-41bc-ae21-6034bd351945.jpg)

#### Bagging

A Bagging classifier is an ensemble meta-estimator that fits base classifiers each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction. Such a meta-estimator can typically be used as a way to reduce the variance of a black-box estimator (e.g., a decision tree), by introducing randomization into its construction procedure and then making an ensemble out of it.

A number of estimators are built on various randomly selected data subsets in ensemble classifiers. But each data subset is not allowed to be balanced by Bagging classifier because the majority classes will be favored by it when implementing training on imbalanced data set.

In contrast, each data subset is allowed to be resample in order to have each ensemble's estimator trained by the Balanced Bagging Classifier. This means the output of an Easy Ensemble sample with an ensemble of classifiers, Bagging Classifier for instance will be combined. So an advantage of Balanced Bagging Classifier over Bagging Classifier from scikit learn is that it takes the same parameters and also another two parameters, sampling strategy and replacement to keep the random under-sampler's behavior under control.

#### Balanced Bagging

A Bagging classifier with additional balancing. This implementation of Bagging is similar to the scikit-learn implementation. It includes an additional step to balance the training set at fit time using a given sampler. This classifier can serves as a basis to implement various methods such as Exactly Balanced Bagging, Roughly Balanced Bagging, Over-Bagging, or SMOTE-Bagging.

![Bagging Classifier](https://user-images.githubusercontent.com/70437668/139254552-6370fdaf-aab8-437e-9fea-8ee0b53f9480.jpg)

#### Random Forest 

Random Forest is another popular ensemble method and it is usually outperforming bagging. Here, I used a vanilla random forest and its balanced counterpart in which each bootstrap sample is balanced. 

A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is controlled with the max_samples parameter if bootstrap=True (default), otherwise the whole dataset is used to build each tree.

#### Balanced Random Forest 

A balanced random forest randomly under-samples each boostrap sample to balance it.

![Random Forest   Balanced Random Forest Confusion Matrix ](https://user-images.githubusercontent.com/70437668/139254564-3db7e51b-dbcc-486b-9547-46916b354b03.jpg)

#### Easy Ensemble

In the same manner, Easy Ensemble classifier is a bag of balanced AdamBoost classifier. However, it will be slower to train than random forest and will achieve worse performance

In another way, bag of balanced boosted learners also known as EasyEnsemble. This algorithm is known as EasyEnsemble [1]. The classifier is an ensemble of AdaBoost learners trained on different balanced boostrap samples. The balancing is achieved by random under-sampling.

https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.ensemble.EasyEnsembleClassifier.html

![Easy Ensemble Confusion Matrix](https://user-images.githubusercontent.com/70437668/139254610-8729394f-7de3-4f55-8f46-f5910f2c4847.jpg)

#### RUS Boost

Several methods taking advantage of boosting have been designed. RUSBoostClassifier randomly under-sample the dataset before to perform a boosting iteration. Random under-sampling integrating in the learning of an AdaBoost classifier. During learning, the problem of class balancing is alleviated by random under-sampling the sample at each iteration of the boosting algorithm.

https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.ensemble.RUSBoostClassifier.html?highlight=rusboost#imblearn.ensemble.RUSBoostClassifier

![RUS Boost Confusion Matrix](https://user-images.githubusercontent.com/70437668/139254624-1849c05c-436f-417b-8fbc-3242c3679600.jpg)

#### XGBoost

"Boosting is a strong alternative to bagging. Instead of aggregating predictions, boosters turn weak learners into strong learners by focusing on where the individual models (usually Decision Trees) went wrong. In Gradient Boosting, individual models train upon the residuals, the difference between the prediction and the actual results. Instead of aggregating trees, gradient boosted trees learns from errors during each boosting round.

XGBoost is short for “eXtreme Gradient Boosting.” The “eXtreme” refers to speed enhancements such as parallel computing and cache awareness that makes XGBoost approximately 10 times faster than traditional Gradient Boosting. In addition, XGBoost includes a unique split-finding algorithm to optimize trees, along with built-in regularization that reduces overfitting. Generally speaking, XGBoost is a faster, more accurate version of Gradient Boosting.

Boosting performs better than bagging on average, and Gradient Boosting is arguably the best boosting ensemble. Since XGBoost is an advanced version of Gradient Boosting, and its results are unparalleled, it’s arguably the best machine learning ensemble that we have."

XGBoost provides a highly efficient implementation of the stochastic gradient boosting algorithm and access to a suite of model hyperparameters designed to provide control over the model training process.

https://machinelearningmastery.com/xgboost-for-imbalanced-classification/

![XGB Confusion Matrix](https://user-images.githubusercontent.com/70437668/139254632-f20cfa20-c252-45d1-bf00-3fa43b774b7a.jpg)

- Mean ROC AUC

- Accuracy scores on Train / Test set (We should not rely on accuracy as it would be high and misleading. Instead, we should look at other metrics as confusion matrix, Balanced accuracy, Geometric mean, Precision, Recall, F1-score.

- Classification report (Accuracy, Balanced accuracy, Geometric mean, Precision, Recall, F1-score)

![DT Report](https://user-images.githubusercontent.com/70437668/139254657-b832acc5-b3d0-4781-afe2-3b9e9bd17660.jpg)

![Bagging rerport](https://user-images.githubusercontent.com/70437668/139254665-0b324e49-5409-48fc-87b4-f4fa918ac30f.jpg)

![Balanced Bagging Report](https://user-images.githubusercontent.com/70437668/139254681-ad7ac256-719d-4521-a40b-7f9b58c92b56.jpg)

![Random Forest   Balanced Random Forest Report](https://user-images.githubusercontent.com/70437668/139254694-3a6bf634-8e82-4792-9764-d04f3f7f5c58.jpg)

![Balanced Random Forest Report](https://user-images.githubusercontent.com/70437668/139254710-776e358d-750b-4840-b216-b7c2d3b6f298.jpg)

![Easy Ensemble Report](https://user-images.githubusercontent.com/70437668/139254724-2ae3b8f0-7d4c-42e6-ab5a-2f3344d9cafc.jpg)

![RUS Boost Rerport](https://user-images.githubusercontent.com/70437668/139254754-798b3155-646f-42cd-96ac-1e83258b0910.jpg)

![XGBoost Report](https://user-images.githubusercontent.com/70437668/139254761-1c245552-d73a-446e-9aa6-7181ba5ecc10.jpg)

#### Deep Neural Network

Deep learning is the application of artificial neural networks using modern hardware. It allows the development, training, and use of neural networks that are much larger (more layers) than was previously thought possible. There are thousands of types of specific neural networks proposed by researchers as modifications or tweaks to existing models. Sometimes wholly new approaches. There are three classes of artificial neural networks that I recommend that you focus on in general. They are:

- Multilayer Perceptrons (MLPs)
- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs)

Multilayer Perceptrons, or MLPs for short, are the classical type of neural network. They are comprised of one or more layers of neurons. Data is fed to the input layer, there may be one or more hidden layers providing levels of abstraction, and predictions are made on the output layer, also called the visible layer.

MLPs are suitable for classification prediction problems where inputs are assigned a class or label. They are also suitable for regression prediction problems where a real-valued quantity is predicted given a set of inputs. Data is often provided in a tabular format, such as you would see in a CSV file or a spreadsheet.

In this project, I used MLPs for the Classification prediction problems on the imbalanced Banking Dataset to predict Subscribed or Non-Subscribed customers. They are very flexible and can be used generally to learn a mapping from inputs to outputs. After feeding to the MLP and training them, I would at the common metrics to compare with those of other classifers. It would be worth at least testing the MLPs on this problem. The results can be used as a baseline point of comparison to confirm that other models that may appear better suited and add more value.

**Initiate optimizer**

- Compile model by this command:

```
model.compile(loss='', optimizer='adam', metrics=''
```

When compiling by the above method, optimizer will use the default learning rate by the system. To input a preferred learning rate, I have to initiate optimizer.

```
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, ...

my_optimizer = Adam(learning_rate = 0.01)
# or
my_optimizer = SGD(learning_rate = 0.01, momentum = 0.9)
# or
my_optimizer = RMSprop(learning_rate = 0.01, momentum = 0.9)

model.compile(loss='', optimizer=my_optimizer, metrics='')
```

Common learning rates include 0.1 or 0.01 or 0.001.

Common momentums include 0.9 or 0.8.

**Initiate model**

There are 2 ways to initiate model in Keras:

*   Method 1: use Sequential

```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(16, activation='relu', input_shape=X.shape[1:]))
...
model.add(Dense(1, activation='sigmoid'))
```

*   Method 2: use Model

```
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

input = Input(shape=X.shape[1:])
layer_1 = Dense(16, activation='relu')(input)
layer_2 = Dense(16, activation='relu')(layer_1)
layer_3 = Dense(16, activation='relu')(layer_2)
output = Dense(1, activation='sigmoid')(layer_3)

model=Model(input, output)
```

**Plot charts after training**

```
model.compile(loss='mse', optimizer='', metrics=['mae',RootMeanSquaredError()])

history = model.fit(X_train, y_train, epochs=50)
```
At this time, the object history will save 3 metrics 'loss',  'mae', 'root_mean_squared_error' after 50 epochs. To see the saved metrics, I use this command:

```
print(history.history.keys())
```

To draw chart for loss after 50 epochs, I use this command:

```
plt.plot(history.history['loss']) # loss is one of ouput's metrics generated after calling the above function print()
```

![Draw Learning Curve](https://user-images.githubusercontent.com/70437668/139254770-25cca523-4764-4b8c-95fe-00b17ee3a97a.jpg)

![Deep Neural Network Confusion Matrix](https://user-images.githubusercontent.com/70437668/139254789-1a371afd-e0f8-4ec3-9395-b2e262352b4f.jpg)

![Deep Neural Network Report](https://user-images.githubusercontent.com/70437668/139254837-ffb1b299-32a5-4d7b-9886-9822f0d270b7.jpg)

Reference: https://machinelearningmastery.com/what-is-deep-learning/ 

**C. Feature Importance**

![Feature Importance - Single Decision Tree](https://user-images.githubusercontent.com/70437668/139254858-49a41b1d-b905-4dbd-8c49-fa944a36efb3.jpg)

![Feature Importance - Random Forest](https://user-images.githubusercontent.com/70437668/139254861-75aec9c5-a48a-4200-aab5-4f4f7fcee125.jpg)

![Feature Importance - Balanced Random Forest](https://user-images.githubusercontent.com/70437668/139254870-c201edba-6c32-4d6b-a11a-7ae2b5a4dd3c.jpg)

![Feature Importance - RUS Boost](https://user-images.githubusercontent.com/70437668/139254873-2f9b1054-4290-4fd4-925e-855d109187eb.jpg)

![Feature Importance - XGBoost](https://user-images.githubusercontent.com/70437668/139254877-8f51e7c7-2015-4d1f-86d9-d07c5921e018.jpg)

**D. Heatmap**

![Heatmap](https://user-images.githubusercontent.com/70437668/139254892-e8a2ebcf-9b39-4317-92cf-59e4bdd13db6.jpg)

**E. Draw Single Decision Tree**

![Trees of Decision Tree](https://user-images.githubusercontent.com/70437668/139254926-c650a12c-7868-4d65-bba7-afa698d28919.jpg)

**F. ROC & AUC between Deep Neural Network, Ensemble Classifiers, XGBoost Classifier**

![ROC   AUC between Deep Neural Network, Ensemble Classifiers, XGBoost Classifier](https://user-images.githubusercontent.com/70437668/139254937-8c0765d7-7f11-484f-8663-305348c805c0.jpg)

**G. Predict**

![Predict](https://user-images.githubusercontent.com/70437668/139255130-152c9d0c-430b-48bc-8298-c947a9185257.jpg)

**H. New Policy on Trial:**

H.1 List out

H.2 Implement that New Policy

H.3 Result

- 443 customers will be targeted for the term deposit campaign instead of 0 with duration = 150 (<= 162.5) when changing the duration (the most influential feature by RUS Boost Classifier policies to predict potential ‘Subscribed’ customers) to 200 for example (as long as the gini >= 0.472). After modifying the duration to more than 800, 5,445 customers will be targeted.

![Result](https://user-images.githubusercontent.com/70437668/139254954-8ec8e891-4cbb-475e-a8a3-6af8f0754e2b.jpg)

**References:**

[1] https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.ensemble.EasyEnsembleClassifier.html

[2] https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.ensemble.BalancedRandomForestClassifier.html

[3] https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.ensemble.BalancedBaggingClassifier.html

[4] https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.ensemble.EasyEnsembleClassifier.html

[5] https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.ensemble.RUSBoostClassifier.html

[6] https://imbalanced-learn.readthedocs.io/en/stable/auto_examples/applications/porto_seguro_keras_under_sampling.html?highlight=mini%20batch

[7] https://www.reddit.com/r/MachineLearning/comments/ihwbsn/d_why_onehot_encoding_is_a_poor_fit_for_random/?utm_source=share&utm_medium=ios_app&utm_name=iossmf

[8] Comparison of ensembling classifiers internally using sampling. https://imbalanced-learn.readthedocs.io/en/stable/auto_examples/ensemble/plot_comparison_ensemble_classifier.html#sphx-glr-auto-examples-ensemble-plot-comparison-ensemble-classifier-py

[9] https://machinelearningmastery.com/xgboost-for-imbalanced-classification/

[10] https://www.kaggle.com/tilii7/hyperparameter-grid-search-with-xgboost

[11] https://www.kaggle.com/saxinou/imbalanced-data-xgboost-tunning

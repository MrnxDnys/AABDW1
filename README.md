# AABDW1
 ## Assignment 1

### 1.1 Problem Statement
In this assignment we are tasked with building a model that learns to predict
which customers of a bank-insurer are likely to churn in the near future. The
dataset consists of tabular data and contains customer information of the bankinsurer.
Our model will be evaluated based on the number of true positives in
the 250 test instances with the highest predicted probability of being a churner.

### 1.2 Processing Data Files
The data we have at our disposal for training consists of three separate data
files, each extracted with one month in between. They all contain the same 39
features, including an identifier variable. The third data file also contains a binary
target which equals one if the corresponding customer is identified as a churner
and zero otherwise. The test data was provided in the same 3-month format,
the only difference being that the third file does not contain the target. Data is
gathered for a total of 90,997 individuals, of which 63,697 (70%) is provided in
the training set and the remaining 27,300 (30%) in the test set. However, not
all 63,697 instances will be available for the actual training as a fraction will be
held out for validation.
A first preprocessing step is to merge the separate data files for training into
one dataset. Hereby we can omit redundant columns for which the values did not
change over time, as well as the identifier variable client_id which is unique
and thus does not contain useful information for future predictions. In table 1, a
list of these time-invariant features is provided. The resulting dataframe contains
100 features and the target. The test data files are merged equivalently, except
that there is no target, only the 100 features.

### 1.3 Data Exploration
In this section we conduct an exploratory data analysis to get an understanding
of the data. To show all possible values of a particular feature and their number
of occurrences, we can use the following command:
    print (df[‘ feature_name ’]. value_counts ( dropna = False ))
For the features having a lot of distinct values, it is more useful to analyze
the summary statistics using the command:
    print (df[‘ feature_name ’]. describe ())
A first observation concerning the target variable is that there is a severe class
imbalance, with around 3% of the instances being identified as churners. The proportion
of positive instances in the test set should be equal, given that a stratified
split was used. Secondly, we observe that several features have missing values,
for which Figure 1 visualises the total counts. The feature customer_education
misses values for around 74% of the instances, compared to only 0.4% for
customer_since_all and customer_since_bank. For the other 7 features, the
missing ratio lies somewhere in between. In the test set, it is the same set of
features that contain missing values.
Besides missing values, the summary statistics show that some features also
contain highly unlikely or even impossible values. For example, 10 individuals
have a postal code equal to 0 and 27 individuals are born before 1900. In addition,
there are 138 cases where the date on which the customer became a client is
before the birth date. The difference ranges between 1 month and 22,5 years.
While we could imagine that a bank allows bank accounts for unborn children,
we do not think this would be possible before they are conceived.

### 1.4 Feature Preprocessing
In this part, we discuss the preprocessing steps relating to the features. Some of
these steps are implemented manually while others are performed automatically
by Catboost [8]. CatBoost is an open source library for gradient boosting on
decision trees, on which we will further elaborate in section 1.5.

#### Transforming features and identifying categorical features
First, three features are given as a date (YYYY-MM) which is not a suitable
format to use in a predictive model. To better represent the temporal dimension,
we convert the feature customer_birth_date to the customer’s age measured in
number of months. Equivalently, we transform the features customer_since_all
and customer_since_bank to the number of months that the customer has been
a client for bank/insurance products. This allows us to treat these features as
numerical values instead of categorical ones. We also considered deleting the
instances with very unlikely values for these features from the training data
with the reasoning that we do not want to train the model on incorrect data.
However, this eventually led to worse performance of the model, so we decided
to keep them.
Conversely, the feature customer_postal_code takes numerical values but
should be treated as a categorical variable. Since there are 1034 different postal
codes in the dataset, which would imply 1034 categories, we group the postal
codes by their first digit. This makes sense because the first digit represents the
province. As mentioned before, there are 10 observations with a value of zero
which is not a valid postal code. Therefore, we decide to replace them with NAs.
Figure 2 displays the new categories and their total counts.
Besides customer_postal_code, we also identify all other features, having
a discrete set of possible values with a size of at most 11, as categorical. In
exception, the 46 dummy variables are treated as numerical features despite
representing categorical data. The remaining 45 features are also considered
numerical. Table 2 provides a list of all categorical features.

#### Processing missing values
During the data exploration we observed that 10 features involve missing values.
Two of them are numerical variables, more specifically, the transformed versions
of customer_since_all and customer_since_bank. CatBoost supports
the processing of missing values only for these numerical features. By default,
their missing values are processed as a value lower than the minimum value for
that feature. Moreover, it is guaranteed that a split that separates missing values
from all other values is considered during the training.
Unlike numerical features, CatBoost does not process missing values for categorical
features in any specific way. Therefore, we consider two different strategies
to deal with them. The first strategy is to replace the missing values by
the mode, with the reasoning that it is the most likely value. For example the
feature customer_occupation_code has the value 9 in more than 93% of the
cases. Consequently, we would replace all missing values for that feature with
the value 9. A second strategy is to keep the missing values and treat them as
a separate valid value. We prefer the second strategy as we believe that the fact
that some value is missing can also be meaningful.

#### Transforming categorical features to numerical features
Because most of the algorithms only accept data with numerical values during
the training process, categorical features should be transformed to numerical
features. Usually, this preprocessing step is done manually, for example using
one-hot-encoding. One-hot-encoding transforms each categorical feature to multiple
dummy variables. This yields a completely numerical dataset and can be
fed to the model. However, CatBoost offers categorical feature support automatically
and does not require extra encoding. We only need to convert our data
into CatBoost’s special Pool datatype by using the Pool() class and specify the
names of the categorical features in the cat_features parameter.
Suppose we have some hypothetical data observations for the categorical
feature customer_education and the target. The first step CatBoost executes
is permuting the rows in a random order. Table 3 shows a possible outcome.
The second step is to transform all categorical feature values to numerical
values one at a time using the following formula:
numerical value =
countInClass + prior
totalCount + 1
– countInClass is the total number of times (before the current instance) that
the target equals 1 for the current categorical feature value. For example,
for the instance with index 3 the countInClass equals 1. Before the current
instance (instance with index 3), we observed two instances with the current
6
categorical feature value (NA) out of which one had a target value equal to
one.
– prior is the preliminary value for the numerator. It is determined by the
starting parameters. Let us assume it is equal to 0.5 here.
– totalCount is the total number of instances (before the current instance)
with a categorical feature value equal to the current one. For the instance
with index 3, the totalCount equals 2 because we observed two instances
with the current categorical feature value (NA) before the current instance.
As a result, each instance receives a new numerical value for the categorical
feature. The final result is shown in Table 4.
This is done for every categorical feature specified in the cat_features.
Moreover, CatBoost is also able to search for meaningful combinations of categorical
features and combinations of categorical and numerical features. For
example, instances have values for two categorical features:
customer_relationship_first_month and
customer_relationship_second_month. The set of possible values for each distinct
feature is {couple, single, NA}. If meaningful, CatBoost can create a new
feature that is a combination of both. The possible values for this new aggregated
feature are consequently all permutations of the individual feature values:
{(couple - couple), (single - single), (couple - single), (single - couple), (couple -
NA), ...}. Any number of features can be combined.

### 1.5 Building The Model
A popular approach for classification problems on heterogeneous tabular data
is using gradient boosting. Gradient boosting is a powerful ensemble machine
learning algorithm based on decision trees. Decision trees are added one at a
time to the ensemble and fit using a differentiable loss function and gradient
descent optimization algorithm. There are many implementations of the gradient
boosting algorithm such as the version provided in the scikit-learn library as
well as versions from third-party libraries that provide computationally efficient
alternative implementations of the algorithm and often achieve better results in
practice. Examples include the XGBoost library [3], the LightGBM library [6],
and the CatBoost library [8].
We chose to make use of the CatBoost library developed by Yandex. The
main reason is that it offers several useful features such as the support of categorical
features and missing values as discussed in the preprocessing part. Furthermore,
it has proven to be one of the fastest boosting algorithms for training
and predicting while at the same time achieving very good performance results.

#### Training
Before training the classifier, we split the dataset in a training and a validation
set. For this, we use a random stratified split with 20% of the instances for the
validation set. While the training set is used for the actual training, the validation
set is constructed for tuning the hyperparameters. By tuning the hyperparameters,
we mean finding those hyperparameters that yield the best score on the
validation set in terms of our evaluation metric. To efficiently do that, there exist
multiple methods such as grid search, random search, and Bayesian methods.
However, grid and random search are only slightly more efficient than manual
tuning, as they choose the next parameters completely uninformed by previous
evaluations. As a result, they often spend a considerable amount of time evaluating
unpromising hyperparameters. In contrast, Bayesian optimization methods
keep track of past trials to build a probabilistic model of the objective function,
which they use to select the next most promising set of hyperparameters.
We make use of the open source optimization framework Optuna [1], which is
an implementation of the Bayesian approach. We perform a total of 300 trials
sampling over the following parameters:
objective: x ∈ {Logloss}
colsample by level: {x ∈ R| 0.01 < x < 0.1}
scale pos weight: {x ∈ I | 25 < x < 50}
depth: {x ∈ I | 1 < x < 12}
boosting type: x ∈ {Ordered, Plain}
bootstrap type: x ∈ {Bayesian, Bernoulli,MVS}
Our model will be evaluated based on the number of true positives in the 250
test instances with the highest predicted probability of being a churner. However,
this is not a practical metric to optimize during the parameter tuning and would
probably also not lead to good generalization on unseen data. Therefore, we
select the average precision score as the objective to optimize on the validation
set. The average precision score summarizes the precision-recall curve as the
weighted mean of precisions achieved at each threshold, with the increase in
recall from the previous threshold used as the weight:
average precision score =
P
n(Rn − Rn−1)Pn
Pn and Rn are respectively the precision and recall at threshold n. A useful
property of this metric is that it is scale-invariant, meaning that it measures how
well predictions are ranked, rather than their absolute values. It is able to evaluate
performance irrespective of which threshold is chosen. For informational purposes
we also include the ROC AUC score, which measures the two-dimensional
area under the ROC curve and which holds the same properties. The ROC AUC
score reported on the public leaderboard allows us to compare the performance
of our model on the validation and test set. To avoid overfitting on the training
data, we instantiate the parameter early_stopping_rounds to 100. This causes
the training to stop if the validation error has not decreased since 100 iterations.

### 1.6 Results
The optimized classifier achieves an average precision score of 7.72% on the validation
set and a ROC AUC score of 71.42%. Unfortunately, we cannot report
the results on the test set. The reason is that we were mistaken by thinking that
we could still submit our predictions on the 10th of May. Our last submitted
predictions were established before we implemented the preprocessing steps discussed
above. So, for those predictions we did not yet transform our variables
measured as a date and just treated them as categorical features. Furthermore,
we did not regroup the customer_postal_code values and processed it as a numerical
feature. From now, we will refer to this model as the ‘initial model’ and
to the model with the correct preprocessing steps as the ‘revisited model’.
Despite these missing preprocessing steps, the initial model achieved an average
precision score of 7.47% on the validation set and a ROC AUC score of
71.16%. After submission, we surprisingly showed up at the second place on the
public leaderboard with a public score of 43 and a ROC AUC score of 70.83%.
However, we jumped to the 15th place with a score of 26 when the score was
recalculated based on the hidden data. This raises the following two questions:
– Why did our score on the leaderboard decrease from 43 to 26 for the initial
model?
– How would the revisited model perform on the test set compared to the initial
model?
To answer these questions, we build both models again but now we construct
a test set ourselves for which we have the labels. We hold out 21,4% of the
instances by means of a random stratified split which corresponds to the size of
the test set for the public and hidden leaderboard (both 13,650 instances). This
is however at the expense of less training and validation data. We will assume
that if the revisited model outperforms the initial model on precision@250 for
the same amount of training and validation data, it is very likely that it will also
perform better when using slightly more training and validation data. Table 5
displays the results for the two models, once with and once without constructing
an extra test set.
A first observation is that the initial model’s performance on the self constructed
test set is similar to the performance on the validation set. This indicates
that the model generalizes well on unseen data despite that some features
where not preprocessed the way they should have been. On the self constructed
test set, it achieves a ROC AUC score of 71.56% and a precision@250 of 32.
Comparing this to the scores on the actual test set, for which more training and
validation data is used, we see that the ROC AUC scores are similar, but the
precision@250 is lower than the precision@250 on the public leaderboard and
higher than the precision@250 on the hidden leaderboard. While it is given that
the training data is split from the actual test data in a stratified way, it is not
given that the public and hidden leaderboard (each containing 50% of the test
data) is split in a stratified way. Consequentially, one possible explanation for
the big gap in precision@250 between the public and hidden leaderboard might
be that the hidden leaderboard contains less churners. However, looking at the
scores of other groups on the public and hidden leaderboard, only 25 of the 36
groups decreased in precision@250 score. A second possible explanation for the
difference in precision@250 might be that the distributions of some important
features in the public leaderboard were in our favor, while this was not the case
for the hidden leaderboard. Finally, a third explanation could be that we have
a considerable number of submission entries (29). By trying a lot of different
things and seeing what works best on the public leaderboard, it is possible that
we overfitted on that data.
In an attempt to answer the second question, we study the results of the
revisited model. Similar results on the self constructed test set and the validation
set indicate good generalization on unseen data. All scores are very similar
to the scores of the initial model, except for the precision@250. The revisited
model achieves a score of 37 which is an improvement over 33 for the initial
model. We would thus expect that, with slightly more training and validation
data, the revisited model would also outperform the initial model on the hidden
leaderboard.
It might also be interesting to have a look at the features that contribute the
most to the performance of both models on the self constructed test set. One
method CatBoost uses to calculate feature importance is LossFunctionChange.
The importance of a particular feature is then computed as the difference between
the prediction loss on the test set when using the original model (which
includes that feature) and the prediction loss on the test set when using the
same model without that feature (the model is then built approximately using
the original model with this feature removed from all the trees in the ensemble).
The higher the difference, the more (positive) impact the feature has on the performance.
Figure 3 displays the 10 most influential features together with their
importance score for the initial model (left) and for the revisited model (right).
Both models have the majority of their most important features in common.
It is however remarkable that customer_since_all occurs in the top 10 of the
initial model as it was not converted into the number of months during the
preprocessing stage and just treated as a categorical feature.

### 1.7 Discussion
Customer churn prediction is a classification task which tries to identify customers
that are likely to leave a service. It is a common practice in real-life
Advanced Analytics Assignment 11
business environments because retaining existing customers is often a lot less
expensive than acquiring new ones. Once the customers are found that are at
risk of leaving, proactive marketing actions can be realized in order to decrease
the chances of actual takeoff.
However, this task may be harder than it seems. It depends a lot on how the
target is constructed. To establish the target for our bank-insurer example, several
things had to be taken into account. First of all, because customers rarely
cancel their complete portfolio, they were labeled as churners based on their
relative decrease in liquidity over time. In addition, to reduce the influence of
sudden temporary jumps in the balance which have nothing to do with churning,
observation periods were introduced. Only customers with a stable balance during
the initial observation period were used for the dataset, and the ones with
a continuous decrease in liquidity during the actual observation period were labeled
as the churners. Furthermore, customers for which the balance decreased
in the feature extraction period were removed as well since they were considered
already too far in their churn process, at which point retention-focused offers
would already be too late. This target definition is focused on cases that are
hard to predict. A small modification to relax this definition could be to allow
customers with a decrease in balance between the final two feature extraction
points. As different customers exhibit different behaviours, the total duration of
their churning process can vary from a week to a year. Our proposed modification
in the target would induce more labelled churners. On the one hand, this
hopefully includes a lot slower churners who can still be convinced to stay, but
on the other hand this would probably also include faster churners for which
actions might come too late.
To evaluate models able to predict this target, a suitable metric should be
selected. In general, ROC AUC and PR AUC are examples of good metrics to
evaluate the ranking of the predictions. Once the AUC score is optimized, a
threshold can be set that labels customers with a probability above that threshold
as risky and customer with a probability below that threshold as not risky.
This threshold is chosen depending on the context. For example, suppose that
the bank-insurer wants to offer some incentive to stay to each client that is predicted
to churn. If that incentive requires a high variable cost per customer, the
cost of a false positive (non churner that receives an incentive) might be higher
than the cost of a false negative (churner that does not receive an incentive). In
that case, a higher precision might be preferred and the threshold will be higher.
Conversely, if the incentive only requires a small cost per customer, and the cost
of a false positive is lower than the cost of a false negative, a higher recall might
be favored and the threshold will be lower. Given the probabilities resulting from
optimizing AUC, a threshold can then be chosen that optimizes a metric such
as Fbeta. Fbeta provides a single score that summarizes the precision and recall
and allows to give more attention to one or the other. In the problem of this
assignment, there was only limited time and resources to investigate possible
churners. Therefore a threshold was chosen that labelled the 250 customer with
the highest probability as risky. The model was then evaluated based on the
number actual churners in these 250 predicted churners (number of hits).
In conclusion, churn prediction is important, but not evident. From the 250
customers that had the highest probability to churn according to our initial
model, only 26 were actual churners. Our revisited model seems slightly more
promising, but a precision score of 50% or more is not to be expected.

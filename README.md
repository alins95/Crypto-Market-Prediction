# Crypto Market Prediction

## Data Eng & Feature Selection

We use the provided data in [this](https://www.kaggle.com/competitions/drw-crypto-market-prediction/data) kaggle competition to train our models. The data for the competition was changed a couple of times and
the time stamps for the test data was removed; we used nearest neighbor to reconstruct the temporal order, and the result is stored in [closest rows.csv](https://github.com/alins95/Crypto-Market-Prediction/blob/main/data/closest_rows.csv).
We used this file to sort the test data, and extract the temporal order. 

The data contains the order book data and 780 masked features X1 to X780. The X features have high correlation among themselves, and any model that uses these features without removing these dependencies 
does not preform very well. We used the correlation matrix of these features stored in [corr.csv](https://github.com/alins95/Crypto-Market-Prediction/blob/main/data/Corr.csv) to cluster the masked features. The result is a set of 174 clusters.
This is done in the [data_eng & feature_selection notebook](https://github.com/alins95/Crypto-Market-Prediction/blob/main/codes/data%20eng%20%26%20feature%20selection.ipynb). The clusters are saved in [clusters.csv](https://github.com/alins95/Crypto-Market-Prediction/blob/main/data/clusters.csv).
We average the masked features over each cluster and call the resulting data set X_clustered. This data set is used to train our first model.

## Lag Regression Ensemble Model

The first model is an ensemble of different regression models trained on lagged versions of X_clustered and order book data. We shift this data set based on various temporal patterns; see the [lag_regression_ensemble notebook](https://github.com/alins95/Crypto-Market-Prediction/blob/main/codes/lag_regression_ensemble.ipynb) for a detailed description.
The resulting model is an ensemble of 252 regression models, and the validation loss is about 0.92.

For the second model, we construct 11 different lag-regression-ensemble models, and train a ridge regression on the predictions of these models. For each of these 11 models, we use a different subset of the masked features and the order book data. Each subset is sampled cluster by cluster untill all the masked features are sampled; then we drop the subsets that have less than 10 features in them. The details can be found in [lag_regression_cluster_sampling notebook](https://github.com/alins95/Crypto-Market-Prediction/blob/main/codes/%20lag_%20regression_cluster_sampling.ipynb).
For an individual comparsion of these models against each other and the first model, checkout [improving_the_lag_ensemble notebook](https://github.com/alins95/Crypto-Market-Prediction/blob/main/codes/improving_the_lag_ensemble.ipynb).

In the final stage, we run ridge regression on the results of these 11 models; the details are in the [ensemble_of_lag regressions_with_all_clusters notebook](https://github.com/alins95/Crypto-Market-Prediction/blob/main/codes/ensemble_of_lag%20regressions_with_all_clusters.ipynb).
The validation loss of the final model is 0.98 which is a great improvment in comparison to the first lag_regression model.

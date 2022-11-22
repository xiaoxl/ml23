# Exercises and Projects


```{exercise}
CHOOSE ONE: Please apply the random forest to one of the following datasets.
- the `iris` dataset.
- the dating dataset.
- the `titanic` dataset.

Please answer the following questions.
1. Please use grid search to find the good `max_leaf_nodes` and `max_depth`.
2. Please record the cross-validation score and the OOB score of your model and compare it with the models you learned before (kNN, Decision Trees). 
3. Please find some typical features (using the Gini importance) and draw the Decision Boundary against the features you choose. 
```



````{exercise}
Please use the following code to get the `mgq` dataset.

```{code-block} python
from sklearn.datasets import make_gaussian_quantiles

X1, y1 = make_gaussian_quantiles(cov=2.0, n_samples=200, n_features=2,
                                 n_classes=2, random_state=1)
X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5, n_samples=300,
                                 n_features=2, n_classes=2, random_state=1)
X = np.concatenate((X1, X2))
y = np.concatenate((y1, -y2 + 1))
```
Please build an `AdaBoost` model.
````



```{exercise}
Please use `RandomForestClassifier`, `ExtraTreesClassifier` and `KNeighbourClassifier` to form a voting classifier, and apply to the `MNIST` dataset. 
```

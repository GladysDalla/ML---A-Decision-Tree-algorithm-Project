import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn import datasets

def main():
    # Prompt the user to provide the data IDs
    data_id_1 = input("Enter the data ID for dataset 1: ")
    data_id_2 = input("Enter the data ID for dataset 2: ")

    # Fetch datasets using provided data IDs
    dia = datasets.fetch_openml(data_id=data_id_1)
    dia2 = datasets.fetch_openml(data_id=data_id_2)

    # Define parameters for grid search
    param_grid = {'min_samples_leaf': [1, 5, 10, 20, 50]}

    # Create a list of datasets
    datasets_list = [(dia, 'Dataset 1'), (dia2, 'Dataset 2')]

    # Perform the task for each dataset
    for dataset, name in datasets_list:
        X = dataset.data
        y = dataset.target

        # Convert labels to binary format
        y_binary = (y == '1').astype(int)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42)

        # Initialize models for entropy and gini index criteria
        tree_entropy = DecisionTreeClassifier(criterion="entropy", random_state=42)
        tree_gini = DecisionTreeClassifier(criterion="gini", random_state=42)

        # Grid search with 10-fold cross-validation for entropy criterion
        grid_search_entropy = GridSearchCV(tree_entropy, param_grid, cv=10, scoring='roc_auc')
        grid_search_entropy.fit(X_train, y_train)

        # Grid search with 10-fold cross-validation for gini index criterion
        grid_search_gini = GridSearchCV(tree_gini, param_grid, cv=10, scoring='roc_auc')
        grid_search_gini.fit(X_train, y_train)
        
        # Perform 10-fold cross-validation for entropy criterion
        cv_results_entropy = cross_val_score(grid_search_entropy.best_estimator_, X_train, y_train, cv=10, scoring='roc_auc')

        # Perform 10-fold cross-validation for gini index criterion
        cv_results_gini = cross_val_score(grid_search_gini.best_estimator_, X_train, y_train, cv=10, scoring='roc_auc')

        # Display AUC values in a table
        auc_df = pd.DataFrame({'Entropy': cv_results_entropy, 'Gini Index': cv_results_gini})
        print("\nAUC Values (10 Folds) for", name)
        print(auc_df)
        #print("\nMean AUC Values:")
        #print(auc_df.mean())
        print("\n=================================================\n")

        # Plotting ROC curve
        plt.figure(figsize=(8, 6))

        # Plotting ROC curves for entropy criterion
        fpr_entropy, tpr_entropy, _ = roc_curve(y_test, grid_search_entropy.predict_proba(X_test)[:, 1])
        roc_auc_entropy = auc(fpr_entropy, tpr_entropy)
        plt.plot(fpr_entropy, tpr_entropy, color='orange', lw=2, label='Entropy (AUC = %0.2f)' % roc_auc_entropy)
        # Print optimal parameters and score
        print('Best Parameter For Entropy: {}'.format(grid_search_entropy.best_params_))
        print('Accuracy For Entropy: {}'.format(grid_search_entropy.best_score_))

        # Plotting ROC curves for gini index criterion
        fpr_gini, tpr_gini, _ = roc_curve(y_test, grid_search_gini.predict_proba(X_test)[:, 1])
        roc_auc_gini = auc(fpr_gini, tpr_gini)
        plt.plot(fpr_gini, tpr_gini, color='blue', lw=2, label='Gini Index (AUC = %0.2f)' % roc_auc_gini)
        # Print optimal parameters and score
        print('Best Parameter For Gini: {}'.format(grid_search_gini.best_params_))
        print('Accuracy For Gini: {}'.format(grid_search_gini.best_score_))

        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('1 - Specificity')
        plt.ylabel('Sensitivity')
        plt.title('ROC for ' + name)
        plt.legend(loc="lower right")
        plt.show()

        # Display AUC values in a table
        auc_df = pd.DataFrame({'Entropy': [roc_auc_entropy], 'Gini Index': [roc_auc_gini]})
        print("\nAUC Values for", name)
        print(auc_df)
        print("\n=================================================\n")

if __name__ == "__main__":
    main()
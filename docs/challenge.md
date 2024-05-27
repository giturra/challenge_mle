# Latam ML and LLM Challenge

- Author: Gabriel Iturra Bocaz
- Contact: g.iturra.bocaz at gmail.com

## Gitflow methodology

In order to code all parts of the challenge, I followed the gitflow workflow. For each new feature added to the project, I created a new branch with the naming convention `feature/<brief-description>`, where `<brief-description>` describes in a few words the new feature added to the project. Once a `feature/<brief-description>` branch was finished, it was merged into the develop branch to keep it updated. Finally, when the development of the project was finished, the `develop` branch was merged into the `main` branch, which is the official branch, to release the official version of the project.

However, in a correct gitflow methodology, it is necessary to create a pull request before adding a new feature to the `develop` branch. When the code reviewer approves the changes, these changes are merged into the `develop` branch. These steps were omitted in order to save development time. I just created a pull request to merge the `develop` branch into the `main` branch and create the release version.

## Notebook Exploration and Bug Fixing

### `barplot` errors

There were some bugs related to bar plots; it seems that argument functions changed in previous versions. Now it is necessary to explicitly define the `x` and `y` parameters in the `barplot` function from te `seaborn` package. For example:

Original version:

```python
sns.barplot(flights_by_airline.index, flights_by_airline.values, alpha=0.9)
```

Fixted version

```python
sns.barplot(x=flights_by_airline.index, y=flights_by_airline.values, alpha=0.9)
```

### get_rate_from_column formula error

The `get_rate_from_column` function had an error when computing the rate of flight delays by city. In the original implementation, it was defined like this:

```python
def get_rate_from_column(data, column):
    delays = {}
    for _, row in data.iterrows():
        if row['delay'] == 1:
            if row[column] not in delays:
                delays[row[column]] = 1
            else:
                delays[row[column]] += 1
    total = data[column].value_counts().to_dict()
    
    rates = {}
    for name, total in total.items():
        if name in delays:
            rates[name] = round(total / delays[name], 2)
        else:
            rates[name] = 0
            
    return pd.DataFrame.from_dict(data = rates, orient = 'index', columns = ['Tasa (%)'])
```

The line `round(total / delays[name], 2)` defined the total as the numerator and the delays as the denominator, which should be inverted and multiplied by 100 in order to transform the result value to percentages. Therefore, the implementation was changed to:

```python
def fixed_get_rate_from_column(data, column):
    delays = {}
    for _, row in data.iterrows():
        if row["delay"] == 1:
            if row[column] not in delays:
                delays[row[column]] = 1
            else:
                delays[row[column]] += 1
    total_countries = data[column].value_counts().to_dict()
    rates = {}
    for name, total in total_countries.items():
        if name in delays:
            rates[name] = round((delays[name] / total) * 100, 2)
        else:
            rates[name] = 0

    return pd.DataFrame.from_dict(data=rates, orient="index", columns=["Tasa (%)"])
```

### Model selection

Given that the two models, `LogisticRegression` and `XGBoost`, have similar performance using feature importance and balancing classes, it is difficult to decide which model should be deployed to production. That is why more analysis is needed.

In order to decide which model is better, I will analyzing them using cross-validation, specifically `StratifiedKFold` implemented in scikit-learn librery. This version maintains the distribution of the original dataset in each partition during the training loop.

#### Cross Validation

The result of applying Cross Validation are depicted on this table:


| model                | fold | metric    | on_time_flights | delayed_flights | accuracy | macro avg | weighted avg |
|----------------------|------|-----------|-----------------|-----------------|----------|-----------|--------------|
|Logistic Regression   | 1    |	precision |	0.871340	    | 0.237555	      | 0.543762 |	0.554447 |	0.754125    |
|                      |      | recall    | 0.516503	    | 0.663892	      | 0.543762 |	0.590198 |	0.543762    |
|                      |      | f1-score  | 0.648560	    | 0.349906	      | 0.543762 |	0.499233 |	0.593326    |
|                      |      | support   | 11119.000000	| 2523.000000	  | 0.543762 |	13642.000000 |	13642.000000 |
|                      | 2    | precision | 0.868087	    |  0.252503	      |0.597683  |	0.560295 |	0.754276    |
|                      |      | recall    | 0.597176	    | 0.599921        |	0.597683 |	0.598548 |	0.597683    |
|                      |      | f1-score  | 0.707587	    | 0.355415        |	0.597683 |	0.531501 |	0.642476    |
|                      |      | support   | 11119.000000	| 2522.000000	  | 0.597683 |	13641.000000 |	13641.000000 |
|                      | 3    | precision | 0.876309	    | 0.240593	      | 0.543142 |	0.558451 |	0.758729    |
|                      |      | recall    | 0.511693	    | 0.681728	      | 0.543142 |	0.596710 |	0.543142    |
|                      |      | f1-score  | 0.646110	    | 0.355666	      | 0.543142 |	0.500888 |	0.592390    |
|                      |      | support   | 11118.000000	| 2523.000000	  | 0.543142 |	13641.000000 |	13641.000000 |
|                      | 4    | precision | 0.876827	    | 0.242330	      | 0.547834 |	0.559579 |	0.759472    |
|                      |      | recall    | 0.517989	    | 0.679350	      | 0.547834 |	0.598669 |	0.547834    |
|                      |      | f1-score  | 0.651250	    | 0.357232	      | 0.547834 |	0.504241 |	0.596869    |
|                      |      | support   | 11118.000000	| 2523.000000	  | 0.547834 |	13641.000000 |	13641.000000 |
|                      | 5    | precision | 0.865441	    | 0.246618	      | 0.587127 |	0.556029 |	0.750985    |
|                      |      | recall    | 0.584278	    | 0.599683	      | 0.587127 |	0.591980 | 	0.587127    |
|                      |      | f1-score  | 0.697595	    | 0.349503        |	0.587127 |	0.523549 |	0.633213    |
|                      |      | support   | 11118.000000	| 2523.000000	  | 0.587127 |	13641.000000 |	13641.000000 |
| XGBoost	           | 1	  | precision | 0.874564	    | 0.240733	      |0.547427	 |0.557649	 | 0.757341     |
|                      |      | recall    |	0.519201        |	0.671819      |	0.547427 |	0.595510 |	0.547427    |
|                      |      | f1-score  |	0.651580        |	0.354454      |	0.547427 |	0.503017 |	0.596629    |
|                      |      | support   |	11119.000000	| 2523.000000	  | 0.547427 |	13642.000000 |	13642.000000 |
|                      | 2	  | precision |	0.882459        |	0.248680      |	0.556997 |	0.565569 |	0.765284    |
|                      |      | recall    |	0.526666        |	0.690722      |	0.556997 |	0.608694 |	0.556997    |
|                      |      | f1-score  |	0.659645        |	0.365697      |	0.556997 | 	0.512671 |	0.605299    |
|                      |      | support   |	11119.000000    |	2522.000000   |	0.556997 |	13641.000000 |	13641.000000 |
|                      | 3	  | precision |	0.878946        |	0.243570      |	0.547541 |	0.561258 |	0.761428    |
|                      |      | recall    |	0.515920        |	0.686881      |	0.547541 |	0.601400 |	0.547541    |
|                      |      | f1-score  |	0.650193        |	0.359618      |	0.547541 |	0.504905 |	0.596449    |
|                      |      | support	  | 11118.000000    |	2523.000000   |	0.547541 |	13641.000000 |	13641.000000 |
|                      | 4	  | precision |	0.877905        |	0.244334	  | 0.552086 |	0.561119 |	0.760722    |
|                      |      | recall    |	0.523206        |	0.679350	  | 0.552086 |	0.601278 | 	0.552086    |
|                      |      | f1-score  |	0.655658        |	0.359404	  | 0.552086 | 	0.507531 |	0.600864    |
|                      |      | support   |	11118.000000    |	2523.000000	  | 0.552086 |	13641.000000 |	13641.000000 |
|                      | 5	  | precision |	0.879091        |	0.239826	  | 0.534785 |	0.559458 |	0.760855    |
|                      |      | recall    |	0.497661        |	0.698375	  | 0.534785 |	0.598018 |	0.534785    |
|                      |      | f1-score  |	0.635539        |	0.357042	  | 0.534785 |	0.496290 |	0.584029    |
|                      |      | support   |	11118.000000    |	2523.000000	  | 0.534785 |	13641.000000 |	13641.000000 |


The above table in the image presents the performance metrics of two binary classification models, Logistic Regression and XGBoost, evaluated using a 5-fold stratified cross-validation approach. The positive class represents delayed flights, which are much less frequent than the negative class (on-time flights), indicating a class imbalance problem. The key metrics shown in the table are accuracy, precision, recall, F1-score, support, macro average, and weighted average for each fold in each classifier.

##### Metrics and Results

- Accuracy: Measures the proportion of correctly classified instances out of the total instances.

- Precision: The ratio of true positive predictions to the total predicted positives. It indicates how many of the predicted delayed flights were actually delayed.

- Recall: The ratio of true positive predictions to the total actual positives. It shows how many of the actual delayed flights were correctly identified.

- F1-Score: The harmonic mean of precision and recall, providing a balance between the two metrics.

- Support: The number of actual occurrences of each class in the dataset.

- Macro Average: The average of the metric (precision, recall, F1-score) calculated for each class independently.

- Weighted Average: The average of the metric (precision, recall, F1-score) weighted by the number of instances in each class.

##### Results for Each Model

##### Logistic Regression

- Fold 1 to 5: The metrics for each fold are provided, showing variations in precision, recall, F1-score, and accuracy.
- Macro Average and Weighted Average: These averages are calculated across all folds, providing an overall performance measure.

##### XGBoost

- Fold 1 to 5: Similar to Logistic Regression, the metrics for each fold are provided.
- Macro Average and Weighted Average: These averages are calculated across all folds, providing an overall performance measure.

##### Conclusion

To determine which model performs better, comparing the overall performance metrics across the two models. 

- Accuracy: XGBoost generally shows higher accuracy across the folds compared to Logistic Regression.
- Precision: XGBoost tends to have higher precision, indicating better performance in predicting delayed flights correctly.
- Recall: XGBoost also shows higher recall, meaning it is better at identifying actual delayed flights.
- F1-Score: The F1-score, which balances precision and recall, is consistently higher for XGBoost.
- Macro and Weighted Averages: Both the macro and weighted averages for precision, recall, and F1-score are higher for XGBoost, indicating better overall performance.

Given these observations, XGBoost appears to be the better-performing model for this particular problem. It consistently outperforms Logistic Regression in terms of accuracy, precision, recall, and F1-score across all folds, making it more suitable for handling the class imbalance and providing more reliable predictions for delayed flights.

In order to support this findings the conclusions in [1] stablish that the recall is most representative metric to measure the performance on this task, thereby if I take average recall for each kfold: 

|model                | recall_on_time_flights  | recall_delayed_flights |
|---------------------|-------------------------|------------------------|
| Logistic Regression |	0.545528                |	0.516531             |
| XGBoost             |	0.644915                |	0.68542              |


Selected model: <strong> XGBoostClassifier </strong>

### Other recomemded features

There are some other recommended features; however, since the Data Scientist team did not use them, I did not analyze them either. These features are *high_season* and *period_day*. Since the challenge specifically mentioned that I did not have to explore the impact of these features on the model performance, I did not include them in the prediction


## Part I: Model Implementation

### Config file

### Implementation details

## Part II: API Implementation

### Config file

### Implementation details

## Part III: Deployment

I deployed the model in Google Cloud by creating a Docker image using the Dockerfile provided in the project. The project was deployed in Python 3.10.

- In the Dockerfile, there are comments explaining what each command does to deploy the application.
- The model is available at [https://fastapi-model-koejv465lq-uc.a.run.app](https://fastapi-model-koejv465lq-uc.a.run.app).
- The commands for deploying the model to GCP are in a cd.yaml file in order to follow the CI/CD pipeline. The explanation of the commands is found in the next section.
- I ran the deployed model using the stress test without any failed requests. This is an example of its output:

```
Response time percentiles (approximated)
 Type     Name                                                              50%    66%    75%    80%    90%    95%    98%    99%  99.9% 99.99%   100% # reqs
--------|------------------------------------------------------------|---------|------|------|------|------|------|------|------|------|------|------|------|
 POST     /predict                                                          160    190    220    230    520    600    650    680   4500   6200   6200   9040
--------|------------------------------------------------------------|---------|------|------|------|------|------|------|------|------|------|------|------|
 None     Aggregated                                                        160    190    220    230    520    600    650    680   4500   6200   6200   9040

```

## Part VI: CI/CD Pipeline

### Continual Integration

Here a completed explination of each command: 

1. **name: CI Pipeline**
    - This sets the name of the workflow as "CI Pipeline."
2. **on:**
    - push:
        - **develop:** The workflow is triggered when code is pushed to the develop branch.
    - pull_request:
        - branches:
            - **develop:** The workflow is also triggered when a pull request is made to the develop branch.
3. **jobs:**
    - build-and-test:
        - runs-on: ubuntu-latest: Specifies that the job will run on the latest version of an Ubuntu runner provided by GitHub.
4. **steps:**
    - name: Checkout code
        - uses: actions/checkout@v3: This step uses the actions/checkout action to check out the repository's code to the runner, allowing subsequent steps to access the code.
    - name: Set up Python
        - uses: actions/setup-python@v4
        - with: python-version: '3.10': This step sets up Python 3.10 on the runner using the actions/setup-python action.
    - name: Install dependencies
        - run: make install: This step runs the make install command to install the project's dependencies. The make install command is typically defined in a Makefile.
    - name: Run tests
        - run: |
            - make model-test
            - make api-test
            - make stress-test
            This step runs a series of tests defined in the Makefile. The make model-test, make api-test, and make stress-test commands execute different test suites for the project.
    - name: Upload build artifacts
            - uses: actions/upload-artifact@v2
                - with:
                - name: build-artifacts
                - path: .
            This step uses the actions/- upload-artifact action to upload build artifacts. The artifacts are named build-artifacts and the path . indicates that all files in the current directory will be included.

### Continual Deployment

## References

[1] Khaksar, Hassan, and Abdolrreza Sheikholeslami. "Airline delay prediction by machine learning algorithms." Scientia Iranica 26.5 (2019): 2689-2702.
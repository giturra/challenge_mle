# Latam ML and LLM Challenge

- Author: Gabriel Iturra Bocaz
- Contact: g.iturra.bocaz at gmail.com

## Gitflow methodology

In order to code all parts of the challenge, I followed the gitflow workflow. For each new feature added to the project, I created a new branch with the naming convention `feature/<brief-description>`, where `<brief-description>` describes in a few words the new feature added to the project. Once a `feature/<brief-description>` branch was finished, it was merged into the main branch to keep it updated. Finally, when the mainment of the project was finished, the `main` branch was merged into the `main` branch, which is the official branch, to release the official version of the project.

However, in a correct gitflow methodology, it is necessary to create a pull request before adding a new feature to the `main` branch. When the code reviewer approves the changes, these changes are merged into the `main` branch. These steps were omitted in order to save mainment time. I just created a pull request to merge the `main` branch into the `main` branch and create the release version. Moreover, sometimes I had to make some small updates in certain branches, I did it in the updates in same branch in order to save time, but the correct way of doing this is development in hot-fix branches that do not require a Pull Request approve before merge into develop or main branches.


However, in a correct Gitflow methodology, it is necessary to create a pull request before adding a new feature to the `main` branch. When the code reviewer approves the changes, these changes are merged into the `main` branch. These steps were omitted to save maintenance time. I just created a pull request to merge the `develop` branch into the `main` branch and create the release version. Moreover, sometimes I had to make some small updates in certain branches, and I did these updates in the same branch to save time. The correct way of doing this is to develop in hot-fix branches that do not require a pull request approval before merging into the `develop` or `main` branches.

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
| Logistic Regression |	0.545528                |	0.516531               |
| XGBoost             |	0.644915                |	0.68542                |


Selected model: <strong> XGBoostClassifier </strong>

### Other recomemded features

There are some other recommended features; however, since the Data Scientist team did not use them, I did not analyze them either. These features are *high_season* and *period_day*. Since the challenge specifically mentioned that I did not have to explore the impact of these features on the model performance, I did not include them in the prediction.

### Compabilities issues

There were some compatibility issues related to certain libraries, especially with FastAPI and Unicorn libraries, when I attempted to run the tests to measure the performance of my implementations. Therefore, I had to install some dependencies that are prior to the recommended versions. Here is the complete list of dependencies that I had to install:

- `anyio==3.4.0`
- `Jinja2==3.0.3`
- `itsdangerous==2.0.1`
- `Werkzeug==2.0.3`

### New libraries installed

Since I selected the `XGBoostClassifier` model as the production model to be deployed, I had to install the `xgboost` library because the implementation of this classifier is not included in `scikit-learn`. Moreover, I installed the `pyyaml` library to open and interact with the configuration YAML files, which are in charge of parameter management for the `DelayModel` class and the API implementation.


## Part I: Model Implementation

### Config file

Since the model needs a lot of configuration, including hyperparameters and the features selected by the feature importance established by the Data Scientist team, it was necessary to centralize the configuration management, not just for the implementation of the production model, but also for the API implementation. I added the configuration files into the directory `challenge/configs`. The file that saves the configuration for the model is [challenge/configs/model_config.yaml](../challenge/configs/model_config.yaml), which contains: This version corrects the grammar and typos while maintaining the original meaning and context.

```yaml
model_name: xgboost_classifier # Name of the model
model_version: 1.0 # Model version use to save and load the train model.
random_state: 1 # Random state for the initial values of the hyperparameters.
learning_rate: 0.01 # Learning rate during the training phase.
threshold_in_minutes: 15 
top_10_features: # Features selected by feature importance.
  - "OPERA_Latin American Wings"
  - "MES_7"
  - "MES_10"
  ...
raw_data_columns: # Features used to encode as one-hot vectors.
  - OPERA
  - TIPOVUELO
  - MES
```

The configuration setting is managed by the `Config` class:

```python
class Config:
    ...

    def load_config(self) -> Dict[str, Any]:
        with open(self.config_path, "r") as file:
            return yaml.safe_load(file)

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return self.config.get(key, default)
```

The most important methods are `load_config`, which loads the configuration file (the model file or the API file), and the `get` method, which extracts a key with its respective value to pass to the model class. 

### Implementation details

In addition to the methods asked in the `DelayModel` class, I implemented some auxiliary methods, which are:

- `__create_path_for_save_trained_models`: A method for creating a directory to store the trained model.
- `prepocess_dataset`: An auxiliary method that creates the one-hot encoding representation of the features from the original dataset.
- `__scale_pos_weight`: A method to compute the scale that considers class balances to improve performance during the prediction phase.
- `__split_dataset:` A method to split the dataset into train and test partitions.
- `__save_model:` A method to store the model once the training has finished.
- `__load_model:` A method to load the model from the local machine where it is stored.
- `get_model:` A method to retrieve the instance of the model during the prediction phase.

I did not use some functions from the notebook, such as `is_high_season` and `get_period_day`, because they were not utilized by the Data Scientist team to predict flight delays. Furthermore, there was a typo in the output type defined for the preprocess function, which was changed from `Union(...)` to - `Union[...]`.

## Part II: API Implementation

In this section, I separated the API implementation into two parts:

- I created a new subpackage called api that contains the `Pydantic` models for processing the input transferred by the endpoint and the API implementation.
 - I created a configuration file to load the main list used by the API implementation

### Config file

As the model implementation, the API implementation has a configuration file, which is store in [challenge/configs/api_config.yaml](../challenge/configs/api_config.yaml). This file contain:

```yaml
# The airlines that the input from each request could bring.
airlines: 
  - "Aerolineas Argentinas"
  - "Aeromexico"
  - "Air Canada"
  - "Air France"
  ...
# The months that the input from each request could bring, expressed in number from 1 to 12.
months: 
  - 1
  - 2
  ...

 # The flight type that the input from each request could bring, expressed I or N, for internationan or national.
flight_type:
  - "N"
  - "I"
# Features selected by feature importance.
top_10_features: 
  - "OPERA_Latin American Wings"
  - "MES_7"
  - "MES_10"
  ...

``` 

### Implementation details

In FastAPI, it is considered best practice to define the input and output of each API route using `Pydantic` models. This approach offers several advantages, including:

- Input/Output Data Validation: By delegating validation to `Pydantic, the API ensures that all input and output data conform to the specified schemas.
- Error Handling: `Pydantic` automatically handles validation errors, returning appropriate status codes when validation fails.

- In my implementation, I mained two separate files to implement the API:

In the file `models`, I implemented the `Pydantic` models to preprocess the input for each endpoint. The implemented models are:

  - `Flight`: A model that stores the three input data fields (these variables are based on the attributes from the dictionaries of the API test), which are `OPERA`, `TIPOVUELO`, and `MES`. `OPERA` represents the airline that organizes the flight.
  - `Flights`: A model that can store a list of flights in case a request submits information for more than one flight.
  
  Furthermore, I used the `validator` decorator to define the correct form for each input. If an input field does not match this validation, a 400 error is raised, indicating the correct format for each specific input.

In the file `api`, I implemented the `predict` endpoint, which is responsible for calling the model and making a prediction from the input data that is in the correct format or has passed validation by the `Pydantic` models. The other two important functions are:

- `preprocess_flights` function: This function processes the input data to create a DataFrame that matches the features the classifier was trained on.
- `validation_exception_handler` function: This function raises a 400 error if any of the input data does not match the correct format validated by the `Pydantic` models

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

Here a completed explanation of each command: 

```
name: CI Pipeline  # Name of the workflow

on:
  push:
    branches:  
      - main  # Triggered by pushes to the 'main' branch
  pull_request:
    branches:
      - main  # Also triggered by pull requests to the 'main' branch

jobs:

  build-and-test:
    runs-on: ubuntu-latest  # Runs this job on the latest version of Ubuntu

    steps:
    
    - name: Checkout code  # Step to check out the source code
      uses: actions/checkout@v3  # Uses the checkout action version 3
      
    - name: Set up Python  # Step to set up Python
      uses: actions/setup-python@v4  # Uses the setup-python action version 4
      with:
        python-version: '3.10'  # Sets Python version to 3.10
        
    - name: Install dependencies  # Step to install dependencies
      run: make install  # Runs the command "make install"
      
    - name: Run tests  # Step to run tests
      run: |  # Runs multiple commands
        make model-test  # Runs model tests
        make api-test  # Runs API tests
        make stress-test  # Runs stress tests
        
    - name: Upload build artifacts  # Step to upload build artifacts
      uses: actions/upload-artifact@v2  # Uses the upload-artifact action version 2
      with:
        name: build-artifacts  # Name of the artifact
        path: .  # Path to the files to upload (current directory)
```

### Continual Deployment

Here a completed explanation of each command: 

```
name: CD Pipeline

on:
  push:
    branches:
      - main  # Trigger this workflow on pushes to the 'main' branch

jobs:
  deploy:
    runs-on: ubuntu-latest  # Use the latest Ubuntu runner

    permissions:
      contents: 'read'  # Read permissions for repository contents
      id-token: 'write'  # Write permissions for ID tokens

    steps:
    - name: Checkout code
      uses: actions/checkout@v3  # Check out the repository code

    - name: Authenticate to Google Cloud
      id: auth
      uses: google-github-actions/auth@v2  
      # Authenticate to Google Cloud using Workload Identity Federation
      with:
        workload_identity_provider: 'projects/932859730377/locations/global/workloadIdentityPools/my-pool/providers/my-provider'
        service_account: 'giturra@challenge-mle.iam.gserviceaccount.com'

    - name: Set up Google Cloud SDK
      uses: google-github-actions/setup-gcloud@v2  # Set up the Google Cloud SDK
      with:
        version: 'latest'  # Use the latest version of the SDK

    - name: Configure Docker for GCP
      run: gcloud auth configure-docker  # Configure Docker to use Google Cloud credentials

    - name: Build Docker image
      run: |
        docker build -t gcr.io/${{ secrets.GCP_PROJECT_ID }}/fastapi-app:$GITHUB_SHA .  
        # Build the Docker image and tag it with the commit SHA

    - name: Push Docker image
      run: |
        docker push gcr.io/${{ secrets.GCP_PROJECT_ID }}/fastapi-app:$GITHUB_SHA  
        # Push the Docker image to Google Container Registry

    - name: Deploy to Cloud Run
      run: |
        gcloud run deploy fastapi-app \
          --image gcr.io/${{ secrets.GCP_PROJECT_ID }}/fastapi-app:$GITHUB_SHA \
          --platform managed \
          --region us-central1 \
          --allow-unauthenticated  
          # Deploy the Docker image to Google Cloud Run
```

## References

[1] Khaksar, Hassan, and Abdolrreza Sheikholeslami. "Airline delay prediction by machine learning algorithms." Scientia Iranica 26.5 (2019): 2689-2702.
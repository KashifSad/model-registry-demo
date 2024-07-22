from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pandas as pd
import mlflow


df = pd.read_csv("https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv")


x = df.drop('Outcome',axis=1)
y = df["Outcome"]

x_train, x_test, y_train, y_test = train_test_split(x,y , test_size=0.2, random_state=42)

rf = RandomForestClassifier(random_state=42)


param_grid = {
    'n_estimators': [10,50,100],
    'max_depth': [None, 10,20,30]
}




grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)


mlflow.set_experiment('diabetes-rf-hp')

with mlflow.start_run(description="Best hyperparameter trained RF model") as parent:

    grid_search.fit(x_train,y_train)


    for i in range(len(grid_search.cv_results_['params'])):
        print(i)
        with mlflow.start_run(nested=True) as child:
             mlflow.log_params(grid_search.cv_results_['params'][i])
             mlflow.log_metric("accuracy",grid_search.cv_results_['mean_test_score'][i])



    best_params = grid_search.best_params_
    best_score = grid_search.best_score_


    mlflow.log_params(best_params)

    mlflow.log_metric("accuracy",best_score)

    train_df = x_train.copy()
    train_df["Outcome"] = y_train

    train_df = mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_df,"training")

    test_df = x_test.copy()
    test_df["Outcome"] = y_test

    test_df = mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_df,"validation")
    

    mlflow.log_artifact(__file__)

     

    signature = mlflow.models.infer_signature(x_train, grid_search.best_estimator_.predict(x_train))
    mlflow.sklearn.log_model(grid_search.best_estimator_,"Random forest",signature=signature)

    mlflow.set_tag("author","KAshif Sadiq")   
    
    print(best_params)
    print(best_score)
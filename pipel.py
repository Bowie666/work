import os
import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def model_validation(**context):
    """Model validation"""

    module_path = "/home/ec2-user/work"
    # module_path = os.path.dirname(__file__)
    filename = os.path.join(module_path, 'iris.csv')
    # filename = '/Users/bowie/Documents/vsfile/u-test/iris.csv'
    names = ['separ-length','separ-width','petal-length','petal-width','class']
    dataset = pd.read_csv(filename, names=names)
    array = dataset.values
    x = array[:, 0:4]
    y = array[:, 4]
    # ValueError: could not convert string to float: 'setosa'

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    rfc = DecisionTreeClassifier().fit(X_train, y_train)
    pred_rfc = rfc.predict(X_test)

    # accuracy = accuracy_score(y_test, preds)
    # name = 'iris-model-{}.model'.format(int(time.time()))
    # joblib.dump(dt, name)


    # rfc = joblib.load(pathlib.Path(run_path, "random_forest_model.sav"))

    # pred_rfc = rfc.predict(X_test)

    # Print classification report
    print("\n" + classification_report(y_test, pred_rfc))

    mlflow.set_tracking_uri("http://161.189.107.57:5000")

    with mlflow.start_run():
        (rmse, mae, r2) = eval_metrics(y_test, pred_rfc)
        mlflow.log_params(rfc.get_params())
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(
            sk_model=rfc,
            artifact_path="model",
            registered_model_name="IrisModel",
        )

if __name__ == "__main__":
    model_validation()
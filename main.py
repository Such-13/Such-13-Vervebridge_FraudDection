import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    df_fraud = data.loc[data.fraud == 1]
    df_non_fraud = data.loc[data.fraud == 0]

    data_reduced = data.drop(['step','zipcodeOri', 'zipMerchant'], axis=1)
    
    col_categorical = data_reduced.select_dtypes(include=['object']).columns
    mappings = {}
    for col in col_categorical:
        data_reduced[col] = data_reduced[col].astype('category')
        mappings[col] = dict(enumerate(data_reduced[col].cat.categories))
        data_reduced[col] = data_reduced[col].cat.codes

    X = data_reduced.drop(['fraud'], axis=1)
    y = data['fraud']
    
    return X, y, df_fraud, df_non_fraud, mappings

def plot_count(data):
    sns.set()
    sns.countplot(x="fraud", data=data)
    plt.title("Count of Fraudulent Payments")
    plt.show()

def plot_histogram(df_fraud, df_non_fraud):
    plt.hist(df_fraud.amount, alpha=0.5, label='fraud', bins=100)
    plt.hist(df_non_fraud.amount, alpha=0.5, label='nonfraud', bins=100)
    plt.title("Histogram for fraud and nonfraud payments")
    plt.ylim(0, 10000)
    plt.xlim(0, 1000)
    plt.legend()
    plt.show()

def plot_roc_auc(y_test, preds):
    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def split_data(X, y):
    return train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True, stratify=y)

def train_knn(X_train, y_train):
    knn = KNeighborsClassifier(n_neighbors=5, p=1)
    knn.fit(X_train, y_train)
    return knn

def train_rf(X_train, y_train):
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, verbose=1, class_weight="balanced")
    rf_clf.fit(X_train, y_train)
    return rf_clf

def train_xgb(X_train, y_train):
    XGBoost_CLF = xgb.XGBClassifier(
        max_depth=6,
        learning_rate=0.05,
        n_estimators=400,
        objective="binary:logistic",  # Use "binary:logistic" for probability output
        booster='gbtree',
        n_jobs=-1,
        random_state=42,
        verbosity=1
    )
    XGBoost_CLF.fit(X_train, y_train)
    return XGBoost_CLF

def create_ensemble(knn, rf_clf, XGBoost_CLF):
    estimators = [("KNN", knn), ("rf", rf_clf), ("xgb", XGBoost_CLF)]
    ens = VotingClassifier(estimators=estimators, voting="soft", weights=[1, 4, 1])
    return ens

def train_ensemble(ens, X_train, y_train):
    ens.fit(X_train, y_train)
    return ens

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Classification Report: \n", classification_report(y_test, y_pred))
    print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
    return y_pred

def setup_logger(name, log_file, filemode='w', level=logging.INFO):
    """Function to set up a logger."""
    handler = logging.FileHandler(log_file, mode=filemode)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levellevelname)s - %(message)s'))

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

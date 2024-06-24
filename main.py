import joblib
from model import load_data, preprocess_data, plot_count, plot_histogram, setup_logger
from model import split_data, train_knn, train_rf, train_xgb, create_ensemble, train_ensemble, evaluate_model, plot_roc_auc

# Setup logger
logger = setup_logger('logs', 'logger.log', filemode='w')

logger.info('Loading and preprocessing data')
# Load and preprocess data
data = load_data("bank_dataset.csv")
X, y, df_fraud, df_non_fraud, mappings = preprocess_data(data)

# Save the mappings
joblib.dump(mappings, 'categorical_mappings.pkl')

logger.info('First 5 rows of the dataset:\n%s', data.head(5))

# Visualize data
plot_count(data)
plot_histogram(df_fraud, df_non_fraud)

# Split data
X_train, X_test, y_train, y_test = split_data(X, y)

# Train individual models
logger.info('Training K-Nearest Neighbours model')
knn = train_knn(X_train, y_train)

logger.info('Training Random Forest model')
rf_clf = train_rf(X_train, y_train)

logger.info('Training XGBoost model')
XGBoost_CLF = train_xgb(X_train, y_train)

# Evaluate individual models
logger.info('Evaluating K-Nearest Neighbours')
evaluate_model(knn, X_test, y_test)
plot_roc_auc(y_test, knn.predict_proba(X_test)[:, 1])

logger.info('Evaluating Random Forest Classifier')
evaluate_model(rf_clf, X_test, y_test)
plot_roc_auc(y_test, rf_clf.predict_proba(X_test)[:, 1])

logger.info('Evaluating XGBoost Classifier')
evaluate_model(XGBoost_CLF, X_test, y_test)
plot_roc_auc(y_test, XGBoost_CLF.predict_proba(X_test)[:, 1])

# Create and train ensemble model
logger.info('Creating and training ensemble model')
ensemble = create_ensemble(knn, rf_clf, XGBoost_CLF)
trained_ensemble = train_ensemble(ensemble, X_train, y_train)

# Save the trained ensemble model
joblib.dump(trained_ensemble, 'trained_ensemble_model.pkl')

# Evaluate ensemble model
logger.info('Evaluating Ensemble Model')
evaluate_model(trained_ensemble, X_test, y_test)
plot_roc_auc(y_test, trained_ensemble.predict_proba(X_test)[:, 1])

# Base score for comparison
base_score = df_non_fraud.fraud.count() / (df_non_fraud.fraud.count() + df_fraud.fraud.count()) * 100
logger.info("Base score we must beat is: %f", base_score)
print("Base score we must beat is: %f", base_score)
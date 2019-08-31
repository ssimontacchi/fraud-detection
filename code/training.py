from config import param_test
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append("..")
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

import warnings
warnings.filterwarnings('ignore')


class Train:
    def __init__(self, data_dir, model_path="", model_name="", save=True,
                 verbose=True):
        ''' Training pipeline that extensively prepares the data, fixes the class
        imbalance issues, runs a stratified KFold,
        Args:
            data_dir (str): directory of 'training_data.csv'
            save (bool): the option to save model when running inference
            model_path (str): where you might like to save your model
            model_name (str): the name of the model
            verbose (bool): option for printing status updates
        '''
        self.df = pd.read_csv(data_dir)
        self.user_ids = self.df['USER_ID']
        self.model_path = model_path
        print(self.model_path)
        self.model_name = model_name
        self.save = save
        self.verbose = verbose

    def prepare_data_for_training(self, df=pd.DataFrame()):
        ''' This method prepares our data for training by:
              - removing unlabeled data
              - removing data points with only one class instance for
                stratified train_test_split
              - performs train_test_split
              - removes USER_ID after stratified splitting on it
              - one-hot encodes categorical variables

        Args:
          df (pd.DataFrame): DataFrame to be prepared for training.

        Returns:
          unlabeled_X (pd.DataFrame): data to be used for inference later
          orig_Xtrain (pd.DataFrame): training data
          orig_Xtest (pd.Series):  test data
          orig_ytrain (pd.DataFrame): training labels
          orig_ytest (pd.Series):  test labels
        '''
        if len(df) == 0:
            df = self.df

        if self.verbose: print('Preparing data for model training.')
        # Find unlabeled data
        unknowns_mask = df['IS_FRAUDSTER'].isna()
        unlabeled_X = df[unknowns_mask]

        # Train on only the labeled data
        y = df[~unknowns_mask]['IS_FRAUDSTER'].astype('bool')
        X = df[~unknowns_mask].drop('IS_FRAUDSTER', axis=1)
        self.user_ids = self.user_ids[~unknowns_mask]

        # Prepare data for stratified split (must be more > 1 class instance)
        counts = X['USER_ID'].value_counts()
        mask = ~X['USER_ID'].isin(counts[counts == 1].index)
        X, y = X[mask], y[mask]
        self.user_ids = self.user_ids[mask]

        # Removes USER_ID because there are too many
        X = X.drop('USER_ID', axis=1)
        unlabeled_X = unlabeled_X.drop('USER_ID', axis=1)

        # One-hot encodes
        X = pd.get_dummies(X, dummy_na=True)
        unlabeled_X = pd.get_dummies(unlabeled_X, dummy_na=True)
        # Call them orig_ because we will train on balanced classes but
        # test our model on the actual unbalanced data.
        indices = np.arange(len(X))
        (orig_Xtrain, orig_Xtest,
         orig_ytrain, orig_ytest,
         indices_train, indices_test) = train_test_split(X, y, indices, stratify=self.user_ids,
                                                         test_size=0.2, random_state=42)

        return unlabeled_X, orig_Xtrain, orig_Xtest, orig_ytrain, orig_ytest

    def balance_training_dataset(self, orig_Xtrain, orig_ytrain):
        """Use undersampling or SMOTE to balance the original datasets before
        training models on them.

        Args:
            orig_Xtrain (object): Pandas DataFrame of training data
            orig_ytrain (object): Pandas DataFrame of training labels

        Returns:
            balanced_df (object): Pandas DataFrame of training data with
                                  balanced classes

        """
        if self.verbose: print("Undersampling training data.")
        # Balance classes taking as many true as fraudulent transactions
        fraud_df = orig_Xtrain[orig_ytrain == True]
        non_fraud_df = orig_Xtrain[orig_ytrain == False].iloc[:len(fraud_df)]
        normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

        # Make sure the targets stay with the correct data through shuffling
        frauds, trues = orig_ytrain == True, orig_ytrain == False
        fewer_non_frauds = orig_ytrain[trues].iloc[:len(fraud_df)]
        targets = pd.concat([orig_ytrain[frauds], fewer_non_frauds])
        bal_classes_df = pd.concat([normal_distributed_df, targets], axis=1)

        # Shuffle the balanced dataframe's rows
        undersampled_df = bal_classes_df.sample(frac=1, random_state=42)
        undersampled_y_train = pd.Series(undersampled_df["IS_FRAUDSTER"].astype(int))
        undersampled_df = undersampled_df.drop(["IS_FRAUDSTER"], axis=1)

        return undersampled_df, undersampled_y_train

    def find_best_model(self, undersampled_df, undersampled_y_train, orig_Xtest,
                        display=False):
        """Runs a Stratified KFolds cross validation and random searches
        for best hyperparameters.

        Args:
            undersampled_df (object): Pandas DataFrame of balanced data
            undersampled_y_train (object): DataFrame of labels for balanced data
            orig_Xtest (object): Pandas DataFrame of test data
        """
        lgbm = lgb.LGBMClassifier(objective='binary', metric='f1',
                                  n_jobs=-1, n_estimators=250, random_state=5)

        # Stratified KFold
        skf = StratifiedKFold(n_splits=3)
        rs = RandomizedSearchCV(lgbm, param_test, cv=skf, n_iter=40)
        rs.fit(undersampled_df, undersampled_y_train)
        update = 'Best score reached: {} with params: {} '
        if self.verbose: print(update.format(rs.best_score_, rs.best_params_))
        self.best_model = lgb.LGBMClassifier(objective='binary', metric='logloss',
                                             n_jobs=-1, n_estimators=250,
                                             random_state=5, **rs.best_params_)
        self.best_params = rs.best_params_
        # Store your model's thoughts
        self.best_model.fit(undersampled_df, undersampled_y_train)
        self.y_pred = self.best_model.predict(orig_Xtest)
        self.y_pred_proba = self.best_model.predict_proba(orig_Xtest)

        # Save it
        if self.save:
            self.save_model('cccv')
            self.save_params('cccv')

    def calibrated_classifier_CV(self, undersampled_df, undersampled_y_train,
                                 orig_Xtest, orig_ytest, model, save=True,
                                 pretrained=True):
        """Calibrated Classifier to calibrate the predict_probas.
        Args:
            undersampled_df (object): Pandas DataFrame of balanced data
            undersampled_y_train (object): labels for DataFrame of balanced data
            orig_Xtest (object): Pandas DataFrame of test data
            orig_ytest (object): Pandas DataFrame of test labels
            model (object): classifier model to be calibrated
            save (bool): save your calibrated classifier
            pretrained (bool): for choosing between calibrating a pretrained model
                               and an untrained model

        Returns:
            balanced_df (object): Pandas DataFrame of training data with
                                  balanced classes

        """
        if pretrained:
            cccv = CalibratedClassifierCV(model, cv='prefit', method='isotonic')
            cccv.fit(orig_Xtest, orig_ytest)
            return cccv
        else:
            cccv = CalibratedClassifierCV(model, cv=5, method='isotonic')
            cccv.fit(undersampled_df, undersampled_y_train)
            return cccv
        if save:
            self.save_model(self.model_name+'calibrclassifier')
            self.save_params(self.model_name+'calibrclassifier')

    def save_model(self, name):
        """Saves model to self.model_path.
        Args:
            name (str): takes an optional name
        """
        joblib.dump(self.best_model, os.path.join(self.model_path, name) + '.pkl')
        if self.verbose: print('Model saved!')

    def save_params(self, name):
        """Saves params to self.model_path.
        Args:
            name (str): takes an optional name
        """
        joblib.dump(self.best_params, os.path.join(self.model_path, name) + 'params.pkl')
        if self.verbose: print('Params saved!')

    def load_model(self, name):
        """Loads model from self.model_path to self.best_model.
        Args:
            name (str): optional name if you want a specific model

        Returns:
            best_model (object): a LightGBM Classifier to be used for inference
        """
        self.best_model = joblib.load(os.path.join('../artifacts/', name) + '.pkl')
        if self.verbose: print('Model loaded.')
        return self.best_model

    def load_params(self, name):
        """Loads params from self.model_path to self.best_params.
        Args:
            name (str): optional name if you want a specific model's params

        Returns:
            best_params (object): params used to train a LightGBM Classifier

        """
        self.best_params = joblib.load(os.path.join(self.model_path, name) + 'params.pkl')
        if self.verbose: print('Params loaded.')
        return self.best_params

    def examine_model(self, y_pred, orig_ytest, display=True):
        """Displays at confusion matrix, precision and recall, and roc_auc.
        Args:
            y_pred (object): model's predictions to be used for model validation
            orig_ytest (object): Pandas Series to be used for model validation
        Return:
            FP_FN_tradoff (tuple): tuple representing the FP and FN tradeoff for
            the model
        """
        conf = confusion_matrix(orig_ytest.astype(int), y_pred)[::-1, ::-1]
        if display:
            print(conf)
            print(conf[0,1], "False Negatives and", conf[1,0], "False Positives.")
            print("Recall:", round(recall_score(orig_ytest.astype(int), y_pred), 3))
            print("Precision:", round(precision_score(orig_ytest.astype(int), y_pred), 3))
            ax = lgb.plot_importance(self.best_model, max_num_features=10)
            plt.show()
            fpr, tpr, thresholds = roc_curve(orig_ytest.astype(int), y_pred)
            roc_auc = auc(fpr, tpr)

            plt.title('Receiver Operating Characteristic')
            plt.plot(fpr, tpr, label='AUC = %0.2f'% roc_auc)
            plt.legend(loc='lower right')
            plt.plot([0,1],[0,1],'r--')
            plt.xlim([-0.001, 1])
            plt.ylim([0, 1.001])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.show();

        # return FN and FP count
        return (conf[0,1], conf[1,0])

    def get_FP_FN_tradeoff_mtx(self):
        """Run models to get confusion matrices to show model tradeoffs.
        Returns:
            tradeoffs_df (object): DataFrame of models and FN/FP tradeoffs
        """
        FN_FP_tradeoff = {}
        # 1. LightGBM with RandomSearchCV
        t = Train("./data/training_data.csv", "./artifacts/", "LGBM")
        unlabeled_X, orig_Xtrain, orig_Xtest, orig_ytrain, orig_ytest = t.prepare_data_for_training()
        undersampled_df, undersampled_y_train = t.balance_training_dataset(orig_Xtrain,
                                                                            orig_ytrain)
        t.find_best_model(undersampled_df, undersampled_y_train, orig_Xtest)
        rs = t.load_model()
        rs_params = t.load_params()
        y_pred = rs.predict(orig_Xtest)
        title = "LightGBM Random Search"
        FN_FP_tradeoff[title] = t.examine_model(y_pred, orig_ytest, display=False)

        # 2. Weighted cross-entropy LGBM
        lgbm = lgb.LGBMClassifier(random_state=12, class_weight = {1: .9, 0: .3})
        lgbm.fit(orig_Xtrain, orig_ytrain)
        y_pred = lgbm.predict(orig_Xtest)
        title = "LightGBM weighted cross-entropy"
        FN_FP_tradeoff[title] = t.examine_model(y_pred, orig_ytest, display=False)

        # 3. Calibrated Classifier on weighted cross-entropy LGBM
        cccv = t.calibrated_classifier_CV(undersampled_df, undersampled_y_train,
                                            orig_Xtest, orig_ytest, lgbm)
        y_pred_cccv = cccv.predict(orig_Xtest)
        title = "Calibrated Classifier w/ weighted cross-entropy LGBM"
        FN_FP_tradeoff[title] = t.examine_model(y_pred_cccv, orig_ytest, display=False)
        self.save_model('cccv')


        # 4. Calibrated Classifier on RS LGBM
        cccvrs = t.calibrated_classifier_CV(undersampled_df, undersampled_y_train,
                                            orig_Xtest, orig_ytest, rs)
        y_pred_cccvrs = cccvrs.predict(orig_Xtest)
        title = "Calibrated Classifier on RS LGBM"
        FN_FP_tradeoff[title] = t.examine_model(y_pred_cccvrs, orig_ytest, display=False)

        # 5. CCCV without pre-trained model
        cccvuntr = lgb.LGBMClassifier(random_state=10)
        cccvuntr = t.calibrated_classifier_CV(undersampled_df, undersampled_y_train,
                                              orig_Xtest, orig_ytest, cccvuntr,
                                              pretrained=False)
        y_pred_cccv_untrained = cccvuntr.predict(orig_Xtest)
        title = "Calibrated Classifier, not pre-trained "
        FN_FP_tradeoff[title] = t.examine_model(y_pred_cccv_untrained, orig_ytest, display=False)

        keys = list(FN_FP_tradeoff.keys())
        FNs = [FN_FP_tradeoff[key][0] for key in keys]
        FPs = [FN_FP_tradeoff[key][1] for key in keys]
        tradeoffs_df = pd.DataFrame(list(zip(keys, FNs, FPs)),
                       columns =['Classifier', '# of FNs (Fraud)',
                                    '# of FPs (locked account)'])

        return tradeoffs_df, rs, cccv, cccvrs, cccvuntr, lgbm

    def get_df(self):
        """Data getter in the form of a DataFrame."""
        return self.df

    def get_user_ids(self):
        """USER_ID getter in the form of a DataFrame."""
        return self.user_ids

if __name__ == "__main__":
    t = Train(data_dir="data/training_data.csv", model_path="/artifacts/", model_name="LGBM")
    (unlabeled_X, orig_Xtrain, orig_Xtest, orig_ytrain, orig_ytest) = t.prepare_data_for_training()
    undersampled_df, undersampled_y_train = t.balance_training_dataset(orig_Xtrain, orig_ytrain)
    t.find_best_model(undersampled_df, undersampled_y_train, orig_Xtest)

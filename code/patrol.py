import sys
sys.path.append('..')

from config import ALERT_AGENT_THRESHOLD, LOCK_ACCOUNT_THRESHOLD
from data_preparation import Data
import numpy as np
import os
import pandas as pd
from training import Train



class FraudChecker:
    def __init__(self):
        self.user_dict = {}

    def get_data(self):
        """Uses Train pipeline to pull in training data.
        Returns:
            orig_Xtrain (object): Pandas DataFrame generally for training.
            orig_Xtest (object): Pandas DataFrame generally for testing.
            model (object): Pickled model for running inference
            user_ids (list): list of all user_ids
        """
        t = Train(data_dir="../data/training_data.csv")
        user_ids = t.get_user_ids()
        (unlabeled_X, orig_Xtrain,
         orig_Xtest, orig_ytrain,
         orig_ytest) = t.prepare_data_for_training()
        model = t.load_model('cccv')
        return model, orig_Xtrain, orig_Xtest, user_ids

    def build_lookup_dict(self):
        """Creates dictionary for O(1) lookup using thresholding. Thresholds
        can be changed in config.py.

        Returns:
            self.user_dict (dictionary): dictionary to quickly look up whether
                                        an actions should be taken towards a user
        """
        model, orig_Xtrain, orig_Xtest, user_ids = self.get_data()

        # Use predict_proba on ALL DATA
        y_pred_probas_cccv_train = model.predict_proba(orig_Xtrain)
        y_pred_probas_cccv_test = model.predict_proba(orig_Xtest)
        train_df = pd.DataFrame(y_pred_probas_cccv_train[:,1])
        test_df = pd.DataFrame(y_pred_probas_cccv_test[:,1])
        all_preds = pd.concat([train_df, test_df])
        join_w_users = all_preds.join(user_ids)

        # Find max fraud probability of a transaction for each user
        person_probs = join_w_users.groupby(['USER_ID'], sort=False)[0].max()

        # Numpy masking for the upper and lower limit
        lower_limit = person_probs > ALERT_AGENT_THRESHOLD
        upper_limit = person_probs > LOCK_ACCOUNT_THRESHOLD
        ACTION = []
        for i in range(len(person_probs)):
            if lower_limit[i]: ACTION.append('LOCK')
            elif upper_limit[i]: ACTION.append('ALERT_AGENT')
            else: ACTION.append("NOTHING")
        action = pd.Series(ACTION).values
        ids = pd.Series(person_probs.keys())
        self.user_dict = pd.Series(action, index=ids).to_dict()

        return self.user_dict

    def check_dict(self, USER_ID):
        '''Returns what action (str) should be taken towards a specified user.'''
        return self.user_dict[USER_ID]


if __name__ == "__main__":
    FC = FraudChecker()
    FC.build_lookup_dict()
    print(FC.check_dict('001926be-3245-43fa-86dd-b40ee160b6f9'))

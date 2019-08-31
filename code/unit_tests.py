from data_preparation import Data
from features import Feature_Extractor
from patrol import FraudChecker
from training import Train
import pytest


def test_clean_data_shape():
    """Tests if cleaned data DataFrame is the correct shape."""
    d = Data('../data/')
    d.get_data()
    df = d.clean_data()
    assert df.shape == (688651, 18), "Check clean_data pipeline."


def test_extracted_features_shape():
    """Tests if extracted features DataFrame is the correct shape."""
    d = Data('../data/')
    d.get_data()
    df = d.clean_data()
    fe = Feature_Extractor() # pass in cleaned DataFrame
    data = fe.extract_features(df)
    assert data.shape == (688651, 24), "Feature extraction pipeline error."


def test_undersampled_shape():
    """Tests if extracted features DataFrame is the correct shape."""
    t = Train("../data/training_data.csv")
    df = t.get_df()
    unlabeled_X, orig_Xtrain, orig_Xtest, orig_ytrain, orig_ytest = t.prepare_data_for_training(df)
    undersampled_df, undersampled_y_train = t.balance_training_dataset(orig_Xtrain, orig_ytrain)
    assert undersampled_df.shape == (22268, 510), "Undersampling error."


def test_action_dict():
    FC = FraudChecker()
    FC.build_lookup_dict()
    assert FC.check_dict('001926be-3245-43fa-86dd-b40ee160b6f9') == 'LOCK', "Get him!"


if __name__ == "__main__":
    test_clean_data_shape()
    test_extracted_features_shape()
    test_undersampled_shape()
    test_action_dict()

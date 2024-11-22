from lightfm import LightFM
import pandas as pd
import numpy as np 
import rectools
from rectools import Columns
from rectools.dataset import Dataset
from rectools.models import ImplicitItemKNNWrapperModel
from implicit.nearest_neighbours import TFIDFRecommender

smm_train_data = pd.read_parquet('train_smm.parquet').drop_duplicates()
smm_test_data = pd.read_parquet('test_smm.parquet').drop_duplicates()

zvuk_train_data = pd.read_parquet('train_zvuk.parquet').drop_duplicates()
zvuk_test_data = pd.read_parquet('test_zvuk.parquet').drop_duplicates()

smm_train_data.columns = [Columns.User, Columns.Item, Columns.Datetime,  Columns.Weight]
smm_test_data.columns = [Columns.User, Columns.Item, Columns.Datetime, Columns.Weight]
zvuk_train_data.columns = [Columns.User, Columns.Datetime, Columns.Item, Columns.Weight]
zvuk_test_data.columns = [Columns.User, Columns.Datetime, Columns.Item, Columns.Weight]


def delete_data(zvuk_train_data, smm_train_data):
    smm_train_data['datetime'] = pd.to_datetime(smm_train_data['datetime']).dt.date
    daily_counts = smm_train_data.groupby('datetime').size().reset_index(name='purchase_count')
    filtered_dates = daily_counts[daily_counts['purchase_count'] <= 4000]['datetime']
    filtered_data = smm_train_data[smm_train_data['datetime'].isin(filtered_dates)]

    zvuk_train_data['datetime'] = pd.to_datetime(zvuk_train_data['datetime']).dt.date
    daily_counts = zvuk_train_data.groupby('datetime').size().reset_index(name='purchase_count')
    filtered_dates = daily_counts[daily_counts['purchase_count'] <= 10000]['datetime']
    filtered_data = zvuk_train_data[zvuk_train_data['datetime'].isin(filtered_dates)]


    g_zvuk = zvuk_train_data.item_id.value_counts(True).reset_index()
    g_zvuk = g_zvuk[g_zvuk.proportion >= 0.000002]
    g_smm = smm_train_data.item_id.value_counts(True).reset_index()
    g_smm = g_smm[g_smm.proportion >= 0.0000025]
    smm_train_data[smm_train_data.item_id.isin(g_smm.item_id)], zvuk_train_data[zvuk_train_data.item_id.isin(g_zvuk.item_id)]
    smm_train_data = smm_train_data[smm_train_data['weight'] >= 2]
    zvuk_train_data = zvuk_train_data[zvuk_train_data['weight'] >= 3]
    zvuk_train_data = zvuk_train_data.groupby('item_id').filter(lambda x: len(x) >= 1000)
    smm_train_data = smm_train_data.groupby('item_id').filter(lambda x: len(x) >= 300)

    return smm_train_data, zvuk_train_data

zvuk_train_data, smm_train_data = delete_data(zvuk_train_data, smm_train_data)

def sparsing(features, column):
    features = pd.get_dummies(features)
    features_frames = []
    for feature in features.columns[1:]:
        feature_frame = features.reindex(columns=[column, feature])
        feature_frame.columns = ["id", "value"]
        feature_frame["feature"] = feature
        features_frames.append(feature_frame)
    return pd.concat(features_frames)


zvuk_user_features_df = sparsing(pd.read_parquet('features/zvuk_user_features.parquet').reset_index().drop(columns=['user_first_interaction', 'user_last_interaction']), "user_id")
zvuk_item_features_df = sparsing(pd.read_parquet('features/zvuk_item_features.parquet').reset_index(), "item_id")
smm_user_features_df = sparsing(pd.read_parquet('features/smm_user_features.parquet').reset_index().drop(columns=['user_first_interaction', 'user_last_interaction']), "user_id")
smm_item_features_df = sparsing(pd.read_parquet('features/smm_item_features.parquet').reset_index(), "item_id")

zvuk_dataset = Dataset.construct(
    interactions_df=zvuk_train_data[zvuk_train_data.user_id.isin(zvuk_test_data.user_id.unique())], 
    user_features_df=zvuk_user_features_df, 
    item_features_df=zvuk_item_features_df,
    )
smm_dataset = Dataset.construct(
    interactions_df=smm_train_data[smm_train_data.user_id.isin(smm_test_data.user_id.unique())], 
    user_features_df=smm_user_features_df, 
    item_features_df=smm_item_features_df,
    )

from rectools.models.lightfm import LightFMWrapperModel

model = LightFMWrapperModel(LightFM(loss='warp', no_components=500), epochs=100, num_threads=12, verbose=True)
model.fit(zvuk_dataset)

# Make recommendations
recos = model.recommend(
    users=zvuk_test_data[Columns.User].unique(),
    dataset=zvuk_dataset,
    k=10,
    filter_viewed=True,
    on_unsupported_targets='ignore'
)
answer = pd.DataFrame(recos.groupby('user_id')['item_id'].apply(list))
answer.to_parquet('submission_zvuk.parquet')

model.fit(smm_dataset)

# Make recommendations
recos = model.recommend(
    users=smm_test_data[Columns.User].unique(),
    dataset=smm_dataset,
    k=10,
    filter_viewed=True,
    on_unsupported_targets='ignore'
)

answer = pd.DataFrame(recos.groupby('user_id')['item_id'].apply(list))
answer.to_parquet('submission_smm.parquet')
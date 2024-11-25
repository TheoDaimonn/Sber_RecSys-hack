import pandas as pd
from datetime import datetime
import numpy as np
smm_train_data = pd.read_parquet('train_smm.parquet')
smm_test_data = pd.read_parquet('test_smm.parquet')
zvuk_train_data = pd.read_parquet('train_zvuk.parquet')
zvuk_test_data = pd.read_parquet('test_zvuk.parquet')

def filter_holiday_periods(df, holiday_periods):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date  
    holiday_masks = [
        (df['date'] >= pd.to_datetime(start).date()) & (df['date'] <= pd.to_datetime(end).date())
        for start, end in holiday_periods.values()
    ]
    combined_mask = pd.concat(holiday_masks, axis=1).any(axis=1)
    return df[combined_mask].drop(columns=['date'])

holiday_periods = {
    'valentine_period': ('2023-02-16', '2023-02-18'),
    'women_period': ('2023-03-06', '2023-03-07'),
}

zvuk_holiday_data = filter_holiday_periods(zvuk_train_data, holiday_periods)
smm_holiday_data = filter_holiday_periods(smm_train_data, holiday_periods)

# Вспомогательные функции
def add_time_features(df):
    df['date'] = df['timestamp'].dt.date
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['week_of_year'] = df['timestamp'].dt.isocalendar().week
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    return df

def normalize_ratings(df):
    rating_mean = df['rating'].mean()
    rating_std = df['rating'].std()
    if rating_std == 0:
        df['rating_normalized'] = 0
    else:
        df['rating_normalized'] = (df['rating'] - rating_mean) / rating_std
    return df

def add_item_total_interactions(df):
    item_interactions = df.groupby('item_id').size().rename('item_total_interactions')
    df = df.merge(item_interactions, on='item_id', how='left')
    return df

def add_item_avg_rating(df):
    item_avg_rating = df.groupby('item_id')['rating_normalized'].mean().rename('item_avg_rating').reset_index()
    df = df.merge(item_avg_rating, on='item_id', how='left')
    return df

def add_item_age(df):
    item_first_appearance = df.groupby('item_id')['timestamp'].min()
    df['item_first_appearance'] = df['item_id'].map(item_first_appearance)
    df['item_age_days'] = (df['timestamp'] - df['item_first_appearance']).dt.days
    return df

def add_item_unique_users(df):
    item_unique_users = df.groupby('item_id')['user_id'].nunique().rename('item_unique_users').reset_index()
    df = df.merge(item_unique_users, on='item_id', how='left')
    return df

def add_item_weighted_avg_rating(df):
    df['timestamp_norm'] = (df['timestamp'] - df['timestamp'].min()) / (df['timestamp'].max() - df['timestamp'].min())
    df['rating_weight'] = df['timestamp_norm']
    
    # Рассчитываем средневзвешенный рейтинг
    item_weighted_avg_rating = df.groupby('item_id', as_index=False).apply(
        lambda x: np.average(x['rating'], weights=x['rating_weight'])
    ).rename('item_weighted_avg_rating').reset_index()
    
    df = df.merge(item_weighted_avg_rating, on='item_id', how='left')
    df.drop(columns=['timestamp_norm'], inplace=True)
    return df

def add_item_lifetime(df):
    item_appearance = df.groupby('item_id')['timestamp'].agg(['min', 'max']).rename(columns={'min': 'item_first_appearance', 'max': 'item_last_appearance'}).reset_index()
    item_appearance['item_lifetime_days'] = (item_appearance['item_last_appearance'] - item_appearance['item_first_appearance']).dt.days
    df = df.merge(item_appearance[['item_id', 'item_lifetime_days']], on='item_id', how='left')
    return df

def add_item_popularity_trend(df):
    # Группируем данные по неделям
    df['year_week'] = df['timestamp'].dt.strftime('%Y-%U')
    item_week_sales = df.groupby(['item_id', 'year_week']).size().reset_index(name='sales')
    
    # Рассчитываем тренд для каждого товара
    def calculate_trend(x):
        y = x['sales'].values
        if len(y) > 1:
            x_vals = np.arange(len(y))
            slope, _, _, _, _ = linregress(x_vals, y)
            return slope
        else:
            return 0  
    
    item_trends = item_week_sales.groupby('item_id', as_index=False).apply(
        calculate_trend,
        include_groups=False  # Добавлено для устранения предупреждения
    ).rename('item_popularity_trend').reset_index()
    
    df = df.merge(item_trends, on='item_id', how='left')
    df.drop(columns=['year_week'], inplace=True, errors='ignore')
    return df

def add_is_universal_item(df, min_sales_per_month=5):
    # Считаем количество продаж каждого товара по месяцам
    df['year_month'] = df['timestamp'].dt.to_period('M')
    item_month_sales = df.groupby(['item_id', 'year_month']).size().unstack(fill_value=0)

    # Определяем первый месяц появления каждого товара
    item_first_month = df.groupby('item_id')['year_month'].min()
    item_first_month = item_first_month.reindex(item_month_sales.index, fill_value=item_month_sales.columns.min())

    # Создаем маску активных месяцев для каждого товара
    months = item_month_sales.columns
    item_active_months = pd.DataFrame(
        {item_id: (months >= item_first_month[item_id]) for item_id in item_month_sales.index},
        index=item_month_sales.columns
    ).T
    item_active_months.index = item_month_sales.index

    # Проверка на универсальность
    is_universal = ((item_month_sales >= min_sales_per_month) | ~item_active_months).all(axis=1)

    # Добавляем фичу в данные
    df['is_universal_item'] = df['item_id'].map(is_universal).fillna(0).astype(int)
    return df

def add_is_seasonal_item(df):
    # Считаем количество уникальных месяцев, в которых продавался товар
    df['year_month'] = df['timestamp'].dt.to_period('M')
    item_months_sold = df.groupby('item_id')['year_month'].nunique()

    # Товар сезонный, если продавался только в одном месяце
    is_seasonal = item_months_sold == 1

    # Альтернативно, если товар имеет большую часть продаж в одном месяце
    item_month_sales = df.groupby(['item_id', 'year_month']).size()
    item_total_sales = df.groupby('item_id').size()
    item_max_month_sales = item_month_sales.groupby('item_id').max()
    item_seasonality_ratio = item_max_month_sales / item_total_sales
    is_seasonal |= (item_seasonality_ratio >= 0.8)

    # Добавляем фичу в данные
    df['is_seasonal_item'] = df['item_id'].map(is_seasonal).astype(int)
    return df

def add_holiday_features(df):
    df['date'] = df['timestamp'].dt.date

    holiday_periods = {
        'valentine_period': ('2023-02-11', '2023-02-14'),
        'defender_period': ('2023-02-18', '2023-02-23'),
        'women_period': ('2023-03-04', '2023-03-08'),
        'early_feb_period': ('2023-02-04', '2023-02-06')
    }

    # Инициализируем колонки
    for period_name in holiday_periods.keys():
        df[f'is_popular_in_{period_name}'] = 0

    # Для каждого периода определяем популярные товары
    for period_name, (start, end) in holiday_periods.items():
        start_date = pd.to_datetime(start).date()
        end_date = pd.to_datetime(end).date()

        # Фильтруем данные за праздничный период
        df_period = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

        # Считаем продажи товаров в период
        period_item_sales = df_period.groupby('item_id').size()

        # Общие продажи товаров
        total_item_sales = df.groupby('item_id').size()

        # Вычисляем долю продаж в праздничный период
        item_sales_ratio = (period_item_sales / total_item_sales).fillna(0)

        # Товары популярны, если доля продаж в период >= 50%
        popular_items = item_sales_ratio[item_sales_ratio >= 0.5].index

        # Устанавливаем флаг
        df.loc[df['item_id'].isin(popular_items), f'is_popular_in_{period_name}'] = 1

    return df

def combine_item_types(df):
    holiday_flags = [col for col in df.columns if col.startswith('is_popular_in_')]
    df['is_holiday_item'] = df[holiday_flags].max(axis=1)

    # Избегаем дублирования, выбирая уникальные товары
    item_features = df.drop_duplicates('item_id')[['item_id', 'is_universal_item', 'is_seasonal_item', 'is_holiday_item']]

    # Функция для определения типа товара
    def determine_item_type(row):
        if row['is_universal_item'] == 1:
            return 'universal'
        elif row['is_holiday_item'] == 1:
            return 'holiday'
        elif row['is_seasonal_item'] == 1:
            return 'seasonal'
        else:
            return 'normal'

    # Применяем функцию к каждому товару
    item_features['item_type'] = item_features.apply(determine_item_type, axis=1)

    # Присоединяем результат обратно к основным данным
    df = df.merge(item_features[['item_id', 'item_type']], on='item_id', how='left')
    df['item_type'] = df['item_type'].astype('category')
    return df

def add_song_age(df):
    song_first_appearance = df.groupby('item_id')['timestamp'].min()
    df['song_first_appearance'] = df['item_id'].map(song_first_appearance)
    df['song_age_days'] = (df['timestamp'].max() - df['song_first_appearance']).dt.days
    return df

def add_advanced_time_features(df):
    df_sorted = df.sort_values(['item_id', 'timestamp'])
    df_sorted['prev_timestamp'] = df_sorted.groupby('item_id')['timestamp'].shift(1)
    df_sorted['time_diff'] = (df_sorted['timestamp'] - df_sorted['prev_timestamp']).dt.total_seconds()

    item_median_time_diff = df_sorted.groupby('item_id')['time_diff'].median().rename('item_median_time_between_plays').reset_index()
    time_diff_trend = df_sorted.groupby('item_id')['time_diff'].apply(lambda x: x.diff().mean()).rename('item_time_diff_trend').reset_index()
    df = df.merge(item_median_time_diff, on='item_id', how='left')
    df = df.merge(time_diff_trend, on='item_id', how='left')
    return df

def drop_unnecessary_columns(df):
    columns_to_drop = ['timestamp_norm', 'rating_weight', 'prev_timestamp', 'time_diff', 'date', 'year_week', 'year_month']
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    return df

def optimize_memory(df):
    df['user_id'] = df['user_id'].astype('int32')
    df['item_id'] = df['item_id'].astype('int32')
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].apply(pd.to_numeric, downcast='float')
    int_cols = df.select_dtypes(include=['int64']).columns
    df[int_cols] = df[int_cols].apply(pd.to_numeric, downcast='integer')
    return df

# Основная функция обработки данных
def process_item_data(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df = add_time_features(df)
    print('add_time_features')
    df = normalize_ratings(df)
    print('normalize_ratings')
    df = add_item_total_interactions(df)
    print('add_item_total_interactions')
    df = add_item_avg_rating(df)
    print('add_item_avg_rating')
    df = add_item_age(df)
    print('add_item_age')
    df = add_item_unique_users(df)
    print('add_item_unique_users')
    df = add_item_weighted_avg_rating(df)
    print('add_item_weighted_avg_rating')
    df = add_item_lifetime(df)
    print('add_item_lifetime')
    df = add_item_popularity_trend(df)
    print('add_item_popularity_trend')
    df = add_is_universal_item(df)
    print('add_is_universal_item')
    df = add_is_seasonal_item(df)
    print('add_is_seasonal_item')
    df = add_holiday_features(df)
    print('add_holiday_features')
    df = combine_item_types(df)
    print('combine_item_types')
    df = add_song_age(df)
    print('add_song_age')
    df = add_advanced_time_features(df)
    print('add_advanced_time_features')
    df = drop_unnecessary_columns(df)
    df = optimize_memory(df)
    
    return df

zvuk_train_data = process_item_data(zvuk_train_data)
smm_train_data = process_item_data(smm_train_data)

def extract_item_features(df):
    required_columns = [
        'item_total_interactions',
        'item_avg_rating',
        'item_weighted_avg_rating',
        'item_lifetime_days',
        'item_popularity_trend',
        'item_unique_users',
        'item_age_days',
        'item_type',
        'item_median_time_between_plays',
        'item_time_diff_trend'
    ]
    
    # Проверка наличия необходимых столбцов
    available_columns = [col for col in required_columns if col in df.columns]
    
    if not available_columns:
        raise ValueError("Нет доступных столбцов для извлечения признаков.")
    
    item_features = df.drop_duplicates('item_id').set_index('item_id')[available_columns]
    return item_features
# Извлечение признаков
zvuk_item_features = extract_item_features(zvuk_train_data)
smm_item_features = extract_item_features(smm_train_data)

# Сохранение в parquet
zvuk_item_features.to_parquet('zvuk_item_features.parquet')
smm_item_features.to_parquet('smm_item_features.parquet')

# айтем
def add_user_total_interactions(df):
    user_interactions = df.groupby('user_id').size().rename('user_total_interactions').reset_index()
    df = df.merge(user_interactions, on='user_id', how='left')
    return df

def add_user_avg_rating(df):
    user_avg_rating = df.groupby('user_id')['rating'].mean().rename('user_avg_rating').reset_index()
    df = df.merge(user_avg_rating, on='user_id', how='left')
    return df

def add_user_unique_items(df):
    user_unique_items = df.groupby('user_id')['item_id'].nunique().rename('user_unique_items').reset_index()
    df = df.merge(user_unique_items, on='user_id', how='left')
    return df

def add_user_last_interaction(df):
    user_last_interaction = df.groupby('user_id')['timestamp'].max().rename('user_last_interaction').reset_index()
    latest_timestamp = df['timestamp'].max()
    user_last_interaction['days_since_last_interaction'] = (latest_timestamp - user_last_interaction['user_last_interaction']).dt.days
    df = df.merge(user_last_interaction, on='user_id', how='left')
    return df

def get_time_of_day(hour):
    if 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'afternoon'
    elif 18 <= hour < 24:
        return 'evening'
    else:
        return 'night'

def add_user_preferred_time_of_day(df):
    df['hour'] = df['timestamp'].dt.hour
    df['time_of_day'] = df['hour'].apply(get_time_of_day)
    user_time_of_day = df.groupby(['user_id', 'time_of_day']).size().unstack(fill_value=0)
    user_time_of_day['preferred_time_of_day'] = user_time_of_day.idxmax(axis=1)
    preferred_time_of_day = user_time_of_day['preferred_time_of_day'].reset_index()
    df = df.merge(preferred_time_of_day, on='user_id', how='left')
    df.drop(columns=['hour', 'time_of_day'], inplace=True)
    return df

def add_user_first_interaction(df):
    user_first_interaction = df.groupby('user_id')['timestamp'].min().rename('user_first_interaction').reset_index()
    latest_timestamp = df['timestamp'].max()
    user_first_interaction['days_since_first_interaction'] = (latest_timestamp - user_first_interaction['user_first_interaction']).dt.days
    df = df.merge(user_first_interaction, on='user_id', how='left')
    return df

def add_user_interaction_frequency(df):
    df['interaction_frequency'] = df['user_total_interactions'] / df['days_since_first_interaction'].replace(0, 1)
    return df

def add_user_avg_item_age(df):
    avg_item_age = df.groupby('user_id')['item_age_days'].mean().rename('user_avg_item_age').reset_index()
    df = df.merge(avg_item_age, on='user_id', how='left')
    return df

def process_user_data(df):
    df = add_user_total_interactions(df)
    df = add_user_avg_rating(df)
    df = add_user_unique_items(df)
    df = add_user_last_interaction(df)
    df = add_user_preferred_time_of_day(df)
    df = add_user_first_interaction(df)
    df = add_user_interaction_frequency(df)
    df = add_user_avg_item_age(df)
    return df

zvuk_train_data = process_user_data(zvuk_train_data)
smm_train_data = process_user_data(smm_train_data)

def extract_user_features(df):
    user_features = df.drop_duplicates('user_id').set_index('user_id')[[
        'user_total_interactions',
        'user_avg_rating',
        'user_last_interaction',
        'days_since_last_interaction',
        'user_unique_items',
        'preferred_time_of_day',
        'user_first_interaction',
        'days_since_first_interaction',
        'interaction_frequency',
        'user_avg_item_age'
    ]]
    return user_features

zvuk_user_features = extract_user_features(zvuk_train_data)
smm_user_features = extract_user_features(smm_train_data)

zvuk_user_features.to_parquet('zvuk_user_features.parquet')
zvuk_user_features.to_parquet('smm_user_features.parquet')
# Объединяем фичи пользователей из обоих датасетов
user_features_combined = pd.concat([zvuk_user_features, smm_user_features], axis=0)

# Удаляем дубликаты и агрегируем фичи для пользователей, присутствующих в обоих датасетах
aggregation_functions = {
    'user_total_interactions': 'sum',
    'user_avg_rating': 'mean',
    'user_last_interaction': 'max',
    'days_since_last_interaction': 'min',
    'user_unique_items': 'sum',
    'preferred_time_of_day': lambda x: x.value_counts().index[0],  # Наиболее частое значение
    'user_first_interaction': 'min',
    'days_since_first_interaction': 'max',
    'interaction_frequency': 'mean',
    'user_avg_item_age': 'mean'
}

user_features_aggregated = user_features_combined.reset_index().groupby('user_id').agg(aggregation_functions)




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


MAX_SMM = max(smm_train_data.item_id)


def delete_data(zvuk_train_data, smm_train_data):
    
    smm_train_data['datetime'] = pd.to_datetime(smm_train_data['datetime']).dt.date

    zvuk_train_data['datetime'] = pd.to_datetime(zvuk_train_data['datetime']).dt.date

    holiday_periods = {
        'valentine_period': ('2023-02-16', '2023-02-18'),
        'women_period': ('2023-03-06', '2023-03-07'),
    }
    holiday_masks = [
        (smm_train_data['datetime'] >= pd.to_datetime(start).date()) & (smm_train_data['datetime'] <= pd.to_datetime(end).date())
        for start, end in holiday_periods.values()
    ]
    combined_mask = pd.concat(holiday_masks, axis=1).any(axis=1)
    smm_train_data = smm_train_data[combined_mask]
    holiday_masks = [
        (zvuk_train_data['datetime'] >= pd.to_datetime(start).date()) & (zvuk_train_data['datetime'] <= pd.to_datetime(end).date())
        for start, end in holiday_periods.values()
    ]
    combined_mask = pd.concat(holiday_masks, axis=1).any(axis=1)
    zvuk_train_data = zvuk_train_data[combined_mask]


    g_zvuk = zvuk_train_data.item_id.value_counts(True).reset_index()
    g_zvuk = g_zvuk[g_zvuk.proportion >= 0.0000005]
    g_smm = smm_train_data.item_id.value_counts(True).reset_index()
    g_smm = g_smm[g_smm.proportion >= 0.000001]
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


zvuk_user_features_df = sparsing(pd.read_parquet('zvuk_user_features.parquet').reset_index().drop(columns=['user_first_interaction', 'user_last_interaction']), "user_id")
zvuk_item_features_df = sparsing(pd.read_parquet('zvuk_item_features.parquet').reset_index(), "item_id")
smm_user_features_df = sparsing(pd.read_parquet('smm_user_features.parquet').reset_index().drop(columns=['user_first_interaction', 'user_last_interaction']), "user_id")
smm_item_features_df = sparsing(pd.read_parquet('smm_item_features.parquet').reset_index(), "item_id")

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

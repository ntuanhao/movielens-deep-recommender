import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import dill  

MODEL_LIST = [
    {'name': 'V2 Enhanced', 'file': 'model_v2_enhanced.keras'},
    {'name': 'NCF',         'file': 'model_ncf.keras'},
    {'name': 'DCN',         'file': 'model_dcn.keras'},
    {'name': 'V1 (Emb 128)','file': 'model_v1.keras'},
    {'name': 'Baseline',    'file': 'baseline_model.keras'}
]

ARTIFACTS_PATH = 'demo_artifacts.joblib'

if not os.path.exists(ARTIFACTS_PATH):
    print(f"File not found: {ARTIFACTS_PATH}")
    exit()

artifacts = joblib.load(ARTIFACTS_PATH)
user_encoder = artifacts['user_encoder']
movies_df = artifacts['movies_df']
movie_genre_features = artifacts['movie_genre_features']
user_watched_map = artifacts['user_watched_map']

class CrossLayer(tf.keras.layers.Layer):
    def __init__(self, kernel_regularizer=None, **kwargs):
        super(CrossLayer, self).__init__(**kwargs)
        self.kernel_regularizer = kernel_regularizer
    def build(self, input_shape):
        dim = input_shape[0][-1]
        self.W = self.add_weight(shape=(dim, 1), initializer='glorot_uniform', regularizer=self.kernel_regularizer, name='cross_kernel')
        self.b = self.add_weight(shape=(dim,), initializer='zeros', name='cross_bias')
    def call(self, inputs):
        x_0, x_l = inputs
        x_l_dot_W = tf.tensordot(x_l, self.W, axes=[1, 0])
        cross_term = x_0 * x_l_dot_W
        return cross_term + self.b + x_l
    def get_config(self):
        config = super(CrossLayer, self).get_config()
        config.update({'kernel_regularizer': self.kernel_regularizer})
        return config

loaded_models = []
for m in MODEL_LIST:
    if os.path.exists(m['file']):
        try:
            model = load_model(m['file'], custom_objects={'CrossLayer': CrossLayer})
            loaded_models.append({'name': m['name'], 'model': model})
        except Exception as e:
            print(f"Error loading {m['name']}: {e}")

if not loaded_models:
    print("No models loaded.")
    exit()

def recommend_for_user(user_id):
    if user_id not in user_watched_map:
        try:
            user_encoder.transform([user_id])
        except:
            print(f"User ID {user_id} not found.")
            return

    try:
        user_idx = user_encoder.transform([user_id])[0]
    except ValueError:
         print(f"Invalid User ID {user_id}.")
         return

    watched_movie_ids = user_watched_map.get(user_id, set())
    mask_not_watched = ~movies_df['movieId'].isin(watched_movie_ids)
    candidate_movies = movies_df[mask_not_watched].copy()
    candidate_genres = movie_genre_features[mask_not_watched]
    
    num_candidates = len(candidate_movies)
    if num_candidates == 0:
        print("No movies left to recommend.")
        return

    user_idx_input = np.full(num_candidates, user_idx)
    X_pred = {
        'user_input': user_idx_input,
        'movie_input': candidate_movies['movie_idx'].values,
        'genre_input': candidate_genres,
        'year_input': candidate_movies['year_scaled'].values
    }
    
    total_score = np.zeros(num_candidates)
    
    for item in loaded_models:
        model = item['model']
        preds = model.predict(X_pred, batch_size=8192, verbose=0).flatten()
        total_score += preds
    
    avg_score = total_score / len(loaded_models)
    candidate_movies['final_score'] = avg_score
    
    top_10 = candidate_movies.sort_values('final_score', ascending=False).head(10)
    
    print(f"\nTOP 10 RECOMMENDATIONS FOR USER {user_id}:")
    print("=" * 80)
    print(f"{'No.':<4} | {'Title':<55} | {'Score':<6} | {'Genres'}")
    print("-" * 80)
    
    for i, row in enumerate(top_10.itertuples(), 1):
        print(f"{i:<4} | {row.title[:53]:<55} | {row.final_score:.2f}   | {row.genres}")
    print("=" * 80)

while True:
    user_input = input("\nEnter User ID (or 'q' to quit): ")
    
    if user_input.lower() in ['q', 'quit', 'exit']:
        break
    
    if not user_input.isdigit():
        continue
        
    user_id = int(user_input)
    recommend_for_user(user_id)
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares

class MatrixFactorization:
    def __init__(self, factors=100, regularization=0.1, iterations=100, category_weight=0.2):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.category_weight = category_weight
        self.model = AlternatingLeastSquares(factors=self.factors, 
                                             regularization=self.regularization, 
                                             iterations=self.iterations,
                                             )
        self.category_dict = {}
        self.user_factors = None
        self.item_factors = None
        self.category_popularity = None
        self.location_popularity = None
        self.train_df = None
    
    def fit(self, train_df, category_df):
        """Train the ALS model and store category information."""
        # Preprocess train_df and category_df
        user_ids = train_df["uid"].astype("category").cat.codes
        location_ids = train_df["lid"].astype("category").cat.codes
        freqs = train_df["freq"].values
        self.train_df = train_df

        # Map locations to categories
        self.category_dict = category_df.groupby("lid")["category"].apply(list).to_dict()
        
        # Create user-category interaction matrix
        user_categories = []
        category_freqs = []
        
        for _, row in train_df.iterrows():
            lid = row["lid"]
            user_id = row["uid"]
            freq = row["freq"]
            if lid in self.category_dict:
                categories = self.category_dict[lid]
                for cat in categories:
                    user_categories.append((user_id, cat))
                    category_freqs.append(freq)

        # Create interaction matrix for users and categories
        user_categories = pd.DataFrame(user_categories, columns=["uid", "category"])
        user_categories["freq"] = category_freqs
        
        user_ids = user_categories["uid"].astype("category").cat.codes
        category_ids = user_categories["category"].astype("category").cat.codes
        freqs = user_categories["freq"].values
        
        # Initialize the interaction matrix between users and categories
        self.num_users = user_ids.max() + 1
        self.num_categories = category_ids.max() + 1
        interaction_matrix = coo_matrix((freqs, (user_ids, category_ids)),
                                        shape=(self.num_users, self.num_categories))
        
        # Train the ALS model with the interaction matrix
        self.model.fit(interaction_matrix)
        self.user_factors = self.model.user_factors
        self.item_factors = self.model.item_factors

        # Store category and location popularity
        self.category_popularity = user_categories["category"].value_counts(normalize=True).to_dict()
        location_popularity = train_df.groupby("lid")["freq"].sum()
        location_popularity = (location_popularity / location_popularity.max()).to_dict()
        self.location_popularity = location_popularity
    
    def get_category_embedding(self, category_id):
        """Get category embedding."""
        if category_id >= self.num_categories:
            return np.zeros(self.factors)  # Handle unseen categories
        
        return self.item_factors[category_id]
    
    def predict(self, uid, category_id):
        """Predict score for a given user and category."""
        if uid >= self.num_users or category_id >= self.num_categories:
            return 0  # Handle unseen users/categories
        
        user_emb = self.user_factors[uid]
        category_emb = self.get_category_embedding(category_id)
        
        return np.dot(user_emb, category_emb)

    def predict_with_category_popularity(self, uid, category_id, alpha=0.1):
        """Blend ALS score with category popularity."""
        base_score = self.predict(uid, category_id)
        
        category_popularity_boost = self.category_popularity.get(category_id, 0) * alpha
        
        return base_score + category_popularity_boost
    
    def predict_with_location_category_popularity(self, uid, lid, alpha=0.1, beta=0.05):
        """predict_with_category_popularity - Blend ALS score with both location and category popularity."""
        base_score = self.predict(uid, lid)

        popularity_boost = self.location_popularity.get(lid, 0) * alpha

        if lid in self.category_dict:
            category_boost = np.mean([self.category_popularity.get(cat, 0) for cat in self.category_dict[lid]]) * beta
        else:
            category_boost = 0

        return base_score + popularity_boost + category_boost
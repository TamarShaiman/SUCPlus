import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares

class MatrixFactorizationALS:
    def __init__(self, factors=100, regularization=0.1, iterations=50, category_weight=0.2):
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

        # Store category popularity
        self.category_popularity = user_categories["category"].value_counts(normalize=True).to_dict()
    
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

    def predict_with_popularity(self, uid, category_id, alpha=0.1):
        """Blend ALS score with category popularity."""
        base_score = self.predict(uid, category_id)
        
        category_popularity_boost = self.category_popularity.get(category_id, 0) * alpha
        
        return base_score + category_popularity_boost


# import numpy as np
# import pandas as pd
# from scipy.sparse import coo_matrix
# from implicit.als import AlternatingLeastSquares

# class MatrixFactorizationALS:
#     def __init__(self, factors=100, regularization=0.1, iterations=50, category_weight=0.2):
#         self.factors = factors
#         self.regularization = regularization
#         self.iterations = iterations
#         self.category_weight = category_weight
#         self.model = AlternatingLeastSquares(factors=self.factors, 
#                                              regularization=self.regularization, 
#                                              iterations=self.iterations,
#                                              )
#         self.category_dict = {}
#         self.user_factors = None
#         self.item_factors = None
#         self.location_popularity = None
#         self.train_df = None
    
#     def fit(self, train_df, category_df):
#         """Train the ALS model and store category information."""
#         user_ids = train_df["uid"].astype("category").cat.codes
#         location_ids = train_df["lid"].astype("category").cat.codes
#         freqs = train_df["freq"].values
#         self.train_df = train_df
#         location_popularity = train_df.groupby("lid")["freq"].sum()
#         location_popularity = (location_popularity / location_popularity.max()).to_dict()
#         self.location_popularity = location_popularity
#         self.num_users = user_ids.max() + 1
#         self.num_locations = location_ids.max() + 1
#         self.category_popularity = category_df["category"].value_counts(normalize=True).to_dict()

        
#         interaction_matrix = coo_matrix((freqs, (user_ids, location_ids)), 
#                                         shape=(self.num_users, self.num_locations))
        
#         self.model.fit(interaction_matrix)
#         self.user_factors = self.model.user_factors
#         self.item_factors = self.model.item_factors
        
#         # Store category mapping
#         self.category_dict = category_df.groupby("lid")["category"].apply(list).to_dict()
    
#     def get_location_embedding(self, lid):
#         """Get location embedding with category-based normalization."""
#         if lid >= self.num_locations:
#             return np.zeros(self.factors)  # Handle unseen locations
        
#         location_emb = self.item_factors[lid].copy()
        
#         if lid in self.category_dict:
#             categories = self.category_dict[lid]
#             category_factor = self.category_weight / max(1, len(categories))  # Normalize by category count
            
#             for cat in categories:
#                 location_emb += category_factor * self.item_factors[cat % self.num_locations]
        
#         return location_emb

    
#     def predict(self, uid, lid):
#         """Predict score for a given user and location."""
#         if uid >= self.num_users or lid >= self.num_locations:
#             return 0  # Handle unseen users/locations
        
#         user_emb = self.user_factors[uid]
#         location_emb = self.get_location_embedding(lid)
        
#         return np.dot(user_emb, location_emb)

#     def predict_with_adaptive_popularity(self, uid, lid, alpha_base=0.1):
#         """Blend ALS score with location popularity, adapting weight per user."""
#         base_score = self.predict(uid, lid)

#         # Users who explore more (high unique visits) get lower alpha
#         unique_visits = len(self.train_df[self.train_df["uid"] == uid]["lid"].unique())
#         alpha = alpha_base / np.log1p(unique_visits + 1)  # Dampen boost for active users

#         popularity_boost = self.location_popularity.get(lid, 0) * alpha
#         return base_score + popularity_boost
   
#     def predict_with_category_popularity(self, uid, lid, alpha=0.1, beta=0.05):
#         """Blend ALS score with both location and category popularity."""
#         base_score = self.predict(uid, lid)

#         popularity_boost = self.location_popularity.get(lid, 0) * alpha

#         if lid in self.category_dict:
#             category_boost = np.mean([self.category_popularity.get(cat, 0) for cat in self.category_dict[lid]]) * beta
#         else:
#             category_boost = 0

#         return base_score + popularity_boost + category_boost
   

#     def predict_with_popularity(self, uid, lid, alpha=0.1):
#         """Blend ALS score with location popularity."""
#         base_score = self.predict(uid, lid)
#         popularity_boost = self.location_popularity.get(lid, 0) * alpha
#         return base_score + popularity_boost    

#     def evaluate_precision_recall(self, test_df, k=10):
#         """Evaluate model using Precision@K and Recall@K."""
#         precision_list = []
#         recall_list = []
        
#         user_true_locations = test_df.groupby("uid")["lid"].apply(set).to_dict()

#         for uid in user_true_locations.keys():
#             if uid >= self.num_users:
#                 continue  # Skip unseen users
            
#             # Predict scores for all locations
#             scores = {lid: self.predict_with_category_popularity(uid, lid) for lid in range(self.num_locations)}
            
#             # Get top-K recommended locations
#             top_k_lids = sorted(scores, key=scores.get, reverse=True)[:k]

#             # True locations for the user
#             true_lids = user_true_locations[uid]

#             # Compute Precision@K and Recall@K
#             hits = len(set(top_k_lids) & true_lids)
#             precision = hits / k
#             recall = hits / len(true_lids) if len(true_lids) > 0 else 0
            
#             precision_list.append(precision)
#             recall_list.append(recall)

#         # Average across users
#         avg_precision = np.mean(precision_list)
#         avg_recall = np.mean(recall_list)

#         return avg_precision, avg_recall


# # # Example usage
# # if __name__ == "__main__":
# #     train_df = pd.read_csv("train.csv")  # uid, lid, freq
# #     category_df = pd.read_csv("category.csv")  # lid, category
    
    
    
# #     # Example prediction
# #     print(mf_model.predict(0, 17))

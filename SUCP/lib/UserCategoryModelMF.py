import numpy as np
import time
import os
from collections import defaultdict
import scipy.sparse as sparse  # For efficient matrix operations
import math


class UserCategoryModelMF:
    def __init__(self, train_file, poi_category_file, output_file, K=30, alpha=20.0, beta=0.2):
        """
        Initializes the model for Matrix Factorization on User-Category data.

        :param train_file: Path to Yelp_train.txt (User-Location-Frequency data).
        :param poi_category_file: Path to Yelp_poi_categories.txt (Location-Category data).
        :param output_file: Path to save the User-Category-Frequency data.
        :param K: Number of latent factors.
        :param alpha: Gamma prior parameter for user/category factors.
        :param beta: Gamma prior parameter for user/category factors.
        """
        self.train_file = train_file
        self.poi_category_file = poi_category_file
        self.output_file = output_file

        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.U, self.C = None, None  # Latent factor matrices for users & categories

        self.user_category_data = defaultdict(dict)
        self.location_to_categories = self.load_location_categories()

        if os.path.exists(self.output_file):
            self.load_user_category_data()

    def load_location_categories(self):
        """Loads location-to-category mapping once to avoid redundant reads."""
        location_to_categories = defaultdict(set)
        with open(self.poi_category_file, 'r') as f:
            for line in f:
                location_id, category_id = map(int, line.strip().split())
                location_to_categories[location_id].add(category_id)
        return location_to_categories

    def get_user_category_counts(self):
        """
        Automatically detects the total number of users and categories from input files.
        :return: (user_num, category_num) - The total number of users and categories.
        """
        max_user_id = 0
        max_category_id = 0

        with open(self.train_file, 'r') as f:
            for line in f:
                user_id, _, _ = map(int, line.strip().split())
                max_user_id = max(max_user_id, user_id)

        with open(self.poi_category_file, 'r') as f:
            for line in f:
                _, category_id = map(int, line.strip().split())
                max_category_id = max(max_category_id, category_id)

        return max_user_id + 1, max_category_id + 1

    def create_user_category_matrix(self):
        """
        Constructs the User-Category matrix (F_uc) from User-Location and Location-Category data
        and saves it as a file.
        """
        ctime = time.time()

        user_num, category_num = self.get_user_category_counts()
        print("Detected", user_num, "users and", category_num, "categories.")

        F_uc = np.zeros((user_num, category_num))

        with open(self.train_file, 'r') as f:
            for line in f:
                user_id, location_id, frequency = map(int, line.strip().split())

                if location_id in self.location_to_categories:
                    for category_id in self.location_to_categories[location_id]:
                        F_uc[user_id, category_id] += frequency

        print("User-Category Matrix Loaded Successfully. Done. Elapsed time:", time.time() - ctime, "s")

        ctime = time.time()
        with open(self.output_file, 'w') as f:
            for user_id in range(user_num):
                for category_id in range(category_num):
                    if F_uc[user_id, category_id] > 0:
                        f.write("{} {} {}\n".format(user_id, category_id, int(F_uc[user_id, category_id])))

        print("User-Category-Frequency file saved to:", self.output_file, "Done. Elapsed time:", time.time() - ctime, "s")
        self.load_user_category_data()
        return sparse.csr_matrix(F_uc)  # Convert to sparse matrix for efficiency

    def load_user_category_data(self):
        """
        Loads the User-Category matrix from a file into memory for fast lookup.
        """
        self.user_category_data = defaultdict(dict)

        with open(self.output_file, 'r') as f:
            for line in f:
                user_id, category_id, freq = map(int, line.strip().split())
                self.user_category_data[user_id][category_id] = freq

        print("User-Category data preloaded into memory.")

    def train(self, max_iters=50, learning_rate=1e-4):
        """
        Trains the Poisson Factorization model for User-Category interactions.

        :param max_iters: Number of training iterations.
        :param learning_rate: Learning rate for SGD updates.
        """
        ctime = time.time()
        print("Training User-Category MF...")

        user_num, category_num = self.get_user_category_counts()

        self.U = 0.3 * np.sqrt(np.random.gamma(self.alpha, self.beta, (user_num, self.K))) / self.K
        self.C = 0.3 * np.sqrt(np.random.gamma(self.alpha, self.beta, (category_num, self.K))) / self.K

        # Convert F_uc to sparse representation
        F_uc = self.create_user_category_matrix()
        F_uc = F_uc.tocoo()  # Convert to coordinate format for efficient operations
        entry_index = list(zip(F_uc.row, F_uc.col))

        F_uc = F_uc.tocsr()
        F_dok = F_uc.todok()


        tau = 10  # Learning rate decay
        last_loss = float('Inf')

        for iters in range(max_iters):
            F_Y = F_dok.copy()
            for i, j in entry_index:
                F_Y[i, j] = 1.0 * F_dok[i, j] / self.U[i].dot(self.C[j]) - 1
            F_Y = F_Y.tocsr()

            learning_rate_k = learning_rate * tau / (tau + iters)
            self.U += learning_rate_k * (F_Y.dot(self.C) + (self.alpha - 1) / self.U - 1 / self.beta)
            self.C += learning_rate_k * ((F_Y.T).dot(self.U) + (self.alpha - 1) / self.C - 1 / self.beta)

            # Compute Loss
            loss = 0.0
            for i, j in entry_index:
                loss += (F_dok[i, j] - self.U[i].dot(self.C[j])) ** 2
            print('Iteration:', iters, 'loss:', loss)

            if loss > last_loss:
                print("Early termination.")
                break
            last_loss = loss

        print("Done. Elapsed time:", time.time() - ctime, "s")

    def predict(self, user_id, location_id, sigmoid=False):
        """
        Predicts the interaction score between a user and a category.

        :param user_id: The user ID.
        :param category_id: The category ID.
        :param sigmoid: Whether to apply a sigmoid function for probability scaling.
        :return: Predicted interaction score.
        """
        # Get categories associated with this location
        categories = self.location_to_categories.get(location_id, None)

        if not categories:
            return 0  # No category information for this location

        # Compute the user's affinity score for each category
        category_scores = [self.U[user_id].dot(self.C[cat]) for cat in categories]

        # Aggregate the scores: Use mean or sum
        return np.mean(category_scores)  # Alternative: np.sum(category_scores)

    def save_model(self, path):
        """Saves the learned user and category latent factors."""
        ctime = time.time()
        print("Saving U and C...")
        np.save(path + "UCM_U", self.U)
        np.save(path + "UCM_C", self.C)
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def load_model(self, path):
        """Loads the learned user and category latent factors."""
        ctime = time.time()
        print("Loading U and C...")
        self.U = np.load(path + "UCM_U.npy")
        self.C = np.load(path + "UCM_C.npy")
        print("Done. Elapsed time:", time.time() - ctime, "s")

#
# # Example usage:
# # Define file paths
# train_file = "C:/project_RS/Dataset/Yelp/Yelp_train.txt"
# poi_category_file = "C:/project_RS/Dataset/Yelp/Yelp_poi_categories.txt"
# output_user_category_file = "C:/project_RS/Dataset/Yelp/user_category_freq.txt"
# model_save_path = "C:/project_RS/Dataset/Yelp/"  # Directory to save model
#
# # Initialize the model
# UCM = UserCategoryModel(train_file, poi_category_file, output_user_category_file, K=30, alpha=20.0, beta=0.2)
#
# # Step 1: Create the user-category matrix
# if not os.path.exists(output_user_category_file):
#     print("Creating User-Category matrix...")
#     UCM.create_user_category_matrix()
# else:
#     print("User-Category matrix already exists, skipping creation.")
#
# # Step 2: Train the MF model
# print("\nTraining Matrix Factorization for User-Category interactions...")
# UCM.train(max_iters=50, learning_rate=1e-4)
#
# # Step 3: Save the trained model
# UCM.save_model(model_save_path)
#
# print("\nTraining completed! Model saved at:", model_save_path)
#
# user_id = 123  # Example user
# category_id = 10  # Example category
#
# score = UCM.predict(user_id, category_id)
# print(user_id, category_id, score)

import numpy as np
from collections import defaultdict
import time
import os  # Ensure file existence

class UserCategoryModel:
    def __init__(self, train_file, poi_category_file, output_file):
        """
        Initializes the model with file paths and prepares for data processing.
        :param train_file: Path to Yelp_train.txt (User-Location-Frequency data).
        :param poi_category_file: Path to Yelp_poi_categories.txt (Location-Category data).
        :param output_file: Path to save the User-Category-Frequency data.
        """
        self.train_file = train_file
        self.poi_category_file = poi_category_file
        self.output_file = output_file

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

        location_to_categories = defaultdict(set)
        with open(self.poi_category_file, 'r') as f:
            for line in f:
                location_id, category_id = map(int, line.strip().split())
                location_to_categories[location_id].add(category_id)

        F_uc = np.zeros((user_num, category_num))

        with open(self.train_file, 'r') as f:
            for line in f:
                user_id, location_id, frequency = map(int, line.strip().split())

                if location_id in location_to_categories:
                    for category_id in location_to_categories[location_id]:
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
        return F_uc

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

    def predict(self, user_id, location_id):
        """
        Predicts a score for a user-location pair based on category interactions, using L1 normalization.

        :param user_id: The user ID.
        :param location_id: The location ID.
        :return: Normalized score for the user-location pair.
        """

        # Check if location_id exists
        if location_id not in self.location_to_categories:
            return 0  # No category data for location

        # Get user-category interactions from preloaded data
        user_category_freq = self.user_category_data.get(user_id, {})
        total_visits = float(sum(user_category_freq.values()))

        # Compute score
        score = float(sum(user_category_freq.get(c, 0) for c in self.location_to_categories[location_id]))

        # Normalize
        if total_visits > 0:
            score = float(score / total_visits)
        else:
            score = 0

        return score


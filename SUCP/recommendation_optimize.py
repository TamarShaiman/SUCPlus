import numpy as np
import torch
from lib.metrics import precisionk, recallk, ndcgk
import time
import scipy.sparse as sparse
from collections import defaultdict
import os
import sys
from lib.PoissonFactorModel import PoissonFactorModel
from lib.MultiGaussianModel import MultiGaussianModel
from lib.TimeAwareMF import TimeAwareMF
from lib.FriendBasedCF import FriendBasedCF
from lib.LocationFriendshipBookmarkColoringAlgorithm import LocationFriendshipBookmarkColoringAlgorithm

# Convert ground truth to a PyTorch tensor format
def convert_ground_truth_to_tensor(ground_truth):
    return {uid: {lid: torch.tensor(relevance, dtype=torch.float32) for lid, relevance in lids.items()}
            for uid, lids in ground_truth.items()}

# Define the overall scoring function
def overall_scores(weights, uid, lid):
    return (weights[0] * PFM.predict(uid, lid) +
            weights[1] * MGMWT.predict(uid, lid) +
            weights[2] * MGMLT.predict(uid, lid) +
            weights[3] * TAMF.predict(uid, lid) +
            weights[4] * LFBCA.predict(uid, lid))

# Use PyTorch’s built-in MSE loss function
mse_loss_fn = torch.nn.MSELoss()

def optimize_weights(ground_truth_tensor, learning_rate=0.01, max_iters=1000, tol=1e-6):
    # Initialize trainable weights as PyTorch tensors
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], requires_grad=True)

    # Define PyTorch SGD optimizer
    optimizer = torch.optim.SGD([weights], lr=learning_rate)

    for it in range(max_iters):
        optimizer.zero_grad()  # Clear previous gradients

        # Collect predicted and actual scores for all users and items
        all_predicted_scores, all_actual_scores = [], []

        for uid in ground_truth_tensor:
            for lid in ground_truth_tensor[uid]:
                predicted_score = overall_scores(weights, uid, lid)
                actual_score = ground_truth_tensor[uid][lid]

                all_predicted_scores.append(predicted_score)
                all_actual_scores.append(actual_score)

        # Convert predictions and ground truth to PyTorch tensors
        predicted_tensor = torch.stack(all_predicted_scores)
        actual_tensor = torch.stack(all_actual_scores)

        # Compute MSE loss using PyTorch’s built-in function
        loss = mse_loss_fn(predicted_tensor, actual_tensor)
        loss.backward()  # Compute gradients automatically

        optimizer.step()  # Update weights using the computed gradients

        # Print progress
        print("Iteration {it + 1}, Loss: {loss.item()}")

        # Convergence check based on gradient magnitude
        if torch.norm(weights.grad) < tol:
            print("Converged.")
            break

    return weights.detach().numpy()  # Return optimized weights as NumPy array

def read_ground_truth():
    ground_truth = defaultdict(set)
    truth_data = open(test_file, 'r').readlines()
    for eachline in truth_data:
        uid, lid, _ = eachline.strip().split()
        uid, lid = int(uid), int(lid)
        ground_truth[uid].add(lid)
    print("The loading of Ground Truth Finished.")
    return ground_truth

def read_poi_coos():
    poi_coos = {}
    poi_data = open(poi_file, 'r').readlines()
    for eachline in poi_data:
        lid, lat, lng = eachline.strip().split()
        lid, lat, lng = int(lid), float(lat), float(lng)
        poi_coos[lid] = (lat, lng)
    return poi_coos

def read_training_data2():
    train_data = open(train_file, 'r').readlines()
    training_matrix = np.zeros((user_num, poi_num))
    for eachline in train_data:
        uid, lid, freq = eachline.strip().split()
        uid, lid, freq = int(uid), int(lid), int(freq)
        training_matrix[uid, lid] = freq
    return training_matrix

def read_friend_data():
    social_data = open(social_file, 'r').readlines()
    social_matrix = np.zeros((user_num, user_num))
    for eachline in social_data:
        uid1, uid2 = eachline.strip().split()
        uid1, uid2 = int(uid1), int(uid2)
        social_matrix[uid1, uid2] = 1.0
        social_matrix[uid2, uid1] = 1.0
    return social_matrix


def read_training_data():
    # load train data
    train_data = open(train_file, 'r').readlines()
    sparse_training_matrix = sparse.dok_matrix((user_num, poi_num))
    training_matrix = np.zeros((user_num, poi_num))
    training_tuples = set()
    for eachline in train_data:
        uid, lid, freq = eachline.strip().split()
        uid, lid, freq = int(uid), int(lid), int(freq)
        sparse_training_matrix[uid, lid] = freq
        training_matrix[uid, lid] = 1.0
        training_tuples.add((uid, lid))

    # load checkins
    # time_list_hour = open("./result/time_hour" + ".txt", 'w')
    check_in_data = open(check_in_file, 'r').readlines()
    training_tuples_with_day = defaultdict(int)
    training_tuples_with_time = defaultdict(int)
    for eachline in check_in_data:
        uid, lid, ctime = eachline.strip().split()
        uid, lid, ctime = int(uid), int(lid), float(ctime)
        if (uid, lid) in training_tuples:
            hour = time.gmtime(ctime).tm_hour
            training_tuples_with_time[(hour, uid, lid)] += 1.0
            if 8 <= hour < 18:
                # working time
                hour = 0
            elif hour >= 18 or hour < 8:
                # leisure time
                hour = 1

            training_tuples_with_day[(hour, uid, lid)] += 1.0

    # Default setting: time is partitioned to 24 hours.
    sparse_training_matrices = [sparse.dok_matrix((user_num, poi_num)) for _ in range(24)]
    for (hour, uid, lid), freq in training_tuples_with_time.items():
        sparse_training_matrices[hour][uid, lid] = 1.0 / (1.0 + 1.0 / freq)

    # Default setting: time is partitioned to WD and WE.
    sparse_training_matrix_WT = sparse.dok_matrix((user_num, poi_num))
    sparse_training_matrix_LT = sparse.dok_matrix((user_num, poi_num))

    for (hour, uid, lid), freq in training_tuples_with_day.items():
        if hour == 0:
            sparse_training_matrix_WT[uid, lid] = freq
        elif hour == 1:
            sparse_training_matrix_LT[uid, lid] = freq

    print ("Data Loader Finished!")
    return sparse_training_matrices, sparse_training_matrix, sparse_training_matrix_WT, sparse_training_matrix_LT, training_tuples, training_matrix

def main(result_dir_name, tmp_dir_name):
    # Reading data
    sparse_training_matrices, sparse_training_matrix, sparse_training_matrix_WT, sparse_training_matrix_LT, training_tuples, training_matrix = read_training_data()
    ground_truth = read_ground_truth()
    ground_truth_tensor = convert_ground_truth_to_tensor(ground_truth)  # Convert to PyTorch tensors
    training_matrix2 = read_training_data2()
    poi_coos = read_poi_coos()
    social_matrix = read_friend_data()

    # Train recommendation models (PFM, TAMF, etc.)
    PFM.train(sparse_training_matrix, max_iters=10, learning_rate=1e-4)
    if not os.path.exists("./tmp2/"):
        os.makedirs("./tmp2/")
    PFM.save_model("./tmp2/")
    PFM.load_model("./tmp2/")

    MGMWT.multi_center_discovering(sparse_training_matrix_WT, poi_coos)
    MGMLT.multi_center_discovering(sparse_training_matrix_LT, poi_coos)

    TAMF.train(sparse_training_matrices, tmp_dir_name, max_iters=30, load_sigma=False)
    TAMF.save_model("./tmp2/")
    TAMF.load_model("./tmp2/")

    LFBCA.precompute_rec_scores(training_matrix2, social_matrix)
    LFBCA.save_result("./tmp2/")

    # Optimize the weights using gradient descent with MSE
    optimized_weights = optimize_weights(ground_truth_tensor)
    print("Optimized Weights:", optimized_weights)

    # Final scoring and evaluation
    evaluate_model(optimized_weights, ground_truth, top_k=10)

def evaluate_model(weights, ground_truth, top_k=10):
    precision_10, recall_10, nDCG_10 = [], [], []

    # Predict and evaluate for each user
    for uid in ground_truth:
        overall_scores_per_user = [
            overall_scores(weights, uid, lid) for lid in range(poi_num)
        ]

        # Rank and get top-k items
        ranked_lids = list(reversed(np.argsort(overall_scores_per_user)))[:top_k]
        actual = ground_truth[uid]

        # Calculate metrics
        precision_10.append(precisionk(actual, ranked_lids))
        recall_10.append(recallk(actual, ranked_lids))
        nDCG_10.append(ndcgk(actual, ranked_lids))

    print("Average Precision@10:", np.mean(precision_10))
    print("Average Recall@10:", np.mean(recall_10))
    print("Average nDCG@10:", np.mean(nDCG_10))

if __name__ == '__main__':
    data_name = sys.argv[1]
    beta_value = sys.argv[2]

    print("======= RUNNING FOR BETA = ", beta_value, ", DATASET = ", data_name, "========")

    if data_name == 'gowalla':
        data_dir = "./Dataset/Gowalla/"
        size_file = data_dir + "Gowalla_data_size.txt"
        check_in_file = data_dir + "Gowalla_checkins.txt"
        train_file = data_dir + "Gowalla_train.txt"
        test_file = data_dir + "Gowalla_test.txt"
        poi_file = data_dir + "Gowalla_poi_coos.txt"
        social_file = data_dir + "Gowalla_social_relations.txt"
    else:
        data_dir = "./Dataset/Yelp/"
        size_file = data_dir + "Yelp_data_size.txt"
        check_in_file = data_dir + "Yelp_checkins.txt"
        train_file = data_dir + "Yelp_train.txt"
        test_file = data_dir + "Yelp_test.txt"
        poi_file = data_dir + "Yelp_poi_coos.txt"
        social_file = data_dir + "Yelp_social_relations.txt"

    user_num, poi_num = map(int, open(size_file).readline().strip().split())
    top_k = 100

    PFM = PoissonFactorModel(K=30, alpha=20.0, beta=0.2)
    MGMWT = MultiGaussianModel(alpha=0.2, theta=0.02, dmax=15)
    MGMLT = MultiGaussianModel(alpha=0.2, theta=0.02, dmax=15)
    TAMF = TimeAwareMF(K=100, Lambda=1.0, beta=2.0, alpha=2.0, T=24)
    LFBCA = LocationFriendshipBookmarkColoringAlgorithm(alpha=0.85, beta=float(beta_value), epsilon=0.001)

    tmp_dir_name = "./tmp2_{data_name}_{beta_value}/"
    result_dir_name = "./result2_{data_name}_{beta_value}/"
    os.makedirs(tmp_dir_name, exist_ok=True)
    os.makedirs(result_dir_name, exist_ok=True)

    main(result_dir_name, tmp_dir_name)

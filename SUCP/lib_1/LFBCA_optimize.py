import time
import numpy as np
from collections import deque
from numpy.linalg import norm
import networkx as nx

class LFBCA_optimize(object):
    def __init__(self, alpha, beta, epsilon):
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.rec_score = None

    def PPR(self, u, friends, sim):
        alpha = self.alpha
        epsilon = self.epsilon

        q = deque()
        q_val = {}
        q.append(u)
        q_val[u] = 1.0
        ppr = np.zeros(sim.shape[0])

        while q:
            i = q.popleft()
            w = q_val[i]
            del q_val[i]

            ppr[i] += alpha * w
            if w > epsilon:
                for j in friends[i]:
                    if j in q_val:
                        q_val[j] += (1 - alpha) * w * sim[i, j]
                    else:
                        q_val[j] = (1 - alpha) * w * sim[i, j]
                        q.append(j)
        return ppr

    # Lior Added:
    # --------------------------------------------------------------------
    def compute_graph_centrality(self, social_matrix, centrality_type='eigenvector'):
        try:
            # Build the graph using NetworkX
            G = nx.from_numpy_array(social_matrix)

            # Compute the chosen centrality measure
            if centrality_type == 'eigenvector':
                centrality = nx.eigenvector_centrality_numpy(G)
            elif centrality_type == 'closeness':
                centrality = nx.closeness_centrality(G)
            elif centrality_type == 'betweenness':
                centrality = nx.betweenness_centrality(G)
            elif centrality_type == 'degree':
                centrality = nx.degree_centrality(G)
            elif centrality_type == 'pagerank':
                centrality = nx.pagerank(G)
            else:
                raise ValueError(
                    "Unknown centrality type: {centrality_type}. Choose from ['eigenvector', 'closeness', 'betweenness', 'degree', 'pagerank']."
                )

            # Convert centrality scores to a numpy array and normalize them
            centrality_scores = np.array(list(centrality.values()))
            centrality_scores /= np.sum(centrality_scores)  # Normalize

            return centrality_scores

        except Exception as e:
            print("Error computing centrality ({centrality_type}): {e}")
            return np.zeros(len(social_matrix))
    # --------------------------------------------------------------------

    def precompute_user_social_similarities(self, check_in_matrix, social_matrix, centrality_type='eigenvector'):
        C = check_in_matrix

        ctime = time.time()
        print("Computing user similarities...", )

        user_sim = C.dot(C.T)
        norms = [norm(C[i]) for i in range(C.shape[0])]

        for i in range(C.shape[0]):
            user_sim[i][i] = 0.0
            for j in range(i+1, C.shape[0]):
                user_sim[i, j] /= (norms[i] * norms[j])
                user_sim[j, i] /= (norms[i] * norms[j])

        for uid in range(user_sim.shape[0]):
            if not sum(user_sim[uid]) == 0:
                user_sim[uid] /= sum(user_sim[uid])
        print("Done. Elapsed time:", time.time() - ctime, "s")

        ctime = time.time()
        print("Computing social similarities...", )
        # Lior Added:
        # --------------------------------------------------------------------
        # Compute centrality scores (default: eigenvector centrality)
        centrality_scores = self.compute_graph_centrality(social_matrix, centrality_type)

        # Scale social similarities by centrality scores
        for uid in range(social_matrix.shape[0]):
            social_matrix[uid, :] *= centrality_scores[uid]  # Scale each user's connections by centrality
        # --------------------------------------------------------------------
        social_sim = social_matrix
        for uid in range(social_sim.shape[0]):
            if not sum(social_sim[uid]) == 0:
                social_sim[uid] /= sum(social_sim[uid])
        print("Done. Elapsed time:", time.time() - ctime, "s")

        print(user_sim, social_sim)
        print(type(user_sim), type(social_sim))
        return self.beta * user_sim + (1 - self.beta) * social_sim

    def compute_ppr_for_all_users(self, sim):
        ctime = time.time()
        print("Computing PPR values for all users...", )
        edges = (sim > 0)
        friends = [np.where(edges[uid, :] > 0)[0] for uid in range(sim.shape[0])]
        all_ppr = [self.PPR(uid, friends, sim) for uid in range(sim.shape[0])]
        print("Done. Elapsed time:", time.time() - ctime, "s")
        return np.array(all_ppr)

    def precompute_rec_scores(self, check_in_matrix, social_matrix, centrality_type='eigenvector'):
        sim = self.precompute_user_social_similarities(check_in_matrix, social_matrix, centrality_type)
        all_ppr = self.compute_ppr_for_all_users(sim)
        normalized_check_in_matrix = np.zeros(check_in_matrix.shape)
        for uid in range(normalized_check_in_matrix.shape[0]):
            normalized_check_in_matrix[uid, :] = check_in_matrix[uid, :] / np.sum(check_in_matrix[uid, :])

        ctime = time.time()
        print("Precomputing recommendation scores...", )
        for uid in range(all_ppr.shape[0]):
            all_ppr[uid, uid] = 0.0
        # self.rec_score = all_ppr.dot(normalized_check_in_matrix)
        # Lior Added:
        # --------------------------------------------------------------------
        gamma = 0.5  # Weight for outward influence

        # Outward influence: users influence others
        outward_influence = all_ppr.dot(normalized_check_in_matrix)

        # Inward influence: users are influenced by others
        inward_influence = normalized_check_in_matrix.dot(all_ppr)

        # Combine both influences
        self.rec_score = gamma * outward_influence + (1 - gamma) * inward_influence
        # --------------------------------------------------------------------
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def save_result(self, path):
        ctime = time.time()
        print("Saving result...",)
        np.save(path + "rec_score", self.rec_score)
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def predict(self, i, j):
        return self.rec_score[i][j]

from scipy.signal import convolve2d
from random import choice
import numpy as np
import unittest

def stats(x):
    mean = np.mean(x)
    median = np.median(x)
    std = np.nanstd(x, ddof=1)
    variance = np.var(x, ddof=1)
    print(f"Mean: {mean}\nMedian: {median}\nStandard Deviation: {std}\nVariance: {variance}")
    return mean, median, std, variance

def manhattan_distance(x, y):
    distance = np.sum([np.abs(a - b) for a, b in zip(x, y)])
    print(f"Manhattan Distance between {x} and {y}: {distance}")
    return distance

def euclidean_distance(x, y):
    distance = np.sqrt(np.sum([(a - b)**2 for a, b in zip(x, y)]))
    print(f"Euclidean Distance between {x} and {y}: {distance}")
    return distance

def n_norm(x, y, n):
    norm = np.power(np.sum([(a - b)**n for a, b in zip(x, y)]), 1/n)
    print(f"{n}-norm L{n} between {x} and {y}: {norm}")
    return norm

def chebyshev_distance(x, y):
    distance = np.max([np.abs(a - b) for a, b in zip(x, y)])
    print(f"Chebyshev Distance between {x} and {y}: {distance}")
    return distance

def levenshtein_rec(x,y):
    if len(x) == 0:
        return len(y)
    if len(y) == 0:
        return len(x)
    if x[0] == y[0]:
        return levenshtein_rec(x[1:], y[1:])
    else:
        return min(levenshtein_rec(x[1:], y) + 1, levenshtein_rec(x, y[1:]) + 1, levenshtein_rec(x[1:], y[1:]) + 1)

def levenshtein_distance(x, y):
    distance = levenshtein_rec(x, y)
    print(f"Levenshtein Distance between {x} and {y}: {distance}")
    return distance

def hamming_distance(x, y):
    if len(x) != len(y):
        raise ValueError("len(x) != len(y)") 
    distance = np.sum([1 for a, b in zip(x, y) if a != b])   
    print(f"Hamming Distance between '{x}' and '{y}': {distance}")
    return distance

def wup_relatedness():
    print(f"go here http://ws4jdemo.appspot.com/")

def covar_mat_eig(a):
    eig_vals, eig_vecs = np.linalg.eig(a)
    print(f"Eigenvalues: {eig_vals}")
    print(f"Eigenvectors: {eig_vecs}")
    return eig_vals, eig_vecs


def linear_regression(x, y):
    # create matrix
    matrix_x = np.empty((len(x), 2))
    matrix_y = np.empty((len(y), 1))
    # add a column of 1s in x matrix
    for i in range(0, len(x)):
        matrix_x[i][0] = 1
        matrix_x[i][1] = x[i]
        matrix_y[i][0] = y[i]
    # use formula
    result = np.matrix(matrix_x.T @ matrix_x).I @ matrix_x.T @ matrix_y
    print('fit_Wh: ')
    print(result)
    print('Equation: y = {} + {} x'.format(result[0, 0], result[1, 0]))
    return result[0, 0], result[1, 0]

def regularised(x, y, sigma, lam=1):
    # create matrix
    matrix_x = np.empty((len(x), 2))
    matrix_y = np.empty((len(y), 1))
    # add a column of 1s in x matrix
    for i in range(0, len(x)):
        matrix_x[i][0] = 1
        matrix_x[i][1] = x[i]
        matrix_y[i][0] = y[i]

    # get reg
    reg = sigma ** 2
    # get diagonal matrix
    diagonal = np.eye(matrix_x.shape[1])
    diagonal = reg * lam * diagonal

    # use formula
    result = np.matrix(matrix_x.T @ matrix_x + diagonal).I @ matrix_x.T @ matrix_y
    print('fit_Wh: ')
    print(result)
    print('Equation: y = {} + {} x'.format(result[0, 0], result[1, 0]))
    return result[0, 0], result[1, 0]

def classification_MLE_two_class(l0, l1):
    likelihood = 0
    likelihood += np.sum([(-np.log(1 + np.e**i)) for i in l0])
    likelihood += np.sum([(-np.log(1 + np.e**(-i))) for i in l1])
    print('Log-Likelihood: {}'.format(likelihood))
    return likelihood

def weight_calculator(distance, b):
    weight = np.e**(-distance/(2*b))
    print(f"Weight: {weight}")
    return weight

def logits_to_probabilities(logits):
    return [1/1+np.exp(-logit) for logit in logits]

def convolute_matrices(a, b, dimensions = 1):
    factor = sum([abs(i) for i in np.array(a).flatten()])
    if dimensions == 1:
        return factor, np.convolve(a, b, mode='valid')
    elif dimensions == 2:
        return factor, convolve2d(a, b, mode='valid')

def k_nearest(xs, labels, n, k):
    distances = []
    for x in xs:
        print(x, n)
        distances.append(euclidean_distance(x, n))
    #print(distances)
    closest_indices = []
    for _ in range(0, k):
        x = min(distances)
        idx = distances.index(x)
        closest_indices.append(idx)
        distances.pop(idx)
    closest_labels = [labels[i] for i in closest_indices]
    belongs_to = max(set(closest_labels), key=closest_labels.count)
    print(f"new datapoint {n} belongs in class {belongs_to}")
    return belongs_to

def weighted_nearest(xs, labels, n, b):
    distances = []
    for x in xs:
        distances.append(euclidean_distance(x, n)**2)
    weights = [weight_calculator(distance, b) for distance in distances]
    label_set = list(set(labels))
    weights_dict = {}
    for label in label_set:
        weights_dict[label] = []
        for i, x in enumerate(xs):
            if labels[i] == label:
                weights_dict[label].append(weights[i])
        weights_dict[label] = np.sum(weights_dict[label])/np.sum(weights)
    belongs_to = max(weights_dict, key=weights_dict.get)
    print(f"new datapoint {n} belongs in class {belongs_to}")
    return belongs_to

def k_means_e_step(xs):
    sum = np.zeros_like(xs[0])
    for x in xs:
        sum = np.add(sum, x)
    print(f"E-Step Result: 1/{len(xs)} * {sum}")
    return sum/len(xs)

def k_means_m_step(x, y):
    return euclidean_distance(x, y)

def k_means(xs, k, labels = None):
    if labels is None:
        labels = [choice(range(k)) for _ in xs]
    clusters = {i: [] for i in range(k)}
    for i, label in enumerate(labels):
        clusters[label].append(xs[i])
    centroids = []
    # E-Step
    for label, group in clusters.items():
        centroid = k_means_e_step(group)
        centroids.append({"label": label, "value": centroid})
    # M-Step
    results = []
    for x in xs:
        distances = []
        for centroid in centroids:
            distance = k_means_m_step(x, centroid["value"])
            distances.append({"label": centroid["label"], "centroid": centroid["value"], "x": x, "distance": distance**2})
        new_label = min(distances, key=lambda x: x["distance"])["label"]
        results.append(
            {"label": new_label, "x": x}
        )
    return results, centroids

def rss(xs, ys, f):
    # f is a function
    res = []
    for x, y in zip(xs, ys):
        res.append((y-(f(x)))**2)
    return sum(res)

class Tests(unittest.TestCase):

    def test_stats(self):
        mean, median, std, variance = stats([-3,2,4,6,-2,0,5])
        self.assertAlmostEqual(mean, 1.71, places=2)
        self.assertAlmostEqual(median, 2.0, places=2)
        self.assertAlmostEqual(std, 3.5, places=2)
        self.assertAlmostEqual(variance, 12.24, places=2)

    def test_manhattan_distance(self):
        self.assertEqual(manhattan_distance([4, 5, 6], [2, -1, 3]), 11)

    def test_euclidean_distance(self):
        self.assertEqual(euclidean_distance([4, 5, 6], [2, -1, 3]), 7.0)

    def test_n_norm(self):
        self.assertAlmostEqual(n_norm([4, 5, 6], [2, -1, 3], 3), 6.3, places=1)

    def test_chebyshev_distance(self):
        self.assertEqual(chebyshev_distance([4, 5, 6], [2, -1, 3]), 6)

    def test_levenshtein_distance(self):
        self.assertEqual(levenshtein_distance("water", "further"), 4)

    def test_hamming_distance(self):
        self.assertEqual(hamming_distance("weather", "further"), 3)
        self.assertRaises(ValueError, hamming_distance, "weather", "further1")
    
    def test_one_d_convolution(self):
        a = [1, 2, 1]
        b = [2, 2, 3, 3, 2]
        factor, conv = (convolute_matrices(a, b))
        self.assertEqual(factor, 4)
        self.assertEqual(conv.tolist(), np.array([ 9, 11, 11]).tolist())

    def test_two_d_convolution(self):
        a = [
            [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]
        ]
        b = [
            [0, 5, 5, 5, 0], 
            [0, 5, 10, 5, 0],
            [0, 10, 10, 10, 0],
            [0, 5, 10, 5, 0],
            [0, 5, 5, 5, 0]
        ]
        factor, conv = convolute_matrices(a, b, dimensions=2)
        self.assertEqual(factor, 8)
        self.assertEqual(conv.tolist(), np.array(
            [[-35,   0,  35],
            [-40,   0,  40],
            [-35,   0,  35]]
        ).tolist())
    
    def test_covariance_matrix_eig(self):
        a = np.array([[3.5, -6, 2],[-6,8.25, -4.5],[2, -4.5, 3.5]])
        eigVals, eigVecs = covar_mat_eig(a)
        expected_eigenvalues = np.array([14.47722755, -0.87333474,  1.64610719]).tolist()
        for i, eigVal in enumerate(eigVals):
            self.assertAlmostEqual(eigVal, expected_eigenvalues[i], places=2)
    
    def test_k_means_e_step(self):
        xs = np.array([
            [-3.4, -1.2],
            [3.2, 2.1],
            [0.6, 0]
        ])
        expected = np.array([0.133, 0.3]).tolist()
        result = k_means_e_step(xs).tolist()
        for i, r in enumerate(result):
            self.assertAlmostEqual(r, expected[i], places=2)
    
    def test_k_means_m_step(self):
        x = np.array([-2.1, -3.2])
        y = np.array([0.133, 0.3])
        self.assertAlmostEqual(euclidean_distance(x, y), 4.15, places = 2)

    def test_k_mean(self):
        xs = np.array([
            [-2.1, -3.2],
            [-3.4, -1.2],
            [-2.6, -2.7],
            [3.2, 2.1],
            [1.2, 3.6],
            [0.6, 0]
        ])
        labels = np.array([1, 0, 1, 0, 1, 0])
        k = 2
        print(k_means(xs, k, labels))

    def test_k_nearest(self):
        xs = np.array([
            [-2.1, -3.2],
            [-3.4, -1.2],
            [-2.6, -2.7],
            [3.2, 2.1],
            [1.2, 3.6],
            [0.6, 0]
        ])

        labels = np.array([0,0,0,1,1,1])
        n = [0.2, -0.3 ]
        k = 3
        self.assertEqual(k_nearest(xs, labels, n, k), 0)

    def test_weighted_nearest(self):
        xs = np.array([
            [-2.1, -3.2],
            [-3.4, -1.2],
            [-2.6, -2.7],
            [3.2, 2.1],
            [1.2, 3.6],
            [0.6, 0]
        ])

        labels = np.array([0,0,0,1,1,1])
        n = [0.2, -0.3 ]
        b = 2

        self.assertEqual(weighted_nearest(xs, labels, n, b), 1)

if __name__ == "__main__":
    unittest.main()


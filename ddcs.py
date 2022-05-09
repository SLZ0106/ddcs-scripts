from nltk.corpus import wordnet as wn
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
    print(f"Manhattan Distance: {distance}")
    return distance

def euclidean_distance(x, y):
    distance = np.sqrt(np.sum([(a - b)**2 for a, b in zip(x, y)]))
    print(f"Euclidean Distance: {distance}")
    return distance

def n_norm(x, y, n):
    norm = np.power(np.sum([(a - b)**n for a, b in zip(x, y)]), 1/n)
    print(f"{n}-norm L{n}: {norm}")
    return norm

def cherbyshev_distance(x, y):
    distance = np.max([np.abs(a - b) for a, b in zip(x, y)])
    print(f"Cherbyshew Distance: {distance}")
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
    print(f"Levenshtein Distance: {distance}")
    return distance

def hamming_distance(x, y):
    if len(x) != len(y):
        raise ValueError("len(x) != len(y)") 
    distance = np.sum([1 for a, b in zip(x, y) if a != b])   
    print(f"Hamming Distance: {distance}")
    return distance

def wup_relatedness():
    print(f"go here http://ws4jdemo.appspot.com/")

class W1Tests(unittest.TestCase):
    """
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

    def test_cherbyshew_distance(self):
        self.assertEqual(cherbyshev_distance([4, 5, 6], [2, -1, 3]), 6)

    def test_levenshtein_distance(self):
        self.assertEqual(levenshtein_distance("water", "further"), 4)

    def test_hamming_distance(self):
        self.assertEqual(hamming_distance("weather", "further"), 3)
        self.assertRaises(ValueError, hamming_distance, "weather", "further1")
    """


if __name__ == "__main__":
    unittest.main()

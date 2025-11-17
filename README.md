### Data Wrangling & Coding Interview (60 mins):
The Data Wrangling & Coding interview is focused on assessing your ability to manipulate, clean, and transform structured datasets using Python, particularly with the Pandas library.
You’ll work with realistic data scenarios, mirroring the kind of messy or incomplete data you might encounter in a production environment.


### Pandas / Python Operations

1. Replace the nulls with the average for its own category in the table table.
2. Return the values that fall outside one standard deviation of mean.

   
基本上考Pandas的一些操作，replace null value with group average


数据处理：
两个column，一个是string format的categorical variable，还有一个是value column，但是他估计给写成string format。里面有null，用‘null’来表示。
问怎样mean replace null for the value column by each category。
这道题比较恶心的他的数据列也用字符串表示，null就是用字符串的‘null’来表示，所以在把这一列转换成float，同时把‘null’转换成np.nan这一步花了一些时间。最后估计也是折在这个上面了。


#### Replace the nulls with the average for its own category
```
df['value'] = df['value'].fillna(df.groupby('category')['value'].transform('mean'))
df_filled_mean = df.fillna(df.mean())

# null就是用字符串的‘null’来表示，所以在把这一列转换成float，同时把‘null’转换成np.nan
df['value'] = pd.to_numeric(df['value'].replace('null', np.nan))
```

#### Return the values that fall outside one standard deviation of mean
```
mean = df['value'].mean()
std = df['value'].std()
# Create a boolean mask for values outside ±1 std
mask = (df['value'] < mean - std) | (df['value'] > mean + std)
outliers = df.loc[mask, 'value']
# combine to 1-liner
outliers = df.loc[(df['value'] - df['value'].mean()).abs() > df['value'].std(), 'value']
```

### Leetcode
#### DP - Triangle

I think for this question, we can use Dynamic Programming to store results for each row. From the question, I think one thing is, the minimum path to reach the bottom, from any element, only directly depends on the two elements below it.

A naive approach would be explore all possible paths from top to bottom, it will be very expensive like exponential time. So I’ll use Dynamic Programming instead.

For DP, we can start from the formula, from we observed:

where f(i,j) = minimum path sum from position (i, j) to the bottom.

we can compute from the last row, which is the base case.
Then move upward row by row, updating each element in the row, as the sum of itself and the minimum of the two possible nodes below.


```
def minimum_total(triangle: List[List[int]]) -> int:
    """
    Bottom-up DP with O(n) space, where n = rows.
    dp[j] = min path sum from current row j..end along an optimal path.
    """
    # Start from the last row
    dp = triangle[-1][:]

    # Work upward, shrinking the problem one row at a time
    for i in range(len(triangle) - 2, -1, -1):
        row = triangle[i]
        for j in range(len(row)):
            # from (i, j) you can go to (i+1, j) or (i+1, j+1)
            dp[j] = row[j] + min(dp[j], dp[j + 1])

    return dp[0]
```

Time complexity is O(N), where N is the total number of elements in the triangle - The bottom-up DP iterates through each element exactly once, performing O(1) work per element


#### Buy & Sell Stock (122+123)
2025 - [MLE] https://www.1point3acres.com/bbs/thread-1132581-1-1.html 
上来就一道hard 妖尔散
先让做妖尔尔，上来就说了brute force 两个for，然后sliding window秒了，需要真的run，类似于jupyter环境。

**122**


I think the idea is, any upward move can be captured. If we always buy before the price move up, sell at the next day. So the sum of all daily positive increases equals the maximum possible profit.
```
def maxProfit(prices):
    profit = 0
    for i in range(1, len(prices)):
        inc = prices[i] - prices[i-1]
        if inc > 0:
            profit += inc
    return profit
```

Complexity. O(n) time, O(1) space.


we can run some test cases for verification: 


Corner cases: single-element list; flat or strictly decreasing prices ⇒ no positive deltas ⇒ profit 0.

**123**


questions: “When you say ‘at most two transactions’, does that mean I can also make fewer than two if it’s not profitable?” 
For each day i, find the best 1-transaction profit on the left (using LC121 logic), and the best 1-transaction profit on the right


we can first create a helper function to compute the best single-transaction profit for any segment of the array
```
def one_transaction(prices):
    min_price = float('inf')
    profit = 0
    for p in prices:
        min_price = min(min_price, p)
        profit = max(profit, p - min_price)
    return profit
```

I run that once from the left and once from the right to get prefix and suffix profits, then combine them.
```
def maxProfit(prices):
    n = len(prices)
    if n < 2: return 0

    # prefix: best profit up to day i (using 121 logic)
    left = [0] * n
    minp = prices[0]
    for i in range(1, n):
        minp = min(minp, prices[i])
        left[i] = max(left[i-1], prices[i] - minp)

    # suffix: best profit after day i (using reversed 121 logic)
    right = [0] * n
    maxp = prices[-1]
    for i in range(n-2, -1, -1):
        maxp = max(maxp, prices[i])
        right[i] = max(right[i+1], maxp - prices[i])

    # combine two segments
    ans = 0
    for i in range(n-1):
        ans = max(ans, left[i] + right[i+1])
    return max(ans, left[-1])
```

O(n) time, O(n) space.
- Left/Right pass (best 1-trade profit up to i): O(n)
- Combine splits: O(n)
- Space for left[] and right[]: O(n)

#### 勾股定理
Given an N element array, find all triples a, b, and c, that satisfy: a**2 + b**2 = c**2.  Assume all elements are unique, positive integers. 注意算法优化。最优应该是o(n*2)

I’ll first sort the array and put all numbers into a set so I can check existence quickly.
 Then I’ll loop over every pair of numbers a and b, and calculate a*a + b*b.
 If that sum is a perfect square — meaning it equals some integer c squared — and c is also in the array, then (a, b, c) is a valid triple.
I’ll use math.isqrt() to safely check if the sum is a perfect square.

```
import math
from typing import List, Tuple

def triples(nums: List[int]) -> List[Tuple[int,int,int]]:
    """
    Return all (a, b, c) with a < b < c from nums such that a^2 + b^2 = c^2.
    Assumes nums has unique, positive integers.
    Roughly O(N²) time (we check every unordered pair (a,b)) and uses O(N) extra space for the set.
    """
    nums = sorted(nums)
    sq_nums = [x * x for x in nums]
    res = set()
    elements = set(nums) # store it in a set, for O(1) lookups

    for i in range(len(nums)):
        for j in range(i + 1, n):  # i<j so we don't reuse the same element
            c2 = sq[i] + sq[j]
            c = math.isqrt(c2) # check if c2 is a perfect square
            if c * c == c2 and c in elements:
                # Since all numbers are positive, c > a,b automatically
                a, b = nums[i], nums[j]
                res.add(tuple(sorted((a, b, c))))

    return sorted(res)

```


#### Valid Triangle Number - Leetcode 611
To form a valid triangle, the sum of any two sides should always be greater than the third side alone. i.e. a+b>c, b+c>a, a+c>b.

I think we should first sort the given array. After sorting the array, we can simplify the inequalities check. If we have the sorted 3 sides, a≤b≤c, then we only need to check if a+b>c. Since c≥b and c≥a, adding any number to c will always produce a sum which is greater than either a or b considered alone. 

And to find the valid triplet, we can fix the largest side and use 2-pointers techniques to find all 3 sides to form a triangle

```
def triangleNumber(nums):
    nums.sort() # so first sort it
    n = len(nums)
    count = 0

    # Then, we can loop through the sorted array from the back. So we fix the largest side, and use a two-pointer approach to count how many smaller pairs can form a triangle with it. 
    for k in range(n - 1, 1, -1):  # largest side at k
        i, j = 0, k - 1 # i is smallest side, j is the middle side
        while i < j:
            if nums[i] + nums[j] > nums[k]:
                # This means all pairs from i to j-1 will also work, hence, for this j, there are (j - i) valid pairs.
                count += j - i  # all i..(j-1) work with this j
                j -= 1 # Then, we move j left
            else:
                i += 1 # If the sum is not greater, we move i right to increase sum
    return count
```

Time Complexity: O(n²) 
- Sorting: O(n log n)
- Two-pointer sweep across all k: each inner loop moves i and j at most n times total per k, but amortizes to O(n²) overall.
- Space Complexity: O(1) (in-place sort aside).

#### 3sum

verify: Before jumping into coding, I’ll first make sure I fully understand the question. So, I need to return all unique triplets that sum up to zero. The order doesn’t matter, and duplicates shouldn’t appear in the final output, right?
explain: Brute force will be too slow, so for a better solution, I think we can solve this using 2-pointers. I think first we need to make this as a sorted array. The key idea is that once the array is sorted, I can fix one number, say nums[i], and then use two pointers to find the other two numbers that make the sum zero. Basically, for each fixed element in array, it becomes a 2-sum problem.

```
def threeSum(nums):

    # I’ll start by sorting the array. Sorting helps me use the two-pointer method and makes it easier to skip duplicates.
    nums.sort()
    res = []
    
    # Iterate through the array.
    for i in range(len(nums)):
        # Skip duplicates for the first element
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        # So we fix the i-th element at a time. Then I’ll try to find two other numbers whose sum equals -nums[i]
        # I’ll use two pointers — left starting from i+1, and right starting from the end of the array. I’ll move them inward depending on whether the total sum is less than, greater than, or equal to zero.
        left, right = i + 1, len(nums) - 1
        while left < right:
            s = nums[i] + nums[left] + nums[right]
            # If the sum is too small, move left rightwards to increase the sum; if too large, move right leftwards
            if s < 0:
                left += 1
            elif s > 0:
                right -= 1
            # If the sum equals zero, record the triplet.
            else:
                res.append([nums[i], nums[left], nums[right]])
                # then, we should make sure to skip over duplicate numbers on both sides so we don’t count the same combination again
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
    return res
```

Time Complexity: O(n²) 
    Sorting takes O(n log n), Then for each number, I use two pointers, which overall takes O(n²) time. So total complexity is O(n²)


### Implement decision tree
2025 - [MLE] https://www.1point3acres.com/bbs/thread-1149619-1-1.html 


#### What is a decision tree (DT)?


A decision tree partitions feature space with axis-aligned splits (e.g., “feature j ≤ τ?”). Each internal node asks a yes/no question; each leaf stores a prediction. Can be either classification or regression
- Classification: leaf predicts a class (optionally class probabilities).
- Regression: leaf predicts a numeric value (usually the mean of training targets in that leaf).

At each node we choose the locally best split according to an impurity/variance reduction measure.

#### impurity / loss (the math)
In a decision tree, each node (or region) contains a subset of training samples with certain labels.
We want every node to be as pure as possible — meaning the samples inside mostly belong to one class (for classification) or have similar target values (for regression).

When building a tree, we repeatedly choose a feature and a threshold to split the data. So we need a criterion to measure: “How good is this split at separating the data into purer groups?”

So we calculate the impurity, then pick the split that gives the biggest reduction in impurity, also called:
Information Gain=Impurity(parent)−Weighted Average Impurity(children)

#### How impurity is calculated
For a node containing a set of samples S with labels y:

[screenshot - need check GDoc]


#### training algorithm (binary splits)
1. Start with root node holding all data.
1. If stopping criteria met → make a leaf (store class histogram or mean).
1. Else, for each feature:
   1. Enumerate candidate thresholds (numeric) or category subsets (categorical trick).
   1. Compute split gain.
1. Choose split with max gain; partition data; recurse on children.
1. (Optional) After full growth, run cost-complexity pruning with CV to pick α\alphaα.

   
#### Hyper-parameters

- Split criterion: gini or entropy (classification), mse (regression).
- Depth/size: max_depth, max_leaf_nodes.
- Sample thresholds: min_samples_split, min_samples_leaf.
- Feature subsampling (usually for ensembles): max_features.
- Splitter strategy: best vs random.
- Pruning: ccp_alpha (cost-complexity).
- Class imbalance: class_weight or thresholding on predicted probabilities.

```
import numpy as np
from collections import Counter

# ------------------------------
# Utility functions
# ------------------------------

def gini(y):
    """
    Compute Gini impurity for a set of class labels y.
    - y: 1D array of integer class labels (e.g., [0, 1, 1, 0, 2])
    Formula: 1 - sum(p_k^2)
    where p_k = proportion of class k in y
    """
    n = len(y)
    if n == 0: 
        return 0.0
    counts = np.bincount(y)             # count samples per class
    p = counts / n                      # convert to probabilities
    return 1.0 - np.sum(p * p)          # probability two random labels differ


def mse(y):
    """
    Compute variance (Mean Squared Error) impurity for regression.
    The mean of y minimizes MSE, so this measures target spread.
    """
    if len(y) == 0:
        return 0.0
    mean = np.mean(y)
    return np.mean((y - mean) ** 2)


# ------------------------------
# Node structure
# ------------------------------

class Node:
    """
    A single node in the decision tree.
    Internal nodes store (feature_index, threshold) and child links.
    Leaf nodes store the prediction value or class histogram.
    """
    __slots__ = ("feat", "thr", "left", "right", "value", "n", "class_counts")

    def __init__(self, feat=None, thr=None, left=None, right=None, value=None, n=0, class_counts=None):
        self.feat = feat              # which feature index to split on
        self.thr = thr                # numeric threshold
        self.left = left              # left child node
        self.right = right            # right child node
        self.value = value            # prediction (class or mean)
        self.n = n                    # number of samples in this node
        self.class_counts = class_counts  # for classification probabilities


# ------------------------------
# Main Decision Tree class
# ------------------------------

class DecisionTree:
    """
    Minimal CART implementation for both classification and regression.
    - task: "classification" or "regression"
    - criterion: "gini" / "entropy" / "mse"
    """
    def __init__(self, task="classification", criterion=None,
                 max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.task = task
        # default criterion
        if criterion is None:
            criterion = "gini" if task == "classification" else "mse"
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    # choose impurity function depending on type
    def _impurity(self, y):
        if self.task == "classification":
            return gini(y)
        else:
            return mse(y)

    # compute what value to store in a leaf node
    def _leaf_value(self, y):
        if self.task == "classification":
            counts = np.bincount(y)
            return np.argmax(counts)      # majority class
        else:
            return float(np.mean(y))      # mean target for regression

    # ----------------------------------------------------
    # Find the best threshold for ONE numeric feature
    # ----------------------------------------------------
    def _best_split_numeric(self, X_col, y):
        """
        X_col: one column (feature vector)
        y: target labels
        Returns: (best_gain, best_threshold)
        """

        n = len(y)
        # sort data by the feature to evaluate thresholds efficiently
        order = np.argsort(X_col)
        Xs, ys = X_col[order], y[order]

        parent_imp = self._impurity(ys)
        best_gain, best_thr = -np.inf, None

        # For classification we can track running counts of classes
        if self.task == "classification":
            K = np.max(y) + 1
            left_counts = np.zeros(K, dtype=np.int64)
            right_counts = np.bincount(ys, minlength=K).astype(np.int64)

            for i in range(n - 1):
                # move one sample from right to left
                c = ys[i]
                left_counts[c] += 1
                right_counts[c] -= 1

                # skip identical feature values (no real split)
                if Xs[i] == Xs[i + 1]:
                    continue

                nl, nr = i + 1, n - (i + 1)
                if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                    continue

                # compute Gini for left and right child using class proportions
                pL = left_counts / nl
                pR = right_counts / nr
                gL = 1 - np.sum(pL * pL)
                gR = 1 - np.sum(pR * pR)

                # weighted average impurity after split
                weighted_imp = (nl / n) * gL + (nr / n) * gR

                # gain = impurity reduction
                gain = parent_imp - weighted_imp
                if gain > best_gain:
                    best_gain = gain
                    best_thr = 0.5 * (Xs[i] + Xs[i + 1])  # midpoint between values

            return best_gain, best_thr

        # For regression use variance reduction
        else:
            ycum = np.cumsum(ys)
            y2cum = np.cumsum(ys ** 2)
            for i in range(n - 1):
                if Xs[i] == Xs[i + 1]:
                    continue

                nl = i + 1
                nr = n - nl
                if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                    continue

                # prefix sums to compute mean and variance of left/right efficiently
                sumL, sum2L = ycum[i], y2cum[i]
                sumR, sum2R = ycum[-1] - sumL, y2cum[-1] - sum2L

                meanL = sumL / nl
                meanR = sumR / nr
                varL = (sum2L / nl) - meanL ** 2
                varR = (sum2R / nr) - meanR ** 2

                # weighted average variance after split
                weighted_imp = (nl / n) * varL + (nr / n) * varR
                gain = parent_imp - weighted_imp

                if gain > best_gain:
                    best_gain = gain
                    best_thr = 0.5 * (Xs[i] + Xs[i + 1])

            return best_gain, best_thr

    # ----------------------------------------------------
    # Recursive tree building
    # ----------------------------------------------------
    def _build(self, X, y, depth):
        """
        Recursively build the decision tree.
        Each call either creates a LEAF or splits into subnodes.
        """
        node = Node(n=len(y))

        # stopping conditions (pre-pruning)
        if (self.max_depth is not None and depth >= self.max_depth) \
           or len(y) < self.min_samples_split \
           or len(np.unique(y)) == 1:     # pure node
            node.value = self._leaf_value(y)
            if self.task == "classification":
                node.class_counts = Counter(y)
            return node

        best_gain, best_feat, best_thr = -np.inf, None, None
        n_features = X.shape[1]

        # evaluate all features and pick the best split
        for j in range(n_features):
            gain, thr = self._best_split_numeric(X[:, j], y)
            if gain is not None and gain > best_gain:
                best_gain, best_feat, best_thr = gain, j, thr

        # if no useful split, make a leaf
        if best_gain <= 0 or best_feat is None:
            node.value = self._leaf_value(y)
            if self.task == "classification":
                node.class_counts = Counter(y)
            return node

        # perform the actual split into left/right subsets
        left_mask = X[:, best_feat] <= best_thr
        right_mask = ~left_mask

        # store split info in node
        node.feat = best_feat
        node.thr = best_thr

        # recursively build children
        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build(X[right_mask], y[right_mask], depth + 1)
        return node

    # public API: train the tree
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.root = self._build(X, y, depth=0)
        return self

    # ----------------------------------------------------
    # Prediction
    # ----------------------------------------------------
    def _predict_row(self, x, node):
        """
        Traverse the tree from root to leaf for a single sample x.
        """
        while node.left is not None and node.right is not None:
            node = node.left if x[node.feat] <= node.thr else node.right
        return node.value, node.class_counts

    def predict(self, X):
        """
        Predict class labels or regression values for all samples in X.
        """
        X = np.asarray(X)
        preds = []
        for i in range(X.shape[0]):
            val, _ = self._predict_row(X[i], self.root)
            preds.append(val)
        return np.array(preds)

    def predict_proba(self, X):
        """
        For classification: return class probability distribution per sample.
        """
        assert self.task == "classification"
        X = np.asarray(X)
        probs = []
        for i in range(X.shape[0]):
            _, counts = self._predict_row(X[i], self.root)
            if counts is None:
                probs.append([1.0])  # safety fallback
                continue
            total = sum(counts.values())
            K = max(counts.keys()) + 1
            p = np.zeros(K)
            for k, c in counts.items():
                p[k] = c / total
            probs.append(p)
        # pad to equal length arrays
        maxK = max(len(p) for p in probs)
        return np.array([np.pad(p, (0, maxK - len(p))) for p in probs])
```

#### Assumptions and properties
- Non-parametric: no linearity/normality assumptions; learns piecewise-constant regions.
- Axis-aligned: splits are univariate; may need many nodes to approximate oblique boundaries.
- Greedy: locally optimal; not globally optimal (can overfit).
- Scale invariance: no need to standardize features.
- Monotone missingness handling not inherent: you must choose a strategy (below).


#### Considerations
- Handling missing values:
    Simple: impute (median/mode) before training.
    Route-to-majority: send missing to the child with more samples.
    Surrogate splits (CART classic): back-up feature that mimics the primary split.
    Probabilistic routing: split missing proportionally (used in GBMs).
  
- Categoricals: one-hot often works well; for high-cardinality consider target encoding (with leakage-safe smoothing) before trees, or use tree libs that natively handle categoricals (e.g., CatBoost).
- Class imbalance: set class_weight='balanced', or tune decision threshold on validation data.
- Complexity: O(nd log n) overall for balanced trees), prediction O(depth), memory O(n)

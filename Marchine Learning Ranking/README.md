# Machine Learning Ranking

In this directory you con find a machine learning ranker for the LOINC clinical terminology based on the approach proposed in [(Cooper et al. (1992))](papers/p198-cooper.pdf).

## Application

![Main app](img/app.png)

## Dataset

The dataset for this algorithm is pretty simple. It is made up of a combination of each query, term pair including the binary relevance of that term for the given query.

**Example:**

*query1, term1, 1* - *query1, term2, 0* - *query2, term1, 0* - ...


## Algorithm

This ranker consists of an implementation of a staged logistic algorithm, used for the probabilistic retrieval of the terms.

The algorithm computes the probability of being relevant of a term given a query as a combination of the probability of being relevant of the term and a linear combination of the partial probabilities of the term of being relevant given a specific word.

![mlranking_prob](img/mlranking_probability.png)

In the above formula, we can see the terms explained in the previous paragraph:
- Probability of being relevant of a term given some words: Log O(R|A1, A2, ..., An)
- Probability of being relevant of a term: Log O(R)
- Probability of being relevant of a term given a particular word: Log O(R|A1)

### Probability of being relevant of a specific term

The estimation of Log O(R) from a learning set is a straightforward matter, for the simple proportion query-term pairs in the learning set that are relevance-related can be used to estimate it.

### Probability of being relevant of a term given a particular word

TO DO


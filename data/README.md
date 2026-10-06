# Data

`default_of_credit_card_clients.csv` is the **Default of Credit Card Clients** dataset:
30,000 anonymised customers of a bank in Taiwan (April to September 2005), 23 features,
and the target column `default payment next month` (1 = defaulted, 22.1% of customers).

- Source: UCI Machine Learning Repository, https://doi.org/10.24432/C55S3H
- Citation: Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository.
- Paper: Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the
  predictive accuracy of probability of default of credit card clients. *Expert Systems with
  Applications*, 36(2), 2473-2480.
- License: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

Changes from the original: converted from the UCI `.xls` file to CSV (title row removed).
No values were changed.

`Churn_Modelling.csv`, used by `original/artificial_neural_network.py`, is not included:
it is synthetic course data whose license is unclear.

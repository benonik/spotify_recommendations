import pandas as pd
import numpy as np

# Import Python's implementation of the apriori algorithm

from apyori import apriori

# Read dataset
dataset = pd.read_csv('data_w_genres.csv')

# The Genres column is in a format that can be manipulated in apriori

genres = dataset['genres']

# Strip the '[]' character so that individual values can be accessed later on
genres = genres.str.strip('[]')

# Calculate lenght of the genres column
index = dataset.index
num_rows = len(index)

# empty transactions list
transactions =[]

# For loop used to append values from each row into the transactions list
for i in range (0, num_rows):
    genres[i]=genres[i].split(",")
    transactions.append(genres[i])
    
# Creating the apriori model
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)

# Display the results
results = list(rules)
results

# Displaay the results in a table

def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])


# print the top 10 recommendation rules
print(resultsinDataFrame.nlargest(n = 10, columns = 'Lift'))
import numpy as np
import pandas as pd
import scipy

#########################################################################
## Change this code to take in all asset price data for one day and    ##
## allocate your portfolio accordingly.                                ##
#########################################################################

df = pd.read_csv('Case3HistoricalPrices.csv')
currentDay = 2520

def allocate_portfolio(asset_prices):
# first section just puts the new returns in the DataFrame and formats them correctly
###############################################################################################
    global df
    global currentDay
    # add the new data to the DataFrame
    allFundsPrice = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8','C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
    allFundsROR = ['S1ROR', 'S2ROR', 'S3ROR', 'S4ROR', 'S5ROR', 'S6ROR', 'S7ROR', 'S8ROR', 'B1ROR', 'B2ROR', 'B3ROR', 'B4ROR', 'B5ROR', 'B6ROR', 'B7ROR', 'B8ROR','C1ROR', 'C2ROR', 'C3ROR', 'C4ROR', 'C5ROR', 'C6ROR', 'C7ROR', 'C8ROR']
    allFundsStdev = ['S1Stdev', 'S2Stdev', 'S3Stdev', 'S4Stdev', 'S5Stdev', 'S6Stdev', 'S7Stdev', 'S8Stdev', 'B1Stdev', 'B2Stdev', 'B3Stdev', 'B4Stdev', 'B5Stdev', 'B6Stdev', 'B7Stdev', 'B8Stdev','C1Stdev', 'C2Stdev', 'C3Stdev', 'C4Stdev', 'C5Stdev', 'C6Stdev', 'C7Stdev', 'C8Stdev']
    allFundsStrength = ['S1Strength', 'S2Strength', 'S3Strength', 'S4Strength', 'S5Strength', 'S6Strength', 'S7Strength', 'S8Strength', 'B1Strength', 'B2Strength', 'B3Strength', 'B4Strength', 'B5Strength', 'B6Strength', 'B7Strength', 'B8Strength','C1Strength', 'C2Strength', 'C3Strength', 'Strength', 'C5Strength', 'C6Strength', 'C7Strength', 'C8Strength']

    dfTemp = pd.DataFrame(data = asset_prices) #add the data to a temp DataFrame
    dfTemp = dfTemp.T #transpose the DataFrame to turn it into a row
    dfTemp.columns = allFundsPrice # set the column name to the name of the current columns
    df = df.append(dfTemp, ignore_index = True) #add the row to the bottom of the DF
###############################################################################################
    # find the 350 day standard deviation of each fund
    df[allFundsROR] = df[allFundsPrice].pct_change()
    df[allFundsStdev] = df[allFundsROR].rolling(350).std()
    # give each fund a weighting factor based off the standard deviation that gives more weights to less volatile funds
    df[allFundsStrength] = 1 / df[allFundsStdev]
    weights = []
    df["StrengthSum"] = df[allFundsStrength].sum(axis = 1)
    # normalize the strengths so they add to 1
    for i in(allFundsStrength):
        weights.append(df.at[currentDay,i] / df.at[currentDay, "StrengthSum"])
    currentDay += 1
    weights[13] += 1 - sum(weights)
    #return the optimal weights in the correct format
    return np.array(weights)

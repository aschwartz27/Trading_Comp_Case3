import numpy as np
import pandas as pd

noEditAllFundsROR = pd.DataFrame() #holds all the returns data to share between functions

class FundGroup:
    def __init__(self, name, leftChild, rightChild, fundsList):
        #######################################################
        # self.name: The name of all the assets aggregated for that portion of the tree
        # self.leftChild: The assets to the left of the group
        # self.rightChild: The assets to the right of the group
        # self.fundsList: Array that stores all the fund names
        # self.weight: Weight given to that portion of the tree
        #######################################################
        self.name = name
        self.leftChild = leftChild
        self.rightChild = rightChild
        self.fundsList = fundsList
        self.weight = 1.0

    def getVar(self, Returns): #Finds the variance of that portion of the tree assigning weights via the minimum variance portfolio strategy
        covDF = Returns.copy()
        covDF = covDF[self.fundsList].cov()
        numerator = np.diag(pd.DataFrame(np.linalg.pinv(covDF.values), covDF.columns, covDF.index))
        denominator = numerator.sum()
        weights = numerator / denominator
        var = weights.T.dot(covDF).dot(weights) #The formula used to find the variance of the minimum variance portfolio
        return var


def getWeights(fundDict, longName, Returns): # Returns the optimal weights of the funds
    if fundDict[longName].leftChild == None and longName[1]: #base case bottom of tree
        return 1
    #Find the variance of both sides of the tree
    leftVar = fundDict[longName].leftChild.getVar(Returns)
    rightVar = fundDict[longName].rightChild.getVar(Returns)
    #weight each side of the tree by giving more weight to the side with less variance
    leftWeight = 1 - leftVar / (leftVar + rightVar)
    rightWeight = 1 - leftWeight
    # multiply out the weight so it is represented throughout the subtrees
    fundDict[longName].leftChild.weight = fundDict[longName].weight * leftWeight
    fundDict[longName].rightChild.weight = fundDict[longName].weight * rightWeight
    #Recursively call the functions to explore the left and right subtrees
    getWeights(fundDict, fundDict[longName].rightChild.name, Returns)
    getWeights(fundDict, fundDict[longName].leftChild.name, Returns)


def treeClusterInit(RORDF, allFundsRORArr): #creates a binary tree of all the funds grouping by correlation
    groupedDict = {} #will hold all the groups of funds. Initalized with each individual fund
    dictList = allFundsRORArr.copy() #holds the name of all the funds
    for i in range(24): #Create a new array that will hold the FundGroup class
        groupedDict[allFundsRORArr[i]] = FundGroup(allFundsRORArr[i], None, None, [allFundsRORArr[i]])
        groupedDict[allFundsRORArr[i]].weight = 1
    corrDF = RORDF.corr() #Find the correlation of the rates of return
    D1 = (.5 * (1 - corrDF)) ** (1.0/2.0) #Determines the "Distance of the funds"
    D2 = D1.copy() #D2 will hold values for the following steps
    #Go through all the funds and determine their distance from each individual fund
    for i in range (24):
        for j in range(24):
            D2.iloc[i,[j]] = 0
            for k in range(24):
                D2.iloc[i,[j]] += (D1.iat[k,i] - D1.iat[k,j]) ** 2
            D2.iloc[i,[j]] = D2.iat[i,j] ** (1.0 / 2.0)
            if (i == j):
                (D2.iloc[i,[j]]) = 100
    #Consolidate the tree
    for i in range(23):
        minCol = (D2[allFundsRORArr].idxmin())[D2[allFundsRORArr].min().idxmin()] #finds the first of the pair of mins
        minRow = D2[allFundsRORArr].min().idxmin() #finds the second of the pair of mins
        newCol = []
        newRow = []
        #Find the values for the new row and column by taking the minimum of the distance for the two rows
        for j in allFundsRORArr:
            minVal = min(D2.at[minCol, j],D2.at[minRow, j])
            newCol.append(minVal)
            newRow.append(minVal)
        D2[minRow + minCol] = newCol #add the new column to the end of the DF
        allFundsRORArr.append(minRow + minCol) #Add the new fund to the list of column names
        dictList.append(minRow + minCol)
        groupedDict[minRow + minCol] = FundGroup(minRow + minCol, groupedDict[minRow], groupedDict[minCol], groupedDict[minRow].fundsList + (groupedDict[minCol].fundsList))
        newRow.append(100) #add 100 to the end of the row becasue they will line up with the other new row
        dfTemp = pd.DataFrame(data = newRow) #add the data to a temp dataframe
        dfTemp = dfTemp.T #transpose the dataframe to turn it into a row
        dfTemp.columns = allFundsRORArr # set the column name to the name of the current columns
        D2 = D2.append(dfTemp) #add the row to the bottom of the DF
        D2 = D2.rename(index = {0:minRow + minCol}) #rename the new row to the new funds
        #Remove the columns and rows that have been aggregated
        D2 = D2.drop([minRow, minCol])
        D2 = D2.drop([minRow, minCol], axis = 1)
        allFundsRORArr.remove(minRow)
        allFundsRORArr.remove(minCol)
    return (groupedDict, allFundsRORArr[0], dictList)


def calculateSharpe(startDate, endDate):
    #Read in the historical price data
    df = pd.read_csv('Case3HistoricalPrices.csv')
    #A list of all the fund names
    allFundsPrice = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
    allFundsROR = []
    stockPrice = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
    bondPrice = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']
    comodityPrice = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']

    # fill out ROR for all funds
    for i in allFundsPrice:
        df[i + 'ROR'] = df[i].pct_change()
        allFundsROR.append(i + 'ROR')
    noEditAllFundsROR = df.iloc[:,[25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48]].copy() #global variable that stores the ROR for all the funds for the given time period
    #Create the initial binary tree of funds and set a variable to the dictionary that holds the tree
    groupedDict = treeClusterInit((df.iloc[startDate - 150:startDate,[25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48]]), allFundsROR.copy())

    # variables that hold the weight of all the funds
    allFundsWeight = []
    allFundsWeightChangeArr = []
    totWeight = 0
    justGroupedDict = groupedDict[0]
    #Go through every day of returns data
    for i in range(startDate, endDate):
        #Recrate the tree every 60 days
        if i % 60 == 0:
            groupedDict = treeClusterInit((df.iloc[i - 150: i,[25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48]]), allFundsROR.copy())
        #get the optimal weights based off the last 150 days of data
        getWeights(groupedDict[0], groupedDict[1], noEditAllFundsROR[i - 150: i].copy())
        #add these new weights to the dataframe
        for j in allFundsROR:
            df.at[i , j[0:2] + "Weight"] = groupedDict[0][j].weight
            allFundsWeight.append(j[0:2] + 'Weight')
    #Reset all the weights to one in the dictionary
    for i in groupedDict[2]:
        groupedDict[0][i].weight = 1

#####Testing begins here#####

    #Determine the change in weights to calculate the sales charges
    for i in(allFundsPrice):
        df[i[0:2] + 'WeightChange'] = np.abs(df[i[0:2] + 'Weight'].diff()) # calculate the abs of the change in weight
        allFundsWeightChangeArr.append(i[0:2] + 'WeightChange')
    df['totalWeightChange'] = df[allFundsWeightChangeArr].sum(axis = 1) #shows the total weight change

    #find index rates of return based off an evenly weighted portfolio within the asset class
    df['stockROR']  = 0
    df['bondROR'] = 0
    df['comodityROR'] = 0
    df['allROR'] = 0
    for i in allFundsROR:
        if i[0] == 'S':
            df['stockROR'] += df[i] / 8
            df['allROR'] += df[i] / 24
        elif i[0] == 'B':
            df['bondROR'] += df[i] / 8
            df['allROR'] += df[i] / 24
        elif i[0] == 'C':
            df['comodityROR'] += df[i] / 8
            df['allROR'] += df[i] / 24

    #find weighted stock ROR
    df['weightedROR'] = 0
    for i in allFundsWeight:
        df['weightedROR'] += df[i] * df[i[0:2] + 'ROR'].shift(-1)
    df['weightedROR'] -= df['totalWeightChange'] * .003 #subtract the exchange fees

    #Find the sharpe ratio of all the different asset classes and the weights we created
    weightedStdev = df.loc[startDate:endDate,['weightedROR']].std(ddof = 1)
    weightedMean = df.loc[startDate:endDate,['weightedROR']].mean()
    weightedSharpe = weightedMean / weightedStdev * (252 ** (1.0/2.0))

    stockStdev = df.loc[startDate:endDate,['stockROR']].std(ddof = 1)
    stockMean = df.loc[startDate:endDate,['stockROR']].mean()
    stockSharpe = stockMean / stockStdev * (252 ** (1.0/2.0))

    bondStdev = df.loc[startDate:endDate,['bondROR']].std(ddof = 1)
    bondMean = df.loc[startDate:endDate,['bondROR']].mean()
    bondSharpe = bondMean / bondStdev * (252 ** (1.0/2.0))

    comodityStdev = df.loc[startDate:endDate,['comodityROR']].std(ddof = 1)
    comodityMean = df.loc[startDate:endDate,['comodityROR']].mean()
    comoditySharpe = comodityMean / comodityStdev * (252 ** (1.0/2.0))

    allStdev = df.loc[startDate:endDate,['allROR']].std(ddof = 1)
    allMean = df.loc[startDate:endDate,['allROR']].mean()
    allSharpe = allMean / allStdev * (252 ** (1.0/2.0))

    return[weightedSharpe, stockSharpe, bondSharpe, comoditySharpe, allSharpe]


print (calculateSharpe(200, 2268))

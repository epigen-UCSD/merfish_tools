"""Use NSForest to identify candidate marker genes for MERFISH gene panels.

This script takes a table of scRNA-seq counts as input and uses the NSForest
algorithm to identify potential candidate marker genes to include in a MERFISH
gene panel. The algorithm also needs metadata identifying the cluster that
each cell belongs to.

Arg1 - A scanpy object of the data. Should be filtered and normalized alread
Arg2 - The column name in the .obs of the scanpy object of the clustering
    labels to design the genes for.
"""
import math
import sys
import pandas as pd
import scanpy as sc
import numpy as np
import graphviz
import numexpr
import itertools
from collections import defaultdict


from subprocess import call
from tqdm import tqdm_notebook as tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score

#####
# Load scanpy object
#
# The scanpy object should be filtered, highly variable genes, log cpm values
#####
cdata = sc.read(sys.argv[1])

#####
# Run NSForest
#####
dataFull = cdata.to_df()
dataFull.columns = [
    x.translate(str.maketrans({".": "_", "-": "_", "@": "_"})) for x in dataFull.columns
]
dataFull["Clusters"] = cdata.obs[sys.argv[2]]

# Creates dummy columns for one vs all Random Forest modeling
dataDummy = pd.get_dummies(dataFull, columns=["Clusters"], prefix="", prefix_sep="")

# Creates matrix of cluster median expression values
medianValues = dataFull.groupby(by="Clusters").median()
medianValues.to_csv("Function_medianValues.csv")

# Finding the number of clusters and printing that to screen (sanity check)
PrecolNum = len(dataFull.columns)
PostcolNum = len(dataDummy.columns)
adjustedColumns = PrecolNum - 1
clusters2Loop = PostcolNum - PrecolNum
# print clusters2Loop


####Random Forest parameters
rfTrees = 1000  # Number of trees
threads = 10  # Number of threads to use, -1 is the greedy option where it will take all available CPUs/RAM

####Filtering and ranking of genes from random forest parameters

Median_Expression_Level = -1
InformativeGenes = 15  # How many top genes from the Random Forest ranked features will be evaluated for binariness
Genes_to_testing = 6  # How many top genes ranked by binary score will be evaluated in permutations by fbeta-score (as the number increases the number of permutation rises exponentially!)

#### fbeta-score parameters

betaValue = 0.5  # Set values for fbeta weighting. 1 is default f-measure. close to zero is Precision, greater than 1 weights toward Recall


def randomForest(column, dataDummy, PrecolNum, rfTrees, threads):
    x_train = dataDummy[list(dataDummy.columns[0 : PrecolNum - 1])]
    names = dataDummy.columns[0 : PrecolNum - 1]
    y_train = dataDummy[column]
    rf = RandomForestClassifier(
        n_estimators=rfTrees, n_jobs=threads, random_state=123456
    )
    rf.fit(x_train, y_train)
    Ranked_Features = sorted(
        zip(map(lambda x: round(x, 4), rf.feature_importances_), names), reverse=True
    )
    return Ranked_Features


def rankInformative(Ranked_Features):
    RankedList = []
    midcounter = 0
    for x in Ranked_Features:
        midcounter += 1
        RankedList.append(x[1])
        rankedDict[column] = RankedList
        if midcounter == 30:
            break
    return RankedList


def negativeOut(x, column, medianValues, Median_Expression_Level):
    Positive_RankedList_Complete = []
    for i in x:
        if medianValues.loc[column, i] > Median_Expression_Level:
            # print(i)
            # print(medianValues.loc[column, i])
            Positive_RankedList_Complete.append(i)
        else:
            # print(i)
            # print(medianValues.loc[column, i])
            # print("Is Right Out!")
            pass
    return Positive_RankedList_Complete


def binaryScore(Positive_RankedList_Complete, informativeGenes, medianValues, column):
    Positive_RankedList = list(Positive_RankedList_Complete[0:InformativeGenes])
    Median_RF_Subset = medianValues.loc[:, Positive_RankedList]
    Rescaled_Matrix = pd.DataFrame()

    for i in Positive_RankedList:
        Target_value = medianValues.loc[column, i]
        Rescaled_values = Median_RF_Subset[[i]].divide(Target_value)
        Rescaled_Matrix = pd.concat([Rescaled_Matrix, Rescaled_values], axis=1)
    difference_matrix = Rescaled_Matrix.apply(lambda x: 1 - x, axis=1)
    difference_matrix_clean = difference_matrix.where(difference_matrix > 0, 0)
    ColumnSums = difference_matrix_clean.sum(0)
    rescaled = ColumnSums / clusters2Loop

    # Double sort so that for ties, the RF ranking prevails!
    Ranked_Features_df = pd.DataFrame(Ranked_Features)
    Ranked_Features_df.rename(columns={1: "Symbol"}, inplace=True)
    Ranked_Features_df_indexed = Ranked_Features_df.set_index("Symbol")
    rescaled_df = pd.DataFrame(rescaled)
    binaryAndinformation_Ranks = rescaled_df.join(
        Ranked_Features_df_indexed, lsuffix="_scaled", rsuffix="_informationGain"
    )
    binaryAndinformation_Ranks.sort_values(
        by=["0_scaled", "0_informationGain"], ascending=[False, False], inplace=True
    )
    Binary_ranked_Genes = binaryAndinformation_Ranks.index.tolist()
    Binary_RankedList = list(Binary_ranked_Genes[0:Genes_to_testing])
    Binary_scores = rescaled.to_dict()
    global Binary_store_DF
    Binary_store_DF = Binary_store_DF.append(binaryAndinformation_Ranks)
    return Binary_RankedList


def DT_cutOffs(x, column):
    cut_dict = {}
    for i in x:
        filename = str(i)
        y_train = dataDummy[column]
        x_train = dataDummy[i]
        X = x_train[:, None]
        clf = tree.DecisionTreeClassifier(max_leaf_nodes=2)
        clf = clf.fit(X, y_train)
        threshold = clf.tree_.threshold
        cut_dict[i] = threshold[0]
    return cut_dict


def queryGenerator(x, cut_dict):
    queryList = []
    for i in x:
        str1 = i
        current_value = cut_dict.get(str1)
        queryString1 = str(str1) + ">=" + str(current_value)
        queryList.append(queryString1)
    return queryList


def permutor(x):
    binarylist2 = x
    combs = []
    for i in range(1, len(x) + 1):
        els = [list(x) for x in itertools.combinations(binarylist2, i)]
        combs.extend(els)
    return combs


def fbetaTest(x, column, testArray, betaValue):
    fbeta_dict = {}
    for list in x:
        testArray["y_pred"] = 0
        betaQuery = "&".join(list)
        Ineq1 = dataFull.query(betaQuery)
        testList = Ineq1.index.tolist()
        testArray.loc[testList, "y_pred"] = 1
        f1 = fbeta_score(
            testArray["y_true"], testArray["y_pred"], average="binary", beta=betaValue
        )
        dictName = column + "&" + betaQuery
        fbeta_dict[dictName] = f1
    return fbeta_dict


# Core analysis
rankedDict = {}  ###gives us the top ten features from RF
f1_store_1D = {}
Binary_score_store_DF = pd.DataFrame()
DT_cutoffs_store = {}

for column in tqdm(dataDummy.columns[PrecolNum - 1 : PostcolNum]):

    ## Run Random Forest and get a ranked list
    Ranked_Features = randomForest(column, dataDummy, PrecolNum, rfTrees, threads)
    RankedList = rankInformative(Ranked_Features)

    ## Setup testArray for f-beta evaluation
    testArray = dataDummy[[column]]
    testArray.columns = ["y_true"]

    # Rerank according to expression level and binary score
    Positive_RankedList_Complete = negativeOut(
        RankedList, column, medianValues, Median_Expression_Level
    )
    Binary_store_DF = pd.DataFrame()
    Binary_RankedList = binaryScore(
        Positive_RankedList_Complete, InformativeGenes, medianValues, column
    )

    Binary_score_store_DF_extra = Binary_store_DF.assign(clusterName=column)
    # print Binary_score_store_DF_extra
    Binary_score_store_DF = Binary_score_store_DF.append(Binary_score_store_DF_extra)

    # Get expression cutoffs for f-beta testing
    cut_dict = DT_cutOffs(Binary_RankedList, column)
    DT_cutoffs_store[column] = cut_dict

    # Generate expression queries and run those queries using fbetaTest() function
    queryInequalities = queryGenerator(Binary_RankedList, cut_dict)
    FullpermutationList = permutor(queryInequalities)
    # print len(FullpermutationList)
    f1_store = fbetaTest(FullpermutationList, column, testArray, betaValue)
    f1_store_1D.update(f1_store)


# Report generation and cleanup

f1_store_1D_df = pd.DataFrame()  # F1 store gives all results.
f1_store_1D_df = pd.DataFrame.from_dict(f1_store_1D, orient="index")
f1_store_1D_df.columns = ["f-measure"]
f1_store_1D_df["markerCount"] = f1_store_1D_df.index.str.count("&")
f1_store_1D_df.reset_index(level=f1_store_1D_df.index.names, inplace=True)

f1_store_1D_df_done = f1_store_1D_df["index"].apply(lambda x: pd.Series(x.split("&")))

NSForest_Results_Table = f1_store_1D_df.join(f1_store_1D_df_done)

NSForest_Results_Table_Fin = pd.DataFrame()
NSForest_Results_Table_Fin = NSForest_Results_Table[NSForest_Results_Table.columns[0:4]]

for i, col in enumerate(NSForest_Results_Table.columns[4:11]):
    splitResults = (
        NSForest_Results_Table[col]
        .astype(str)
        .apply(lambda x: pd.Series(x.split(">=")))
    )
    firstOnly = splitResults[0]
    Ascolumn = firstOnly.to_frame()
    Ascolumn.columns = [col]
    NSForest_Results_Table_Fin = NSForest_Results_Table_Fin.join(Ascolumn)


NSForest_Results_Table_Fin.rename(
    columns={0: "clusterName"}, inplace=True
)  # rename columns by position
NSForest_Results_Table_Fin.sort_values(
    by=["clusterName", "f-measure", "markerCount"],
    ascending=[True, False, True],
    inplace=True,
)

# Write outs
Binary_score_store_DF.to_csv("Binary_scores_Supplmental_results.csv")
NSForest_Results_Table_Fin.to_csv("NS-Forest_v2_results.csv")


# Subsets of full results
max_grouped = NSForest_Results_Table_Fin.groupby(by="clusterName")["f-measure"].max()
max_grouped.df = pd.DataFrame(max_grouped)
max_grouped.df.to_csv("NSForest_v2_maxF-scores.csv")

NSForest_Results_Table_Fin["f-measureRank"] = NSForest_Results_Table_Fin.groupby(
    by="clusterName"
)["f-measure"].rank(ascending=False)
topResults = NSForest_Results_Table_Fin["f-measureRank"] < 50
NSForest_Results_Table_top = NSForest_Results_Table_Fin[topResults]
NSForest_Results_Table_top.to_csv("NSForest_v2_topResults.csv")


#####
# Convert the NSForest output into a simpler format
#####
nsf = pd.read_csv("NSForest_v2_topResults.csv", index_col=0)
genes = defaultdict(list)
done = defaultdict(set)
for name, group in nsf.groupby("clusterName"):
    for row in group.itertuples():
        g = set([x for x in row[5:11] if x not in done[name] and isinstance(x, str)])
        done[name].update(g)
        if g:
            genes[name].append(g)
genes
total = set()
for k, v in genes.items():
    for i, g in enumerate(v, start=1):
        print("\n".join(",".join([str(k), str(i), a]) for a in g))

### Step 0: Get Started

# Check if Microsoft R Server (RRE 8.0) is installed.
if (!require("RevoScaleR")) {
    cat("RevoScaleR package does not seem to exist. 
      \nThis means that the functions starting with 'rx' will not run. 
      \nIf you have Microsoft R Server installed, please switch the R engine.
      \nFor example, in R Tools for Visual Studio: 
      \nR Tools -> Options -> R Engine. 
      \nIf Microsoft R Server is not installed, you can download it from: 
      \nhttps://www.microsoft.com/en-us/server-cloud/products/r-server/
      \n")
    quit()
}

# Initial some variables.
# github
loadFile <- paste0("datasets/ldeaths_Sample.csv")
loadFile <- paste0("datasets/mdeaths_Sample.csv")

# Create a temporary directory to store the intermediate .xdf files.
td <- tempdir() 
outFileldeaths <- paste0(td, "/ldeaths.xdf");
outFilemdeaths <- paste0(td, "/mdeaths.xdf");
outFileOrigin <- paste0(td, "/originData.xdf");
outFileDest <- paste0(td, "/destData.xdf");
outFileFinal <- paste0(td, "/finalData.xdf");


### Step 1: Import Data

# Import the flight data.
ldeaths_mrs <- rxImport(
  inData = paste0("datasets/ldeaths_Sample.csv"), outFile = outFileldeaths,
  missingValueString = "M", stringsAsFactors = FALSE,
# Remove columns that are possible target leakers from the flight data.
  varsToDrop = c("RowNum"),
# Define "Carrier" as categorical.
  colInfo = list(x = list(type = "factor")),
# Round down scheduled departure time to full hour.
#  transforms = list(CRSDepTime = floor(CRSDepTime / 100)),
  overwrite = TRUE
)

# Review the first 6 rows of flight data.
head(ldeaths_mrs)

# Summary the flight data.
rxSummary(~., data = ldeaths_mrs, blocksPerRead = 2)

# Import the weather data.
xform <- function(dataList) {
    # Create a function to normalize some numerical features.
    featureNames <- c(
  "x"
  )
    dataList[featureNames] <- lapply(dataList[featureNames], scale)
    return(dataList)
}

mdeaths_mrs <- rxImport(
  inData = paste0("datasets/mdeaths_Sample.csv"), outFile = outFilemdeaths,
  missingValueString = "M", stringsAsFactors = FALSE,
# Eliminate some features due to redundance.
  varsToDrop = c("RowNum"),
# Create a new column "DestAirportID" in weather data.
  transforms = list(x = x),
# Apply the normalization function.
  transformFunc = xform,
  transformVars = c(
    "x"
    ),
  overwrite = TRUE
)

# Review the variable information of weather data.
rxGetVarInfo(mdeaths_mrs)


### Step 2: Pre-process Data

# Rename some column names in the weather data to prepare it for merging.
newVarInfo <- list(
  x = list(newName = "deaths")
)
rxSetVarInfo(varInfo = newVarInfo, data = mdeaths_mrs)

# Concatenate/Merge flight records and weather data.
# 1). Join flight records and weather data at origin of the flight 
#     (OriginAirportID).
originData_mrs <- rxMerge(
  inData1 = ldeaths_mrs, inData2 = mdeaths_mrs, outFile = outFileOrigin,
  type = "inner", autoSort = TRUE,
  matchVars = c("deaths","deaths"),
  varsToDrop2 = "RowNum",
  overwrite = TRUE
)

# 2). Join flight records and weather data using the destination of 
#     the flight (DestAirportID).
destData_mrs <- rxMerge(
  inData1 = originData_mrs, inData2 = mdeaths_mrs, outFile = outFileDest,
  type = "inner", autoSort = TRUE,
  matchVars = c("deaths","deaths"),
  varsToDrop2 = c("RowNum"),
  duplicateVarExt = c("deaths","deaths"),
  overwrite = TRUE
)

# Call "rxFactors" function to convert "OriginAirportID" and 
# "DestAirportID" as categorical.
rxFactors(inData = destData_mrs, outFile = outFileFinal, sortLevels = TRUE,
          factorInfo = c("deaths", "deaths"),
          overwrite = TRUE)



### Step 3: Prepare Training and Test Datasets

# Randomly split 80% data as training set and the remaining 20% as test set.
rxSplit(inData = outFileFinal,
        outFilesBase = paste0(td, "/modelData"),
        outFileSuffixes = c("Train", "Test"),
        splitByFactor = "splitVar",
        overwrite = TRUE,
        transforms = list(
          splitVar = factor(sample(c("Train", "Test"),
                                   size = .rxNumRows,
                                   replace = TRUE,
                                   prob = c(.80, .20)),
                            levels = c("Train", "Test"))),
        rngSeed = 17,
        consoleOutput = TRUE)

# Point to the .xdf files for the training and test set.
train <- RxXdfData(paste0(td, "/modelData.splitVar.Train.xdf"))
test <- RxXdfData(paste0(td, "/modelData.splitVar.Test.xdf"))


### Step 4A: Choose and apply a learning algorithm (Logistic Regression)

# Build the formula.
modelFormula <- formula(train, depVars = "deaths",
                        varsToDrop = c("RowNum", "splitVar"))

# Fit a Logistic Regression model.
logitModel_mrs <- rxLogit(modelFormula, data = train)

# Review the model results.
summary(logitModel_mrs)


### Step 5A: Predict over new data (Logistic Regression)

# Predict the probability on the test dataset.
rxPredict(logitModel_mrs, data = test,
          type = "response",
          predVarNames = "deaths_Pred_Logit",
          overwrite = TRUE)

# Calculate Area Under the Curve (AUC).
paste0("AUC of Logistic Regression Model:",
      rxAuc(rxRoc("deaths", "deaths_Pred_Logit", test)))

# Plot the ROC curve.
rxRocCurve("deaths", "deaths_Pred_Logit", data = test,
           title = "ROC curve - Logistic regression")


### Step 4B: Choose and apply a learning algorithm (Decision Tree)

# Build a decision tree model.
dTree1_mrs <- rxDTree(modelFormula, data = test, reportProgress = 1)

# Find the Best Value of cp for Pruning rxDTree Object.
treeCp_mrs <- rxDTreeBestCp(dTree1_mrs)

# Prune a decision tree created by rxDTree and return the smaller tree.
dTree2_mrs <- prune.rxDTree(dTree1_mrs, cp = treeCp_mrs)


### Step 5B: Predict over new data (Decision Tree)

# Predict the probability on the test dataset.
rxPredict(dTree2_mrs, data = test,
          predVarNames = "deaths_Pred_Tree",
          overwrite = TRUE)

# Calculate Area Under the Curve (AUC).
paste0("AUC of Decision Tree Model:",
       rxAuc(rxRoc(" deaths ", " deaths_Pred_Tree ", test)))

# Plot the ROC curve.
rxRocCurve("deaths",
           predVarNames = c("deaths_Pred_Tree", "deaths_Pred_Logit"),
           data = test,
           title = "ROC curve - Logistic regression")

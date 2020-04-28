# Read in the necessary libraries

# Install these libraries (only do this once)
# install.packages("ChannelAttribution", repos = "http://cran.us.r-project.org")
# install.packages("reshape", repos = "http://cran.us.r-project.org")
# install.packages("ggplot2", repos = "http://cran.us.r-project.org")

# Load these libraries (every time you start RStudio)
library(ChannelAttribution)
library(reshape)
library(ggplot2)

# This loads the demo data. You can load your own data by importing a dataset or reading in a file
data(PathData)
# Set Working Directory
setwd <- setwd('/Users/htahir/Documents/Python/Markov_chains')
# Read in our CSV file outputted by the python script
df <- read.csv('Paths.csv')
# Select only the necessary columns
df <- df[c(1,2)]
# Run the Markov Model function
M <- markov_model(df, 'Path', var_value = 'Conversion', var_conv = 'Conversion', sep = '>', order=1, out_more = TRUE)
# Output the model output as a csv file, to be read back into Python
write.csv(M$result, file = "Markov - Output - Conversion values.csv", row.names=FALSE)
# Output the transition matrix as well, for visualization purposes
write.csv(M$transition_matrix, file = "Markov - Output - Transition matrix.csv", row.names=FALSE)

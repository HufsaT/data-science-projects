import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import subprocess

df = pd.read_csv('Channel_attribution.csv')

cols = df.columns


for col in cols:
    df[col] = df[col].astype(str)
    df[col] = df[col].map(lambda x: str(x)[:-2] if '.' in x else str(x))

# Create a total path variable
df['Path'] = ''
for i in df.index:
    for x in cols:
        df.at[i, 'Path'] = df.at[i, 'Path'] + df.at[i, x] + ' > '

df['Path'] = df['Path'].map(lambda x: x.split(' > 21')[0])

df['Conversion'] = 1

df = df[['Path', 'Conversion']]
df = df.groupby('Path').sum().reset_index()

df = df[df['Path'] != '.']
# print(df.head())
df.to_csv('paths.csv',index=False)
print('Cleaned up data and wrote to file.')

# Define the path to the R script that will run the Markov Model
path2script = '/Users/htahir/Documents/Python/Markov_chains/markovchains.R'

# Call the R script
subprocess.call(['Rscript --vanilla '+ path2script], shell=True)
print('Ran R script.')

# Load in the CSV file with the model output from R
markov = pd.read_csv('Markov - Output - Conversion values.csv')

# Select only the necessary columns and rename them
markov = markov[['channel_name', 'total_conversions']]
markov.columns = ['Channel', 'Conversion']

# calculate first touch
df['First Touch'] = df['Path'].map(lambda x: x.split(' > ')[0])
df_ft = pd.DataFrame()
df_ft['Channel'] = df['First Touch']
df_ft['Attribution'] = 'First Touch'
df_ft['Conversion'] = 1
df_ft = df_ft.groupby(['Channel', 'Attribution']).sum().reset_index()

# Last Touch Attribution
df['Last Touch'] = df['Path'].map(lambda x: x.split(' > ')[-1])
df_lt = pd.DataFrame()
df_lt['Channel'] = df['Last Touch']
df_lt['Attribution'] = 'Last Touch'
df_lt['Conversion'] = 1
df_lt = df_lt.groupby(['Channel', 'Attribution']).sum().reset_index()

# linear approach
conversion = []
channel = []

for i in df.index:
    for x in df.at[i,'Path'].split(' > '):
        channel.append(x)
        conversion.append(1/len(df.at[i,'Path'].split(' > ')))

lin_att_df = pd.DataFrame()
lin_att_df['Channel'] = channel
lin_att_df['Attribution'] = 'Linear'
lin_att_df['Conversion'] = conversion
lin_att_df = lin_att_df.groupby(['Channel', 'Attribution']).sum().reset_index()

# print(lin_att_df.head())

df_total_attr = pd.concat([df_ft, df_lt, lin_att_df, markov])
df_total_attr.dropna(inplace=True)
df_total_attr['Channel'] = df_total_attr['Channel'].astype(int)
df_total_attr.sort_values(by='Channel', ascending=True, inplace=True)
print(df_total_attr.head(10))

# # visualize
# sns.set_style('whitegrid')
# plt.rc('legend',fontsize=15)
# fig, ax = plt.subplots(figsize=(10,5))
# sns.barplot(x='Channel', y='Conversion', hue='Attribution', data=df_total_attr)
# plt.show()

# Read in transition matrix CSV
trans_prob = pd.read_csv('Markov - Output - Transition matrix.csv')

# Convert data to floats
trans_prob['transition_probability'] = trans_prob['transition_probability'].astype(float)

# Convert start and conversion event to numeric values so we can sort and iterate through
trans_prob.replace('(start)', '0', inplace=True)
trans_prob.replace('(conversion)', '21', inplace=True)

# Get unique origin channels
channel_from_unique = trans_prob['channel_from'].unique().tolist()
channel_from_unique.sort(key=float)

# Get unique destination channels
channel_to_unique = trans_prob['channel_to'].unique().tolist()
channel_to_unique.sort(key=float)

# Create new matrix with origin and destination channels as columns and index
trans_matrix = pd.DataFrame(columns=channel_to_unique, index=channel_from_unique)

# Assign the probabilities to the corresponding cells in our transition matrix
for f in channel_from_unique:
    for t in channel_to_unique:
        x = trans_prob[(trans_prob['channel_from'] == f) & (trans_prob['channel_to'] == t)]
        prob = x['transition_probability'].values
        if prob.size > 0:
            trans_matrix[t][f] = prob[0]
        else:
            trans_matrix[t][f] = 0

# Convert all probabilities to floats
trans_matrix = trans_matrix.apply(pd.to_numeric)

# Rename our start and conversion events
trans_matrix.rename(index={'0': 'Start'}, inplace=True)
trans_matrix.rename(columns={'21': 'Conversion'}, inplace=True)

# Visualize this transition matrix in a heatmap
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(trans_matrix, cmap="RdBu_r")
plt.show()
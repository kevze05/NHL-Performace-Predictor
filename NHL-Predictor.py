# %%
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# %%
def format_df (url):
    dfs = pd.read_html(url)

    temp_df = dfs[0]

    headers = temp_df.columns.get_level_values(level = 1)

    df = pd.DataFrame(temp_df.values, None, headers)

    df = df.dropna()
    df = df.rename(columns={'Unnamed: 1_level_1' : 'Name'})

    o_data = df.loc[:, ['S', 'S%', 'PPO', 'PP%', 'SA', 'SV%', 'PPOA', 'PK%', 'PTS%', 'GP']]

    o_data['GP'] = o_data['GP'].astype(float)
    
    for col in o_data:
        o_data[col] = o_data[col].astype(float) 

    o_data['S'] = o_data['S']/o_data['GP']
    o_data['PPO'] = o_data['PPO']/o_data['GP']

    o_data['SA'] = o_data['SA']/o_data['GP']
    o_data['PPOA'] = o_data['PPOA']/o_data['GP']
        
    X = o_data.iloc[:, 0:8]
    y = o_data.loc[:, 'PTS%']
    return(X, y)

# %%
#2022-2023
base_url = "http://hkref.com/pi/share/O9rGt"

#url list from 2021-2022 -> 2018-2019
url_list = ["http://hkref.com/pi/share/wmL1R",
            "http://hkref.com/pi/share/DcQl1",
            "http://hkref.com/pi/share/uRko1",
            "http://hkref.com/pi/share/8frtU"
            ]


df = format_df(base_url)
df0 = df[0]
df1 = df[1]

for url in url_list:
    df = format_df(url)
    df0 = pd.concat([df0, df[0]])
    df1 = pd.concat([df1, df[1]])


import sklearn as sk
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(df0, df1, test_size = 0.2)



# %%
from keras import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(64, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))

model.compile(optimizer='adam', loss='mae')
history = model.fit(x = X_train, y = y_train, epochs = 100, validation_data = (X_val, y_val))


# %%
url_23_24 = "http://hkref.com/pi/share/d0qJU"

df2 = format_df(url_23_24)

X_test = df2[0]
y_test = df2[1]

model.evaluate(x = X_test, y = y_test)
y_pred = model.predict(x = X_test)

y_pred


# %%
import seaborn as sns
import matplotlib.pyplot as plt


pred_res = pd.DataFrame(y_pred, columns=["Predicted"])
actual_res = pd.DataFrame(y_test)

graph_data = pd.concat([pred_res, actual_res], axis = 1)
graph_data = graph_data.dropna()

scat_plot = sns.scatterplot(data = graph_data)
plt.show()


# %%
difference_data = graph_data["Predicted"] - graph_data["PTS%"]
difference_data

diff_plot = sns.scatterplot(difference_data)
diff_plot.axhline(y = 0, color = 'red')

plt.show()

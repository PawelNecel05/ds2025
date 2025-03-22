# Exercise 3. - GroupBy

### Introduction:

GroupBy can be summarized as Split-Apply-Combine.

Special thanks to: https://github.com/justmarkham for sharing the dataset and materials.

Check out this [Diagram](http://i.imgur.com/yjNkiwL.png)  

Check out [Alcohol Consumption Exercises Video Tutorial](https://youtu.be/az67CMdmS6s) to watch a data scientist go through the exercises


### Step 1. Import the necessary libraries


```python
import pandas as pd
```

### Step 2. Import the dataset from this [address](https://raw.githubusercontent.com/justmarkham/DAT8/master/data/drinks.csv). 

### Step 3. Assign it to a variable called drinks.


```python
drinks = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/drinks.csv', keep_default_na=False, na_values=[])
# interestingly, we must pass keep_default_na=False, because I believe that in this dataset 'NA' stands for North America
```

### Step 4. Which continent drinks more beer on average?


```python
q = drinks.groupby('continent').beer_servings.mean().idxmax()
q
```




    'EU'



### Step 5. For each continent print the statistics for wine consumption.


```python
drinks.groupby('continent').wine_servings.describe()


```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>continent</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AF</th>
      <td>53.0</td>
      <td>16.264151</td>
      <td>38.846419</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>13.00</td>
      <td>233.0</td>
    </tr>
    <tr>
      <th>AS</th>
      <td>44.0</td>
      <td>9.068182</td>
      <td>21.667034</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>8.00</td>
      <td>123.0</td>
    </tr>
    <tr>
      <th>EU</th>
      <td>45.0</td>
      <td>142.222222</td>
      <td>97.421738</td>
      <td>0.0</td>
      <td>59.0</td>
      <td>128.0</td>
      <td>195.00</td>
      <td>370.0</td>
    </tr>
    <tr>
      <th>NA</th>
      <td>23.0</td>
      <td>24.521739</td>
      <td>28.266378</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>11.0</td>
      <td>34.00</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>OC</th>
      <td>16.0</td>
      <td>35.625000</td>
      <td>64.555790</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>8.5</td>
      <td>23.25</td>
      <td>212.0</td>
    </tr>
    <tr>
      <th>SA</th>
      <td>12.0</td>
      <td>62.416667</td>
      <td>88.620189</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>12.0</td>
      <td>98.50</td>
      <td>221.0</td>
    </tr>
  </tbody>
</table>
</div>



### Step 6. Print the mean alcohol consumption per continent for every column


```python
# Select only numeric columns
# Perform the groupby operation on the numeric columns
result = drinks.groupby('continent')[drinks.select_dtypes(include=['number']).columns].mean()

result
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>beer_servings</th>
      <th>spirit_servings</th>
      <th>wine_servings</th>
      <th>total_litres_of_pure_alcohol</th>
    </tr>
    <tr>
      <th>continent</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AF</th>
      <td>61.471698</td>
      <td>16.339623</td>
      <td>16.264151</td>
      <td>3.007547</td>
    </tr>
    <tr>
      <th>AS</th>
      <td>37.045455</td>
      <td>60.840909</td>
      <td>9.068182</td>
      <td>2.170455</td>
    </tr>
    <tr>
      <th>EU</th>
      <td>193.777778</td>
      <td>132.555556</td>
      <td>142.222222</td>
      <td>8.617778</td>
    </tr>
    <tr>
      <th>NA</th>
      <td>145.434783</td>
      <td>165.739130</td>
      <td>24.521739</td>
      <td>5.995652</td>
    </tr>
    <tr>
      <th>OC</th>
      <td>89.687500</td>
      <td>58.437500</td>
      <td>35.625000</td>
      <td>3.381250</td>
    </tr>
    <tr>
      <th>SA</th>
      <td>175.083333</td>
      <td>114.750000</td>
      <td>62.416667</td>
      <td>6.308333</td>
    </tr>
  </tbody>
</table>
</div>



### Step 7. Print the median alcohol consumption per continent for every column


```python
drinks.groupby('continent')[drinks.select_dtypes(include=['number']).columns].median()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>beer_servings</th>
      <th>spirit_servings</th>
      <th>wine_servings</th>
      <th>total_litres_of_pure_alcohol</th>
    </tr>
    <tr>
      <th>continent</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AF</th>
      <td>32.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>2.30</td>
    </tr>
    <tr>
      <th>AS</th>
      <td>17.5</td>
      <td>16.0</td>
      <td>1.0</td>
      <td>1.20</td>
    </tr>
    <tr>
      <th>EU</th>
      <td>219.0</td>
      <td>122.0</td>
      <td>128.0</td>
      <td>10.00</td>
    </tr>
    <tr>
      <th>NA</th>
      <td>143.0</td>
      <td>137.0</td>
      <td>11.0</td>
      <td>6.30</td>
    </tr>
    <tr>
      <th>OC</th>
      <td>52.5</td>
      <td>37.0</td>
      <td>8.5</td>
      <td>1.75</td>
    </tr>
    <tr>
      <th>SA</th>
      <td>162.5</td>
      <td>108.5</td>
      <td>12.0</td>
      <td>6.85</td>
    </tr>
  </tbody>
</table>
</div>



### Step 8. Print the mean, min and max values for spirit consumption.
#### This time output a DataFrame


```python
drinks.spirit_servings.describe()
```




    count    193.000000
    mean      80.994819
    std       88.284312
    min        0.000000
    25%        4.000000
    50%       56.000000
    75%      128.000000
    max      438.000000
    Name: spirit_servings, dtype: float64



## 3-Way ANOVA

The three-way ANOVA is an extension of the two-way ANOVA for assessing whether there is an interaction effect between three independent categorical variables on a continuous outcome variable.

We’ll use the **headache dataset** [datarium package], which contains the measures of migraine headache episode pain score in 72 participants treated with three different treatments. The participants include 36 males and 36 females. Males and females were further subdivided into whether they were at low or high risk of migraine.

We want to understand how each independent variable (type of treatments, risk of migraine and gender) interact to predict the pain score.


```python
import pandas as pd
data = pd.read_csv('headache.csv')
print(data.shape)
```

    (72, 5)
    

### Descriptive statistics


```python
g = data.groupby(['gender', 'risk', 'treatment'])['pain_score']
ds = g.agg(n='count', mean='mean', sd='std', median='median', q1=lambda s: s.quantile(.25), q3=lambda s: s.quantile(.75), min='min', max='max')
ds['iqr'] = ds.q3 - ds.q1
ds['se'] = ds.sd / (ds.n ** 0.5)
ds['ci_low'] = ds['mean'] - (1.96 * ds.se)
ds['ci_high'] = ds['mean'] + (1.96 * ds.se)
ds.sort_index()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th></th>
      <th>n</th>
      <th>mean</th>
      <th>sd</th>
      <th>median</th>
      <th>q1</th>
      <th>q3</th>
      <th>min</th>
      <th>max</th>
      <th>iqr</th>
      <th>se</th>
      <th>ci_low</th>
      <th>ci_high</th>
    </tr>
    <tr>
      <th>gender</th>
      <th>risk</th>
      <th>treatment</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
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
      <th rowspan="6" valign="top">female</th>
      <th rowspan="3" valign="top">high</th>
      <th>X</th>
      <td>6</td>
      <td>78.865059</td>
      <td>5.316489</td>
      <td>81.055686</td>
      <td>79.142521</td>
      <td>81.431621</td>
      <td>68.360185</td>
      <td>82.657063</td>
      <td>2.289099</td>
      <td>2.170447</td>
      <td>74.610982</td>
      <td>83.119135</td>
    </tr>
    <tr>
      <th>Y</th>
      <td>6</td>
      <td>81.175304</td>
      <td>4.619387</td>
      <td>81.809575</td>
      <td>80.011223</td>
      <td>83.666457</td>
      <td>73.144392</td>
      <td>86.591089</td>
      <td>3.655234</td>
      <td>1.885857</td>
      <td>77.479024</td>
      <td>84.871583</td>
    </tr>
    <tr>
      <th>Z</th>
      <td>6</td>
      <td>81.035142</td>
      <td>3.984886</td>
      <td>80.842595</td>
      <td>79.811669</td>
      <td>82.406134</td>
      <td>74.988057</td>
      <td>87.142265</td>
      <td>2.594465</td>
      <td>1.626823</td>
      <td>77.846569</td>
      <td>84.223715</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">low</th>
      <th>X</th>
      <td>6</td>
      <td>74.156231</td>
      <td>3.690272</td>
      <td>74.624574</td>
      <td>72.005373</td>
      <td>77.059079</td>
      <td>68.613936</td>
      <td>78.071412</td>
      <td>5.053705</td>
      <td>1.506547</td>
      <td>71.203398</td>
      <td>77.109063</td>
    </tr>
    <tr>
      <th>Y</th>
      <td>6</td>
      <td>68.361925</td>
      <td>4.081842</td>
      <td>68.733945</td>
      <td>65.236456</td>
      <td>69.799297</td>
      <td>63.732617</td>
      <td>74.746005</td>
      <td>4.562841</td>
      <td>1.666405</td>
      <td>65.095771</td>
      <td>71.628079</td>
    </tr>
    <tr>
      <th>Z</th>
      <td>6</td>
      <td>69.779555</td>
      <td>2.719645</td>
      <td>69.587288</td>
      <td>68.866039</td>
      <td>71.645371</td>
      <td>65.449408</td>
      <td>73.096326</td>
      <td>2.779333</td>
      <td>1.110290</td>
      <td>67.603386</td>
      <td>71.955725</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">male</th>
      <th rowspan="3" valign="top">high</th>
      <th>X</th>
      <td>6</td>
      <td>92.738847</td>
      <td>5.116095</td>
      <td>93.392498</td>
      <td>88.907525</td>
      <td>95.304508</td>
      <td>86.293706</td>
      <td>100.000000</td>
      <td>6.396984</td>
      <td>2.088637</td>
      <td>88.645119</td>
      <td>96.832575</td>
    </tr>
    <tr>
      <th>Y</th>
      <td>6</td>
      <td>82.341542</td>
      <td>5.000898</td>
      <td>81.160483</td>
      <td>78.951861</td>
      <td>83.897360</td>
      <td>77.524462</td>
      <td>91.178517</td>
      <td>4.945499</td>
      <td>2.041608</td>
      <td>78.339990</td>
      <td>86.343094</td>
    </tr>
    <tr>
      <th>Z</th>
      <td>6</td>
      <td>79.680736</td>
      <td>4.045885</td>
      <td>80.370195</td>
      <td>76.602057</td>
      <td>81.983814</td>
      <td>74.419865</td>
      <td>85.056463</td>
      <td>5.381757</td>
      <td>1.651725</td>
      <td>76.443354</td>
      <td>82.918118</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">low</th>
      <th>X</th>
      <td>6</td>
      <td>76.051783</td>
      <td>3.854876</td>
      <td>75.947806</td>
      <td>73.593322</td>
      <td>78.694621</td>
      <td>70.832421</td>
      <td>81.163944</td>
      <td>5.101299</td>
      <td>1.573747</td>
      <td>72.967240</td>
      <td>79.136326</td>
    </tr>
    <tr>
      <th>Y</th>
      <td>6</td>
      <td>73.138772</td>
      <td>4.765061</td>
      <td>73.360982</td>
      <td>69.455207</td>
      <td>74.857744</td>
      <td>67.923560</td>
      <td>80.677163</td>
      <td>5.402536</td>
      <td>1.945328</td>
      <td>69.325930</td>
      <td>76.951615</td>
    </tr>
    <tr>
      <th>Z</th>
      <td>6</td>
      <td>74.455863</td>
      <td>4.888865</td>
      <td>74.850275</td>
      <td>70.706772</td>
      <td>77.945192</td>
      <td>68.299416</td>
      <td>80.432775</td>
      <td>7.238420</td>
      <td>1.995871</td>
      <td>70.543956</td>
      <td>78.367769</td>
    </tr>
  </tbody>
</table>
</div>



### Assumptions


```python
import matplotlib.pyplot as plt
import scipy.stats as ss
import statsmodels.formula.api as smf
m = smf.ols("pain_score ~ C(gender)*C(risk)*C(treatment)", data=data).fit()
ss.probplot(m.resid, dist="norm", plot=plt)
plt.title("Q-Q plot of ANOVA residuals")
plt.show()
print(ss.shapiro(m.resid)) # plot looks good and even shapiro-wilk test returns p-value ~0.4 so the data is normal 

cells = data.groupby(['gender','risk','treatment'])['pain_score'].apply(list)
print(ss.levene(*cells, center='median')) # p-value ~1 -> homogenous variance
```


    
![png](output_5_0.png)
    


    ShapiroResult(statistic=np.float64(0.9821218558347028), pvalue=np.float64(0.39807840029779584))
    LeveneResult(statistic=np.float64(0.17859528084995452), pvalue=np.float64(0.9982136500456728))
    

#### Outliers


```python
plt.scatter(m.fittedvalues, m.resid)
plt.axhline(0, color="k")
plt.xlabel("Fitted")
plt.ylabel("Residual")
plt.show()
from statsmodels.stats.outliers_influence import OLSInfluence
infl = OLSInfluence(m)
out = infl.summary_frame() 
out[['standard_resid','student_resid','cooks_d']].sort_values('cooks_d', ascending=False).head(10)
# no |student_resid| > 3 -> no outliers  
```


    
![png](output_7_0.png)
    





<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>standard_resid</th>
      <th>student_resid</th>
      <th>cooks_d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>56</th>
      <td>-2.616140</td>
      <td>-2.756206</td>
      <td>0.114070</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2.200766</td>
      <td>2.276150</td>
      <td>0.080723</td>
    </tr>
    <tr>
      <th>61</th>
      <td>-2.000023</td>
      <td>-2.052898</td>
      <td>0.066668</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.877365</td>
      <td>1.918867</td>
      <td>0.058742</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1.808322</td>
      <td>1.844147</td>
      <td>0.054500</td>
    </tr>
    <tr>
      <th>22</th>
      <td>-1.605102</td>
      <td>-1.626984</td>
      <td>0.042939</td>
    </tr>
    <tr>
      <th>45</th>
      <td>1.589895</td>
      <td>1.610888</td>
      <td>0.042129</td>
    </tr>
    <tr>
      <th>13</th>
      <td>-1.533205</td>
      <td>-1.551063</td>
      <td>0.039179</td>
    </tr>
    <tr>
      <th>70</th>
      <td>1.520922</td>
      <td>1.538136</td>
      <td>0.038553</td>
    </tr>
    <tr>
      <th>66</th>
      <td>-1.505970</td>
      <td>-1.522417</td>
      <td>0.037799</td>
    </tr>
  </tbody>
</table>
</div>



#### Normality


```python
# taken care of in "Assumptions" above
```

#### Homogeneity of variance


```python
# taken care of in "Assumptions" above
```

### Anova


```python
from statsmodels.stats.anova import anova_lm
anova_lm(m, typ=3) # 3-way interaction is significant, wwe will consider conditional effects
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum_sq</th>
      <th>df</th>
      <th>F</th>
      <th>PR(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Intercept</th>
      <td>37318.184764</td>
      <td>1.0</td>
      <td>1928.766332</td>
      <td>2.538132e-47</td>
    </tr>
    <tr>
      <th>C(gender)</th>
      <td>577.446016</td>
      <td>1.0</td>
      <td>29.844925</td>
      <td>9.483691e-07</td>
    </tr>
    <tr>
      <th>C(risk)</th>
      <td>66.519182</td>
      <td>1.0</td>
      <td>3.438001</td>
      <td>6.862955e-02</td>
    </tr>
    <tr>
      <th>C(treatment)</th>
      <td>20.132278</td>
      <td>2.0</td>
      <td>0.520262</td>
      <td>5.970215e-01</td>
    </tr>
    <tr>
      <th>C(gender):C(risk)</th>
      <td>215.217212</td>
      <td>1.0</td>
      <td>11.123363</td>
      <td>1.466050e-03</td>
    </tr>
    <tr>
      <th>C(gender):C(treatment)</th>
      <td>399.733382</td>
      <td>2.0</td>
      <td>10.329981</td>
      <td>1.395716e-04</td>
    </tr>
    <tr>
      <th>C(risk):C(treatment)</th>
      <td>110.970499</td>
      <td>2.0</td>
      <td>2.867719</td>
      <td>6.464732e-02</td>
    </tr>
    <tr>
      <th>C(gender):C(risk):C(treatment)</th>
      <td>286.595625</td>
      <td>2.0</td>
      <td>7.406255</td>
      <td>1.334476e-03</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>1160.892871</td>
      <td>60.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### Post-hoc tests

If there is a significant 3-way interaction effect, you can decompose it into:

-    Simple two-way interaction: run two-way interaction at each level of third variable,
-    Simple simple main effect: run one-way model at each level of second variable,
-    Simple simple pairwise comparisons: run pairwise or other post-hoc comparisons if necessary.

If you do not have a statistically significant three-way interaction, you need to determine whether you have any statistically significant two-way interaction from the ANOVA output. You can follow up a significant two-way interaction by simple main effects analyses and pairwise comparisons between groups if necessary.


#### Two-way interactions


```python
# with a significant 3-way interaction, each 2-way interaction is conditional on the 3rd factor
# so we test each 2-way interaction *within* each level of the 3rd variable.

def _simple_two_way(data_subset, formula, interaction_term, label):
    mod = smf.ols(formula, data=data_subset).fit()
    aov = anova_lm(mod, typ=3)
    p = aov.loc[interaction_term, 'PR(>F)']
    print(f"{label}: {interaction_term} p = {p:.4g}")
    display(aov)
    return p

# 1) gender × risk, separately for each treatment level
for t in sorted(data['treatment'].unique()):
    sub = data.loc[data['treatment'] == t].copy()
    _simple_two_way(
        sub,
        "pain_score ~ C(gender)*C(risk)",
        "C(gender):C(risk)",
        label=f"Within treatment={t}",
    )

# 2) gender × treatment, separately for each risk level
for r in sorted(data['risk'].unique()):
    sub = data.loc[data['risk'] == r].copy()
    _simple_two_way(
        sub,
        "pain_score ~ C(gender)*C(treatment)",
        "C(gender):C(treatment)",
        label=f"Within risk={r}",
    )

# 3) risk × treatment, separately for each gender level
for g in sorted(data['gender'].unique()):
    sub = data.loc[data['gender'] == g].copy()
    _simple_two_way(
        sub,
        "pain_score ~ C(risk)*C(treatment)",
        "C(risk):C(treatment)",
        label=f"Within gender={g}",
    )
```

    Within treatment=X: C(gender):C(risk) p = 0.004274
    


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum_sq</th>
      <th>df</th>
      <th>F</th>
      <th>PR(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Intercept</th>
      <td>37318.184764</td>
      <td>1.0</td>
      <td>1800.252865</td>
      <td>4.541104e-21</td>
    </tr>
    <tr>
      <th>C(gender)</th>
      <td>577.446016</td>
      <td>1.0</td>
      <td>27.856361</td>
      <td>3.640480e-05</td>
    </tr>
    <tr>
      <th>C(risk)</th>
      <td>66.519182</td>
      <td>1.0</td>
      <td>3.208927</td>
      <td>8.838384e-02</td>
    </tr>
    <tr>
      <th>C(gender):C(risk)</th>
      <td>215.217212</td>
      <td>1.0</td>
      <td>10.382215</td>
      <td>4.274481e-03</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>414.588256</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


    Within treatment=Y: C(gender):C(risk) p = 0.3508
    


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum_sq</th>
      <th>df</th>
      <th>F</th>
      <th>PR(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Intercept</th>
      <td>39536.579469</td>
      <td>1.0</td>
      <td>1845.025912</td>
      <td>3.561044e-21</td>
    </tr>
    <tr>
      <th>C(gender)</th>
      <td>4.080335</td>
      <td>1.0</td>
      <td>0.190414</td>
      <td>6.672464e-01</td>
    </tr>
    <tr>
      <th>C(risk)</th>
      <td>492.547990</td>
      <td>1.0</td>
      <td>22.985393</td>
      <td>1.104515e-04</td>
    </tr>
    <tr>
      <th>C(gender):C(risk)</th>
      <td>19.554744</td>
      <td>1.0</td>
      <td>0.912548</td>
      <td>3.508443e-01</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>428.574788</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


    Within treatment=Z: C(gender):C(risk) p = 0.07868
    


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum_sq</th>
      <th>df</th>
      <th>F</th>
      <th>PR(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Intercept</th>
      <td>39400.165336</td>
      <td>1.0</td>
      <td>2480.104915</td>
      <td>1.898022e-22</td>
    </tr>
    <tr>
      <th>C(gender)</th>
      <td>5.503246</td>
      <td>1.0</td>
      <td>0.346410</td>
      <td>5.627409e-01</td>
    </tr>
    <tr>
      <th>C(risk)</th>
      <td>380.064683</td>
      <td>1.0</td>
      <td>23.923765</td>
      <td>8.830056e-05</td>
    </tr>
    <tr>
      <th>C(gender):C(risk)</th>
      <td>54.554253</td>
      <td>1.0</td>
      <td>3.434003</td>
      <td>7.868063e-02</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>317.729827</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


    Within risk=high: C(gender):C(treatment) p = 0.0008597
    


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum_sq</th>
      <th>df</th>
      <th>F</th>
      <th>PR(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Intercept</th>
      <td>37318.184764</td>
      <td>1.0</td>
      <td>1683.075465</td>
      <td>6.509194e-28</td>
    </tr>
    <tr>
      <th>C(gender)</th>
      <td>577.446016</td>
      <td>1.0</td>
      <td>26.043207</td>
      <td>1.740421e-05</td>
    </tr>
    <tr>
      <th>C(treatment)</th>
      <td>20.132278</td>
      <td>2.0</td>
      <td>0.453990</td>
      <td>6.393808e-01</td>
    </tr>
    <tr>
      <th>C(gender):C(treatment)</th>
      <td>399.733382</td>
      <td>2.0</td>
      <td>9.014123</td>
      <td>8.597413e-04</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>665.178458</td>
      <td>30.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


    Within risk=low: C(gender):C(treatment) p = 0.6201
    


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum_sq</th>
      <th>df</th>
      <th>F</th>
      <th>PR(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Intercept</th>
      <td>32994.879212</td>
      <td>1.0</td>
      <td>1996.807738</td>
      <td>5.217062e-29</td>
    </tr>
    <tr>
      <th>C(gender)</th>
      <td>10.779355</td>
      <td>1.0</td>
      <td>0.652353</td>
      <td>4.256309e-01</td>
    </tr>
    <tr>
      <th>C(treatment)</th>
      <td>109.477870</td>
      <td>2.0</td>
      <td>3.312730</td>
      <td>5.012709e-02</td>
    </tr>
    <tr>
      <th>C(gender):C(treatment)</th>
      <td>16.044566</td>
      <td>2.0</td>
      <td>0.485498</td>
      <td>6.201420e-01</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>495.714413</td>
      <td>30.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


    Within gender=female: C(risk):C(treatment) p = 0.05378
    


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum_sq</th>
      <th>df</th>
      <th>F</th>
      <th>PR(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Intercept</th>
      <td>37318.184764</td>
      <td>1.0</td>
      <td>2170.521906</td>
      <td>1.518755e-29</td>
    </tr>
    <tr>
      <th>C(risk)</th>
      <td>66.519182</td>
      <td>1.0</td>
      <td>3.868927</td>
      <td>5.849696e-02</td>
    </tr>
    <tr>
      <th>C(treatment)</th>
      <td>20.132278</td>
      <td>2.0</td>
      <td>0.585473</td>
      <td>5.630789e-01</td>
    </tr>
    <tr>
      <th>C(risk):C(treatment)</th>
      <td>110.970499</td>
      <td>2.0</td>
      <td>3.227165</td>
      <td>5.377519e-02</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>515.795551</td>
      <td>30.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


    Within gender=male: C(risk):C(treatment) p = 0.01644
    


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum_sq</th>
      <th>df</th>
      <th>F</th>
      <th>PR(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Intercept</th>
      <td>51602.962433</td>
      <td>1.0</td>
      <td>2399.775699</td>
      <td>3.432538e-30</td>
    </tr>
    <tr>
      <th>C(risk)</th>
      <td>835.374326</td>
      <td>1.0</td>
      <td>38.848758</td>
      <td>7.285360e-07</td>
    </tr>
    <tr>
      <th>C(treatment)</th>
      <td>571.396212</td>
      <td>2.0</td>
      <td>13.286279</td>
      <td>7.374482e-05</td>
    </tr>
    <tr>
      <th>C(risk):C(treatment)</th>
      <td>203.220142</td>
      <td>2.0</td>
      <td>4.725337</td>
      <td>1.644381e-02</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>645.097320</td>
      <td>30.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


#### Main effects


```python
# Simple-simple main effects: test the effect of one factor at each combination of the other two.

def _one_way(data_subset, formula):
    mod = smf.ols(formula, data=data_subset).fit()
    aov = anova_lm(mod, typ=2)
    p = aov.iloc[0]['PR(>F)']
    return p, aov

rows = []

# A) Treatment effect within each (gender, risk) cell
for g in sorted(data['gender'].unique()):
    for r in sorted(data['risk'].unique()):
        sub = data.loc[(data['gender'] == g) & (data['risk'] == r)].copy()
        p, aov = _one_way(sub, "pain_score ~ C(treatment)")
        rows.append({"effect": "treatment", "within": f"gender={g}, risk={r}", "p": p})

# B) Risk effect within each (gender, treatment) cell
for g in sorted(data['gender'].unique()):
    for t in sorted(data['treatment'].unique()):
        sub = data.loc[(data['gender'] == g) & (data['treatment'] == t)].copy()
        p, aov = _one_way(sub, "pain_score ~ C(risk)")
        rows.append({"effect": "risk", "within": f"gender={g}, treatment={t}", "p": p})

# C) Gender effect within each (risk, treatment) cell
for r in sorted(data['risk'].unique()):
    for t in sorted(data['treatment'].unique()):
        sub = data.loc[(data['risk'] == r) & (data['treatment'] == t)].copy()
        p, aov = _one_way(sub, "pain_score ~ C(gender)")
        rows.append({"effect": "gender", "within": f"risk={r}, treatment={t}", "p": p})
simple_effects = pd.DataFrame(rows).sort_values(["effect", "p"])
display(simple_effects)
```


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>effect</th>
      <th>within</th>
      <th>p</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>gender</td>
      <td>risk=high, treatment=X</td>
      <td>0.000971</td>
    </tr>
    <tr>
      <th>15</th>
      <td>gender</td>
      <td>risk=low, treatment=Z</td>
      <td>0.067791</td>
    </tr>
    <tr>
      <th>14</th>
      <td>gender</td>
      <td>risk=low, treatment=Y</td>
      <td>0.091778</td>
    </tr>
    <tr>
      <th>13</th>
      <td>gender</td>
      <td>risk=low, treatment=X</td>
      <td>0.404662</td>
    </tr>
    <tr>
      <th>12</th>
      <td>gender</td>
      <td>risk=high, treatment=Z</td>
      <td>0.572021</td>
    </tr>
    <tr>
      <th>11</th>
      <td>gender</td>
      <td>risk=high, treatment=Y</td>
      <td>0.683641</td>
    </tr>
    <tr>
      <th>7</th>
      <td>risk</td>
      <td>gender=male, treatment=X</td>
      <td>0.000080</td>
    </tr>
    <tr>
      <th>6</th>
      <td>risk</td>
      <td>gender=female, treatment=Z</td>
      <td>0.000194</td>
    </tr>
    <tr>
      <th>5</th>
      <td>risk</td>
      <td>gender=female, treatment=Y</td>
      <td>0.000470</td>
    </tr>
    <tr>
      <th>8</th>
      <td>risk</td>
      <td>gender=male, treatment=Y</td>
      <td>0.008525</td>
    </tr>
    <tr>
      <th>9</th>
      <td>risk</td>
      <td>gender=male, treatment=Z</td>
      <td>0.071362</td>
    </tr>
    <tr>
      <th>4</th>
      <td>risk</td>
      <td>gender=female, treatment=X</td>
      <td>0.105042</td>
    </tr>
    <tr>
      <th>2</th>
      <td>treatment</td>
      <td>gender=male, risk=high</td>
      <td>0.000595</td>
    </tr>
    <tr>
      <th>1</th>
      <td>treatment</td>
      <td>gender=female, risk=low</td>
      <td>0.032188</td>
    </tr>
    <tr>
      <th>3</th>
      <td>treatment</td>
      <td>gender=male, risk=low</td>
      <td>0.549583</td>
    </tr>
    <tr>
      <th>0</th>
      <td>treatment</td>
      <td>gender=female, risk=high</td>
      <td>0.639198</td>
    </tr>
  </tbody>
</table>
</div>


#### Pairwise comparisons


```python
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd

alpha = 0.05
def tukey_in_slice(df, dv, factor, slice_label=""):
    # df: already-filtered slice
    levels = df[factor].dropna().unique()
    if len(levels) < 2:
        print(f"Skip (need >=2 levels): {factor} {slice_label}")
        return None

    res = pairwise_tukeyhsd(endog=df[dv], groups=df[factor], alpha=alpha)
    tbl = pd.DataFrame(res.summary().data[1:], columns=res.summary().data[0])

    header = f"Tukey HSD: {factor} {slice_label} (alpha={alpha})"
    print(header)
    display(tbl)
    return tbl

# 1) Treatment pairwise comparisons within each (gender, risk)
for gender in sorted(data["gender"].unique()):
    for risk in sorted(data["risk"].unique()):
        sub = data[(data["gender"] == gender) & (data["risk"] == risk)].copy()
        tukey_in_slice(sub, dv="pain_score", factor="treatment",
                       slice_label=f"within gender={gender}, risk={risk}")

# 2) Risk (2 levels) within each (gender, treatment)
for gender in sorted(data["gender"].unique()):
    for treatment in sorted(data["treatment"].unique()):
        sub = data[(data["gender"] == gender) & (data["treatment"] == treatment)].copy()
        tukey_in_slice(sub, dv="pain_score", factor="risk",
                       slice_label=f"within gender={gender}, treatment={treatment}")

# 3) Gender (2 levels) within each (risk, treatment)
for risk in sorted(data["risk"].unique()):
    for treatment in sorted(data["treatment"].unique()):
        sub = data[(data["risk"] == risk) & (data["treatment"] == treatment)].copy()
        tukey_in_slice(sub, dv="pain_score", factor="gender",
                       slice_label=f"within risk={risk}, treatment={treatment}")
```

    Tukey HSD: treatment within gender=female, risk=high (alpha=0.05)
    


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>group1</th>
      <th>group2</th>
      <th>meandiff</th>
      <th>p-adj</th>
      <th>lower</th>
      <th>upper</th>
      <th>reject</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>X</td>
      <td>Y</td>
      <td>2.3102</td>
      <td>0.6748</td>
      <td>-4.6961</td>
      <td>9.3166</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>X</td>
      <td>Z</td>
      <td>2.1701</td>
      <td>0.7060</td>
      <td>-4.8363</td>
      <td>9.1765</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Y</td>
      <td>Z</td>
      <td>-0.1402</td>
      <td>0.9985</td>
      <td>-7.1465</td>
      <td>6.8662</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>


    Tukey HSD: treatment within gender=female, risk=low (alpha=0.05)
    


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>group1</th>
      <th>group2</th>
      <th>meandiff</th>
      <th>p-adj</th>
      <th>lower</th>
      <th>upper</th>
      <th>reject</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>X</td>
      <td>Y</td>
      <td>-5.7943</td>
      <td>0.0319</td>
      <td>-11.1088</td>
      <td>-0.4798</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>X</td>
      <td>Z</td>
      <td>-4.3767</td>
      <td>0.1153</td>
      <td>-9.6912</td>
      <td>0.9378</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Y</td>
      <td>Z</td>
      <td>1.4176</td>
      <td>0.7712</td>
      <td>-3.8969</td>
      <td>6.7321</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>


    Tukey HSD: treatment within gender=male, risk=high (alpha=0.05)
    


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>group1</th>
      <th>group2</th>
      <th>meandiff</th>
      <th>p-adj</th>
      <th>lower</th>
      <th>upper</th>
      <th>reject</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>X</td>
      <td>Y</td>
      <td>-10.3973</td>
      <td>0.0047</td>
      <td>-17.5135</td>
      <td>-3.2811</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>X</td>
      <td>Z</td>
      <td>-13.0581</td>
      <td>0.0007</td>
      <td>-20.1743</td>
      <td>-5.9419</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Y</td>
      <td>Z</td>
      <td>-2.6608</td>
      <td>0.6054</td>
      <td>-9.7770</td>
      <td>4.4554</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>


    Tukey HSD: treatment within gender=male, risk=low (alpha=0.05)
    


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>group1</th>
      <th>group2</th>
      <th>meandiff</th>
      <th>p-adj</th>
      <th>lower</th>
      <th>upper</th>
      <th>reject</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>X</td>
      <td>Y</td>
      <td>-2.9130</td>
      <td>0.5201</td>
      <td>-9.7011</td>
      <td>3.8751</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>X</td>
      <td>Z</td>
      <td>-1.5959</td>
      <td>0.8166</td>
      <td>-8.3840</td>
      <td>5.1922</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Y</td>
      <td>Z</td>
      <td>1.3171</td>
      <td>0.8705</td>
      <td>-5.4710</td>
      <td>8.1052</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>


    Tukey HSD: risk within gender=female, treatment=X (alpha=0.05)
    


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>group1</th>
      <th>group2</th>
      <th>meandiff</th>
      <th>p-adj</th>
      <th>lower</th>
      <th>upper</th>
      <th>reject</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>high</td>
      <td>low</td>
      <td>-4.7088</td>
      <td>0.105</td>
      <td>-10.5957</td>
      <td>1.1781</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>


    Tukey HSD: risk within gender=female, treatment=Y (alpha=0.05)
    


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>group1</th>
      <th>group2</th>
      <th>meandiff</th>
      <th>p-adj</th>
      <th>lower</th>
      <th>upper</th>
      <th>reject</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>high</td>
      <td>low</td>
      <td>-12.8134</td>
      <td>0.0005</td>
      <td>-18.4208</td>
      <td>-7.206</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>


    Tukey HSD: risk within gender=female, treatment=Z (alpha=0.05)
    


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>group1</th>
      <th>group2</th>
      <th>meandiff</th>
      <th>p-adj</th>
      <th>lower</th>
      <th>upper</th>
      <th>reject</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>high</td>
      <td>low</td>
      <td>-11.2556</td>
      <td>0.0002</td>
      <td>-15.6441</td>
      <td>-6.8671</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>


    Tukey HSD: risk within gender=male, treatment=X (alpha=0.05)
    


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>group1</th>
      <th>group2</th>
      <th>meandiff</th>
      <th>p-adj</th>
      <th>lower</th>
      <th>upper</th>
      <th>reject</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>high</td>
      <td>low</td>
      <td>-16.6871</td>
      <td>0.0001</td>
      <td>-22.514</td>
      <td>-10.8601</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>


    Tukey HSD: risk within gender=male, treatment=Y (alpha=0.05)
    


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>group1</th>
      <th>group2</th>
      <th>meandiff</th>
      <th>p-adj</th>
      <th>lower</th>
      <th>upper</th>
      <th>reject</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>high</td>
      <td>low</td>
      <td>-9.2028</td>
      <td>0.0085</td>
      <td>-15.4861</td>
      <td>-2.9194</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>


    Tukey HSD: risk within gender=male, treatment=Z (alpha=0.05)
    


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>group1</th>
      <th>group2</th>
      <th>meandiff</th>
      <th>p-adj</th>
      <th>lower</th>
      <th>upper</th>
      <th>reject</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>high</td>
      <td>low</td>
      <td>-5.2249</td>
      <td>0.0714</td>
      <td>-10.9973</td>
      <td>0.5476</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>


    Tukey HSD: gender within risk=high, treatment=X (alpha=0.05)
    


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>group1</th>
      <th>group2</th>
      <th>meandiff</th>
      <th>p-adj</th>
      <th>lower</th>
      <th>upper</th>
      <th>reject</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>female</td>
      <td>male</td>
      <td>13.8738</td>
      <td>0.001</td>
      <td>7.1622</td>
      <td>20.5854</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>


    Tukey HSD: gender within risk=high, treatment=Y (alpha=0.05)
    


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>group1</th>
      <th>group2</th>
      <th>meandiff</th>
      <th>p-adj</th>
      <th>lower</th>
      <th>upper</th>
      <th>reject</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>female</td>
      <td>male</td>
      <td>1.1662</td>
      <td>0.6836</td>
      <td>-5.0265</td>
      <td>7.3589</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>


    Tukey HSD: gender within risk=high, treatment=Z (alpha=0.05)
    


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>group1</th>
      <th>group2</th>
      <th>meandiff</th>
      <th>p-adj</th>
      <th>lower</th>
      <th>upper</th>
      <th>reject</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>female</td>
      <td>male</td>
      <td>-1.3544</td>
      <td>0.572</td>
      <td>-6.52</td>
      <td>3.8112</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>


    Tukey HSD: gender within risk=low, treatment=X (alpha=0.05)
    


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>group1</th>
      <th>group2</th>
      <th>meandiff</th>
      <th>p-adj</th>
      <th>lower</th>
      <th>upper</th>
      <th>reject</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>female</td>
      <td>male</td>
      <td>1.8956</td>
      <td>0.4047</td>
      <td>-2.9587</td>
      <td>6.7498</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>


    Tukey HSD: gender within risk=low, treatment=Y (alpha=0.05)
    


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>group1</th>
      <th>group2</th>
      <th>meandiff</th>
      <th>p-adj</th>
      <th>lower</th>
      <th>upper</th>
      <th>reject</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>female</td>
      <td>male</td>
      <td>4.7768</td>
      <td>0.0918</td>
      <td>-0.9305</td>
      <td>10.4842</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>


    Tukey HSD: gender within risk=low, treatment=Z (alpha=0.05)
    


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>group1</th>
      <th>group2</th>
      <th>meandiff</th>
      <th>p-adj</th>
      <th>lower</th>
      <th>upper</th>
      <th>reject</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>female</td>
      <td>male</td>
      <td>4.6763</td>
      <td>0.0678</td>
      <td>-0.4126</td>
      <td>9.7652</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>


### Interaction visualisation


```python
import matplotlib.pyplot as plt
import seaborn as sns
df = data.copy()
treatment_order = sorted(df["treatment"].dropna().unique())
risk_order = sorted(df["risk"].dropna().unique())
gender_order = sorted(df["gender"].dropna().unique())
sns.set_theme(style="whitegrid")
# Facet by gender, show risk as lines, treatment on x-axis (mean ± 95% CI)
g = sns.catplot(
    data=df,
    x="treatment",
    y="pain_score",
    hue="risk",
    col="gender",
    kind="point",
    order=treatment_order,
    hue_order=risk_order,
    col_order=gender_order,
    errorbar=("ci", 95),   # seaborn >= 0.12
    dodge=0.25,
    capsize=0.12,
    height=4,
    aspect=1.2,
)

g.set_axis_labels("Treatment", "Mean pain score")
g.set_titles("Gender = {col_name}")
g.fig.suptitle(
    "3-way interaction: Treatment × Risk within Gender (mean ± 95% CI)",
    y=1.05
)
plt.show()

```


    
![png](output_22_0.png)
    


### Interpretation

There is a significant gender × risk × treatment interaction on pain score. Follow-up simple-effects analyses show treatment differences are present mainly for high-risk males (X < Y and X < Z) and low-risk females (X < Y), with no detectable treatment differences in the other gender×risk slices.

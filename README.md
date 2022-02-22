# Final---Project

Project Title: Credit Card Fraud Detection
Author: Arnab Bera
Introduction
In this notebook, I am going to predict whether a credit card is fraud or not using Machine Learning. The datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions. Due to confidentiality issues, the input variables are transformed into numerical using PCA transformations.

The dataset is taken from kaggle here.

Importing Libraies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
import warnings
warnings.simplefilter('ignore')
Reading and Preprocessing the Dataset
data=pd.read_csv("creditcard.csv")
df=pd.DataFrame(data)
df
Time	V1	V2	V3	V4	V5	V6	V7	V8	V9	...	V21	V22	V23	V24	V25	V26	V27	V28	Amount	Class
0	0.0	-1.359807	-0.072781	2.536347	1.378155	-0.338321	0.462388	0.239599	0.098698	0.363787	...	-0.018307	0.277838	-0.110474	0.066928	0.128539	-0.189115	0.133558	-0.021053	149.62	0
1	0.0	1.191857	0.266151	0.166480	0.448154	0.060018	-0.082361	-0.078803	0.085102	-0.255425	...	-0.225775	-0.638672	0.101288	-0.339846	0.167170	0.125895	-0.008983	0.014724	2.69	0
2	1.0	-1.358354	-1.340163	1.773209	0.379780	-0.503198	1.800499	0.791461	0.247676	-1.514654	...	0.247998	0.771679	0.909412	-0.689281	-0.327642	-0.139097	-0.055353	-0.059752	378.66	0
3	1.0	-0.966272	-0.185226	1.792993	-0.863291	-0.010309	1.247203	0.237609	0.377436	-1.387024	...	-0.108300	0.005274	-0.190321	-1.175575	0.647376	-0.221929	0.062723	0.061458	123.50	0
4	2.0	-1.158233	0.877737	1.548718	0.403034	-0.407193	0.095921	0.592941	-0.270533	0.817739	...	-0.009431	0.798278	-0.137458	0.141267	-0.206010	0.502292	0.219422	0.215153	69.99	0
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
284802	172786.0	-11.881118	10.071785	-9.834783	-2.066656	-5.364473	-2.606837	-4.918215	7.305334	1.914428	...	0.213454	0.111864	1.014480	-0.509348	1.436807	0.250034	0.943651	0.823731	0.77	0
284803	172787.0	-0.732789	-0.055080	2.035030	-0.738589	0.868229	1.058415	0.024330	0.294869	0.584800	...	0.214205	0.924384	0.012463	-1.016226	-0.606624	-0.395255	0.068472	-0.053527	24.79	0
284804	172788.0	1.919565	-0.301254	-3.249640	-0.557828	2.630515	3.031260	-0.296827	0.708417	0.432454	...	0.232045	0.578229	-0.037501	0.640134	0.265745	-0.087371	0.004455	-0.026561	67.88	0
284805	172788.0	-0.240440	0.530483	0.702510	0.689799	-0.377961	0.623708	-0.686180	0.679145	0.392087	...	0.265245	0.800049	-0.163298	0.123205	-0.569159	0.546668	0.108821	0.104533	10.00	0
284806	172792.0	-0.533413	-0.189733	0.703337	-0.506271	-0.012546	-0.649617	1.577006	-0.414650	0.486180	...	0.261057	0.643078	0.376777	0.008797	-0.473649	-0.818267	-0.002415	0.013649	217.00	0
284807 rows × 31 columns

Basic Infornation about the Dataset
df.shape
(284807, 31)
Thus there are 284807 rows and 31 columns.

df.describe()
Time	V1	V2	V3	V4	V5	V6	V7	V8	V9	...	V21	V22	V23	V24	V25	V26	V27	V28	Amount	Class
count	284807.000000	2.848070e+05	2.848070e+05	2.848070e+05	2.848070e+05	2.848070e+05	2.848070e+05	2.848070e+05	2.848070e+05	2.848070e+05	...	2.848070e+05	2.848070e+05	2.848070e+05	2.848070e+05	2.848070e+05	2.848070e+05	2.848070e+05	2.848070e+05	284807.000000	284807.000000
mean	94813.859575	3.918649e-15	5.682686e-16	-8.761736e-15	2.811118e-15	-1.552103e-15	2.040130e-15	-1.698953e-15	-1.893285e-16	-3.147640e-15	...	1.473120e-16	8.042109e-16	5.282512e-16	4.456271e-15	1.426896e-15	1.701640e-15	-3.662252e-16	-1.217809e-16	88.349619	0.001727
std	47488.145955	1.958696e+00	1.651309e+00	1.516255e+00	1.415869e+00	1.380247e+00	1.332271e+00	1.237094e+00	1.194353e+00	1.098632e+00	...	7.345240e-01	7.257016e-01	6.244603e-01	6.056471e-01	5.212781e-01	4.822270e-01	4.036325e-01	3.300833e-01	250.120109	0.041527
min	0.000000	-5.640751e+01	-7.271573e+01	-4.832559e+01	-5.683171e+00	-1.137433e+02	-2.616051e+01	-4.355724e+01	-7.321672e+01	-1.343407e+01	...	-3.483038e+01	-1.093314e+01	-4.480774e+01	-2.836627e+00	-1.029540e+01	-2.604551e+00	-2.256568e+01	-1.543008e+01	0.000000	0.000000
25%	54201.500000	-9.203734e-01	-5.985499e-01	-8.903648e-01	-8.486401e-01	-6.915971e-01	-7.682956e-01	-5.540759e-01	-2.086297e-01	-6.430976e-01	...	-2.283949e-01	-5.423504e-01	-1.618463e-01	-3.545861e-01	-3.171451e-01	-3.269839e-01	-7.083953e-02	-5.295979e-02	5.600000	0.000000
50%	84692.000000	1.810880e-02	6.548556e-02	1.798463e-01	-1.984653e-02	-5.433583e-02	-2.741871e-01	4.010308e-02	2.235804e-02	-5.142873e-02	...	-2.945017e-02	6.781943e-03	-1.119293e-02	4.097606e-02	1.659350e-02	-5.213911e-02	1.342146e-03	1.124383e-02	22.000000	0.000000
75%	139320.500000	1.315642e+00	8.037239e-01	1.027196e+00	7.433413e-01	6.119264e-01	3.985649e-01	5.704361e-01	3.273459e-01	5.971390e-01	...	1.863772e-01	5.285536e-01	1.476421e-01	4.395266e-01	3.507156e-01	2.409522e-01	9.104512e-02	7.827995e-02	77.165000	0.000000
max	172792.000000	2.454930e+00	2.205773e+01	9.382558e+00	1.687534e+01	3.480167e+01	7.330163e+01	1.205895e+02	2.000721e+01	1.559499e+01	...	2.720284e+01	1.050309e+01	2.252841e+01	4.584549e+00	7.519589e+00	3.517346e+00	3.161220e+01	3.384781e+01	25691.160000	1.000000
8 rows × 31 columns

df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 284807 entries, 0 to 284806
Data columns (total 31 columns):
 #   Column  Non-Null Count   Dtype  
---  ------  --------------   -----  
 0   Time    284807 non-null  float64
 1   V1      284807 non-null  float64
 2   V2      284807 non-null  float64
 3   V3      284807 non-null  float64
 4   V4      284807 non-null  float64
 5   V5      284807 non-null  float64
 6   V6      284807 non-null  float64
 7   V7      284807 non-null  float64
 8   V8      284807 non-null  float64
 9   V9      284807 non-null  float64
 10  V10     284807 non-null  float64
 11  V11     284807 non-null  float64
 12  V12     284807 non-null  float64
 13  V13     284807 non-null  float64
 14  V14     284807 non-null  float64
 15  V15     284807 non-null  float64
 16  V16     284807 non-null  float64
 17  V17     284807 non-null  float64
 18  V18     284807 non-null  float64
 19  V19     284807 non-null  float64
 20  V20     284807 non-null  float64
 21  V21     284807 non-null  float64
 22  V22     284807 non-null  float64
 23  V23     284807 non-null  float64
 24  V24     284807 non-null  float64
 25  V25     284807 non-null  float64
 26  V26     284807 non-null  float64
 27  V27     284807 non-null  float64
 28  V28     284807 non-null  float64
 29  Amount  284807 non-null  float64
 30  Class   284807 non-null  int64  
dtypes: float64(30), int64(1)
memory usage: 67.4 MB
df.columns
Index(['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',
       'Class'],
      dtype='object')
df.head()
Time	V1	V2	V3	V4	V5	V6	V7	V8	V9	...	V21	V22	V23	V24	V25	V26	V27	V28	Amount	Class
0	0.0	-1.359807	-0.072781	2.536347	1.378155	-0.338321	0.462388	0.239599	0.098698	0.363787	...	-0.018307	0.277838	-0.110474	0.066928	0.128539	-0.189115	0.133558	-0.021053	149.62	0
1	0.0	1.191857	0.266151	0.166480	0.448154	0.060018	-0.082361	-0.078803	0.085102	-0.255425	...	-0.225775	-0.638672	0.101288	-0.339846	0.167170	0.125895	-0.008983	0.014724	2.69	0
2	1.0	-1.358354	-1.340163	1.773209	0.379780	-0.503198	1.800499	0.791461	0.247676	-1.514654	...	0.247998	0.771679	0.909412	-0.689281	-0.327642	-0.139097	-0.055353	-0.059752	378.66	0
3	1.0	-0.966272	-0.185226	1.792993	-0.863291	-0.010309	1.247203	0.237609	0.377436	-1.387024	...	-0.108300	0.005274	-0.190321	-1.175575	0.647376	-0.221929	0.062723	0.061458	123.50	0
4	2.0	-1.158233	0.877737	1.548718	0.403034	-0.407193	0.095921	0.592941	-0.270533	0.817739	...	-0.009431	0.798278	-0.137458	0.141267	-0.206010	0.502292	0.219422	0.215153	69.99	0
5 rows × 31 columns

df.tail()
Time	V1	V2	V3	V4	V5	V6	V7	V8	V9	...	V21	V22	V23	V24	V25	V26	V27	V28	Amount	Class
284802	172786.0	-11.881118	10.071785	-9.834783	-2.066656	-5.364473	-2.606837	-4.918215	7.305334	1.914428	...	0.213454	0.111864	1.014480	-0.509348	1.436807	0.250034	0.943651	0.823731	0.77	0
284803	172787.0	-0.732789	-0.055080	2.035030	-0.738589	0.868229	1.058415	0.024330	0.294869	0.584800	...	0.214205	0.924384	0.012463	-1.016226	-0.606624	-0.395255	0.068472	-0.053527	24.79	0
284804	172788.0	1.919565	-0.301254	-3.249640	-0.557828	2.630515	3.031260	-0.296827	0.708417	0.432454	...	0.232045	0.578229	-0.037501	0.640134	0.265745	-0.087371	0.004455	-0.026561	67.88	0
284805	172788.0	-0.240440	0.530483	0.702510	0.689799	-0.377961	0.623708	-0.686180	0.679145	0.392087	...	0.265245	0.800049	-0.163298	0.123205	-0.569159	0.546668	0.108821	0.104533	10.00	0
284806	172792.0	-0.533413	-0.189733	0.703337	-0.506271	-0.012546	-0.649617	1.577006	-0.414650	0.486180	...	0.261057	0.643078	0.376777	0.008797	-0.473649	-0.818267	-0.002415	0.013649	217.00	0
5 rows × 31 columns

Checking Null Values
df.isnull().sum()
Time      0
V1        0
V2        0
V3        0
V4        0
V5        0
V6        0
V7        0
V8        0
V9        0
V10       0
V11       0
V12       0
V13       0
V14       0
V15       0
V16       0
V17       0
V18       0
V19       0
V20       0
V21       0
V22       0
V23       0
V24       0
V25       0
V26       0
V27       0
V28       0
Amount    0
Class     0
dtype: int64
Thus there are no null values in the dataset.

EDA
Checking Fraud & Non Fraud Cases
fraud_cases=len(data[data['Class']==1])
print('Number of Fraud Cases:',fraud_cases)
Number of Fraud Cases: 492
non_fraud_cases=len(data[data['Class']==0])
print('Number of Non Fraud Cases:',non_fraud_cases)
Number of Non Fraud Cases: 284315
df['Class'].value_counts().plot(kind="bar")
plt.xticks(range(2),['Non Fraud','Fraud'])
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

By looking at the above statistics, we can see that the data is highly imbalanced. Only 492 out of 284807 are fraud.

fraud=data[data['Class']==1]
fraud
Time	V1	V2	V3	V4	V5	V6	V7	V8	V9	...	V21	V22	V23	V24	V25	V26	V27	V28	Amount	Class
541	406.0	-2.312227	1.951992	-1.609851	3.997906	-0.522188	-1.426545	-2.537387	1.391657	-2.770089	...	0.517232	-0.035049	-0.465211	0.320198	0.044519	0.177840	0.261145	-0.143276	0.00	1
623	472.0	-3.043541	-3.157307	1.088463	2.288644	1.359805	-1.064823	0.325574	-0.067794	-0.270953	...	0.661696	0.435477	1.375966	-0.293803	0.279798	-0.145362	-0.252773	0.035764	529.00	1
4920	4462.0	-2.303350	1.759247	-0.359745	2.330243	-0.821628	-0.075788	0.562320	-0.399147	-0.238253	...	-0.294166	-0.932391	0.172726	-0.087330	-0.156114	-0.542628	0.039566	-0.153029	239.93	1
6108	6986.0	-4.397974	1.358367	-2.592844	2.679787	-1.128131	-1.706536	-3.496197	-0.248778	-0.247768	...	0.573574	0.176968	-0.436207	-0.053502	0.252405	-0.657488	-0.827136	0.849573	59.00	1
6329	7519.0	1.234235	3.019740	-4.304597	4.732795	3.624201	-1.357746	1.713445	-0.496358	-1.282858	...	-0.379068	-0.704181	-0.656805	-1.632653	1.488901	0.566797	-0.010016	0.146793	1.00	1
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
279863	169142.0	-1.927883	1.125653	-4.518331	1.749293	-1.566487	-2.010494	-0.882850	0.697211	-2.064945	...	0.778584	-0.319189	0.639419	-0.294885	0.537503	0.788395	0.292680	0.147968	390.00	1
280143	169347.0	1.378559	1.289381	-5.004247	1.411850	0.442581	-1.326536	-1.413170	0.248525	-1.127396	...	0.370612	0.028234	-0.145640	-0.081049	0.521875	0.739467	0.389152	0.186637	0.76	1
280149	169351.0	-0.676143	1.126366	-2.213700	0.468308	-1.120541	-0.003346	-2.234739	1.210158	-0.652250	...	0.751826	0.834108	0.190944	0.032070	-0.739695	0.471111	0.385107	0.194361	77.89	1
281144	169966.0	-3.113832	0.585864	-5.399730	1.817092	-0.840618	-2.943548	-2.208002	1.058733	-1.632333	...	0.583276	-0.269209	-0.456108	-0.183659	-0.328168	0.606116	0.884876	-0.253700	245.00	1
281674	170348.0	1.991976	0.158476	-2.583441	0.408670	1.151147	-0.096695	0.223050	-0.068384	0.577829	...	-0.164350	-0.295135	-0.072173	-0.450261	0.313267	-0.289617	0.002988	-0.015309	42.53	1
492 rows × 31 columns

fraud["Amount"].describe()
count     492.000000
mean      122.211321
std       256.683288
min         0.000000
25%         1.000000
50%         9.250000
75%       105.890000
max      2125.870000
Name: Amount, dtype: float64
def amount(var):
    if var<9:
        return 'low'
    elif var<105:
        return 'medium'
    else:
        return 'high'
fraud["Amount"]=fraud["Amount"].apply(amount)
fraud
Time	V1	V2	V3	V4	V5	V6	V7	V8	V9	...	V21	V22	V23	V24	V25	V26	V27	V28	Amount	Class
541	406.0	-2.312227	1.951992	-1.609851	3.997906	-0.522188	-1.426545	-2.537387	1.391657	-2.770089	...	0.517232	-0.035049	-0.465211	0.320198	0.044519	0.177840	0.261145	-0.143276	low	1
623	472.0	-3.043541	-3.157307	1.088463	2.288644	1.359805	-1.064823	0.325574	-0.067794	-0.270953	...	0.661696	0.435477	1.375966	-0.293803	0.279798	-0.145362	-0.252773	0.035764	high	1
4920	4462.0	-2.303350	1.759247	-0.359745	2.330243	-0.821628	-0.075788	0.562320	-0.399147	-0.238253	...	-0.294166	-0.932391	0.172726	-0.087330	-0.156114	-0.542628	0.039566	-0.153029	high	1
6108	6986.0	-4.397974	1.358367	-2.592844	2.679787	-1.128131	-1.706536	-3.496197	-0.248778	-0.247768	...	0.573574	0.176968	-0.436207	-0.053502	0.252405	-0.657488	-0.827136	0.849573	medium	1
6329	7519.0	1.234235	3.019740	-4.304597	4.732795	3.624201	-1.357746	1.713445	-0.496358	-1.282858	...	-0.379068	-0.704181	-0.656805	-1.632653	1.488901	0.566797	-0.010016	0.146793	low	1
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
279863	169142.0	-1.927883	1.125653	-4.518331	1.749293	-1.566487	-2.010494	-0.882850	0.697211	-2.064945	...	0.778584	-0.319189	0.639419	-0.294885	0.537503	0.788395	0.292680	0.147968	high	1
280143	169347.0	1.378559	1.289381	-5.004247	1.411850	0.442581	-1.326536	-1.413170	0.248525	-1.127396	...	0.370612	0.028234	-0.145640	-0.081049	0.521875	0.739467	0.389152	0.186637	low	1
280149	169351.0	-0.676143	1.126366	-2.213700	0.468308	-1.120541	-0.003346	-2.234739	1.210158	-0.652250	...	0.751826	0.834108	0.190944	0.032070	-0.739695	0.471111	0.385107	0.194361	medium	1
281144	169966.0	-3.113832	0.585864	-5.399730	1.817092	-0.840618	-2.943548	-2.208002	1.058733	-1.632333	...	0.583276	-0.269209	-0.456108	-0.183659	-0.328168	0.606116	0.884876	-0.253700	high	1
281674	170348.0	1.991976	0.158476	-2.583441	0.408670	1.151147	-0.096695	0.223050	-0.068384	0.577829	...	-0.164350	-0.295135	-0.072173	-0.450261	0.313267	-0.289617	0.002988	-0.015309	medium	1
492 rows × 31 columns

pd.crosstab(fraud["Amount"],fraud["Class"]).plot(kind="bar")
<AxesSubplot:xlabel='Amount'>

From the above graph we can see that for the lower amount the chances of fraud is more.

data.hist(figsize=(20,20),color='lime')
plt.show()

From the above histograms we can see that Time is more distributed but another features are not that much distributed.

Fraud=data[data['Class']==1]
NonFraud=data[data['Class']==0]
rcParams['figure.figsize'] = 12, 6
f,(ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Amount by class')
ax1.scatter(Fraud.Time, Fraud.Amount)
ax1.set_title('Fraud')
ax2.scatter(NonFraud.Time, NonFraud.Amount)
ax2.set_title('NonFraud')
plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()

From the Scatter plot we can see that for the fraud cases amount is more distributed.

CORRELATION

corr=data.corr()
corr
Time	V1	V2	V3	V4	V5	V6	V7	V8	V9	...	V21	V22	V23	V24	V25	V26	V27	V28	Amount	Class
Time	1.000000	1.173963e-01	-1.059333e-02	-4.196182e-01	-1.052602e-01	1.730721e-01	-6.301647e-02	8.471437e-02	-3.694943e-02	-8.660434e-03	...	4.473573e-02	1.440591e-01	5.114236e-02	-1.618187e-02	-2.330828e-01	-4.140710e-02	-5.134591e-03	-9.412688e-03	-0.010596	-0.012323
V1	0.117396	1.000000e+00	4.135835e-16	-1.227819e-15	-9.215150e-16	1.812612e-17	-6.506567e-16	-1.005191e-15	-2.433822e-16	-1.513678e-16	...	-2.457409e-16	-4.290944e-16	6.168652e-16	-4.425156e-17	-9.605737e-16	-1.581290e-17	1.198124e-16	2.083082e-15	-0.227709	-0.101347
V2	-0.010593	4.135835e-16	1.000000e+00	3.243764e-16	-1.121065e-15	5.157519e-16	2.787346e-16	2.055934e-16	-5.377041e-17	1.978488e-17	...	-8.480447e-17	1.526333e-16	1.634231e-16	1.247925e-17	-4.478846e-16	2.057310e-16	-4.966953e-16	-5.093836e-16	-0.531409	0.091289
V3	-0.419618	-1.227819e-15	3.243764e-16	1.000000e+00	4.711293e-16	-6.539009e-17	1.627627e-15	4.895305e-16	-1.268779e-15	5.568367e-16	...	5.706192e-17	-1.133902e-15	-4.983035e-16	2.686834e-19	-1.104734e-15	-1.238062e-16	1.045747e-15	9.775546e-16	-0.210880	-0.192961
V4	-0.105260	-9.215150e-16	-1.121065e-15	4.711293e-16	1.000000e+00	-1.719944e-15	-7.491959e-16	-4.104503e-16	5.697192e-16	6.923247e-16	...	-1.949553e-16	-6.276051e-17	9.164206e-17	1.584638e-16	6.070716e-16	-4.247268e-16	3.977061e-17	-2.761403e-18	0.098732	0.133447
V5	0.173072	1.812612e-17	5.157519e-16	-6.539009e-17	-1.719944e-15	1.000000e+00	2.408382e-16	2.715541e-16	7.437229e-16	7.391702e-16	...	-3.920976e-16	1.253751e-16	-8.428683e-18	-1.149255e-15	4.808532e-16	4.319541e-16	6.590482e-16	-5.613951e-18	-0.386356	-0.094974
V6	-0.063016	-6.506567e-16	2.787346e-16	1.627627e-15	-7.491959e-16	2.408382e-16	1.000000e+00	1.191668e-16	-1.104219e-16	4.131207e-16	...	5.833316e-17	-4.705235e-19	1.046712e-16	-1.071589e-15	4.562861e-16	-1.357067e-16	-4.452461e-16	2.594754e-16	0.215981	-0.043643
V7	0.084714	-1.005191e-15	2.055934e-16	4.895305e-16	-4.104503e-16	2.715541e-16	1.191668e-16	1.000000e+00	3.344412e-16	1.122501e-15	...	-2.027779e-16	-8.898922e-16	-4.387401e-16	7.434913e-18	-3.094082e-16	-9.657637e-16	-1.782106e-15	-2.776530e-16	0.397311	-0.187257
V8	-0.036949	-2.433822e-16	-5.377041e-17	-1.268779e-15	5.697192e-16	7.437229e-16	-1.104219e-16	3.344412e-16	1.000000e+00	4.356078e-16	...	3.892798e-16	2.026927e-16	6.377260e-17	-1.047097e-16	-4.653279e-16	-1.727276e-16	1.299943e-16	-6.200930e-16	-0.103079	0.019875
V9	-0.008660	-1.513678e-16	1.978488e-17	5.568367e-16	6.923247e-16	7.391702e-16	4.131207e-16	1.122501e-15	4.356078e-16	1.000000e+00	...	1.936953e-16	-7.071869e-16	-5.214137e-16	-1.430343e-16	6.757763e-16	-7.888853e-16	-6.709655e-17	1.110541e-15	-0.044246	-0.097733
V10	0.030617	7.388135e-17	-3.991394e-16	1.156587e-15	2.232685e-16	-5.202306e-16	5.932243e-17	-7.492834e-17	-2.801370e-16	-4.642274e-16	...	1.177547e-15	-6.418202e-16	3.214491e-16	-1.355885e-16	-2.846052e-16	-3.028119e-16	-2.197977e-16	4.864782e-17	-0.101502	-0.216883
V11	-0.247689	2.125498e-16	1.975426e-16	1.576830e-15	3.459380e-16	7.203963e-16	1.980503e-15	1.425248e-16	2.487043e-16	1.354680e-16	...	-5.658364e-16	7.772895e-16	-4.505332e-16	1.933267e-15	-5.600475e-16	-1.003221e-16	-2.640281e-16	-3.792314e-16	0.000104	0.154876
V12	0.124348	2.053457e-16	-9.568710e-17	6.310231e-16	-5.625518e-16	7.412552e-16	2.375468e-16	-3.536655e-18	1.839891e-16	-1.079314e-15	...	7.300527e-16	1.644699e-16	1.800885e-16	4.436512e-16	-5.712973e-16	-2.359969e-16	-4.672391e-16	6.415167e-16	-0.009542	-0.260593
V13	-0.065902	-2.425603e-17	6.295388e-16	2.807652e-16	1.303306e-16	5.886991e-16	-1.211182e-16	1.266462e-17	-2.921856e-16	2.251072e-15	...	1.008461e-16	6.747721e-17	-7.132064e-16	-1.397470e-16	-5.497612e-16	-1.769255e-16	-4.720898e-16	1.144372e-15	0.005293	-0.004570
V14	-0.098757	-5.020280e-16	-1.730566e-16	4.739859e-16	2.282280e-16	6.565143e-16	2.621312e-16	2.607772e-16	-8.599156e-16	3.784757e-15	...	-3.356561e-16	3.740383e-16	3.883204e-16	2.003482e-16	-8.547932e-16	-1.660327e-16	1.044274e-16	2.289427e-15	0.033751	-0.302544
V15	-0.183453	3.547782e-16	-4.995814e-17	9.068793e-16	1.377649e-16	-8.720275e-16	-1.531188e-15	-1.690540e-16	4.127777e-16	-1.051167e-15	...	6.605263e-17	-4.208921e-16	-3.912243e-16	-4.478263e-16	3.206423e-16	2.817791e-16	-1.143519e-15	-1.194130e-15	-0.002986	-0.004223
V16	0.011903	7.212815e-17	1.177316e-17	8.299445e-16	-9.614528e-16	2.246261e-15	2.623672e-18	5.869302e-17	-5.254741e-16	-1.214086e-15	...	-4.715090e-16	-7.923387e-17	5.020770e-16	-3.005985e-16	-1.345418e-15	-7.290010e-16	6.789513e-16	7.588849e-16	-0.003910	-0.196539
V17	-0.073297	-3.879840e-16	-2.685296e-16	7.614712e-16	-2.699612e-16	1.281914e-16	2.015618e-16	2.177192e-16	-2.269549e-16	1.113695e-15	...	-8.230527e-16	-8.743398e-16	3.706214e-16	-2.403828e-16	2.666806e-16	6.932833e-16	6.148525e-16	-5.534540e-17	0.007309	-0.326481
V18	0.090438	3.230206e-17	3.284605e-16	1.509897e-16	-5.103644e-16	5.308590e-16	1.223814e-16	7.604126e-17	-3.667974e-16	4.993240e-16	...	-9.408680e-16	-4.819365e-16	-1.912006e-16	-8.986916e-17	-6.629212e-17	2.990167e-16	2.242791e-16	7.976796e-16	0.035650	-0.111485
V19	0.028975	1.502024e-16	-7.118719e-18	3.463522e-16	-3.980557e-16	-1.450421e-16	-1.865597e-16	-1.881008e-16	-3.875186e-16	-1.376135e-16	...	5.115885e-16	-1.163768e-15	7.032035e-16	2.587708e-17	9.577163e-16	5.898033e-16	-2.959370e-16	-1.405379e-15	-0.056151	0.034783
V20	-0.050866	4.654551e-16	2.506675e-16	-9.316409e-16	-1.857247e-16	-3.554057e-16	-1.858755e-16	9.379684e-16	2.033737e-16	-2.343720e-16	...	-7.614597e-16	1.009285e-15	2.712885e-16	1.277215e-16	1.410054e-16	-2.803504e-16	-1.138829e-15	-2.436795e-16	0.339403	0.020090
V21	0.044736	-2.457409e-16	-8.480447e-17	5.706192e-17	-1.949553e-16	-3.920976e-16	5.833316e-17	-2.027779e-16	3.892798e-16	1.936953e-16	...	1.000000e+00	3.649908e-15	8.119580e-16	1.761054e-16	-1.686082e-16	-5.557329e-16	-1.211281e-15	5.278775e-16	0.105999	0.040413
V22	0.144059	-4.290944e-16	1.526333e-16	-1.133902e-15	-6.276051e-17	1.253751e-16	-4.705235e-19	-8.898922e-16	2.026927e-16	-7.071869e-16	...	3.649908e-15	1.000000e+00	-7.303916e-17	9.970809e-17	-5.018575e-16	-2.503187e-17	8.461337e-17	-6.627203e-16	-0.064801	0.000805
V23	0.051142	6.168652e-16	1.634231e-16	-4.983035e-16	9.164206e-17	-8.428683e-18	1.046712e-16	-4.387401e-16	6.377260e-17	-5.214137e-16	...	8.119580e-16	-7.303916e-17	1.000000e+00	2.130519e-17	-8.232727e-17	1.114524e-15	2.839721e-16	1.481903e-15	-0.112633	-0.002685
V24	-0.016182	-4.425156e-17	1.247925e-17	2.686834e-19	1.584638e-16	-1.149255e-15	-1.071589e-15	7.434913e-18	-1.047097e-16	-1.430343e-16	...	1.761054e-16	9.970809e-17	2.130519e-17	1.000000e+00	1.015391e-15	1.343722e-16	-2.274142e-16	-2.819805e-16	0.005146	-0.007221
V25	-0.233083	-9.605737e-16	-4.478846e-16	-1.104734e-15	6.070716e-16	4.808532e-16	4.562861e-16	-3.094082e-16	-4.653279e-16	6.757763e-16	...	-1.686082e-16	-5.018575e-16	-8.232727e-17	1.015391e-15	1.000000e+00	2.646517e-15	-6.406679e-16	-7.008939e-16	-0.047837	0.003308
V26	-0.041407	-1.581290e-17	2.057310e-16	-1.238062e-16	-4.247268e-16	4.319541e-16	-1.357067e-16	-9.657637e-16	-1.727276e-16	-7.888853e-16	...	-5.557329e-16	-2.503187e-17	1.114524e-15	1.343722e-16	2.646517e-15	1.000000e+00	-3.667715e-16	-2.782204e-16	-0.003208	0.004455
V27	-0.005135	1.198124e-16	-4.966953e-16	1.045747e-15	3.977061e-17	6.590482e-16	-4.452461e-16	-1.782106e-15	1.299943e-16	-6.709655e-17	...	-1.211281e-15	8.461337e-17	2.839721e-16	-2.274142e-16	-6.406679e-16	-3.667715e-16	1.000000e+00	-3.061287e-16	0.028825	0.017580
V28	-0.009413	2.083082e-15	-5.093836e-16	9.775546e-16	-2.761403e-18	-5.613951e-18	2.594754e-16	-2.776530e-16	-6.200930e-16	1.110541e-15	...	5.278775e-16	-6.627203e-16	1.481903e-15	-2.819805e-16	-7.008939e-16	-2.782204e-16	-3.061287e-16	1.000000e+00	0.010258	0.009536
Amount	-0.010596	-2.277087e-01	-5.314089e-01	-2.108805e-01	9.873167e-02	-3.863563e-01	2.159812e-01	3.973113e-01	-1.030791e-01	-4.424560e-02	...	1.059989e-01	-6.480065e-02	-1.126326e-01	5.146217e-03	-4.783686e-02	-3.208037e-03	2.882546e-02	1.025822e-02	1.000000	0.005632
Class	-0.012323	-1.013473e-01	9.128865e-02	-1.929608e-01	1.334475e-01	-9.497430e-02	-4.364316e-02	-1.872566e-01	1.987512e-02	-9.773269e-02	...	4.041338e-02	8.053175e-04	-2.685156e-03	-7.220907e-03	3.307706e-03	4.455398e-03	1.757973e-02	9.536041e-03	0.005632	1.000000
31 rows × 31 columns

plt.figure(figsize=(18,14))
sns.heatmap(corr,cmap="coolwarm",annot=True)
plt.show()

From the Correlation graph we can see that Time is strongly correlated with V3 and amount is stronlgy correlated with V2 and V5.

Machine Learning Model
x = df.drop(labels='Class', axis=1)
y = df['Class']
Breaking the dataset into Training and Testing data.

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest=train_test_split(x,y,random_state=42, test_size=0.3, shuffle=True)
Applying Machine Learning Model.

Logistic Regression
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
# Training the algorithm
lr_model.fit(xtrain, ytrain)
LogisticRegression()
# Predictions on training and testing data
lr_pred_train = lr_model.predict(xtrain)
lr_pred_test = lr_model.predict(xtest)
# Importing the required metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
Confusion matrix

tn, fp, fn, tp = confusion_matrix(ytest, lr_pred_test).ravel()
conf_matrix = pd.DataFrame(
    {
        'Predicted Fraud': [tp, fp],
        'Predicted Not Fraud': [fn, tn]
    }, index=['Fraud', 'Not Fraud'])
conf_matrix
Predicted Fraud	Predicted Not Fraud
Fraud	93	43
Not Fraud	52	85255
plt.figure(figsize=(10,6))
sns.heatmap(conf_matrix, annot=True)
<AxesSubplot:>

Heatmap also suggests that the data is highly imbalanced.

Accuracy score

lr_accuracy = accuracy_score(ytest, lr_pred_test)
lr_accuracy
0.9988881476539916
We can see here that accuracy is great. Around 99%. But We know that the dataset is highly unbalanced and accuracy takes into account the whole confusion matrix. So we can say that this measure is not suitable.

precision

lr_precision = precision_score(ytest, lr_pred_test)
lr_precision
0.6413793103448275
Recall

lr_recall = recall_score(ytest, lr_pred_test)
lr_recall
0.6838235294117647
Recall is very low in case of logistic regression. However, we may try to increase it by increasing the complexity of the model.

Although, let's check the recall for training dataset to get the idea of any overfitting we may be having.

lr_recall_train = recall_score(ytrain, lr_pred_train)
lr_recall_train
0.6882022471910112
We can see that the delta is small, only around 0.03. So, we can say that the model is not overfitting.

f1_score

from sklearn.metrics import f1_score
lr_f1 = f1_score(ytest, lr_pred_test)
lr_f1
0.6619217081850534
Classification Report

from sklearn.metrics import classification_report
print(classification_report(ytest, lr_pred_test))
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85307
           1       0.64      0.68      0.66       136

    accuracy                           1.00     85443
   macro avg       0.82      0.84      0.83     85443
weighted avg       1.00      1.00      1.00     85443

ROC Curve

For the ROC Curve, we need the probabilites of Fraud happening (which is the probability of occurance of 1)

lr_pred_test_prob = lr_model.predict_proba(xtest)[:, 1]
lr_pred_test_prob 
array([1.00000000e+00, 9.52091630e-05, 5.60554277e-03, ...,
       1.46778949e-03, 2.45075224e-05, 1.02963468e-04])
To draw the ROC Curve, we need to have True Positive Rate and False Positive Rate.

from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, threshold = roc_curve(ytest, lr_pred_test_prob)
Also, let's get the auc score.

lr_auc = roc_auc_score(ytest, lr_pred_test_prob)
lr_auc
0.9324629590427376
Function to plot the ROC Curve.

def plot_roc_curve(fpr, tpr, label=None):
    plt.figure(figsize=(8, 6))
    plt.title('ROC Curve', fontsize=15)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.xticks(np.arange(0, 1, 0.05), rotation=90)
    plt.xlabel('False Positive Rates', fontsize=15)
    plt.ylabel('True Positive Rates', fontsize=15)
    plt.legend(loc='best')
    
    plt.show()
Plotting ROC Curve

plot_roc_curve(fpr=fpr, tpr=tpr, label="AUC = %.3f" % lr_auc)

AUC is quite good. i.e. 0.932. Based on the data being highly imbalanced, we'll only check the AUC metric in later algorithms.

Naive Bayes Algorithm
from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()
nb_model.fit(xtrain,ytrain)
GaussianNB()
# Predictions on training and testing data
nb_pred_train = nb_model.predict(xtrain)
nb_pred_test = nb_model.predict(xtest)
# Importing the required metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
tn, fp, fn, tp = confusion_matrix(ytest, nb_pred_test).ravel()
conf_matrix1 = pd.DataFrame(
    {
        'Predicted Fraud': [tp, fp],
        'Predicted Not Fraud': [fn, tn]
    }, index=['Fraud', 'Not Fraud'])
conf_matrix1
Predicted Fraud	Predicted Not Fraud
Fraud	90	46
Not Fraud	548	84759
plt.figure(figsize=(10,6))
sns.heatmap(conf_matrix1, annot=True)
<AxesSubplot:>

Accuracy

nb_accuracy = accuracy_score(ytest, nb_pred_test)
nb_accuracy
0.9930479969102208
We can see here that accuracy is great. Around 99%. But We know that the dataset is highly unbalanced and accuracy takes into account the whole confusion matrix. So we can say that this measure is not suitable.

Precision

nb_precision = precision_score(ytest, nb_pred_test)
nb_precision
0.14106583072100312
Recall

nb_recall = recall_score(ytest, nb_pred_test)
nb_recall
0.6617647058823529
Recall is very low in case of Naive Bayes Algorithm.

f1_score

from sklearn.metrics import f1_score
nb_f1 = f1_score(ytest, nb_pred_test)
nb_f1
0.2325581395348837
f1_score is also very low in case of Naive Bayes Algorithm.

from sklearn.metrics import classification_report
print(classification_report(ytest, nb_pred_test))
              precision    recall  f1-score   support

           0       1.00      0.99      1.00     85307
           1       0.14      0.66      0.23       136

    accuracy                           0.99     85443
   macro avg       0.57      0.83      0.61     85443
weighted avg       1.00      0.99      1.00     85443

From the classification report we can see that the Naive Bayes Algorithm model is not enoungh good for the dataset.
ROC Curve

nb_pred_test_prob = nb_model.predict_proba(xtest)[:, 1]
nb_pred_test_prob
array([1.00000000e+00, 6.07470313e-13, 5.64185401e-13, ...,
       6.01387224e-13, 3.67382604e-13, 4.16619083e-13])
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, threshold = roc_curve(ytest, nb_pred_test_prob)
nb_auc = roc_auc_score(ytest, nb_pred_test)
nb_auc
0.82767042426006
Function to plot the ROC Curve.

def plot_roc_curve(fpr, tpr, label=None):
    plt.figure(figsize=(8, 6))
    plt.title('ROC Curve', fontsize=15)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.xticks(np.arange(0, 1, 0.05), rotation=90)
    plt.xlabel('False Positive Rates', fontsize=15)
    plt.ylabel('True Positive Rates', fontsize=15)
    plt.legend(loc='best')
    
    plt.show()
Plotting ROC Curve.

fpr, tpr, threshold = roc_curve(ytest, nb_pred_test)
plot_roc_curve(fpr, tpr, 'AUC = %.3f' % nb_auc)

Conclusion: Based upon all the parameters of two machine learning model we can conclude that Logistic Regression performs better than Naive Bayes Algorithm. Logistic Regression model fits good for the dataset.

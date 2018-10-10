
# coding: utf-8

# # Home Credit Group
# Home Credit B.V. is an international non-bank financial institution founded in 1997 in the Czech Republic.The company operates in 14 countries and focuses on lending primarily to people with little or no credit history. As of 2016 the company has over 15 million active customers, with two-thirds of them in Asia and 7.3 million in China. Major shareholder of company is PPF, a privately held international financial and investment group, which controls an 88.62% stake.
#
# In 1999, Home Credit a.s. was founded in the Czech Republic and in 1999 company expanded to Slovakia. In 2000s company started to expand to Commonwealth of Independent States countries - Russia, Kazakhstan, Ukraine and Belarus.As of 2007 the company was the second largest consumer lender in Russia.In 2010s company expanded to Asia - China, India, Indonesia, Philippines and Vietnam.In 2010 the company was first foreign company to set up as a consumer finance lender in China. In 2015 company launched its operations in the United States of America through a partnership with Sprint Corporation.
#
# Home Credit Group Loans:
#
# Many people struggle to get loans due to insufficient or non-existent credit histories. And, unfortunately, this population is often taken advantage of by untrustworthy lenders.
#
# Home Credit strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. In order to make sure this underserved population has a positive loan experience, Home Credit makes use of a variety of alternative data--including telco and transactional information--to predict their clients' repayment abilities.
#
# While Home Credit is currently using various statistical and machine learning methods to make these predictions, they're challenging Kagglers to help them unlock the full potential of their data. Doing so will ensure that clients capable of repayment are not rejected and that loans are given with a principal, maturity, and repayment calendar that will empower their clients to be successful.

# In[1]:

from IPython.display import YouTubeVideo
# YouTubeVideo('QOO14BQITz4', width=700, height=400)


# In[2]:

# IMPORTING LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
from PIL import  Image
# get_ipython().magic('matplotlib inline')
import pandas as pd
import seaborn as sns
import itertools
import warnings
warnings.filterwarnings("ignore")
from wordcloud import WordCloud,STOPWORDS
import io
import base64
from matplotlib import rc,animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.basemap import Basemap
import folium
import folium.plugins
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import os
print(os.listdir("Data/"))


# # Importing Data

# In[3]:

application_train     = pd.read_csv(r"Data/application_train.csv")
application_test      = pd.read_csv(r"Data/application_test.csv")
bureau                = pd.read_csv(r"Data/bureau.csv")
bureau_balance        = pd.read_csv(r"Data/bureau_balance.csv")
credit_card_balance   = pd.read_csv(r"Data/credit_card_balance.csv")
installments_payments = pd.read_csv(r"Data/installments_payments.csv")
pos_cash_balance      = pd.read_csv(r"Data/POS_CASH_balance.csv")
previous_application  = pd.read_csv(r"Data/previous_application.csv")


# # Data Dimensions

# In[4]:

print ("application_train     :",application_train.shape)
print ("application_test      :",application_test.shape)
print ("bureau                :",bureau.shape)
print ("bureau_balance        :",bureau_balance.shape)
print ("credit_card_balance   :",credit_card_balance.shape)
print ("installments_payments :",installments_payments.shape)
print ("pos_cash_balance      :",pos_cash_balance.shape)
print ("previous_application  :",previous_application.shape)


# # First Few rows of Data

# In[5]:

# from IPython.core.display import display


# In[6]:

# display("application_train")
# display(application_train.head(3))
# display("application_test")
# display(application_test.head(3))
# display("bureau")
# display(bureau.head(3))
# display("bureau_balance")
# display(bureau_balance.head(3))
# display("credit_card_balance")
# display(credit_card_balance.head(3))
# display("installments_payments")
# display(installments_payments.head(3))
# display("pos_cash_balance")
# display(pos_cash_balance.head(3))
# display("previous_application")
# display(previous_application.head(3))


# # Data Sequence

# In[7]:

image = np.array(Image.open(r"Data/home_credit.png"))
fig = plt.figure(figsize=(15,8))
plt.imshow(image,interpolation="bilinear")
plt.axis("off")
fig.set_facecolor("lightgrey")
plt.title("Data Sequence")
# plt.show()
plt.savefig('Output/homeCredit.png')

# # Percentage of Missing values in application train and test data

# In[8]:

fig = plt.figure(figsize=(18,6))
miss_train = pd.DataFrame((application_train.isnull().sum())*100/application_train.shape[0]).reset_index()
miss_test = pd.DataFrame((application_test.isnull().sum())*100/application_test.shape[0]).reset_index()
miss_train["type"] = "train"
miss_test["type"]  =  "test"
missing = pd.concat([miss_train,miss_test],axis=0)
ax = sns.pointplot("index",0,data=missing,hue="type")
plt.xticks(rotation =90,fontsize =7)
plt.title("Percentage of Missing values in application train and test data")
plt.ylabel("PERCENTAGE")
plt.xlabel("COLUMNS")
ax.set_facecolor("k")
fig.set_facecolor("lightgrey")
plt.savefig('Output/PercentageofNAsInAppTrainandTestData.png')


# # Percentage of missing values in other data sets

# In[9]:

sns.set()


# In[10]:

plt.figure(figsize=(15,20))

plt.subplot(231)
sns.heatmap(pd.DataFrame(bureau.isnull().sum()/bureau.shape[0]*100),annot=True,
            linewidth=1,linecolor="white") # cmap=sns.color_palette("cool"),
plt.title("bureau")

plt.subplot(232)
sns.heatmap(pd.DataFrame(bureau_balance.isnull().sum()/bureau_balance.shape[0]*100),annot=True,
            linewidth=1,linecolor="white") # cmap=sns.color_palette("cool"),
plt.title("bureau_balance")

plt.subplot(233)
sns.heatmap(pd.DataFrame(credit_card_balance.isnull().sum()/credit_card_balance.shape[0]*100),annot=True,
            linewidth=1,linecolor="white") # cmap=sns.color_palette("cool"),
plt.title("credit_card_balance")

plt.subplot(234)
sns.heatmap(pd.DataFrame(installments_payments.isnull().sum()/installments_payments.shape[0]*100),annot=True,
            linewidth=1,linecolor="white") # cmap=sns.color_palette("cool"),
plt.title("installments_payments")

plt.subplot(235)
sns.heatmap(pd.DataFrame(pos_cash_balance.isnull().sum()/pos_cash_balance.shape[0]*100),annot=True,
            linewidth=1,linecolor="white") # cmap=sns.color_palette("cool"),
plt.title("pos_cash_balance")

plt.subplot(236)
sns.heatmap(pd.DataFrame(previous_application.isnull().sum()/previous_application.shape[0]*100),annot=True,
            linewidth=1,linecolor="white") # cmap=sns.color_palette("cool"),
plt.title("previous_application")

plt.subplots_adjust(wspace = 1.6)
plt.savefig('Output/bureau.png')

# # Distribution of Target variable
# >TARGET :Target variable (1 - client with payment difficulties: he/she had late payment more than X days on at least one of the first Y installments of the loan in sample, 0 - all other cases)
# >8% out  of total client  population  have difficulties in repaying loans.

# In[11]:

plt.figure(figsize=(14,7))
plt.subplot(121)
application_train["TARGET"].value_counts().plot.pie(autopct = "%1.0f%%",colors = sns.color_palette("prism",7),startangle = 60,labels=["repayer","defaulter"],
                                                                        wedgeprops={"linewidth":2,"edgecolor":"k"},explode=[.1,0],shadow =True)
plt.title("Distribution of target variable")

plt.subplot(122)
ax = application_train["TARGET"].value_counts().plot(kind="barh")

for i,j in enumerate(application_train["TARGET"].value_counts().values):
    ax.text(.7,i,j,weight = "bold",fontsize=20)

plt.title("Count of target variable")
# plt.show()
plt.savefig('Output/CountofTargetVariable.png')


# In[12]:

#Concatenating train and test data
application_train_x = application_train[[x for x in application_train.columns if x not in ["TARGET"]]]
application_train_x["type"] = "train"
application_test["type"]    = "test"
data = pd.concat([application_train_x,application_test],axis=0)


# # Distribution in Contract types in training and test data
# >NAME_CONTRACT_TYPE :	Identification if loan is cash or revolving
# >In training data the percentage of revolving loans and cash loans are 10% & 90%.
# >In test data the percentage of revolving loans and cash loans are 1% & 99%.

# In[13]:

plt.figure(figsize=(14,7))
plt.subplot(121)
data[data["type"] == "train"]["NAME_CONTRACT_TYPE"].value_counts().plot.pie(autopct = "%1.0f%%",colors = ["grey","orange"],startangle = 60,
                                                                        wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)
circ = plt.Circle((0,0),.7,color="white")
plt.gca().add_artist(circ)
plt.title("distribution of contract types in train data")

plt.subplot(122)
data[data["type"] == "test"]["NAME_CONTRACT_TYPE"].value_counts().plot.pie(autopct = "%1.0f%%",colors = ["grey","orange"],startangle = 60,
                                                                        wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)
circ = plt.Circle((0,0),.7,color="white")
plt.gca().add_artist(circ)
plt.ylabel("")
plt.title("distribution of contract types in test data")
# plt.show()
plt.savefig('Output/DistributionofContractTypesinTestData.png')


# # Gender Distribution  in training and test data
# >Train data - Male : 66% ,FEMALE : 34%    , Test data - Male : 67% ,FEMALE : 33%

# In[14]:

fig = plt.figure(figsize=(13,6))
plt.subplot(121)
data[data["type"] == "train"]["CODE_GENDER"].value_counts().plot.pie(autopct = "%1.0f%%",colors = ["white","r"],startangle = 60,
                                                                        wedgeprops={"linewidth":2,"edgecolor":"k"},explode=[.05,0,0],shadow =True)
plt.title("distribution of gender in train data")

plt.subplot(122)
data[data["type"] == "test"]["CODE_GENDER"].value_counts().plot.pie(autopct = "%1.0f%%",colors = ["white","r"],startangle = 60,
                                                                        wedgeprops={"linewidth":2,"edgecolor":"k"},explode=[.05,0],shadow =True)
plt.ylabel("")
plt.title("distribution of gender in test data")
fig.set_facecolor("lightgrey")
plt.savefig('Output/DistributionofGenderinTrainData.png')

# # Distribution of Contract type by gender

# In[15]:

fig  = plt.figure(figsize=(13,6))
plt.subplot(121)
ax = sns.countplot("NAME_CONTRACT_TYPE",hue="CODE_GENDER",data=data[data["type"] == "train"],palette=["r","b","g"])
ax.set_facecolor("lightgrey")
ax.set_title("Distribution of Contract type by gender -train data")

plt.subplot(122)
ax1 = sns.countplot("NAME_CONTRACT_TYPE",hue="CODE_GENDER",data=data[data["type"] == "test"],palette=["b","r"])
ax1.set_facecolor("lightgrey")
ax1.set_title("Distribution of Contract type by gender -test data")
# plt.show()
plt.savefig('Output/DistributionofContractTypebyGenderTrainData.png')


# # Distribution of client owning a car and by gender
# >FLAG_OWN_CAR	Flag if the client owns a car .
# >SUBPLOT 1 : Distribution of client owning a car . 34% of clients own a car .
# >SUBPLOT 1 : Distribution of client owning a car by gender .  Out of total clients who own car 57% are male and 43% are female.

# In[16]:

fig = plt.figure(figsize=(13,6))

plt.subplot(121)
data["FLAG_OWN_CAR"].value_counts().plot.pie(autopct = "%1.0f%%",colors = ["gold","orangered"],startangle = 60,
                                                                        wedgeprops={"linewidth":2,"edgecolor":"k"},explode=[.05,0],shadow =True)
plt.title("distribution of client owning a car")

plt.subplot(122)
data[data["FLAG_OWN_CAR"] == "Y"]["CODE_GENDER"].value_counts().plot.pie(autopct = "%1.0f%%",colors = ["b","orangered"],startangle = 90,
                                                                        wedgeprops={"linewidth":2,"edgecolor":"k"},explode=[.05,0,0],shadow =True)
plt.title("distribution of client owning a car by gender")
plt.savefig('Output/DistributionofClientOwningCarGender.png')
#plt.show()


# # Distribution of client owning a house or flat and by gender
# >FLAG_OWN_REALTY - Flag if client owns a house or flat
# >SUBPLOT 1 : Distribution of client owning a house or flat  . 69% of clients own a flat or house .
# >SUBPLOT 1 : Distribution of client owning a house or flat by gender .  Out of total clients who own house 67% are female and 33% are male.

# In[17]:

plt.figure(figsize=(13,6))
plt.subplot(121)
data["FLAG_OWN_REALTY"].value_counts().plot.pie(autopct = "%1.0f%%",colors = ["skyblue","gold"],startangle = 90,
                                              wedgeprops={"linewidth":2,"edgecolor":"k"},explode=[0.05,0],shadow =True)
plt.title("Distribution of client owns a house or flat")

plt.subplot(122)
data[data["FLAG_OWN_REALTY"] == "Y"]["CODE_GENDER"].value_counts().plot.pie(autopct = "%1.0f%%",colors = ["orangered","b"],startangle = 90,
                                                                        wedgeprops={"linewidth":2,"edgecolor":"k"},explode=[.05,0,0],shadow =True)
plt.title("Distribution of client owning a house or flat by gender")
# plt.show()
plt.savefig('Output/DistributionofClientOwnHouseorFlat.png')


# # Distribution of Number of children and family members of client by repayment status.
# >CNT_CHILDREN - Number of children the client has.
# >CNT_FAM_MEMBERS	- How many family members does client have.

# In[18]:

fig = plt.figure(figsize=(12,10))
plt.subplot(211)
sns.countplot(application_train["CNT_CHILDREN"],palette="Set1",hue=application_train["TARGET"])
plt.legend(loc="upper center")
plt.title(" Distribution of Number of children client has  by repayment status")
plt.subplot(212)
sns.countplot(application_train["CNT_FAM_MEMBERS"],palette="Set1",hue=application_train["TARGET"])
plt.legend(loc="upper center")
plt.title(" Distribution of Number of family members client has  by repayment status")
fig.set_facecolor("lightblue")
plt.savefig('Output/DistributionNumbChildrenbyRepaymentStatus.png')

# ## Distribution of contract type ,gender ,own car ,own house with respect to Repayment status(Target variable)
# >Percentage of males is 10%  more in defaults than non defaulters.

# In[19]:

default = application_train[application_train["TARGET"]==1][[ 'NAME_CONTRACT_TYPE', 'CODE_GENDER','FLAG_OWN_CAR', 'FLAG_OWN_REALTY']]
non_default = application_train[application_train["TARGET"]==0][[ 'NAME_CONTRACT_TYPE', 'CODE_GENDER','FLAG_OWN_CAR', 'FLAG_OWN_REALTY']]

d_cols = ['NAME_CONTRACT_TYPE', 'CODE_GENDER','FLAG_OWN_CAR', 'FLAG_OWN_REALTY']
d_length = len(d_cols)

fig = plt.figure(figsize=(16,4))
for i,j in itertools.zip_longest(d_cols,range(d_length)):
    plt.subplot(1,4,j+1)
    default[i].value_counts().plot.pie(autopct = "%1.0f%%",colors = sns.color_palette("prism"),startangle = 90,
                                        wedgeprops={"linewidth":1,"edgecolor":"white"},shadow =True)
    circ = plt.Circle((0,0),.7,color="white")
    plt.gca().add_artist(circ)
    plt.ylabel("")
    plt.title(i+"-Defaulter")
plt.savefig('Output/DistributionofContractTypewrtTargetVar1.png')

fig = plt.figure(figsize=(16,4))
for i,j in itertools.zip_longest(d_cols,range(d_length)):
    plt.subplot(1,4,j+1)
    non_default[i].value_counts().plot.pie(autopct = "%1.0f%%",colors = sns.color_palette("prism",3),startangle = 90,
                                           wedgeprops={"linewidth":1,"edgecolor":"white"},shadow =True)
    circ = plt.Circle((0,0),.7,color="white")
    plt.gca().add_artist(circ)
    plt.ylabel("")
    plt.title(i+"-Repayer")
plt.savefig('Output/DistributionofContractTypewrtTargetVar2.png')

# # Distribution of amount data
# >AMT_INCOME_TOTAL - Income of the client
# >AMT_CREDIT                - Credit amount of the loan
# >AMT_ANNUITY             - Loan annuity
# >AMT_GOODS_PRICE   - For consumer loans it is the price of the goods for which the loan is given

# In[20]:

cols = [ 'AMT_INCOME_TOTAL', 'AMT_CREDIT','AMT_ANNUITY', 'AMT_GOODS_PRICE']
length = len(cols)
cs = ["r","b","g","k"]

ax = plt.figure(figsize=(12,12))
ax.set_facecolor("lightgrey")
for i,j,k in itertools.zip_longest(cols,range(length),cs):
    plt.subplot(2,2,j+1)
    sns.distplot(data[data[i].notnull()][i],color=k)
    plt.axvline(data[i].mean(),label = "mean",linestyle="dashed",color="k")
    plt.legend(loc="best")
    plt.title(i)
    plt.subplots_adjust(hspace = .2)
plt.savefig('Output/DistributionofAmountofData.png')

# # Comparing summary statistics between defaulters and non - defaulters for loan amounts .
# >Income of client  -
# 1  . average income of clients who default and who do not are almost same.
# 2 . standard deviation in income of client who default is very high compared to who do not default.
# 3 . clients who default also has maximum income earnings
#
# >Credit amount of the loan ,Loan annuity,Amount goods price -
# 1 .statistics between  credit amounts,Loan annuity and Amount goods price given to cilents who default and who dont are almost similar.

# In[21]:

cols = [ 'AMT_INCOME_TOTAL', 'AMT_CREDIT','AMT_ANNUITY', 'AMT_GOODS_PRICE']

df = application_train.groupby("TARGET")[cols].describe().transpose().reset_index()
df = df[df["level_1"].isin([ 'mean', 'std', 'min', 'max'])]
df_x = df[["level_0","level_1",0]]
df_y = df[["level_0","level_1",1]]
df_x = df_x.rename(columns={'level_0':"amount_type", 'level_1':"statistic", 0:"amount"})
df_x["type"] = "REPAYER"
df_y = df_y.rename(columns={'level_0':"amount_type", 'level_1':"statistic", 1:"amount"})
df_y["type"] = "DEFAULTER"
df_new = pd.concat([df_x,df_y],axis = 0)

stat = df_new["statistic"].unique().tolist()
length = len(stat)

plt.figure(figsize=(13,15))

for i,j in itertools.zip_longest(stat,range(length)):
    plt.subplot(2,2,j+1)
    fig = sns.barplot(df_new[df_new["statistic"] == i]["amount_type"],df_new[df_new["statistic"] == i]["amount"],
                hue=df_new[df_new["statistic"] == i]["type"],palette=["g","r"])
    plt.title(i + "--Defaulters vs Non defaulters")
    plt.subplots_adjust(hspace = .4)
    fig.set_facecolor("lightgrey")
plt.savefig('Output/CreditAmountDefaultvsNonDefaulter.png')

# # Average Income,credit,annuity & goods_price by gender

# In[22]:

cols = [ 'AMT_INCOME_TOTAL', 'AMT_CREDIT','AMT_ANNUITY', 'AMT_GOODS_PRICE']

df1 = data.groupby("CODE_GENDER")[cols].mean().transpose().reset_index()

df_f   = df1[["index","F"]]
df_f   = df_f.rename(columns={'index':"amt_type", 'F':"amount"})
df_f["gender"] = "FEMALE"
df_m   = df1[["index","M"]]
df_m   = df_m.rename(columns={'index':"amt_type", 'M':"amount"})
df_m["gender"] = "MALE"
df_xna = df1[["index","XNA"]]
df_xna = df_xna.rename(columns={'index':"amt_type", 'XNA':"amount"})
df_xna["gender"] = "XNA"

df_gen = pd.concat([df_m,df_f,df_xna],axis=0)

plt.figure(figsize=(12,5))
ax = sns.barplot("amt_type","amount",data=df_gen,hue="gender",palette="Set1")
plt.title("Average Income,credit,annuity & goods_price by gender")
# plt.show()
plt.savefig('Output/AverageIncomebyGender.png')

# # Scatter plot between credit amount and annuity amount

# In[23]:

fig = plt.figure(figsize=(10,8))
plt.scatter(application_train[application_train["TARGET"]==0]['AMT_ANNUITY'],application_train[application_train["TARGET"]==0]['AMT_CREDIT'],s=35,
            color="b",alpha=.5,label="REPAYER",linewidth=.5,edgecolor="k")
plt.scatter(application_train[application_train["TARGET"]==1]['AMT_ANNUITY'],application_train[application_train["TARGET"]==1]['AMT_CREDIT'],s=35,
            color="r",alpha=.2,label="DEFAULTER",linewidth=.5,edgecolor="k")
plt.legend(loc="best",prop={"size":15})
plt.xlabel("AMT_ANNUITY")
plt.ylabel("AMT_CREDIT")
plt.title("Scatter plot between credit amount and annuity amount")
#plt.show()
plt.savefig('Output/CreditAmountvsAnnuityAmount.png')


# # Pair Plot between amount variables
# >AMT_INCOME_TOTAL - Income of the client
# >AMT_CREDIT                - Credit amount of the loan
# >AMT_ANNUITY             - Loan annuity
# >AMT_GOODS_PRICE   - For consumer loans it is the price of the goods for which the loan is given

# In[24]:

amt = application_train[[ 'AMT_INCOME_TOTAL','AMT_CREDIT',
                         'AMT_ANNUITY', 'AMT_GOODS_PRICE',"TARGET"]]
amt = amt[(amt["AMT_GOODS_PRICE"].notnull()) & (amt["AMT_ANNUITY"].notnull())]
sns.pairplot(amt,hue="TARGET",palette=["b","r"])
# plt.show()
plt.savefig('Output/PairPlotAmountVariables.png')


# # Distribution of Suite type
# >NAME_TYPE_SUITE - Who was accompanying client when he was applying for the loan.

# In[25]:

plt.figure(figsize=(12,5))
plt.subplot(121)
sns.countplot(y=data["NAME_TYPE_SUITE"],
              palette="Set2",
              order=data["NAME_TYPE_SUITE"].value_counts().index[:5])
plt.title("Distribution of Suite type")

plt.subplot(122)
sns.countplot(y=data["NAME_TYPE_SUITE"],
              hue=data["CODE_GENDER"],palette="Set2",
              order=data["NAME_TYPE_SUITE"].value_counts().index[:5])
plt.ylabel("")
plt.title("Distribution of Suite type by gender")
plt.subplots_adjust(wspace = .4)
plt.savefig('Output/DistributionofSuitebyGender.png')

# # Distribution of client income type
# >NAME_INCOME_TYPE	Clients income type (businessman, working, maternity leave,…)

# In[26]:

plt.figure(figsize=(12,5))
plt.subplot(121)
sns.countplot(y=data["NAME_INCOME_TYPE"],
              palette="Set2",
              order=data["NAME_INCOME_TYPE"].value_counts().index[:4])
plt.title("Distribution of client income type")

plt.subplot(122)
sns.countplot(y=data["NAME_INCOME_TYPE"],
              hue=data["CODE_GENDER"],
              palette="Set2",
              order=data["NAME_INCOME_TYPE"].value_counts().index[:4])
plt.ylabel("")
plt.title("Distribution of client income  type by gender")
plt.subplots_adjust(wspace = .4)
plt.savefig('Output/DistributionofClientIncomebyGender.png')

# # Distribution of Education type by loan repayment status
# >NAME_EDUCATION_TYPE	Level of highest education the client achieved..
# >Clients who default have proportionally 9% less higher education compared to clients who do not default.

# In[27]:

plt.figure(figsize=(16,8))
plt.subplot(121)
application_train[application_train["TARGET"]==0]["NAME_EDUCATION_TYPE"].value_counts().plot.pie(fontsize=9,autopct = "%1.0f%%",
                                                                                                 colors = sns.color_palette("Set1"),
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)
circ = plt.Circle((0,0),.7,color="white")
plt.gca().add_artist(circ)
plt.title("Distribution of Education type for Repayers",color="b")

plt.subplot(122)
application_train[application_train["TARGET"]==1]["NAME_EDUCATION_TYPE"].value_counts().plot.pie(fontsize=9,autopct = "%1.0f%%",
                                                                                                 colors = sns.color_palette("Set1"),
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)
circ = plt.Circle((0,0),.7,color="white")
plt.gca().add_artist(circ)
plt.title("Distribution of Education type for Defaulters",color="b")
plt.ylabel("")
#plt.show()
plt.savefig('Output/DistributionofEducationbyLoanRepayments.png')

# # Average Earnings by different professions and education types

# In[28]:

edu = data.groupby(['NAME_EDUCATION_TYPE','NAME_INCOME_TYPE'])['AMT_INCOME_TOTAL'].mean().reset_index().sort_values(by='AMT_INCOME_TOTAL',ascending=False)
fig = plt.figure(figsize=(13,7))
ax = sns.barplot('NAME_INCOME_TYPE','AMT_INCOME_TOTAL',data=edu,hue='NAME_EDUCATION_TYPE',palette="seismic")
ax.set_facecolor("k")
plt.title(" Average Earnings by different professions and education types")
#plt.show()
plt.savefig('Output/AverageEarningsbyProfessionandEducation.png')


# # Distribution of Education type by loan repayment status
# >NAME_FAMILY_STATUS - Family status of the client
# >Percentage of single people are more in defaulters than non defaulters.

# In[29]:

plt.figure(figsize=(16,8))
plt.subplot(121)
application_train[application_train["TARGET"]==0]["NAME_FAMILY_STATUS"].value_counts().plot.pie(autopct = "%1.0f%%",
                                                             startangle=120,colors = sns.color_palette("Set2",7),
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True,explode=[0,.07,0,0,0,0])

plt.title("Distribution of Family status for Repayers",color="b")

plt.subplot(122)
application_train[application_train["TARGET"]==1]["NAME_FAMILY_STATUS"].value_counts().plot.pie(autopct = "%1.0f%%",
                                                    startangle=120,colors = sns.color_palette("Set2",7),
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True,explode=[0,.07,0,0,0])


plt.title("Distribution of Family status for Defaulters",color="b")
plt.ylabel("")
#plt.show()
plt.savefig('Output/DistributionEducationbyLoanRepaymentStatus.png')


# # Distribution of Housing type by loan repayment status
# >NAME_HOUSING_TYPE - What is the housing situation of the client (renting, living with parents, ...)

# In[30]:

plt.figure(figsize=(16,8))
plt.subplot(121)
application_train[application_train["TARGET"]==0]["NAME_HOUSING_TYPE"].value_counts().plot.pie(autopct = "%1.0f%%",fontsize=8,
                                                             colors = sns.color_palette("Spectral"),
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)

plt.title("Distribution of housing type  for Repayer",color="b")

plt.subplot(122)
application_train[application_train["TARGET"]==1]["NAME_HOUSING_TYPE"].value_counts().plot.pie(autopct = "%1.0f%%",fontsize=8,
                                                    colors = sns.color_palette("Spectral"),
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)


plt.title("Distribution of housing type for Defaulters",color="b")
plt.ylabel("")
#plt.show()
plt.savefig('Output/DistributionofHousingbyLoanRepayment.png')


# # Distribution normalized population of region where client lives by loan repayment status
# >REGION_POPULATION_RELATIVE - Normalized population of region where client lives (higher number means the client lives in more populated region).
# >In High  population density regions people are less likely to default on loans.
#

# In[31]:

fig = plt.figure(figsize=(13,8))

plt.subplot(121)
sns.violinplot(y=application_train[application_train["TARGET"]==0]["REGION_POPULATION_RELATIVE"]
               ,x=application_train[application_train["TARGET"]==0]["NAME_CONTRACT_TYPE"],
               palette="Set1")
plt.title("Distribution of region population for Non Default loans",color="b")
plt.subplot(122)
sns.violinplot(y = application_train[application_train["TARGET"]==1]["REGION_POPULATION_RELATIVE"]
               ,x=application_train[application_train["TARGET"]==1]["NAME_CONTRACT_TYPE"]
               ,palette="Set1")
plt.title("Distribution of region population  for  Default loans",color="b")

plt.subplots_adjust(wspace = .2)
fig.set_facecolor("lightgrey")
plt.savefig('Output/DistributionPopulationRegionbyLoanRepayment.png')

# # Client's age
# >DAYS_BIRTH - Client's age in days at the time of application.
# >average clients age is comparatively less in non repayers than repayers in every aspect.
# >younger people  tend to default more than elder  people.

# In[32]:

fig = plt.figure(figsize=(13,15))

plt.subplot(221)
sns.distplot(application_train[application_train["TARGET"]==0]["DAYS_BIRTH"],color="b")
plt.title("Age Distribution of repayers")

plt.subplot(222)
sns.distplot(application_train[application_train["TARGET"]==1]["DAYS_BIRTH"],color="r")
plt.title("Age Distribution of defaulters")

plt.subplot(223)
sns.lvplot(application_train["TARGET"],application_train["DAYS_BIRTH"],hue=application_train["CODE_GENDER"],palette=["b","grey","m"])
plt.axhline(application_train["DAYS_BIRTH"].mean(),linestyle="dashed",color="k",label ="average age of client")
plt.legend(loc="lower right")
plt.title("Client age vs Loan repayment status(hue=gender)")

plt.subplot(224)
sns.lvplot(application_train["TARGET"],application_train["DAYS_BIRTH"],hue=application_train["NAME_CONTRACT_TYPE"],palette=["r","g"])
plt.axhline(application_train["DAYS_BIRTH"].mean(),linestyle="dashed",color="k",label ="average age of client")
plt.legend(loc="lower right")
plt.title("Client age vs Loan repayment status(hue=contract type)")

plt.subplots_adjust(wspace = .2,hspace = .3)

fig.set_facecolor("lightgrey")
plt.savefig('Output/ClientsAge.png')

# # Distribution of days employed for target variable.
# >DAYS_EMPLOYED - How many days before the applicationfor target variable the person started current employment

# In[33]:

fig = plt.figure(figsize=(13,5))

plt.subplot(121)
sns.distplot(application_train[application_train["TARGET"]==0]["DAYS_EMPLOYED"],color="b")
plt.title("days employed distribution of repayers")

plt.subplot(122)
sns.distplot(application_train[application_train["TARGET"]==1]["DAYS_EMPLOYED"],color="r")
plt.title("days employed distribution of defaulters")

fig.set_facecolor("ghostwhite")
plt.savefig('Output/DistributionofDaysEmployedforTargetVar.png')

# # Distribution of registration days for target variable.
# >DAYS_REGISTRATION	How many days before the application did client change his registration

# In[34]:

fig = plt.figure(figsize=(13,5))

plt.subplot(121)
sns.distplot(application_train[application_train["TARGET"]==0]["DAYS_REGISTRATION"],color="b")
plt.title("registration days distribution of repayers")

plt.subplot(122)
sns.distplot(application_train[application_train["TARGET"]==1]["DAYS_REGISTRATION"],color="r")
plt.title("registration days distribution of defaulter")

fig.set_facecolor("ghostwhite")
plt.savefig('Output/DistributionofRegistrationforTargetVar.png')

# # Distribution of registration days for target variable.
# >OWN_CAR_AGE  - Age of client's car.
# >Mean car age of non repayers is slightly higher than repayers.

# In[35]:

fig = plt.figure(figsize=(15,7))
plt.subplot(121)
sns.violinplot(y = application_train[application_train["OWN_CAR_AGE"].notnull()]["OWN_CAR_AGE"],
               x=application_train[application_train["OWN_CAR_AGE"].notnull()]["TARGET"])
plt.axhline(application_train[(application_train["OWN_CAR_AGE"].notnull())&(application_train["TARGET"] ==0)]["OWN_CAR_AGE"].mean(),color="b",
            linestyle="dashed",label = "average car age of repayers")
plt.axhline(application_train[(application_train["OWN_CAR_AGE"].notnull())&(application_train["TARGET"] ==1)]["OWN_CAR_AGE"].mean(),color="r",
            linestyle="dashed",label = "average car age of defaulters")
plt.legend(loc="best")
plt.title("Distribution of car age by repayment status")

plt.subplot(122)
sns.distplot(application_train[application_train["OWN_CAR_AGE"].notnull()]["OWN_CAR_AGE"],color="k")
plt.title("Distribution of car age")

fig.set_facecolor("lightgrey")
plt.savefig('Output/DistributionofCarAgeforTargetVar.png')

# # Distribution in contact information provided by client
# >FLAG_MOBIL                  - Did client provide mobile phone (1=YES, 0=NO)
# >FLAG_EMP_PHONE       - Did client provide work phone (1=YES, 0=NO)
# >FLAG_WORK_PHONE   - Did client provide home phone (1=YES, 0=NO)
# >FLAG_CONT_MOBILE   - Was mobile phone reachable (1=YES, 0=NO)
# >FLAG_PHONE                - Did client provide home phone (1=YES, 0=NO)
# >FLAG_EMAIL                  - Did client provide email (1=YES, 0=NO)

# In[36]:

x   = application_train[['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',
       'FLAG_PHONE', 'FLAG_EMAIL',"TARGET"]]
x["TARGET"] = x["TARGET"].replace({0:"repayers",1:"defaulters"})
x  = x.replace({1:"YES",0:"NO"})

cols = ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',
       'FLAG_PHONE', 'FLAG_EMAIL']
length = len(cols)

fig = plt.figure(figsize=(15,12))
fig.set_facecolor("lightgrey")

for i,j in itertools.zip_longest(cols,range(length)):
    plt.subplot(2,3,j+1)
    sns.countplot(x[i],hue=x["TARGET"],palette=["r","g"])
    plt.title(i,color="b")
plt.savefig('Output/DistributionofContactInfobyClient.png')

# # Occupation percentage in data with respect to repayment status
# >OCCUPATION_TYPE -	What kind of occupation does the client have.
# >occupations like Cleaning staff ,Cooking staff, Drivers ,Laborers , Low-skill Laborers ,Sales staff ,Security staff are more likely to default in loans.

# In[37]:


fig = plt.figure(figsize=(13,7))
occ = application_train[application_train["TARGET"]==0]["OCCUPATION_TYPE"].value_counts().reset_index()
occ = occ.sort_values(by = "index",ascending=True)
occ1 = application_train[application_train["TARGET"]==1]["OCCUPATION_TYPE"].value_counts().reset_index()
occ1 = occ1.sort_values(by = "index",ascending=True)
occ["percentage"]  = (occ["OCCUPATION_TYPE"]*100/occ["OCCUPATION_TYPE"].sum())
occ1["percentage"] = (occ1["OCCUPATION_TYPE"]*100/occ1["OCCUPATION_TYPE"].sum())
occ["type"]        = "Repayers"
occ1["type"]       = "defaulters"
occupation = pd.concat([occ,occ1],axis=0)

ax = sns.barplot("index","percentage",data=occupation,hue="type",palette=["b","r"])
plt.xticks(rotation = 70)
plt.xlabel("occupation")
ax.set_facecolor("k")
fig.set_facecolor("ghostwhite")
plt.title("Occupation percentage in data with respect to repayment status")
#plt.show()
plt.savefig('Output/OccupationwrtRepayment.png')


# # Distribution of registration days for target variable.
# >REGION_RATING_CLIENT - Home credit rating of the region where client lives (1,2,3).
# >REGION_RATING_CLIENT_W_CITY - Home credit rating of the region where client lives with taking city into account (1,2,3).
# >Percentage of defaulters are less  in 1-rated regions compared to repayers.
# >Percentage of defaulters are more in 3-rated regions compared to repayers.

# In[38]:

fig = plt.figure(figsize=(13,13))
plt.subplot(221)
application_train[application_train["TARGET"]==0]["REGION_RATING_CLIENT"].value_counts().plot.pie(autopct = "%1.0f%%",fontsize=8,
                                                             colors = sns.color_palette("Pastel1"),
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)

plt.title("Distribution of region rating  for Repayers",color="b")

plt.subplot(222)
application_train[application_train["TARGET"]==1]["REGION_RATING_CLIENT"].value_counts().plot.pie(autopct = "%1.0f%%",fontsize=8,
                                                    colors = sns.color_palette("Pastel1"),
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)


plt.title("Distribution of region rating  for Defaulters",color="b")
plt.ylabel("")

plt.subplot(223)
application_train[application_train["TARGET"]==0]["REGION_RATING_CLIENT_W_CITY"].value_counts().plot.pie(autopct = "%1.0f%%",fontsize=8,
                                                             colors = sns.color_palette("Paired"),
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)

plt.title("Distribution of city region rating   for Repayers",color="b")

plt.subplot(224)
application_train[application_train["TARGET"]==1]["REGION_RATING_CLIENT_W_CITY"].value_counts().plot.pie(autopct = "%1.0f%%",fontsize=8,
                                                    colors = sns.color_palette("Paired"),
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)


plt.title("Distribution of city region rating  for Defaulters",color="b")
plt.ylabel("")
fig.set_facecolor("ivory")
plt.savefig('Output/DistributionofRegistrationforTargetVar2.png')

# # Peak days and hours for applying loans (defaulters vs repayers)
# >WEEKDAY_APPR_PROCESS_START - On which day of the week did the client apply for the loan.
# >HOUR_APPR_PROCESS_START    - Approximately at what hour did the client apply for the loan.
# >On tuesdays , percentage of defaulters applying for loans is greater than that of repayers.
# >from morning 4'O clock to 9'O clock percentage of defaulters applying for loans is greater than that of repayers.

# In[39]:

day = application_train.groupby("TARGET").agg({"WEEKDAY_APPR_PROCESS_START":"value_counts"})
day = day.rename(columns={"WEEKDAY_APPR_PROCESS_START":"value_counts"})
day = day.reset_index()
day_0 = day[:7]
day_1 = day[7:]
day_0["percentage"] = day_0["value_counts"]*100/day_0["value_counts"].sum()
day_1["percentage"] = day_1["value_counts"]*100/day_1["value_counts"].sum()
days = pd.concat([day_0,day_1],axis=0)
days["TARGET"] = days.replace({1:"defaulters",0:"repayers"})

fig = plt.figure(figsize=(13,15))
plt.subplot(211)
order = ['SUNDAY', 'MONDAY','TUESDAY', 'WEDNESDAY','THURSDAY', 'FRIDAY', 'SATURDAY']
ax= sns.barplot("WEEKDAY_APPR_PROCESS_START","percentage",data=days,
                hue="TARGET",order=order,palette="prism")
ax.set_facecolor("k")
ax.set_title("Peak days for applying loans (defaulters vs repayers)")

hr = application_train.groupby("TARGET").agg({"HOUR_APPR_PROCESS_START":"value_counts"})
hr = hr.rename(columns={"HOUR_APPR_PROCESS_START":"value_counts"}).reset_index()
hr_0 = hr[hr["TARGET"]==0]
hr_1 = hr[hr["TARGET"]==1]
hr_0["percentage"] = hr_0["value_counts"]*100/hr_0["value_counts"].sum()
hr_1["percentage"] = hr_1["value_counts"]*100/hr_1["value_counts"].sum()
hrs = pd.concat([hr_0,hr_1],axis=0)
hrs["TARGET"] = hrs["TARGET"].replace({1:"defaulters",0:"repayers"})
hrs = hrs.sort_values(by="HOUR_APPR_PROCESS_START",ascending=True)

plt.subplot(212)
ax1 = sns.pointplot("HOUR_APPR_PROCESS_START","percentage",
                    data=hrs,hue="TARGET",palette="prism")
ax1.set_facecolor("k")
ax1.set_title("Peak hours for applying loans (defaulters vs repayers)")
fig.set_facecolor("snow")
plt.savefig('Output/PeakDaysforApplyingLoans.png')

# # Distribution in organization types for repayers and defaulters
# >ORGANIZATION_TYPE - Type of organization where client works.
# >organizations like Business Entity Type 3,Construction,Self-employed percentage of defaulters are higher than repayers.

# In[40]:

org = application_train.groupby("TARGET").agg({"ORGANIZATION_TYPE":"value_counts"})
org = org.rename(columns = {"ORGANIZATION_TYPE":"value_counts"}).reset_index()
org_0 = org[org["TARGET"] == 0]
org_1 = org[org["TARGET"] == 1]
org_0["percentage"] = org_0["value_counts"]*100/org_0["value_counts"].sum()
org_1["percentage"] = org_1["value_counts"]*100/org_1["value_counts"].sum()

organization = pd.concat([org_0,org_1],axis=0)
organization = organization.sort_values(by="ORGANIZATION_TYPE",ascending=True)

organization["TARGET"] = organization["TARGET"].replace({0:"repayers",1:"defaulters"})

organization
plt.figure(figsize=(13,7))
ax = sns.pointplot("ORGANIZATION_TYPE","percentage",
                   data=organization,hue="TARGET",palette=["b","r"])
plt.xticks(rotation=90)
plt.grid(True,alpha=.3)
ax.set_facecolor("k")
ax.set_title("Distribution in organization types for repayers and defaulters")
#plt.show()
plt.savefig('Output/DistributioninOrganization.png')


# # Distribution of Normalized score from external data source for repayer and defaulter
# >EXT_SOURCE_1	Normalized score from external data source.
# >EXT_SOURCE_2	Normalized score from external data source.
# >EXT_SOURCE_3	Normalized score from external data source.
# >Average value of normalized score from external data sources of defaulters is less than repayers.

# In[41]:

application_train[["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"]]

fig = plt.figure(figsize=(13,20))

plt.subplot(321)
sns.distplot(application_train[(application_train["EXT_SOURCE_1"].notnull()) & (application_train["TARGET"] ==0 )]["EXT_SOURCE_1"],color="b")
plt.axvline(application_train[(application_train["EXT_SOURCE_1"].notnull()) & (application_train["TARGET"] ==0 )]["EXT_SOURCE_1"].mean(),
           linestyle="dashed",color="k",
           label = ["MEAN :",application_train[(application_train["EXT_SOURCE_1"].notnull()) & (application_train["TARGET"] ==0 )]["EXT_SOURCE_1"].mean()])
plt.legend(loc="best")
plt.title("Repayer EXT_SOURCE_1 distribution")

plt.subplot(322)
sns.distplot(application_train[(application_train["EXT_SOURCE_1"].notnull()) & (application_train["TARGET"] ==1 )]["EXT_SOURCE_1"],color="r")
plt.axvline(application_train[(application_train["EXT_SOURCE_1"].notnull()) & (application_train["TARGET"] ==1 )]["EXT_SOURCE_1"].mean(),
           linestyle="dashed",color="k",
           label = ["MEAN :",application_train[(application_train["EXT_SOURCE_1"].notnull()) & (application_train["TARGET"] ==1 )]["EXT_SOURCE_1"].mean()])
plt.legend(loc="best")
plt.title("Defaulter EXT_SOURCE_1 distribution")
####
plt.subplot(323)
sns.distplot(application_train[(application_train["EXT_SOURCE_2"].notnull()) & (application_train["TARGET"] ==0 )]["EXT_SOURCE_2"],color="b")
plt.axvline(application_train[(application_train["EXT_SOURCE_2"].notnull()) & (application_train["TARGET"] ==0 )]["EXT_SOURCE_2"].mean(),
           linestyle="dashed",color="k",
           label = ["MEAN :",application_train[(application_train["EXT_SOURCE_2"].notnull()) & (application_train["TARGET"] ==0 )]["EXT_SOURCE_2"].mean()])
plt.legend(loc="best")
plt.title("Repayer EXT_SOURCE_2 distribution")

plt.subplot(324)
sns.distplot(application_train[(application_train["EXT_SOURCE_2"].notnull()) & (application_train["TARGET"] ==1 )]["EXT_SOURCE_2"],color="r")
plt.axvline(application_train[(application_train["EXT_SOURCE_2"].notnull()) & (application_train["TARGET"] ==1 )]["EXT_SOURCE_2"].mean(),
           linestyle="dashed",color="k",
           label = ["MEAN :",application_train[(application_train["EXT_SOURCE_2"].notnull()) & (application_train["TARGET"] ==1 )]["EXT_SOURCE_2"].mean()])
plt.legend(loc="best")
plt.title("Defaulter EXT_SOURCE_2 distribution")

###
plt.subplot(325)
sns.distplot(application_train[(application_train["EXT_SOURCE_3"].notnull()) & (application_train["TARGET"] ==0 )]["EXT_SOURCE_3"],color="b")
plt.axvline(application_train[(application_train["EXT_SOURCE_3"].notnull()) & (application_train["TARGET"] ==0 )]["EXT_SOURCE_3"].mean(),
           linestyle="dashed",color="k",
           label = ["MEAN :",application_train[(application_train["EXT_SOURCE_3"].notnull()) & (application_train["TARGET"] ==0 )]["EXT_SOURCE_3"].mean()])
plt.legend(loc="best")
plt.title("Repayer EXT_SOURCE_3 distribution")

plt.subplot(326)
sns.distplot(application_train[(application_train["EXT_SOURCE_3"].notnull()) & (application_train["TARGET"] ==1 )]["EXT_SOURCE_3"],color="r")
plt.axvline(application_train[(application_train["EXT_SOURCE_3"].notnull()) & (application_train["TARGET"] ==1 )]["EXT_SOURCE_3"].mean(),
           linestyle="dashed",color="k",
           label = ["MEAN :",application_train[(application_train["EXT_SOURCE_3"].notnull()) & (application_train["TARGET"] ==1 )]["EXT_SOURCE_3"].mean()])
plt.legend(loc="best")
plt.title("Defaulter EXT_SOURCE_3 distribution")

plt.subplots_adjust(hspace = .3)
fig.set_facecolor("lightgrey")
plt.savefig('Output/DistributionFromExternalDataSource.png')

# # Average  Normalized information about building where the client lives.
# APARTMENTS_AVG    - apartment size.
# BASEMENTAREA_AVG  - basement area .
# YEARS_BEGINEXPLUATATION_AVG - years begin expluatation .
# YEARS_BUILD_AVG - build years.
# COMMONAREA_AVG - common area.
# ELEVATORS_AVG - number of elevators.
# ENTRANCES_AVG -  number of entrances.
# FLOORSMAX_AVG -  maximum floors.
# FLOORSMIN_AVG - minimum floors.
# LANDAREA_AVG  - land area .
# LIVINGAPARTMENTS_AVG - living apartaents.
# LIVINGAREA_AVG - living area.
# NONLIVINGAPARTMENTS_AVG  - non living apartments.
# NONLIVINGAREA_AVG -non living area.

# In[42]:

fig = plt.figure(figsize=(12,13))

cols = ['APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG',
       'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG',
       'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG',
       'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG',
       'NONLIVINGAREA_AVG']

plt.subplot(211)
sns.heatmap(application_train[application_train["TARGET"] == 0][cols].describe()[1:].transpose(),
            annot=True, # cmap=sns.color_palette("Set1"),
            linecolor="k",linewidth=1)
plt.title("descriptive stats for Average  Normalized information about building where the repayers lives.",color="b")

plt.subplot(212)
sns.heatmap(application_train[application_train["TARGET"] == 1][cols].describe()[1:].transpose(),
            annot=True,# cmap=sns.color_palette("Set1"),
           linecolor="k",linewidth=1)
plt.title("descriptive stats for Average  Normalized information about building where the defaulters lives.",color="b")
fig.set_facecolor("ghostwhite")
plt.savefig('Output/WhereClientLivesAverage.png')

# # Mode Normalized information about building where the client lives.

# In[43]:

fig = plt.figure(figsize=(13,13))

cols1 = ['APARTMENTS_MODE', 'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE',
       'YEARS_BUILD_MODE', 'COMMONAREA_MODE', 'ELEVATORS_MODE',
       'ENTRANCES_MODE', 'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE',
       'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE']

plt.subplot(211)
sns.heatmap(application_train[application_train["TARGET"] == 0][cols1].describe()[1:].transpose(),annot=True, # cmap=sns.color_palette("viridis"),
            linecolor="k",linewidth=1)
plt.title("descriptive stats for Mode  Normalized information about building where the repayers lives.",color="b")

plt.subplot(212)
sns.heatmap(application_train[application_train["TARGET"] == 1][cols1].describe()[1:].transpose(),annot=True, # cmap=sns.color_palette("viridis"),
           linecolor="k",linewidth=1)
plt.title("descriptive stats for Mode  Normalized information about building where the defaulters lives.",color="b")
fig.set_facecolor("ghostwhite")
plt.savefig('Output/WhereClientLivesModeNormalized.png')

# # Median Normalized information about building where the client lives.

# In[44]:

fig = plt.figure(figsize=(12,13))

cols2 = ['APARTMENTS_MEDI', 'BASEMENTAREA_MEDI',
       'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI',
       'ELEVATORS_MEDI', 'ENTRANCES_MEDI', 'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI',
       'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MEDI',
       'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI']

plt.subplot(211)
sns.heatmap(application_train[application_train["TARGET"] == 0][cols2].describe()[1:].transpose(),
            annot=True, # cmap=sns.color_palette("magma"),
            linecolor="k",linewidth=1)
plt.title("descriptive stats for Median Normalized information about building where the repayers lives.",color="b")

plt.subplot(212)
sns.heatmap(application_train[application_train["TARGET"] == 1][cols2].describe()[1:].transpose(),
            annot=True, # cmap=sns.color_palette("magma"),
           linecolor="k",linewidth=1)
plt.title("descriptive stats for Median  Normalized information about building where the defaulters lives.",color="b")
fig.set_facecolor("ghostwhite")
plt.savefig('Output/WhereClientLivesMedianNormalized.png')

# # Comparing mean,standard deviation of normalized values between repayers and defaulters.
# >Mean values of defaulters are slightly less than repayers.
# >The standard deviation of  YEARS_BEGINEXPLUATATION for defaulters is higher than repayers.

# In[45]:

col = cols + cols1 +cols2

avg_mean = application_train.groupby("TARGET")[col].mean().stack().reset_index()
avg_mean["TARGET"] = avg_mean["TARGET"].replace({1:"defaulters",0:"repayers"})
avg_std = application_train.groupby("TARGET")[col].std().stack().reset_index()
avg_std["TARGET"] = avg_std["TARGET"].replace({1:"defaulters",0:"repayers"})

fig =plt.figure(figsize=(14,15))
plt.subplot(211)
ax = sns.barplot("level_1",0,data=avg_mean,hue="TARGET",palette=["white","k"])
plt.xticks(rotation=90)
plt.xlabel("")
plt.ylabel("mean")
plt.title("comparing mean values of normalized values between repayers and defaulters",color="b")
ax.set_facecolor("r")

plt.subplot(212)
ax1 = sns.pointplot("level_1",0,data=avg_std,hue="TARGET",palette=["white","k"])
plt.xticks(rotation=90)
plt.xlabel("")
plt.ylabel("standard deviation")
plt.title("comparing standard deviation of normalized values between repayers and defaulters",color="b")
ax1.set_facecolor("r")
plt.subplots_adjust(hspace = .7)
plt.savefig('Output/WhereClientLivesComparison.png')

# # Square plot for distribution in house attributes.
#

# In[46]:

import squarify
cs = [ 'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE']
length = len(cs)
pal = ["Set1","Set3","Set2"]

plt.figure(figsize=(13,10))
for i,j,k in itertools.zip_longest(cs,range(length),pal):
    plt.subplot(2,2,j+1)
    squarify.plot(sizes=data[i].value_counts().values,label=data[i].value_counts().keys(),
                  value=data[i].value_counts().values,
                  color=sns.color_palette(k),linewidth=2,edgecolor="k",alpha=.8)
    plt.title(i)
    plt.axis("off")
plt.savefig('Output/SquarePlotDistributionHouse.png')

# # Normalized average total area by house attributes and repayment status

# In[47]:

cs = [ 'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE']
length = len(cs)

fig = plt.figure(figsize=(13,14))
fig.set_facecolor("lightgrey")
for i,j in itertools.zip_longest(cs,range(length)):
    plt.subplot(2,2,j+1)
    ax = sns.barplot(i,"TOTALAREA_MODE",data=application_train.groupby(["TARGET",i])["TOTALAREA_MODE"].mean().reset_index(),
                hue="TARGET",palette=["b","r"])
    ax.set_facecolor("yellow")
    plt.title(i)
plt.savefig('Output/AverageAreaHouse.png')

# # Distribution client's social surroundings with observed and defaulted 30 DPD (days past due)
# >OBS_30_CNT_SOCIAL_CIRCLE- How many observation of client's social surroundings with observable 30 DPD (days past due) default.
# >DEF_30_CNT_SOCIAL_CIRCLE-How many observation of client's social surroundings defaulted on 30 DPD (days past due) .
# >OBS_60_CNT_SOCIAL_CIRCLE -	How many observation of client's social surroundings with observable 60 DPD (days past due) default.
# >DEF_60_CNT_SOCIAL_CIRCLE - How many observation of client's social surroundings defaulted on 60 (days past due) DPD.

# In[48]:

fig = plt.figure(figsize=(13,20))
plt.subplot(421)
ax = sns.distplot(application_train[(application_train["TARGET"] == 1 ) & (application_train["OBS_30_CNT_SOCIAL_CIRCLE"].notnull())]["OBS_30_CNT_SOCIAL_CIRCLE"],
                color="r",label="defaulter")
plt.axvline(application_train[(application_train["TARGET"] == 1 ) ]["OBS_30_CNT_SOCIAL_CIRCLE"].mean(),label="mean",color="k",linestyle="dashed")
plt.legend(loc="best")
plt.title("OBS_30_CNT_SOCIAL_CIRCLE - defaulter")
plt.xlabel("")
ax.set_facecolor("lightyellow")

plt.subplot(422)
ax = sns.distplot(application_train[(application_train["TARGET"] == 0 ) & (application_train["OBS_30_CNT_SOCIAL_CIRCLE"].notnull())]["OBS_30_CNT_SOCIAL_CIRCLE"],
                color="b",label="repayer")
plt.axvline(application_train[(application_train["TARGET"] == 0 ) ]["OBS_30_CNT_SOCIAL_CIRCLE"].mean(),label="mean",color="k",linestyle="dashed")
plt.legend(loc="best")
plt.title("OBS_30_CNT_SOCIAL_CIRCLE - repayer")
plt.xlabel("")
ax.set_facecolor("lightyellow")

plt.subplot(423)
ax = sns.distplot(application_train[(application_train["TARGET"] == 1 ) & (application_train["DEF_30_CNT_SOCIAL_CIRCLE"].notnull())]["DEF_30_CNT_SOCIAL_CIRCLE"],
                color="r",label="defaulter")
plt.axvline(application_train[(application_train["TARGET"] == 1 ) ]["DEF_30_CNT_SOCIAL_CIRCLE"].mean(),label="mean",color="k",linestyle="dashed")
plt.legend(loc="best")
plt.title("DEF_30_CNT_SOCIAL_CIRCLE - defaulter")
plt.xlabel("")
ax.set_facecolor("lightyellow")

plt.subplot(424)
ax = sns.distplot(application_train[(application_train["TARGET"] == 0 ) & (application_train["DEF_30_CNT_SOCIAL_CIRCLE"].notnull())]["DEF_30_CNT_SOCIAL_CIRCLE"],
                color="b",label="repayer")
plt.axvline(application_train[(application_train["TARGET"] == 0 ) ]["DEF_30_CNT_SOCIAL_CIRCLE"].mean(),label="mean",color="k",linestyle="dashed")
plt.legend(loc="best")
plt.title("DEF_30_CNT_SOCIAL_CIRCLE - repayer")
plt.xlabel("")
ax.set_facecolor("lightyellow")

plt.subplot(425)
ax = sns.distplot(application_train[(application_train["TARGET"] == 1 ) & (application_train["OBS_60_CNT_SOCIAL_CIRCLE"].notnull())]["OBS_60_CNT_SOCIAL_CIRCLE"],
                color="r",label="defaulter")
plt.axvline(application_train[(application_train["TARGET"] == 1 ) ]["OBS_60_CNT_SOCIAL_CIRCLE"].mean(),label="mean",color="k",linestyle="dashed")
plt.legend(loc="best")
plt.title("OBS_60_CNT_SOCIAL_CIRCLE - defaulter")
plt.xlabel("")
ax.set_facecolor("lightyellow")

plt.subplot(426)
ax = sns.distplot(application_train[(application_train["TARGET"] == 0 ) & (application_train["OBS_60_CNT_SOCIAL_CIRCLE"].notnull())]["OBS_60_CNT_SOCIAL_CIRCLE"],
                color="b",label="repayer")
plt.axvline(application_train[(application_train["TARGET"] == 0 ) ]["OBS_60_CNT_SOCIAL_CIRCLE"].mean(),label="mean",color="k",linestyle="dashed")
plt.legend(loc="best")
plt.title("OBS_60_CNT_SOCIAL_CIRCLE - repayer")
plt.xlabel("")
ax.set_facecolor("lightyellow")



plt.subplot(427)
ax = sns.distplot(application_train[(application_train["TARGET"] == 1 ) & (application_train["DEF_60_CNT_SOCIAL_CIRCLE"].notnull())]["DEF_60_CNT_SOCIAL_CIRCLE"],
                color="r",label="defaulter")
plt.axvline(application_train[(application_train["TARGET"] == 1 ) ]["DEF_60_CNT_SOCIAL_CIRCLE"].mean(),label="mean",color="k",linestyle="dashed")
plt.legend(loc="best")
plt.title("DEF_60_CNT_SOCIAL_CIRCLE - defaulter")
plt.xlabel("")
ax.set_facecolor("lightyellow")

plt.subplot(428)

ax = sns.distplot(application_train[(application_train["TARGET"] == 0 ) & (application_train["DEF_60_CNT_SOCIAL_CIRCLE"].notnull())]["DEF_60_CNT_SOCIAL_CIRCLE"],
                color="b",label="repayer")
plt.axvline(application_train[(application_train["TARGET"] == 0 ) ]["DEF_60_CNT_SOCIAL_CIRCLE"].mean(),label="mean",color="k",linestyle="dashed")
plt.legend(loc="best")
plt.title("DEF_60_CNT_SOCIAL_CIRCLE - repayer")
plt.xlabel("")
ax.set_facecolor("lightyellow")
fig.set_facecolor("lightgrey")
plt.savefig('Output/ClientSocialSorrounding.png')

# # Number of days before application client changed phone .
# >DAYS_LAST_PHONE_CHANGE	 - How many days before application did client change phone.
# >average days of defaulters phone change is less than average days of repayers phone change.

# In[49]:

plt.figure(figsize=(13,7))
plt.subplot(121)
ax = sns.violinplot(application_train["TARGET"],
                    application_train["DAYS_LAST_PHONE_CHANGE"],palette=["g","r"])
ax.set_facecolor("oldlace")
ax.set_title("days before application client changed phone -violin plot")
plt.subplot(122)
ax1 = sns.lvplot(application_train["TARGET"],
                 application_train["DAYS_LAST_PHONE_CHANGE"],palette=["g","r"])
ax1.set_facecolor("oldlace")
ax1.set_ylabel("")
ax1.set_title("days before application client changed phone -box plot")
plt.subplots_adjust(wspace = .2)
plt.savefig('Output/ChangingPhoneClient.png')

# # Documents provided by the clients.
# FLAG_DOCUMENT - 	Did client provide documents.(1,0)

# In[50]:

cols = [ 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3',
       'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6',
       'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9',
       'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12',
       'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15',
       'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18',
       'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']

df_flag = application_train[cols+["TARGET"]]

length = len(cols)

df_flag["TARGET"] = df_flag["TARGET"].replace({1:"defaulter",0:"repayer"})

fig = plt.figure(figsize=(13,24))
fig.set_facecolor("lightgrey")
for i,j in itertools.zip_longest(cols,range(length)):
    plt.subplot(5,4,j+1)
    ax = sns.countplot(df_flag[i],hue=df_flag["TARGET"],palette=["r","b"])
    plt.yticks(fontsize=5)
    plt.xlabel("")
    plt.title(i)
    ax.set_facecolor("k")
plt.savefig('Output/DocumentProvidedbyClient.png')

# # Equiries to Credit Bureau about the client before application.
# >AMT_REQ_CREDIT_BUREAU_HOUR	 - Number of enquiries to Credit Bureau about the client one hour before application.
# >AMT_REQ_CREDIT_BUREAU_DAY - 	Number of enquiries to Credit Bureau about the client one day before application (excluding one hour before application).
# >AMT_REQ_CREDIT_BUREAU_WEEK -	Number of enquiries to Credit Bureau about the client one week before application (excluding one day before application).
# >AMT_REQ_CREDIT_BUREAU_MON	- Number of enquiries to Credit Bureau about the client one month before application (excluding one week before application).
# >AMT_REQ_CREDIT_BUREAU_QRT	 - Number of enquiries to Credit Bureau about the client 3 month before application (excluding one month before application).
# >AMT_REQ_CREDIT_BUREAU_YEAR	 - Number of enquiries to Credit Bureau about the client one day year (excluding last 3 months before application).
#

# In[51]:

cols = ['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
       'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
       'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']
application_train.groupby("TARGET")[cols].max().transpose().plot(kind="barh",
                                                                 figsize=(10,5),width=.8)
plt.title("Maximum enquries made by defaulters and repayers")
application_train.groupby("TARGET")[cols].mean().transpose().plot(kind="barh",
                                                                  figsize=(10,5),width=.8)
plt.title("average enquries made by defaulters and repayers")
application_train.groupby("TARGET")[cols].std().transpose().plot(kind="barh",
                                                                 figsize=(10,5),width=.8)
plt.title("standard deviation in enquries made by defaulters and repayers")
#plt.show()
plt.savefig('Output/EnquiriesmadebyClients.png')


# # Previous credits by repayers and defaulters
# >SK_ID_CURR - ID of loan in our sample - one loan in our sample can have 0,1,2 or more related previous credits in credit bureau .
# >SK_BUREAU_ID - Recoded ID of previous Credit Bureau credit related to our loan (unique coding for each loan application).

# In[52]:

#Merging bureau andapplication train data
app_tar = application_train[["TARGET","SK_ID_CURR"]]
app_bureau = bureau.merge(app_tar,left_on="SK_ID_CURR",right_on="SK_ID_CURR",how="left")


prev_cre = app_bureau.groupby(["TARGET","SK_ID_CURR"])["SK_ID_BUREAU"].nunique().reset_index()
fig = plt.figure(figsize=(12,12))
plt.subplot(211)
sns.distplot(prev_cre[prev_cre["TARGET"]==0]["SK_ID_BUREAU"],color="g")
plt.axvline(prev_cre[prev_cre["TARGET"]==0]["SK_ID_BUREAU"].mean(),linestyle="dashed",color="k",label = "mean")
plt.title("Number of previous credits by repayers")
plt.legend(loc="best")

plt.subplot(212)
sns.distplot(prev_cre[prev_cre["TARGET"]==1]["SK_ID_BUREAU"],color="r")
plt.axvline(prev_cre[prev_cre["TARGET"]==1]["SK_ID_BUREAU"].mean(),linestyle="dashed",color="k",label = "mean")
plt.legend(loc="best")
plt.title("Number of previous credits by defaulters")
fig.set_facecolor("lightgrey")
plt.savefig('Output/PreviousCredits.png')

# # Credit status for repayers and defaulters
# >CREDIT_ACTIVE  - Status of the Credit Bureau (CB) reported credits.
# >defaulters have more active previous loans compared to repayers.

# In[53]:

plt.figure(figsize=(12,6))
plt.subplot(121)
app_bureau[app_bureau["TARGET"]  ==0]["CREDIT_ACTIVE"].value_counts().plot.pie(autopct = "%1.1f%%",fontsize=8,
                                                             colors = sns.color_palette("Set1"),
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)
plt.title("previous credit status for repayers")

plt.subplot(122)
app_bureau[app_bureau["TARGET"]  ==1]["CREDIT_ACTIVE"].value_counts().plot.pie(autopct = "%1.0f%%",fontsize=8,
                                                             colors = sns.color_palette("Set1"),
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)
plt.title("previous credit status for defaulters")
#plt.show()
plt.savefig('Output/PreviousCreditStatusForDefaulters.png')


# # Distribution of currency of the Credit Bureau credit.
# >CREDIT_CURRENCY - Recoded currency of the Credit Bureau credit.
# >99.9% of  currency types  of the Credit Bureau credit are of type Currency type1.

# In[54]:

plt.figure(figsize=(8,8))
app_bureau["CREDIT_CURRENCY"].value_counts().plot.pie(autopct = "%1.1f%%",fontsize=8,
                                                             colors = sns.color_palette("Set1"),
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)
plt.ylabel("")
plt.title("distribution of currency of the Credit Bureau credit")
#plt.show()
plt.savefig('Output/DistributionofCurrency.png')


# # How many days before current application did client apply for Credit Bureau credit.
# >DAYS_CREDIT - How many days before current application did client apply for Credit Bureau credit.
# >The average days of defaulters applying for bureau before current application is less compared to repayers.

# In[55]:

app_bureau["TARGET"] = app_bureau["TARGET"].replace({1:"defaulter",0:"repayer"})

plt.figure(figsize=(13,8))

plt.subplot(121)
sns.violinplot(y=app_bureau["DAYS_CREDIT"],x=app_bureau["TARGET"],palette="husl")
plt.title("days before current application client applied for bureau")

plt.subplot(122)
sns.boxplot(y=app_bureau["DAYS_CREDIT"],x=app_bureau["TARGET"],palette="husl")
plt.ylabel("")
plt.title("days before current application client applied for bureau")
#plt.show()
plt.savefig('Output/DaysBeforeClientApplyforCreditBureau.png')


# # Distribution of credit duration variables
# >CREDIT_DAY_OVERDUE - Number of days past due on CB credit at the time of application for related loan in our sample.
# >DAYS_CREDIT_ENDDATE - Remaining duration of CB credit (in days) at the time of application in Home Credit.
# >DAYS_ENDDATE_FACT - Days since CB credit ended at the time of application in Home Credit (only for closed credit).
# >DAYS_CREDIT_UPDATE - 	How many days before loan application did last information about the Credit Bureau credit come.

# In[56]:


types = ["defaulter","repayer"]
length = len(types)
cs = ["r","b"]

fig = plt.figure(figsize=(13,22))
plt.subplot(411)
for i,j,k in itertools.zip_longest(types,(range(length)),cs):
    ax = sns.kdeplot(app_bureau[(app_bureau["TARGET"]==i)&(app_bureau["CREDIT_DAY_OVERDUE"].notnull())]["CREDIT_DAY_OVERDUE"],shade=True,color=k,label=i)
    plt.axvline(app_bureau[(app_bureau["TARGET"]==i)&(app_bureau["CREDIT_DAY_OVERDUE"].notnull())]["CREDIT_DAY_OVERDUE"].mean(),color=k,linestyle="dashed",label="mean")
    plt.title("Number of days past due on Bureau credit at the time of application for related loan")
    ax.set_facecolor("lightgrey")

plt.subplot(412)
for i,j,k in itertools.zip_longest(types,(range(length)),cs):
    ax = sns.kdeplot(app_bureau[(app_bureau["TARGET"]==i)&(app_bureau["DAYS_CREDIT_ENDDATE"].notnull())]["DAYS_CREDIT_ENDDATE"],shade=True,color=k,label=i)
    plt.axvline(app_bureau[(app_bureau["TARGET"]==i)&(app_bureau["DAYS_CREDIT_ENDDATE"].notnull())]["DAYS_CREDIT_ENDDATE"].mean(),color=k,linestyle="dashed",label="mean")
    ax.set_facecolor("lightgrey")
    plt.title("Remaining duration of Bureau credit (in days) at the time of application in Home Credit")

plt.subplot(413)
for i,j,k in itertools.zip_longest(types,(range(length)),cs):
    ax = sns.kdeplot(app_bureau[(app_bureau["TARGET"]==i)&(app_bureau["DAYS_ENDDATE_FACT"].notnull())]["DAYS_ENDDATE_FACT"],shade=True,color=k,label=i)
    plt.axvline(app_bureau[(app_bureau["TARGET"]==i)&(app_bureau["DAYS_ENDDATE_FACT"].notnull())]["DAYS_ENDDATE_FACT"].mean(),color=k,linestyle="dashed",label="mean")
    ax.set_facecolor("lightgrey")
    plt.title("Days since Bureau credit ended at the time of application in Home Credit (only for closed credit")

plt.subplot(414)
for i,j,k in itertools.zip_longest(types,(range(length)),cs):
    ax = sns.kdeplot(app_bureau[(app_bureau["TARGET"]==i)&(app_bureau["DAYS_CREDIT_UPDATE"].notnull())]["DAYS_CREDIT_UPDATE"],shade=True,color=k,label=i)
    plt.axvline(app_bureau[(app_bureau["TARGET"]==i)&(app_bureau["DAYS_CREDIT_UPDATE"].notnull())]["DAYS_CREDIT_UPDATE"].mean(),color=k,linestyle="dashed",label="mean")
    ax.set_facecolor("lightgrey")
    plt.title("Number of days before loan application did last information about the Credit Bureau credit come.")

fig.set_facecolor("lightyellow")
plt.savefig('Output/DistributionCreditDurationVariables.png')

# # Percentage of  Credit Bureau - credit  types for defaulters & repayers.
# >CREDIT_TYPE- Type of Credit Bureau credit (Car, cash,...).                                                                                              >Percentages for  Credit card & micro loan types are more for defaulters than repayers.

# In[57]:


rep = ((app_bureau[app_bureau["TARGET"] == "repayer"]["CREDIT_TYPE"].value_counts()*100)/app_bureau[app_bureau["TARGET"] == "repayer"]["CREDIT_TYPE"].value_counts().sum()).reset_index()
rep["type"] = "repayers"

defl = ((app_bureau[app_bureau["TARGET"] == "defaulter"]["CREDIT_TYPE"].value_counts()*100)/app_bureau[app_bureau["TARGET"] == "defaulter"]["CREDIT_TYPE"].value_counts().sum()).reset_index()
defl["type"] = "defaulters"

credit_types = pd.concat([rep,defl],axis=0)
credit_types = credit_types.sort_values(by="CREDIT_TYPE",ascending =False)
plt.figure(figsize=(10,8))
ax = sns.barplot("CREDIT_TYPE","index",data=credit_types[:10],hue="type",palette=["b","r"])
ax.set_ylabel("credit types")
ax.set_xlabel("percentage")
ax.set_facecolor("k")
ax.set_title('Type of Credit Bureau credit')
#plt.show()
plt.savefig('Output/CreditTypes.png')


# # Credit amount variables by Credit Bureau.
# >AMT_CREDIT_MAX_OVERDUE - Maximal amount overdue on the Credit Bureau credit so far (at application date of loan in our sample).
# >AMT_CREDIT_SUM - Current credit amount for the Credit Bureau credit.
# >AMT_CREDIT_SUM_DEBT - Current debt on Credit Bureau credit.
# >AMT_CREDIT_SUM_LIMIT - Current credit limit of credit card reported in Credit Bureau.
# >AMT_CREDIT_SUM_OVERDUE - Current amount overdue on Credit Bureau credit.

# In[58]:

cols = ['AMT_CREDIT_MAX_OVERDUE', 'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT','AMT_CREDIT_SUM_OVERDUE']
plt.figure(figsize=(12,6))
sns.heatmap(app_bureau[cols].describe().transpose(),annot=True,
            linecolor="k",linewidth=2) # ,cmap=sns.color_palette("Set1")
plt.title("summary of amount variables")
#plt.show()
plt.savefig('Output/CreditAmountVariables.png')


# # Percentage of number of times Credit Bureau credit prolonged.
# >CNT_CREDIT_PROLONG - How many times was the Credit Bureau credit prolonged.

# In[ ]:

d = app_bureau[(app_bureau["TARGET"]=="defaulter") & (app_bureau["CNT_CREDIT_PROLONG"].notnull())]["CNT_CREDIT_PROLONG"].value_counts()*100/app_bureau[(app_bureau["TARGET"]=="defaulter") & (app_bureau["CNT_CREDIT_PROLONG"].notnull())]["CNT_CREDIT_PROLONG"].value_counts().sum()
r = app_bureau[(app_bureau["TARGET"]=="repayer") & (app_bureau["CNT_CREDIT_PROLONG"].notnull())]["CNT_CREDIT_PROLONG"].value_counts()*100/app_bureau[(app_bureau["TARGET"]=="repayer") & (app_bureau["CNT_CREDIT_PROLONG"].notnull())]["CNT_CREDIT_PROLONG"].value_counts().sum()

plt.figure(figsize=(10,8))
plt.subplot(121)
sns.heatmap(pd.DataFrame(d),annot=True,linecolor="k",linewidths=2) # ,cmap=sns.color_palette("Set1")
plt.title("%age of number of times prolonged - defaulters")

plt.subplot(122)
sns.heatmap(pd.DataFrame(r),annot=True,linecolor="k",linewidths=2) # ,cmap=sns.color_palette("Set2")
plt.title("%ge of number of times prolonged - repayers")
plt.subplots_adjust(wspace = .7)
plt.savefig('Output/TimesCreditBureauProlonged.png')

# In[ ]:

app_bureau_balance = app_bureau.merge(bureau_balance,left_on="SK_ID_BUREAU",right_on="SK_ID_BUREAU",how="left")


# # Month of  balances for status types.
# >MONTHS_BALANCE	 - Month of balance relative to application date (-1 means the freshest balance date).
# >STATUS	 - Status of Credit Bureau loan during the month (active, closed, DPD0-30,… [C means closed, X means status unknown, 0 means no DPD, 1 means maximal did during month between 1-30, 2 means DPD 31-60,… 5 means DPD 120+ or sold or written off ] ).

# In[ ]:

plt.figure(figsize=(13,7))
sns.boxplot(y=bureau_balance["MONTHS_BALANCE"],x=bureau_balance["STATUS"],palette="husl")
plt.title("Months balance for status types")
#plt.show()
plt.savefig('Output/BalancesforStatus.png')


# # Distribution of Month of balances
# >MONTHS_BALANCE	 - Month of balance relative to application date (-1 means the freshest balance date).

# In[ ]:

plt.figure(figsize=(12,6))
sns.distplot(bureau_balance["MONTHS_BALANCE"],color="b")
plt.title("Distribution of Month of balances")
#plt.show()
plt.savefig('Output/DistributionofMonthofBalances.png')


# # Current loan id having previous loan applications.
# >SK_ID_PREV - ID of previous credit in Home credit related to loan in our sample. (One loan in our sample can have 0,1,2 or more previous loan applications in Home Credit, previous application could, but not necessarily have to lead to credit).
# >SK_ID_CURR	ID of loan in our sample.
# >On average current loan ids have 4 to 5 loan applications previously.

# In[ ]:

x = previous_application.groupby("SK_ID_CURR")["SK_ID_PREV"].count().reset_index()
plt.figure(figsize=(13,7))
ax = sns.distplot(x["SK_ID_PREV"],color="orange")
plt.axvline(x["SK_ID_PREV"].mean(),linestyle="dashed",color="r",label="average")
plt.axvline(x["SK_ID_PREV"].std(),linestyle="dashed",color="b",label="standard deviation")
plt.axvline(x["SK_ID_PREV"].max(),linestyle="dashed",color="g",label="maximum")
plt.legend(loc="best")
plt.title("Current loan id having previous loan applications")
ax.set_facecolor("k")
plt.savefig('Output/PreviousLoanApplications.png')


# # Contract types in previous applications
# >NAME_CONTRACT_TYPE	Contract product type (Cash loan, consumer loan [POS] ,...) of the previous application.
# >Cash loan applications  are maximum followed by consumer loan applications.

# In[ ]:

cnts = previous_application["NAME_CONTRACT_TYPE"].value_counts()
import squarify
plt.figure(figsize=(8,6))
squarify.plot(cnts.values,label=cnts.keys(),value=cnts.values,linewidth=2,edgecolor="k",alpha=.8,color=sns.color_palette("Set1"))
plt.axis("off")
plt.title("Contract types in previous applications")
#plt.show()
plt.savefig('Output/ContractTypesinPrevApplications.png')


# # Previous loan amounts applied and loan amounts credited.
# >AMT_APPLICATION-For how much credit did client ask on the previous application.
# >AMT_CREDIT-Final credit amount on the previous application. This differs from AMT_APPLICATION in a way that the AMT_APPLICATION is the amount for which the client initially applied for, but during our approval process he could have received different amount - AMT_CREDIT.

# In[ ]:

plt.figure(figsize=(12,13))
plt.subplot(211)
ax = sns.kdeplot(previous_application["AMT_APPLICATION"],color="b",linewidth=3)
ax = sns.kdeplot(previous_application[previous_application["AMT_CREDIT"].notnull()]["AMT_CREDIT"],color="r",linewidth=3)
plt.axvline(previous_application[previous_application["AMT_CREDIT"].notnull()]["AMT_CREDIT"].mean(),color="r",linestyle="dashed",label="AMT_APPLICATION_MEAN")
plt.axvline(previous_application["AMT_APPLICATION"].mean(),color="b",linestyle="dashed",label="AMT_APPLICATION_MEAN")
plt.legend(loc="best")
plt.title("Previous loan amounts applied and loan amounts credited.")
ax.set_facecolor("k")

plt.subplot(212)
diff = (previous_application["AMT_CREDIT"] - previous_application["AMT_APPLICATION"]).reset_index()
diff = diff[diff[0].notnull()]
ax1 = sns.kdeplot(diff[0],color="g",linewidth=3,label = "difference in amount requested by client and amount credited")
plt.axvline(diff[0].mean(),color="white",linestyle="dashed",label = "mean")
plt.title("difference in amount requested by client and amount credited")
ax1.legend(loc="best")
ax1.set_facecolor("k")
plt.savefig('Output/PreviousLoanAmountAplliedandCredited.png')


# # Total and average amounts applied and credited in previousapplications
# >AMT_APPLICATION-For how much credit did client ask on the previous application.                                                >AMT_CREDIT-Final credit amount on the previous application. This differs from AMT_APPLICATION in a way that the AMT_APPLICATION is the amount for which the client.

# In[ ]:

mn = previous_application.groupby("NAME_CONTRACT_TYPE")[["AMT_APPLICATION","AMT_CREDIT"]].mean().stack().reset_index()
tt = previous_application.groupby("NAME_CONTRACT_TYPE")[["AMT_APPLICATION","AMT_CREDIT"]].sum().stack().reset_index()
fig = plt.figure(figsize=(10,13))
fig.set_facecolor("ghostwhite")
plt.subplot(211)
ax = sns.barplot(0,"NAME_CONTRACT_TYPE",data=mn[:6],hue="level_1",palette="inferno")
ax.set_facecolor("k")
ax.set_xlabel("average amounts")
ax.set_title("Average amounts by contract types")

plt.subplot(212)
ax1 = sns.barplot(0,"NAME_CONTRACT_TYPE",data=tt[:6],hue="level_1",palette="magma")
ax1.set_facecolor("k")
ax1.set_xlabel("total amounts")
ax1.set_title("total amounts by contract types")
plt.subplots_adjust(hspace = .2)
plt.savefig('Output/AmountsAppliedandCreditedPrevious.png')

# # Annuity of previous application
# >AMT_ANNUITY - Annuity of previous application

# In[ ]:

plt.figure(figsize=(14,5))
plt.subplot(121)
previous_application.groupby("NAME_CONTRACT_TYPE")["AMT_ANNUITY"].sum().plot(kind="bar")
plt.xticks(rotation=0)
plt.title("Total annuity amount by contract types in previous applications")
plt.subplot(122)
previous_application.groupby("NAME_CONTRACT_TYPE")["AMT_ANNUITY"].mean().plot(kind="bar")
plt.title("average annuity amount by contract types in previous applications")
plt.xticks(rotation=0)
#plt.show()
plt.savefig('Output/AnnuityPreviousApplication.png')


# # Count of application status by application type.
# >NAME_CONTRACT_TYPE -Contract product type (Cash loan, consumer loan [POS] ,...) of the previous application.
# >NAME_CONTRACT_STATUS -Contract status (approved, cancelled, ...) of previous application.
# >Consumer loan applications are most approved loans and cash loans are most cancelled and refused loans.

# In[ ]:

ax = pd.crosstab(previous_application["NAME_CONTRACT_TYPE"],previous_application["NAME_CONTRACT_STATUS"]).plot(kind="barh",figsize=(10,7),stacked=True)
plt.xticks(rotation =0)
plt.ylabel("count")
plt.title("Count of application status by application type")
ax.set_facecolor("k")
plt.savefig('Output/CountofApplicationStatusbyType.png')

# # Contract status by weekdays
# WEEKDAY_APPR_PROCESS_START - On which day of the week did the client apply for previous application

# In[ ]:

ax = pd.crosstab(previous_application["WEEKDAY_APPR_PROCESS_START"],previous_application["NAME_CONTRACT_STATUS"]).plot(kind="barh",colors=["g","r","b","orange"],
                                                                                                                  stacked =True,figsize=(12,8))
ax.set_facecolor("k")

ax.set_title("Contract status by weekdays")
#plt.show()
plt.savefig('Output/ContractStatusbyWeekdays.png')


# # Contract status by hour of the day
# >HOUR_APPR_PROCESS_START -  Approximately at what day hour did the client apply for the previous application.
# >Morning 11'o  clock have maximum number of approvals.
# >Morning 10'o clock have maximum number of refused and cancelled contracts.

# In[ ]:

hr = pd.crosstab(previous_application["HOUR_APPR_PROCESS_START"],previous_application["NAME_CONTRACT_STATUS"]).stack().reset_index()
plt.figure(figsize=(12,8))
ax = sns.pointplot(hr["HOUR_APPR_PROCESS_START"],hr[0],hue=hr["NAME_CONTRACT_STATUS"],palette=["g","r","b","orange"],scale=1)
ax.set_facecolor("k")
ax.set_ylabel("count")
ax.set_title("Contract status by day hours.")
plt.grid(True,alpha=.2)
plt.savefig('Output/ContractStatusbyDayHours.png')

# # Peak hours for week days for applying loans.

# In[ ]:

ax = pd.crosstab(previous_application["HOUR_APPR_PROCESS_START"],previous_application["WEEKDAY_APPR_PROCESS_START"]).plot(kind="bar",colors=sns.color_palette("rainbow",7),
                                                                                                                     figsize=(13,8),stacked=True)
ax.set_facecolor("k")
ax.set_title("Peak hours for week days ")
#plt.show()
plt.savefig('Output/PeakHoursForWeekdays.png')


# # Interest rate normalized on previous credit
# >RATE_INTEREST_PRIMARY - Interest rate normalized on previous credit.
# >RATE_INTEREST_PRIVILEGED - Interest rate normalized on previous credit.

# In[ ]:

plt.figure(figsize=(13,6))
plt.subplot(121)
sns.violinplot(previous_application["RATE_INTEREST_PRIMARY"],alpha=.01,color="orange")
plt.axvline(previous_application[previous_application["RATE_INTEREST_PRIMARY"].notnull()]["RATE_INTEREST_PRIMARY"].mean(),color="k",linestyle="dashed")
plt.title("RATE_INTEREST_PRIMARY")
plt.subplot(122)
sns.violinplot(previous_application["RATE_INTEREST_PRIVILEGED"],color="c")
plt.axvline(previous_application[previous_application["RATE_INTEREST_PRIVILEGED"].notnull()]["RATE_INTEREST_PRIVILEGED"].mean(),color="k",linestyle="dashed")
plt.title("RATE_INTEREST_PRIVILEGED")
#plt.show()
plt.savefig('Output/IRonPreviousCredit.png')


# # Percentage of applications accepted,cancelled,refused and unused for different loan purposes.
# >NAME_CASH_LOAN_PURPOSE	 - Purpose of the cash loan.
# >NAME_CONTRACT_STATUS	 - Contract status (approved, cancelled, ...) of previous application.
# > Purposes like XAP ,electronic eqipment ,everey day expences and education have maximum loan acceptance.
# >Loan puposes like payment of other loans ,refusal to name goal ,buying new home or car have most refusals.
# >40% of XNA purpose loans are cancalled.

# In[ ]:

previous_application[["NAME_CASH_LOAN_PURPOSE","NAME_CONTRACT_STATUS"]]
purpose = pd.crosstab(previous_application["NAME_CASH_LOAN_PURPOSE"],previous_application["NAME_CONTRACT_STATUS"])
purpose["a"] = (purpose["Approved"]*100)/(purpose["Approved"]+purpose["Canceled"]+purpose["Refused"]+purpose["Unused offer"])
purpose["c"] = (purpose["Canceled"]*100)/(purpose["Approved"]+purpose["Canceled"]+purpose["Refused"]+purpose["Unused offer"])
purpose["r"] = (purpose["Refused"]*100)/(purpose["Approved"]+purpose["Canceled"]+purpose["Refused"]+purpose["Unused offer"])
purpose["u"] = (purpose["Unused offer"]*100)/(purpose["Approved"]+purpose["Canceled"]+purpose["Refused"]+purpose["Unused offer"])
purpose_new = purpose[["a","c","r","u"]]
purpose_new = purpose_new.stack().reset_index()
purpose_new["NAME_CONTRACT_STATUS"] = purpose_new["NAME_CONTRACT_STATUS"].replace({"a":"accepted_percentage","c":"cancelled_percentage",
                                                               "r":"refused_percentage","u":"unused_percentage"})

lst = purpose_new["NAME_CONTRACT_STATUS"].unique().tolist()
length = len(lst)
cs = ["lime","orange","r","b"]

fig = plt.figure(figsize=(14,18))
fig.set_facecolor("lightgrey")
for i,j,k in itertools.zip_longest(lst,range(length),cs):
    plt.subplot(2,2,j+1)
    dat = purpose_new[purpose_new["NAME_CONTRACT_STATUS"] == i]
    ax = sns.barplot(0,"NAME_CASH_LOAN_PURPOSE",data=dat.sort_values(by=0,ascending=False),color=k)
    plt.ylabel("")
    plt.xlabel("percentage")
    plt.title(i+" by purpose")
    plt.subplots_adjust(wspace = .7)
    ax.set_facecolor("k")
plt.savefig('Output/PercentageofApplicationForDifferentLoans.png')

# # Contract status relative to decision made about previous application.
# >DAYS_DECISION - Relative to current application when was the decision about previous application made.
# >On average approved contract types have higher number of decision days compared to cancelled and refused contracts.

# In[ ]:

plt.figure(figsize=(13,6))
sns.violinplot(y= previous_application["DAYS_DECISION"],
               x = previous_application["NAME_CONTRACT_STATUS"],palette=["r","g","b","y"])
plt.axhline(previous_application[previous_application["NAME_CONTRACT_STATUS"] == "Approved"]["DAYS_DECISION"].mean(),
            color="r",linestyle="dashed",label="accepted_average")
plt.axhline(previous_application[previous_application["NAME_CONTRACT_STATUS"] == "Refused"]["DAYS_DECISION"].mean(),
            color="g",linestyle="dashed",label="refused_average")
plt.axhline(previous_application[previous_application["NAME_CONTRACT_STATUS"] == "Cancelled"]["DAYS_DECISION"].mean(),color="b",
            linestyle="dashed",label="cancelled_average")
plt.axhline(previous_application[previous_application["NAME_CONTRACT_STATUS"] == "Unused offer"]["DAYS_DECISION"].mean(),color="y",
            linestyle="dashed",label="un used_average")
plt.legend(loc="best")

plt.title("Contract status relative to decision made about previous application.")
#plt.show()
plt.savefig('Output/ContractStatusRelativetoPreviousApplication.png')


# # Client payment methods & reasons for application rejections
# >NAME_PAYMENT_TYPE - Payment method that client chose to pay for the previous application.
# >CODE_REJECT_REASON - Why was the previous application rejected.
# >Around 81% of rejected applications the reason is XAP.
# >62% of chose to pay through cash by bank for previous applications.

# In[ ]:

plt.figure(figsize=(8,12))
plt.subplot(211)
rej = previous_application["CODE_REJECT_REASON"].value_counts().reset_index()
ax = sns.barplot("CODE_REJECT_REASON","index",data=rej[:6],palette="husl")
for i,j in enumerate(np.around((rej["CODE_REJECT_REASON"][:6].values*100/(rej["CODE_REJECT_REASON"][:6].sum())))):
    ax.text(.7,i,j,weight="bold")
plt.xlabel("percentage")
plt.ylabel("CODE_REJECT_REASON")
plt.title("Reasons for application rejections")

plt.subplot(212)
pay = previous_application["NAME_PAYMENT_TYPE"].value_counts().reset_index()
ax1 = sns.barplot("NAME_PAYMENT_TYPE","index",data=pay,palette="husl")
for i,j in enumerate(np.around((pay["NAME_PAYMENT_TYPE"].values*100/(pay["NAME_PAYMENT_TYPE"].sum())))):
    ax1.text(.7,i,j,weight="bold")
plt.xlabel("percentage")
plt.ylabel("NAME_PAYMENT_TYPE")
plt.title("Clients payment methods")
plt.subplots_adjust(hspace = .3)
plt.savefig('Output/RejectionReasons.png')

# # Distribution in Client suite type & client type.
# >NAME_TYPE_SUITE - Who accompanied client when applying for the previous application.
# >NAME_CLIENT_TYPE - Was the client old or new client when applying for the previous application.
# >about 60% clients are un-accompained when applying for loans.
# >73% clients are old clients

# In[ ]:

plt.figure(figsize=(13,6))
plt.subplot(121)
previous_application["NAME_TYPE_SUITE"].value_counts().plot.pie(autopct = "%1.1f%%",fontsize=8,
                                                             colors = sns.color_palette("inferno"),
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)
circ = plt.Circle((0,0),.7,color="white")
plt.gca().add_artist(circ)
plt.title("NAME_TYPE_SUITE")

plt.subplot(122)
previous_application["NAME_CLIENT_TYPE"].value_counts().plot.pie(autopct = "%1.1f%%",fontsize=8,
                                                             colors = sns.color_palette("inferno"),
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)
circ = plt.Circle((0,0),.7,color="white")
plt.gca().add_artist(circ)
plt.title("NAME_CLIENT_TYPE")
#plt.show()
plt.savefig('Output/DistributionClientSuitandClient.png')


# # popular goods for applying loans
# >NAME_GOODS_CATEGORY - What kind of goods did the client apply for in the previous application.
# >XNA ,Mobiles ,Computers and consumer electronics are popular goods for applying loans

# In[ ]:

goods = previous_application["NAME_GOODS_CATEGORY"].value_counts().reset_index()
goods["percentage"] = round(goods["NAME_GOODS_CATEGORY"]*100/goods["NAME_GOODS_CATEGORY"].sum(),2)
fig = plt.figure(figsize=(12,5))
ax = sns.pointplot("index","percentage",data=goods,color="yellow")
plt.xticks(rotation = 80)
plt.xlabel("NAME_GOODS_CATEGORY")
plt.ylabel("percentage")
plt.title("popular goods for applying loans")
ax.set_facecolor("k")
fig.set_facecolor('lightgrey')
plt.savefig('Output/PopularGoodsforLoans.png')

# # Previous applications portfolio and product types
# NAME_PORTFOLIO - Was the previous application for CASH, POS, CAR, …
# NAME_PRODUCT_TYPE - Was the previous application x-sell o walk-in.

# In[ ]:

plt.figure(figsize=(12,6))
plt.subplot(121)
previous_application["NAME_PORTFOLIO"].value_counts().plot.pie(autopct = "%1.1f%%",fontsize=8,
                                                             colors = sns.color_palette("prism",5),
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},
                                                               shadow =True)
plt.title("previous applications portfolio")
plt.subplot(122)
previous_application["NAME_PRODUCT_TYPE"].value_counts().plot.pie(autopct = "%1.1f%%",fontsize=8,
                                                             colors = sns.color_palette("prism",3),
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},
                                                                  shadow =True)
plt.title("previous applications product types")
#plt.show()
plt.savefig('Output/PreviousApplicationsPortfolio.png')


# # Approval,canceled and refusal rates by channel types.
# >CHANNEL_TYPE	- Through which channel we acquired the client on the previous application.
# >NAME_CONTRACT_STATUS- Contract status (approved, cancelled, ...) of previous application.
# >Channel types like Stone ,regional and country-wide have maximum approval rates.
# >Channel of coorporate sales have maximum refusal rate.
# >Credit-cash centres and Contact centres have maximum cancellation rates.

# In[ ]:

app = pd.crosstab(previous_application["CHANNEL_TYPE"],previous_application["NAME_CONTRACT_STATUS"])
app1 = app
app1["approval_rate"] = app1["Approved"]*100/(app1["Approved"]+app1["Refused"]+app1["Canceled"])
app1["refused_rate"]  = app1["Refused"]*100/(app1["Approved"]+app1["Refused"]+app1["Canceled"])
app1["cacelled_rate"] = app1["Canceled"]*100/(app1["Approved"]+app1["Refused"]+app1["Canceled"])
app2 = app[["approval_rate","refused_rate","cacelled_rate"]]
ax = app2.plot(kind="barh",stacked=True,figsize=(10,7))
ax.set_facecolor("k")
ax.set_xlabel("percentage")
ax.set_title("approval,cancel and refusal rates by channel types")
#plt.show()
plt.savefig('Output/RatesbyChannel.png')


# # Highest amount credited seller areas and industries.
# >SELLERPLACE_AREA - Selling area of seller place of the previous application.
# >NAME_SELLER_INDUSTRY - The industry of the seller.

# In[ ]:

fig = plt.figure(figsize=(13,5))
plt.subplot(121)
are = previous_application.groupby("SELLERPLACE_AREA")["AMT_CREDIT"].sum().reset_index()
are = are.sort_values(by ="AMT_CREDIT",ascending = False)
ax = sns.barplot(y= "AMT_CREDIT",x ="SELLERPLACE_AREA",data=are[:15],color="r")
ax.set_facecolor("k")
ax.set_title("Highest amount credited seller place areas")

plt.subplot(122)
sell = previous_application.groupby("NAME_SELLER_INDUSTRY")["AMT_CREDIT"].sum().reset_index().sort_values(by = "AMT_CREDIT",ascending = False)
ax1=sns.barplot(y = "AMT_CREDIT",x = "NAME_SELLER_INDUSTRY",data=sell,color="b")
ax1.set_facecolor("k")
ax1.set_title("Highest amount credited seller industrys")
plt.xticks(rotation=90)
plt.subplots_adjust(wspace = .5)
fig.set_facecolor("lightgrey")
plt.savefig('Output/HighestAmountCreditedAreasandIndustries.png')

# # Popular terms of previous credit at application.
# >CNT_PAYMENT - Term of previous credit at application of the previous application.
# >popular term of previous credit are 6months ,10months ,1year ,2years & 3 years.

# In[ ]:

plt.figure(figsize=(13,5))
ax = sns.countplot(previous_application["CNT_PAYMENT"],palette="Set1",order=previous_application["CNT_PAYMENT"].value_counts().index)
ax.set_facecolor("k")
plt.xticks(rotation = 90)
plt.title("popular terms of previous credit at application")
#plt.show()
plt.savefig('Output/PopularTermsofPreviousCredit.png')


# # Detailed product combination of the previous application

# In[ ]:

plt.figure(figsize=(10,8))
sns.countplot(y = previous_application["PRODUCT_COMBINATION"],order=previous_application["PRODUCT_COMBINATION"].value_counts().index)
plt.title("Detailed product combination of the previous application -count")
#plt.show()
plt.savefig('Output/DetailedCombinationofPreviousApplication.png')


# # Frequency distribution of intrest rates and client insurance requests
# >NAME_YIELD_GROUP  - Grouped interest rate into small medium and high of the previous application.
# >NFLAG_INSURED_ON_APPROVAL - Did the client requested insurance during the previous application.

# In[ ]:

plt.figure(figsize=(12,6))
plt.subplot(121)
previous_application["NFLAG_INSURED_ON_APPROVAL"].value_counts().plot.pie(autopct = "%1.1f%%",fontsize=8,
                                                             colors = sns.color_palette("prism",4),
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)
circ = plt.Circle((0,0),.7,color="white")
plt.gca().add_artist(circ)
plt.title("client requesting insurance")

plt.subplot(122)
previous_application["NAME_YIELD_GROUP"].value_counts().plot.pie(autopct = "%1.1f%%",fontsize=8,
                                                             colors = sns.color_palette("prism",4),
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)
circ = plt.Circle((0,0),.7,color="white")
plt.gca().add_artist(circ)
plt.title("interest rates")
#plt.show()
plt.savefig('Output/FrequencyDistributionofIRandInsurance.png')


# # Days variables - Relative to application date of current application
# >DAYS_FIRST_DRAWING - Relative to application date of current application when was the first disbursement of the previous application.
# >DAYS_FIRST_DUE - Relative to application date of current application when was the first due supposed to be of the previous application.
# >DAYS_LAST_DUE_1ST_VERSION - Relative to application date of current application when was the first due of the previous application.
# >DAYS_LAST_DUE -Relative to application date of current application when was the last due date of the previous application.
# >DAYS_TERMINATION - Relative to application date of current application when was the expected termination of the previous application.

# In[ ]:

cols = ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION','DAYS_LAST_DUE', 'DAYS_TERMINATION']
plt.figure(figsize=(12,6))
sns.heatmap(previous_application[cols].describe()[1:].transpose(),
            annot=True,linewidth=2,linecolor="k") # ,cmap=sns.color_palette("inferno")
#plt.show()
plt.savefig('Output/VariableDaysRelativetoApplication.png')


# # Days variables - Relative to application date of current application
# >DAYS_FIRST_DRAWING - Relative to application date of current application when was the first disbursement of the previous application.
# >DAYS_FIRST_DUE - Relative to application date of current application when was the first due supposed to be of the previous application.
# >DAYS_LAST_DUE_1ST_VERSION - Relative to application date of current application when was the first due of the previous application.
# >DAYS_LAST_DUE -Relative to application date of current application when was the last due date of the previous application.
# >DAYS_TERMINATION - Relative to application date of current application when was the expected termination of the previous application.

# In[ ]:

cols = ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION','DAYS_LAST_DUE', 'DAYS_TERMINATION']
length = len(cols)
cs = ["r","g","b","c","m"]

plt.figure(figsize=(13,6))
for i,j,k in itertools.zip_longest(cols,range(length),cs):
    ax = sns.distplot(previous_application[previous_application[i].notnull()][i],color=k,label=i)
    plt.legend(loc="best")
    ax.set_facecolor("k")
    plt.xlabel("days")
    plt.title("Days variables - Relative to application date of current application")
plt.savefig('Output/VariableDaysRelativetoApplication2.png')

# # Month of balance relative to application date
# >MONTHS_BALANCE -	Month of balance relative to application date (-1 means the information to the freshest monthly snapshot, 0 means the information at application - often it will be the same as -1 as many banks are not updating the information to Credit Bureau regularly ).

# In[ ]:

plt.figure(figsize=(14,6))
ax = sns.countplot(pos_cash_balance["MONTHS_BALANCE"],palette="rainbow")
plt.xticks(rotation = 90,fontsize=8)
plt.title("frequency distribution in Month of balance relative to application date for previous applications")
ax.set_facecolor("k")
#plt.show()
plt.savefig('Output/BalanceRelativetoApplication.png')


# # frequency of Past and Future installments
# >CNT_INSTALMENT - Term of previous credit (can change over time).
# >CNT_INSTALMENT_FUTURE - Installments left to pay on the previous credit.

# In[ ]:

fig = plt.figure(figsize=(12,7))
plt.subplot(122)
sns.countplot(pos_cash_balance["CNT_INSTALMENT"],
              order=pos_cash_balance["CNT_INSTALMENT"].value_counts().index[:10],palette="husl")
plt.title("Term of previous credit")
plt.subplot(121)
sns.countplot(pos_cash_balance["CNT_INSTALMENT_FUTURE"],
              order=pos_cash_balance["CNT_INSTALMENT_FUTURE"].value_counts().index[:10],palette="husl")
plt.title("Installments left to pay on the previous credit.")
fig.set_facecolor("lightgrey")
plt.savefig('Output/FrequencyPastandFutureInstallments.png')

# # Total prescribed installment amount and total amount actually paid to client on previous credits.
# >AMT_INSTALMENT - What was the prescribed installment amount of previous credit on this installment.
# >AMT_PAYMENT - What the client actually paid on previous credit on this installment.
# >X-axis  - total installment amount prescribed by current id on previous applications.
# >Y-axis  - total  amount client actually paid to  current id on previous applications.
# >color - Number of previous applications by current id.

# In[ ]:

amt_ins = installments_payments.groupby("SK_ID_CURR").agg({"AMT_INSTALMENT":"sum","AMT_PAYMENT":"sum","SK_ID_PREV":"count"}).reset_index()

plt.figure(figsize = (12,8))
plt.scatter(amt_ins["AMT_INSTALMENT"],amt_ins["AMT_PAYMENT"],
            c=amt_ins["SK_ID_PREV"],edgecolor="k",cmap="viridis",s=60)
plt.colorbar()
plt.xlabel("AMT_INSTALMENT")
plt.ylabel("AMT_PAYMENT")
plt.title("Total prescribed installment amount and total amount actually paid to client on previous credits.")
#plt.show()
plt.savefig('Output/PrescribedInstallmentonPreviousCredits.png')

# # Average days previous credit was supposed to be paid and credit actually paid by current ids.
# >DAYS_INSTALMENT - When the installment of previous credit was supposed to be paid (relative to application date of current loan).
# >DAYS_ENTRY_PAYMENT - When was the installments of previous credit paid actually (relative to application date of current loan).
#
# >X-axis  - average days by current id to be paid.
# >Y-axis  - average days by current id actually  paid.

# In[ ]:

days_ins = installments_payments.groupby("SK_ID_CURR")[["DAYS_ENTRY_PAYMENT","DAYS_INSTALMENT"]].mean().reset_index()
days_ins = days_ins[days_ins["DAYS_ENTRY_PAYMENT"].notnull()]
plt.figure(figsize=(12,9))
plt.hist2d(days_ins["DAYS_ENTRY_PAYMENT"],days_ins["DAYS_INSTALMENT"],bins=(20,20),cmap="hot")
plt.colorbar()
plt.xlabel("AVERAGE_DAYS_ENTRY_PAYMENT")
plt.ylabel("AVERAGE_DAYS_INSTALMENT")
plt.title("density plot between Average days previous credit was supposed to be paid and credit actually paid by current ids.")
#plt.show()
plt.savefig('Output/AverageDaysCreditPayed.png')


# # Credit card balances data
# >AMT_BALANCE - Balance during the month of previous credit .
# >AMT_CREDIT_LIMIT_ACTUAL - Credit card limit during the month of the previous credit.
# >AMT_INST_MIN_REGULARITY - Minimal installment for this month of the previous credit.
# >AMT_RECEIVABLE_PRINCIPAL - Amount receivable for principal on the previous credit.
# >AMT_RECIVABLE - Amount receivable on the previous credit.
# >AMT_TOTAL_RECEIVABLE - Total amount receivable on the previous credit.

# In[ ]:

cols = ['AMT_BALANCE','AMT_CREDIT_LIMIT_ACTUAL', 'AMT_INST_MIN_REGULARITY',
       'AMT_RECEIVABLE_PRINCIPAL', 'AMT_RECIVABLE', 'AMT_TOTAL_RECEIVABLE']

length = len(cols)
cs = ["r","g","b","c","m","y"]

fig = plt.figure(figsize=(13,14))
fig.set_facecolor("lightgrey")

for i,j,k in itertools.zip_longest(cols,range(length),cs):
    plt.subplot(3,2,j+1)
    ax = sns.distplot(credit_card_balance[credit_card_balance[i].notnull()][i],color=k)
    ax.set_facecolor("k")
    plt.xlabel("")
    plt.title(i)
plt.savefig('Output/CreditCardBalance.png')

# In[ ]:

application_train["type"] = "train"
application_test["type"]  = "test"
#conactenating train & test data
data = pd.concat([application_train,application_test],axis=0)


# In[ ]:

#Removing columns with missing values more than 40%
missing_cols = [ 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
       'APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG',
       'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG',
       'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG',
       'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG',
       'NONLIVINGAREA_AVG', 'APARTMENTS_MODE', 'BASEMENTAREA_MODE',
       'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE', 'COMMONAREA_MODE',
       'ELEVATORS_MODE', 'ENTRANCES_MODE', 'FLOORSMAX_MODE', 'FLOORSMIN_MODE',
       'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE',
       'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE', 'APARTMENTS_MEDI',
       'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI',
       'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI', 'FLOORSMAX_MEDI',
       'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI',
       'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI',
       'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'TOTALAREA_MODE',
       'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE',"OWN_CAR_AGE","OCCUPATION_TYPE"]

data_new  = data[[i for i in data.columns if i not in missing_cols]]


# In[ ]:

#Separating numberical and categorical columns
obj_dtypes = [i for i in data_new.select_dtypes(include=np.object).columns if i not in ["type"] ]
num_dtypes = [i for i in data_new.select_dtypes(include = np.number).columns if i not in ['SK_ID_CURR'] + [ 'TARGET']]


# In[ ]:

#MISSING values treatment
amt_cs = ["AMT_ANNUITY","AMT_GOODS_PRICE"]
for i in amt_cs:
    data_new[i] = data_new.groupby("type").transform(lambda x:x.fillna(x.mean()))

enq_cs =['AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_HOUR',
       'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT',
       'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_YEAR']
for i in enq_cs:
    data_new[i] = data_new[i].fillna(0)

cols = ["DEF_30_CNT_SOCIAL_CIRCLE","DEF_60_CNT_SOCIAL_CIRCLE","OBS_30_CNT_SOCIAL_CIRCLE",
        "OBS_60_CNT_SOCIAL_CIRCLE","NAME_TYPE_SUITE","CNT_FAM_MEMBERS",
       "DAYS_LAST_PHONE_CHANGE","DAYS_LAST_PHONE_CHANGE"]
for i in cols :
    data_new[i]  = data_new[i].fillna(data_new[i].mode()[0])


# In[ ]:

#Label encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for i in obj_dtypes:
    data_new[i] = le.fit_transform(data_new[i])


# In[ ]:

#one hot encoding for categorical variables
data_new = pd.get_dummies(data=data_new,columns=obj_dtypes)


# In[ ]:

#splitting new train and test data
application_train_newdf = data_new[data_new["type"] == "train"]
application_test_newdf  = data_new[data_new["type"] == "test"]


# In[ ]:

#splitting application_train_newdf into train and test
from sklearn.model_selection import train_test_split
train,test = train_test_split(application_train_newdf,test_size=.3,random_state = 123)

train = train.drop(columns="type",axis=1)
test  = test.drop(columns="type",axis=1)

#seperating dependent and independent variables
train_X = train[[i for i in train.columns if i not in ['SK_ID_CURR'] + [ 'TARGET']]]
train_Y = train[["TARGET"]]

test_X  = test[[i for i in test.columns if i not in ['SK_ID_CURR'] + [ 'TARGET']]]
test_Y  = test[["TARGET"]]


# In[ ]:

# Up-sample Minority Class
from sklearn.utils import resample

#separating majority and minority classes
df_majority = train[train["TARGET"] == 0]
df_minority = train[train["TARGET"] == 1]

#upsample minority data
df_minority_upsampled = resample(df_minority,
                                 replace=True,
                                 n_samples =197969,
                                 random_state=123)

df_upsampled = pd.concat([df_majority,df_minority_upsampled],axis=0)

#splitting dependent and independent variables
df_upsampled_X = df_upsampled[[i for i in df_upsampled.columns if i not in ['SK_ID_CURR'] + [ 'TARGET']]]
df_upsampled_Y = df_upsampled[["TARGET"]]


# In[ ]:

# Down-sample Majority Class
from sklearn.utils import resample

#separating majority and minority classes
df_majority = train[train["TARGET"] == 0]
df_minority = train[train["TARGET"] == 1]

df_majority_downsampled = resample(df_majority,
                                   replace=False,
                                   n_samples=17288,
                                   random_state=123)

df_downsampled = pd.concat([df_minority,df_majority_downsampled],axis=0)

#splitting dependent and independent variables

df_downsampled_X = df_downsampled[[i for i in df_downsampled.columns if i not in ['SK_ID_CURR'] + [ 'TARGET']]]
df_downsampled_Y = df_downsampled[["TARGET"]]


# In[ ]:

from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,roc_auc_score,classification_report,roc_auc_score,roc_curve,auc

#Model function
def model(name,algorithm,dtrain_X,dtrain_Y,dtest_X,dtest_Y,cols=None):

    algorithm.fit(dtrain_X[cols],dtrain_Y)
    predictions = algorithm.predict(dtest_X[cols])
    print (algorithm)

    print ("Accuracy score : ", accuracy_score(predictions,dtest_Y))
    print ("Recall score   : ", recall_score(predictions,dtest_Y))
    print ("classification report :\n",classification_report(predictions,dtest_Y))

    fig = plt.figure(figsize=(10,8))
    ax  = fig.add_subplot(111)
    prediction_probabilities = algorithm.predict_proba(dtest_X[cols])[:,1]
    fpr , tpr , thresholds   = roc_curve(dtest_Y,prediction_probabilities)
    ax.plot(fpr,tpr,label   = ["Area under curve : ",auc(fpr,tpr)],linewidth=2,linestyle="dotted")
    ax.plot([0,1],[0,1],linewidth=2,linestyle="dashed")
    plt.legend(loc="best")
    plt.title("ROC-CURVE & AREA UNDER CURVE")
    ax.set_facecolor("k")
    plt.savefig('Output/ROCCurve'+ name +'.png')


# # LogisticRegression

# In[ ]:

from sklearn.linear_model import LogisticRegression
logit = LogisticRegression()
model('LogisticRegression',logit,df_downsampled_X,df_downsampled_Y,test_X,test_Y,df_downsampled_X.columns)


# # Random Forest Classifier

# In[ ]:

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
model('RandomForestClassifier',rfc,df_downsampled_X,df_downsampled_Y,test_X,test_Y,df_downsampled_X.columns)


# # Decision Tree Classifier

# In[ ]:

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
model('DecisionTreeClassifier',dtc,df_downsampled_X,df_downsampled_Y,test_X,test_Y,df_downsampled_X.columns)


# # Gaussian Naive Bayes

# In[ ]:

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
model('GaussianNB',gnb,df_downsampled_X,df_downsampled_Y,test_X,test_Y,df_downsampled_X.columns)


# # XGBoost Classifier

# In[ ]:

from xgboost import XGBClassifier
xgb = XGBClassifier()
model('XGBoost',xgb,df_downsampled_X,df_downsampled_Y,test_X,test_Y,df_downsampled_X.columns)


# # Gradient Boosting Classifier

# In[ ]:

from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
model('GradientBoostingClassifier',gbc,df_downsampled_X,df_downsampled_Y,test_X,test_Y,df_downsampled_X.columns)


# In[ ]:

test_sub_X = application_test_newdf[[i for i in df_downsampled.columns if i not in ['SK_ID_CURR'] + [ 'TARGET']]]
xgb1 = XGBClassifier()
xgb1.fit(df_downsampled_X,df_downsampled_Y)
sub_prob = xgb1.predict_proba(test_sub_X)[:,1]
sub_prob = pd.DataFrame(sub_prob)
ids = application_test[["SK_ID_CURR"]]
subm  = ids.merge(sub_prob,left_index=True,right_index=True,how="left")
sample_submission  = subm.rename(columns={"SK_ID_CURR":'SK_ID_CURR',0:'TARGET'})


# In[ ]:

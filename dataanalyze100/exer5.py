import pandas as pd

from dateutil.relativedelta import relativedelta

from sklearn.tree import DecisionTreeClassifier
import sklearn.model_selection

from sklearn import tree
import matplotlib.pyplot as plt
import japanize_matplotlib

#==========================================================

customer = pd.read_csv('./data/customer_join2.csv')
print(customer.head())

uselog_months = pd.read_csv('./data/use_log_months.csv')
print(uselog_months.head())

year_months = list(uselog_months["年月"].unique())
uselog = pd.DataFrame()

for i in range(1, len(year_months)):
    tmp = uselog_months.loc[ uselog_months["年月"]==year_months[i] ].copy()
    tmp.rename(columns={"count":"count_0"}, inplace=True)
    tmp_before = uselog_months.loc[uselog_months["年月"]==year_months[i-1]].copy()
    del tmp_before["年月"]
    tmp_before.rename(columns={"count":"count_1"}, inplace=True)
    tmp = pd.merge(tmp, tmp_before, on="customer_id", how="left")
    uselog = pd.concat([uselog, tmp], ignore_index=True)

print( len(uselog) )
print(uselog.head())

print("=====================================================")

exit_customer = customer.loc[ customer["is_deleted"]==1 ].copy()
exit_customer["exit_date"] = None

exit_customer["end_date"] = pd.to_datetime(exit_customer["end_date"])

for i in exit_customer.index:
    exit_customer.loc[i, "exit_date"] = exit_customer.loc[i, "end_date"] - relativedelta(months=1)

exit_customer["exit_date"] = pd.to_datetime(exit_customer["exit_date"])
exit_customer["年月"] = exit_customer["exit_date"].dt.strftime("%Y%m")
uselog["年月"] = uselog["年月"].astype(str)

print(exit_customer.head())

exit_uselog = pd.merge(uselog, exit_customer, on=["customer_id","年月"], how="left")

print( "uselog length=" , len(uselog) )
print( "exit_uselog length=" , len(exit_uselog) )

print( exit_uselog.head() )

exit_uselog = exit_uselog.dropna(subset=["name"])

print( "exit_uselog length=" , len(exit_uselog) )
print( exit_uselog["customer_id"].unique() )

print( exit_uselog.head() )

print("=====================================================")

conti_customer = customer.loc[ customer["is_deleted"]==0 ]
conti_uselog = pd.merge( uselog, conti_customer, on=["customer_id"], how="left" )
print( len(conti_uselog) )
conti_uselog = conti_uselog.dropna(subset=["name"])
print( len(conti_uselog) )

conti_uselog = conti_uselog.sample( frac=1, random_state=0).reset_index(drop=True)
conti_uselog = conti_uselog.drop_duplicates(subset="customer_id")
print( len(conti_uselog) )

print( conti_uselog.head() )

predict_data = pd.concat([conti_uselog, exit_uselog], ignore_index=True)
print( len(predict_data) )
print( predict_data.head() )
print( predict_data.tail() )

print("=====================================================")

predict_data["period"] = 0
predict_data["now_date"] = pd.to_datetime( predict_data["年月"], format="%Y%m")
predict_data["start_date"] = pd.to_datetime(predict_data["start_date"])

for i in range( len(predict_data) ):
    delta = relativedelta( predict_data.loc[ i, "now_date"], predict_data.loc[ i, "start_date"] )
    predict_data.loc[i, "period"] = int( delta.years*12 + delta.months )
print( predict_data.head() )
print( predict_data.tail() )

print("=====================================================")

print( predict_data.isna().sum() )

predict_data = predict_data.dropna( subset=["count_1"] )
print( predict_data.isna().sum() )

print("=====================================================")

target_col = ["campaign_name","class_name","gender","count_1","routine_flg","period","is_deleted"]
predict_data = predict_data[target_col]
print( predict_data.head() )

predict_data = pd.get_dummies(predict_data)
print( predict_data.head() )

del predict_data["campaign_name_通常"]
del predict_data["class_name_ナイト"]
del predict_data["gender_M"]
print( predict_data.head() )

print("=====================================================")

exit = predict_data.loc[ predict_data["is_deleted"]==1 ]
conti = predict_data.loc[ predict_data["is_deleted"]==0 ].sample( len(exit), random_state=0 )

X = pd.concat( [exit, conti], ignore_index=True )
y = X["is_deleted"]
del X["is_deleted"]
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=0)

#model = DecisionTreeClassifier(random_state=0)
model = DecisionTreeClassifier(random_state=0, max_depth=5)
model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)
print(y_test_pred)

print("=====================================================")

results_test = pd.DataFrame({"y_test":y_test,"y_pred":y_test_pred})
print( results_test.head() )

correct = len( results_test.loc[ results_test["y_test"]==results_test["y_pred"] ] )
data_count = len( results_test )
print( correct/data_count )


print( model.score(X_test, y_test) )
print( model.score(X_train, y_train) )

print("=====================================================")

importance = pd.DataFrame( {"feature_names":X.columns, "coefficient":model.feature_importances_} )
print( importance )

plt.figure( figsize=(20,8) )
tree.plot_tree( model, feature_names=X.columns, fontsize=8 )

plt.savefig("exer5.png")

print("=====================================================")

count_1 = 3
routine_flg = 1
period = 10
campaign_name = "入会費無料"
class_name = "オールタイム"
gender = "M"

if campaign_name == "入会費半額":
    campaign_name_list = [1, 0]
elif campaign_name == "入会費無料":
    campaign_name_list = [0, 1]
elif campaign_name == "通常":
    campaign_name_list = [1, 1]

if class_name == "オールタイム":
    class_name_list = [1,0]
elif class_name == "デイタイム":
    class_name_list = [0,1]
elif class_name == "ナイト":
    class_name_list = [0,0]    
    
if gender == "F":
    gender_list = [1]
elif gender == "M":
    gender_list = [0]
    
input_data = [count_1, routine_flg, period]
input_data.extend(campaign_name_list)
input_data.extend(class_name_list)
input_data.extend(gender_list)
input_data = pd.DataFrame(data=[input_data], columns=X.columns)

print( model.predict(input_data) )
print( model.predict_proba(input_data) )
    
print("=====================================================")



    

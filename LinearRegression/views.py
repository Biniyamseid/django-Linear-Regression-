
from django.http import JsonResponse
# from sklearn.externals import joblib
import pandas as pd
import pickle
import pickle
import pandas as pd
import json
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import boxcox
from django.views.decorators.csrf import csrf_exempt


import json
import pandas as pd
from sklearn.linear_model import LinearRegression
from django.http import JsonResponse
from scipy.stats import boxcox
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import sklearn



from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def predict(request):
    # Load the data
    df = pd.read_csv('LinearRegression/data/data.csv')
    
    # Convert categorical variables into dummy/indicator variables
    # df = pd.get_dummies(df, drop_first=True)
    # Select columns with numerical values
    numerical_columns = df.select_dtypes(include=['number'])
    corr = numerical_columns.corr()

    #dummy variable
    catagorical_columns = ["sex","children","smoker","region"]
    df_encode = pd.get_dummies(data=df,columns = catagorical_columns,drop_first = True,dtype = 'int8')

    y_bc,lam,ci = boxcox(df_encode['expenses'],alpha = 0.05)
    df_encode['expenses'] = np.log(df_encode['expenses'])

    # Train the model
    X = df.drop('expenses', axis=1)
    y = df['expenses']
    model = LinearRegression().fit(X, y)

    X = df_encode.drop('expenses',axis=1)
    Y = df_encode['expenses']
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state=23)

    lin_reg = LinearRegression()
    lin_reg.fit(X_train,Y_train)
    sk_theta = [lin_reg.intercept_]+list(lin_reg.coef_)


    # Get the JSON data from the request
    data = json.loads(request.body)

    # Convert categorical variables in the input data into dummy/indicator variables
    data = pd.get_dummies(pd.DataFrame(data, index=[0]), drop_first=True).to_dict(orient='records')[0]


    # Make the prediction
    prediction = model.predict(data)

    # Return the prediction
    return JsonResponse({'prediction': prediction.tolist()})
@csrf_exempt
def train(request):
    # Load the data
    df = pd.read_csv('LinearRegression/data/data.csv')
    
    # Convert categorical variables into dummy/indicator variables
    # df = pd.get_dummies(df, drop_first=True)
    # Select columns with numerical values
    numerical_columns = df.select_dtypes(include=['number'])
    corr = numerical_columns.corr()

    #dummy variable
    catagorical_columns = ["sex","children","smoker","region"]
    df_encode = pd.get_dummies(data=df,columns = catagorical_columns,drop_first = True,dtype = 'int8')

    y_bc,lam,ci = boxcox(df_encode['expenses'],alpha = 0.05)
    df_encode['expenses'] = np.log(df_encode['expenses'])

    # Train the model
    X = df_encode.drop('expenses',axis=1)
    Y = df_encode['expenses']
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state=23)
    lin_reg = LinearRegression()
    lin_reg.fit(X_train,Y_train)
    sk_theta = [lin_reg.intercept_]+list(lin_reg.coef_)
    y_pred_sk = lin_reg.predict(X_test)


    j_mse_sk = mean_squared_error(y_pred_sk,Y_test)
    R_square_sk = lin_reg.score(X_test,Y_test)
        # Print some sample values from X_test as JSON
    sample_values = X_test.sample(5).to_dict(orient='records')

    print('The Mean Square Error(MSE) or J(theta) is: ',j_mse_sk)
    print('R square obtain for scikit learn library is :',R_square_sk)
    # Print some sample values from X_test as JSON
    print(X_test.sample(5).to_json(orient='records'))
    # joblib.dump(lin_reg, './trained_model.pkl')
    # Save the trained model to a file
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(lin_reg, f)
    return JsonResponse({'The Mean Square Error(MSE) or J(theta) is: ': j_mse_sk,
                         'sample x_test values':sample_values
                         })



@csrf_exempt
def make_prediction(request):
    # Load the trained model
    # model = joblib.load('trained_model.pkl')

    # Load the trained model
    with open('trained_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Get the JSON data from the request
    data = json.loads(request.body)

    # Convert categorical variables in the input data into dummy/indicator variables
    data = pd.get_dummies(pd.DataFrame(data, index=[0]), drop_first=True).to_dict(orient='records')[0]

    # Make the prediction
    prediction = model.predict(data)

    # Return the prediction
    return JsonResponse({'prediction': prediction.tolist()})






@csrf_exempt
def prediction2(request):
    data = json.loads(request.body)
    print(data)
    with open('trained_model.pkl', 'rb') as f:
        mod = pickle.load(f)

    df = pd.DataFrame([data])
   # Define the correct order of the features
    columns_order = ['age', 'bmi', 'sex_male', 'children_1', 'children_2', 'children_3', 'children_4', 'children_5', 'smoker_yes', 'region_northwest', 'region_southeast', 'region_southwest']

    df = pd.DataFrame([data])
    df['sex_male'] = (df['sex'] == 'male').astype(int)
    df['smoker_yes'] = (df['smoker'] == 'yes').astype(int)
    df['region_northwest'] = (df['region'] == 'northwest').astype(int)
    df['region_southeast'] = (df['region'] == 'southeast').astype(int)
    df['region_southwest'] = (df['region'] == 'southwest').astype(int)
    df['children_1'] = (df['children'] == 1).astype(int)
    df['children_2'] = (df['children'] == 2).astype(int)
    df['children_3'] = (df['children'] == 3).astype(int)
    df['children_4'] = (df['children'] == 4).astype(int)
    df['children_5'] = (df['children'] >= 5).astype(int)
    df.drop(columns=['sex', 'smoker', 'region', 'children'], inplace=True)

    # Reorder the columns to match the order the model was trained on
    df = df[columns_order]

    y_pred = mod.predict(df)
    print(y_pred)
    y_pred = y_pred.tolist()



   
   

    return JsonResponse({'The predicted expense is ': y_pred})






        
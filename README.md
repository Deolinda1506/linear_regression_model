
Task 2: Create an API
Create a function to make a prediction using a linear regression model, which is a Python function as follows. Use Fast API to create an API endpoint and upload source code files to a free hosting platform (or paid where possible)

#import things. . . …..
app = FastAPI(‘insert something here)
@app.post(‘/predict’)
def predict(*args, **kwargs):
 #insert your code here
return prediction
#replace *args and **kwargs where you deem necessary

The data I used is meant for predicting wine quality based on parameters:

fixed_acidity:
volatile_acidity:
residual_sugar:
chlorides:
free_SO2:
sulphates:
alcohol:
colour:
API Documentation
Link = render fastapi hosted link

The fastapi swagger shows the following:

Use the API endpoint created in task 2,that is, url + path_to_predict on your flutter app.(https://linear-regression-model-13.onrender.com/predict)

This is how the API works:

    GET/ class
    Content type
    application/json
    null

   GET /
   Content type
   application/json
   null

   POST /predict
   Content type
   application/json
   {
  "fixed_acidity": 999,
  "volatile_acidity": 999,
  "residual_sugar": 999,
  "chlorides": 999,
  "free_SO2": 999,
  "sulphates": 999,
  "alcohol": 999,
  "colour": 1
   }
   null

  {
  "predicted Quality": 
  }

Task 3: Flutter App
Instruction of the flutter app creation are:

TextFields for inputting values needed for the prediction.
A Button with the text "Predict".
A display area for showing the predicted value or an error message if the values are out of range or if one or more expected values are missing.
Make sure you reload the endpoint server before opening the Flutter app.

Then test the 8 inputs in the test fields.

Press predict to get the quality of the wine.

Submission Details
A GitHub link containing the notebook, API code files, and Flutter app with directories well labeled ** Empty cell outputs on the notebook will be considered failed run outputs**
On the README :
Provide a publicly available API endpoint that returns predictions given input values. Tests will be assessed using Postman; alternatively, you can provide steps to access the API.
Contributing
Make a pull request before contributing

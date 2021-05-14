from flask import Flask, render_template,request
from flasgger import Swagger
import pickle
import pandas as pd

app = Flask(__name__)
test_df = pd.read_csv("test_dataset.csv")
pickle_file = open('Credit_risk.pkl','rb')
classifier = pickle.load(pickle_file)
Swagger(app)

@app.route("/")
def base_route():
    return "Welcome to Credit Risk Prediction API",200

@app.route("/predictForSample",methods=['GET'])
def predictRate():
    """Swagger App for Credit Risk Prediction
    --------
    parameters:
    -   name: ExistingCreditsCount
        description: Waht is the existing credit count
        in: query
        type: integer
        required: true
    -   name: CurrentResidenceDuration
        description : Curresnt resident duration
        in: query
        type: integer
        required: true
    -   name: Age
        description : Age of the customer
        in: query
        type: integer
        required: true
    -   name: Dependents
        description : How amny dependents are there
        in: query
        type: integer
        required: true
    -   name: LoanDuration
        description : Duration of previous loan
        in: query
        type: integer
        required: true
    -   name: LoanAmount
        description : Loan amount for previous loan
        in: query
        type: integer
        required: true
    -   name: InstallmentPercent
        description : What was the percentage of installment amount
        in: query
        type: integer
        required: true
    -   name: Risk
        description : Target variable
        in: query
        type: integer
        required: true
    responses:
        200:
            description : Predicted for Sample Customers
        201:
            description : Predicted for file containing all Customers
    """

    ExistingCreditsCount = request.args.get("ExistingCreditsCoun")
    CurrentResidenceDuration = request.args.get("CurrentResidenceDuration")
    Age = request.args.get("Age")
    Dependents = request.args.get("Dependents")
    LoanDuration = request.args.get("LoanDuration")
    LoanAmount = request.args.get("LoanAmount")
    InstallmentPercent = request.args.get("InstallmentPercent")
    Risk = request.args.get("Risk")


    result = classifier.predict([[ExistingCreditsCount, CurrentResidenceDuration, Age, Dependents,
       LoanDuration, LoanAmount, InstallmentPercent, Risk]])

    if(result in [0.0,"0.0"]) : return "No Risk"
    if(result in [1.0,"1.0"]) : return "Risk"

if __name__ == "__main__":
    app.run(debug=True, host= "127.0.0.1", port= 5000)
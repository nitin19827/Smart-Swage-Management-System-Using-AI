from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score


app = Flask(__name__, template_folder='templates')


@app.route("/")
def index():
  return render_template('index.html')


@app.route("/pda")
def pda():
  return render_template('pda.html')

@app.route("/results", methods=['POST'])
def results():
  if request.method == 'POST':
    ph = float(request.form['ph'])
    Solids = float(request.form['Solids'])
    Hardness = float(request.form['Hardness'])
    Sulfate = float(request.form['Sulfate'])
    Chloramines = float(request.form['Chloramines'])
    Organic_carbon = float(request.form['Organic_carbon'])
    Potability = request.form['Potability']
    if Potability == 'Not_Suitable':
      Potability = 1
    else:
      Potability = 0


    dt = pickle.load(open('ML Models/decision_tree_model.pkl', 'rb'))
    knn = pickle.load(open('ML Models/knn_model.pkl', 'rb'))
    lr = pickle.load(open('ML Models/logistic_regression_model.pkl', 'rb'))
    rf = pickle.load(open('ML Models/random_forest_model.pkl', 'rb'))
    svm = pickle.load(open('ML Models/support_vector_machine.pkl', 'rb'))
    ada = pickle.load(open('ML Models/adaboost_classifier.pkl', 'rb'))
    xg = pickle.load(open('ML Models/xgboost_classifier.pkl', 'rb'))


    data = [[ph,Hardness,Solids,Chloramines,Sulfate,Organic_carbon]]

    dt_prediction = dt.predict(data)[0]
    knn_prediction = knn.predict(data)[0]
    lr_prediction = lr.predict(data)[0]
    rf_prediction = rf.predict(data)[0]
    svm_prediction = svm.predict(data)[0]
    ada_prediction = ada.predict(data)[0]
    xg_prediction = xg.predict(data)[0]

    y_true = [Potability]  
    dt_accuracy = accuracy_score([y_true], [dt_prediction])
    knn_accuracy = accuracy_score([y_true], [knn_prediction])
    lr_accuracy = accuracy_score([y_true], [lr_prediction])
    rf_accuracy = accuracy_score([y_true], [rf_prediction])
    svm_accuracy = accuracy_score([y_true], [svm_prediction])
    ada_accuracy = accuracy_score([y_true], [ada_prediction])
    xg_accuracy = accuracy_score([y_true], [xg_prediction])
    
    if dt_prediction == 1:
      dt_prediction = 'Blocking Occurred'
    else:
      dt_prediction = 'No Blocking Occurred'

    if lr_prediction == 1:
      lr_prediction = 'Blocking Occurred'
    else:
      lr_prediction = 'No Blocking Occurred'

    if knn_prediction == 1:
      knn_prediction = 'Blocking Occurred'
    else:
      knn_prediction = 'No Blocking Occurred'

    if rf_prediction == 1:
      rf_prediction = 'Blocking Occurred'
    else:
      rf_prediction = 'No Blocking Occurred'
    
    if svm_prediction == 1:
      svm_prediction = 'Blocking Occurred'
    else:
      svm_prediction = 'No Blocking Occurred'

    if ada_prediction == 1:
      ada_prediction = 'Blocking Occurred'
    else:
      ada_prediction = 'No Blocking Occurred'

    if xg_prediction == 1:
      xg_prediction = 'Blocking Occurred'
    else:
      xg_prediction = 'No Blocking Occurred'


    return render_template('results.html', dt_prediction=dt_prediction, knn_prediction=knn_prediction, lr_prediction=lr_prediction, rf_prediction=rf_prediction, svm_prediction=svm_prediction, ada_prediction=ada_prediction, xg_prediction=xg_prediction, dt_accuracy=dt_accuracy, knn_accuracy=knn_accuracy, rf_accuracy=rf_accuracy, lr_accuracy=lr_accuracy, svm_accuracy=svm_accuracy, ada_accuracy=ada_accuracy, xg_accuracy=xg_accuracy)



if __name__ == '__main__':
    app.run(port=5000, debug=True)
import pandas as pd
from werkzeug.utils import secure_filename
from flask import Flask,render_template,request,redirect,url_for,send_file
import module

app = Flask(__name__)
user_req = ''

reg = ['LinearRegression','RandomForestRegressor','DecisionTreeRegressor','SupportVectorRegressor']
cla = ['LogisticRegression','RandomForestClassifier','KNeighborsClassifier','DecisionTreeClassifier','SupportVectorClassifier']

@app.route('/')
def start():
    return render_template('index.html')

filename = ''
dependent = ''
independent = []


def detect_columns(name):
    df = pd.read_csv(name)
    return list(df.columns)


def clean_data():
    pass

@app.route('/file',methods = ['POST','GET'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        global filename 
        filename = str(f.filename)
        columns_detected = detect_columns(f.filename)
        columns = [i.replace(' ','_') for i in columns_detected]
        return render_template('dependent.html' ,value = columns)
    return render_template('index.html')
        

@app.route("/dependent", methods=["POST",'GET'])
def dependent():
    y = request.form["dependent"]
    columns_detected= detect_columns(filename)
    columns = [i.replace(' ','_') for i in columns_detected]
    columns.remove(y)
    global dependent
    dependent = y
    return render_template('independent.html',y = y,value = columns)

cla = ['LogisticRegression','RandomForestClassifier','KNeighborsClassifier','DecisionTreeClassifier','SupportVectorClassifier']

def classifier(csv_file,independent_variables,dependent_variables,test_size):
    accuracy = dict()
    accuracy['LogisticRegression'] = module.LogisticRegression_train(csv_file,independent_variables,dependent_variables,test_size)
    accuracy['RandomForestClassifier'] = module.RandomForestClassifier_train(csv_file,independent_variables,dependent_variables,test_size)
    accuracy['KNeighborsClassifier'] = module.KNeighborsClassifier_train(csv_file,independent_variables,dependent_variables,test_size)
    accuracy['DecisionTreeClassifier'] = module.DecisionTreeClassifier_train(csv_file,independent_variables,dependent_variables,test_size)
    accuracy['SupportVectorClassifier'] = module.SupportVectorClassifier_train(csv_file,independent_variables,dependent_variables,test_size)
    return accuracy

reg = ['LinearRegression','RandomForestRegressor','DecisionTreeRegressor','SupportVectorRegressor']

def regressor(csv_file,independent_variables,dependent_variables,test_size):
    accuracy = dict()
    accuracy['LinearRegression'] = module.LinearRegression_train(csv_file,independent_variables,dependent_variables,test_size)
    accuracy['RandomForestRegressor'] = module.RandomForestRegressor_train(csv_file,independent_variables,dependent_variables,test_size)
    accuracy['DecisionTreeRegressor'] = module.DecisionTreeRegressor_train(csv_file,independent_variables,dependent_variables,test_size)
    accuracy['SupportVectorRegressor'] = module.SupportVectorRegressor_train(csv_file,independent_variables,dependent_variables,test_size)
    return accuracy

@app.route("/independent",methods = ['POST','GET'])
def independent():
    num = 0
    y = list(request.form.getlist("independent"))
    x = [i.replace('_',' ') for i in y]
    model = request.form.get('difference')
    global independent 
    independent = x
    if model == 'classification':
        final = classifier(filename,independent,dependent,20)
        num = 5
        models = cla
    elif model == 'regression':
        final = regressor(filename,independent,dependent,20)
        num = 4
        models = reg
    else:
        pass
    return render_template('final.html',final= final,num = num,models = models)
    # return render_template('text.html',value = independent,model = model)


@app.route('/user',methods = ['POST','GET'])
def user():
    if request.method == 'POST':
        global user_req
        user_req = request.form.get('final-model')

        return render_template('output.html',user = user_req)
    
@app.route('/joblib_file')
def joblib_file():
    path = user_req + '-joblib'
    return send_file(path,as_attachment=True)


@app.route('/pickle_file')
def pickle_file():
    path = user_req + '.pkl'
    return send_file(path,as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
    
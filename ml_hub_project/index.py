from flask import Flask,render_template,request,jsonify 
from ml.classifier import Classifer
from ml.regressor import Regression
from ml.decision_tree_class import Tree
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/input")
def second():
    return render_template("input.html")


@app.route("/classifier")
def third():
    return render_template("classification.html")


@app.route("/decsiion_tree")
def fourth():
    return render_template("tree.html")


@app.route("/predict",methods=['GET','POST'])
def prediction():
    result = None
    if request.method == 'POST':        
        model = request.form.get("model")
        f1 = float(request.form.get("f1"))
        f2 = float(request.form.get("f2"))
        f3 = float(request.form.get("f3"))
        obj = Regression(f1,f2,f3)
        if model == "linear":
            result = obj.Linear_regression()
            result = abs(result)
        elif model == "logistic":
            result = obj.Logistic_Regression()
        elif model == "rf":
            result = obj.Random_Forest_Regressor()
            result = abs(result)
        elif model == "knn":
            result = obj.KNN_Regressor()
            result = abs(result)
        else:
            result = "Invalid Model Select"
    return render_template("result.html",result=result)


@app.route("/decision_predict",methods=['POST'])
def decision_pred():
    result = None
    if request.method == 'POST':
       model = request.form.get("model")
       f1 = float(request.form.get("f1"))
       f2 = float(request.form.get("f2"))
       f3 = float(request.form.get("f3"))
       obj = Tree(f1,f2,f3)
       if model == "decision":
           result = obj.Decision_tree_Regressor()
           result = abs(result)
       elif model == "random_forest":
           result = obj.Random_Forest_Regressor()
           result = abs(result) 
       else:
           result = "Invalid Model Select"
    return render_template("result_decision_tree.html",result=result)
                

@app.route("/classify_predict",methods=['POST'])
def classify_prediction():
     result = None
     if request.method == 'POST':        
        model = request.form.get("model")
        f1 = float(request.form.get("f1"))
        f2 = float(request.form.get("f2"))
        f3 = float(request.form.get("f3"))
        obj = Classifer(f1,f2,f3)
        if model == "linear":
            result = obj.Logistic_Regression()
        elif model == "logistic":
            result = obj.Random_Forest_Classifier()
        elif model == "rf":
            result = obj.KNN_Classifier()
        else:
            result = "Invalid Model Select"
     return render_template("result_classify.html",result=result)

    
@app.errorhandler(404)
def page_not_found(e):
    return render_template("page.html"),404


if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0")
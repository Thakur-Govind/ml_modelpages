from django.shortcuts import render,get_object_or_404
from .models import mlmodels
from logic import X_y_sep, svc_ml, acc, lr_ml, dt_ml, rf_ml
import numpy as np
# Create your views here.
def home(request):
    model = mlmodels.objects
    return render(request,'mlmodels/home.html',{'mlmodels': model})

def params(request, ml_id):
    model = get_object_or_404(mlmodels, pk = ml_id)
    par1_name = ""
    par2_name = ""
    if (ml_id == 1):
        para1 = request.POST.get('parameter1',False)
        para2 = request.POST.get('parameter2',False)
        par1_name = "C"
        par2_name = "Gamma"

    elif (ml_id == 3):
        para1 = request.POST.get('parameter1',False)
        para2 = request.POST.get('parameter2',False)
        par1_name = "Max Depth"
        par2_name = "Minimum Samples-Split"
    elif (ml_id == 4):
        para1 = request.POST.get('parameter1',False)
        para2 = request.POST.get('parameter2',False)
        par1_name = "Max Depth in a Tree"
        par2_name = "Minimum Samples-Split in a Tree"
    model.save()

    return render( request, 'mlmodels/ml_params.html', {'par1':par1_name, 'par2': par2_name, 'model_id': ml_id})

#def test(request):
    return render()
def mlexec(request, ml_id):
    model = get_object_or_404(mlmodels, pk = ml_id)
    #out=[]
    if(ml_id == 1):
        if model.para1 == '"' or model.para2 == '""':
            p1 = 0.6 if model.para1 == '""' else model.para1
            p2= 20 if model.para2 == '""' else model.para2
        else:
            p1 = float(model.para1)
            p2 = int(model.para2)
        out,y_test = svc_ml(0.2, np.array([1,2,3,6,3,8,9,10,7,7,4]).reshape(-1,1),[1,0,1,0,1,0,1,1,0,0,1], p1,p2)
        ac = acc(out,y_test)


    elif ml_id == 2:
        model = get_object_or_404(mlmodels, pk = ml_id)
        out,y_test = lr_ml(0.2, np.array([1,2,3,6,11,8,9,10,7,7]).reshape(-1,1),[1,0,1,0,1,0,1,1,0,0])
        ac = acc(out, y_test)


    elif ml_id == 3:
        if model.para1 == '"' or model.para2 == '""':
            p1 = 0.6 if model.para1 == '""' else model.para1
            p2= 20 if model.para2 == '""' else model.para2
        else:
            p1 = float(model.para1)
            p2 = int(model.para2)
        model = get_object_or_404(mlmodels, pk = ml_id)
        out,y_test = dt_ml(0.2, np.array([1,2,3,6,11,8,9,10,7,12,5,4]).reshape(-1,1),[1,0,1,0,1,0,1,1,0,0,1,1], p1,p2)
        ac = acc(out, y_test)

    elif ml_id == 4:
        p1 = int(model.para1)
        p2 = int(model.para2)
        model = get_object_or_404(mlmodels, pk = ml_id)
        out,y_test = rf_ml(0.2, np.array([1,2,3,6,11,8,9,10,7,12,5,4]).reshape(-1,1),[1,0,1,0,1,0,1,1,0,0,1,1], 24, p1,p2)
        ac = acc(out, y_test)
    print(acc)

    return render(request, 'mlmodels/mlexec.html', {'output': out, 'accuracy':ac,'model':model})

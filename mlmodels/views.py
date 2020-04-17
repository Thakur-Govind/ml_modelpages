from django.shortcuts import render,get_object_or_404
from .models import mlmodels,dataset,modelhist
from logic import X_y_sep, svc_ml, tmetrics, lr_ml, dt_ml, rf_ml,get_data
import numpy as np
# Create your views here.
def home(request):
    model = mlmodels.objects
    hist = modelhist.objects
    return render(request,'mlmodels/home.html',{'mlmodels': model,'entries':hist})

def params(request, ml_id):
    model = get_object_or_404(mlmodels, pk = ml_id)
    par1_name = ""
    par2_name = ""

    if (ml_id == 1):
        par1_name = "C"
        par2_name = "Gamma"

    elif (ml_id == 3):
        par1_name = "Max Depth"
        par2_name = "Minimum Samples-Split"
    elif (ml_id == 4):
        par1_name = "Max Depth in a Tree"
        par2_name = "Minimum Samples-Split in a Tree"
    return render( request, 'mlmodels/ml_params.html', {'par1':par1_name, 'par2': par2_name, 'model_id': ml_id, 'name':model.name})
#def test(request):
    #return render()
def mlexec(request, ml_id):
    model = get_object_or_404(mlmodels, pk = ml_id)
    #out=[]
    model.para1 = request.POST.get("p1",False)
    model.para2 = request.POST.get('parameter2',False)
    model.save()
    X,y = get_data('lr.csv') #to be updated
    if(ml_id == 1):
        if model.para1 == '' or model.para2 == '':
            p1 = 0.6 if model.para1 == '' else model.para1
            p2= 20 if model.para2 == '' else model.para2
        else:
            p1 = int(model.para1)
            p2 = int(model.para2)
        out,y_test = svc_ml(0.2,X,y, p1,p2)
        ac,f,fbl,fbh = tmetrics(out, y_test)
    elif ml_id == 2:
        out,y_test = lr_ml(0.2,X,y)
        ac,f,fbl,fbh = tmetrics(out, y_test)
    elif ml_id == 3:
        if model.para1 == '' or model.para2 == '':
            p1 = 5 if model.para1 == '' else model.para1
            p2= 20 if model.para2 == '' else model.para2
        else:
            p1 = float(model.para1)
            p2 = int(model.para2)
        out,y_test = dt_ml(0.2, X, y, p1,p2)
        ac,f,fbl,fbh = tmetrics(out, y_test)
    elif ml_id == 4:
        if model.para1 == '' or model.para2 == '':
            p1 = 3 if model.para1 == '' else model.para1
            p2= 10 if model.para2 == '' else model.para2
        else:
            p1 = int(model.para1)
            p2 = int(model.para2)
        out,y_test = rf_ml(0.2, X, y, 24, p1,p2)
        ac,f,fbl,fbh = tmetrics(out, y_test)
    entry = modelhist()
    entry.m_name = model.name
    if ml_id !=2:
        entry.m_para1 = p1
        entry.m_para2 = p2
    else:
        entry.m_para1 = "LR"
        entry.m_para2 = "LR"
    entry.accuracy = ac
    entry.f_score = f
    entry.f_b_h_score = fbh
    entry.f_b_l_score = fbl
    entry.save()
    if ml_id != 2:
        return render(request, 'mlmodels/mlexec.html', {'output': out, 'accuracy':ac,'fscore':f,'fbeta_l':fbl,'fbeta_h':fbh,'model':model, 'para1':p1, 'para2':p2})
    else:
        return render(request, 'mlmodels/mlexec.html', {'output': out, 'accuracy':ac,'fscore':f,'fbeta_l':fbl,'fbeta_h':fbh,'model':model})

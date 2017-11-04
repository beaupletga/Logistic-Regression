#cython main.pyx --embed && gcc -Os -I /usr/include/python2.7 -o test main.c -lpython2.7 && ./test
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
np.random.seed(1)


iris = datasets.load_iris()
x=iris['data']
y=iris['target']
y=y[y<=1].reshape((-1,1))
x=x[:y.shape[0]]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=1)


cdef f(coef,x):
    c=np.array(())
    cdef int i
    for i in range(x.shape[0]):
        c=np.insert(c,len(c),1/(1+np.exp(-coef.dot(x[i].T))))
    return c.reshape((-1,1))

cdef e(coef,x):
    c=np.array(())
    cdef int i
    for i in range(x.shape[0]):
        c=np.insert(c,len(c),np.exp(-coef.dot(x[i].T)))
    return c.reshape((-1,1))


def get_erreur(y_true,y_pred):
    return -np.mean(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred))

def assess_model(x_test,y_test,coef):
    return np.sum(1-np.abs(np.round(f(coef,x_test))-y_test))/x_test.shape[0]


cdef gradient_descent(x,y):
    coef=2*np.random.rand((x.shape[1])).reshape((1,-1))-1
    lr=1e-1
    cdef int i
    for i in range(30):
        y_pred=f(coef,x)
        erreur=get_erreur(y,y_pred)
        if i==0:
            print "Erreur à la première itération :",erreur
        coef-=-np.mean(-x+x*e(coef,x)*f(coef,x)+y*x,axis=0)
    print "Erreur à la dernière itération :",erreur
    return coef

coef=gradient_descent(x_train,y_train)
print "Taux de bonnes réponses (Validation):",assess_model(x_test,y_test,coef)

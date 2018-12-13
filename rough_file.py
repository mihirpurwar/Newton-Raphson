import numpy as np
'''
def solve(f, value):
    f = '('+f+')'
    operators = ['+','-','*','(',')','.']
    num = ['0','1','2','3','4','5','6','7','8','9']
    i = 0
    new_f = ''
    while i < len(f):
        if f[i] in operators+num:
            new_f+=f[i]
            i+=1
        else:
            j = i
            temp = ''
            while f[i] not in operators:
                temp = temp+f[i]
                i+=1

            new_f+="value['"+temp+"']"

    result = eval(new_f)
    return(result)

f = '(2*x+u1)'
r = solve(f,{'u1':1,'x':2})
print(r)
#x = np.array([1,2,3])
#u = np.array([1,2,3])
#r = eval(f)
#print(r)
'''
'''
operators = ['+','-','*','(',')','.']
num = ['0','1','2','3','4','5','6','7','8','9']
f = '0.5*((1+xytt23+u553)**3)'
i = 0
new_f = ''
while i < len(f):
    if f[i] in operators+num:
        new_f+=f[i]
        i+=1
    else:
        j = i
        temp = ''
        while f[i] not in operators:
            temp = temp+f[i]
            i+=1
        print(temp)
        new_f+="value['"+temp+"']"
print(new_f)
'''
#f = "value['a']+value['b']"
#value = {'a':2, 'b':6}
#print(eval(f))

# ---------------- D I F F E R E N T I A T I O N -----------------
from sympy import *

#f = "0.5*((1+value['x']+value['u'])**3)"
#value = {'x':Symbol('x'),'u':Symbol('u')}
x = Symbol('x')
f = 'x+u'
u = 4
y = eval(f)
#yprime = y.diff(value['x'])
print(y)

'''
def differentiate(f, wrt):
    f = '('+f+')'
    operators = ['+','-','*','(',')','.']
    num = ['0','1','2','3','4','5','6','7','8','9']
    i = 0
    new_f = ''
    sym = {}
    while i < len(f):
        if f[i] in operators+num:
            new_f+=f[i]
            i+=1
        else:
            temp = ''
            while f[i] not in operators:
                temp = temp+f[i]
                i+=1
            new_f+="sym['"+temp+"']"
            sym[temp] = Symbol(temp)
    res = eval(new_f)
    resPrime = res.diff(sym[wrt])
    return(str(resPrime))
f = '(u1**2)+12*u1-6*u2-24'
print(differentiate(f,'u1'))
'''

import numpy as np
from sympy import *

def solve(f, value):
    f = '('+f+')'
    operators = ['+','-','*','(',')','.']
    num = ['0','1','2','3','4','5','6','7','8','9']
    i = 0
    new_f = ''
    while i < len(f):
        if f[i] in operators+num and f[i] != ' ':
            new_f+=f[i]
            i+=1
        else:
            temp = ''
            while f[i] not in operators+[' ']:
                temp = temp+f[i]
                i+=1
            new_f+="value['"+temp+"']"
    result = eval(new_f)
    return(result)

def differentiate(f, wrt):
    f = '('+f+')'
    operators = ['+','-','*','(',')','.']
    num = ['0','1','2','3','4','5','6','7','8','9']
    i = 0
    new_f = ''
    sym = {}
    while i < len(f):
        if f[i] in operators+num and f[i] != ' ':
            new_f+=f[i]
            i+=1
        else:
            temp = ''
            while f[i] not in operators+[' ']:
                temp = temp+f[i]
                i+=1
            new_f+="sym['"+temp+"']"
            sym[temp] = Symbol(temp)
    res = eval(new_f)
    resPrime = res.diff(sym[wrt])
    return(str(resPrime))


# ---------------------------------------------------------
# constants
#f_expr = '0.5*((1+x+u)**3)'
f_expr = '1.5*(u**2)'
#a0, a1, gamma1 = -1, -1, -0.5
a0, a1, gamma1 = 1, 0, 4
#b0, b1, gamma2 = 1, 1, 1
b0, b1, gamma2 = 1, 0, 1
#h = 0.5
h = (1/3)
x_l, x_u = 0, 1

# number of iterations
iteration = 3
decimal_threshold = 4

# assumptions
c = h
alpha = h
# ---------------------------------------------------------
# create x-matrix
x = np.linspace(x_l, x_u, num = ((x_u-x_l)*(1/h))+1)

size = x.shape[0]
if a1 == 0 or b1 == 0:

    size1 = size - 2
else:
    size1 = size

# initilize
#u = np.array([0.001, -0.1, 0.001, 0.1])
if a1 == 0 or b1 == 0:
    #u = np.append(np.append(gamma1,np.random.uniform(1, 3, size1)),gamma2)
    u = np.array([gamma1,2,1.5,gamma2])
else:
    u = np.random.uniform(-0.1, 0.1, size)
f = np.zeros(size)
d_f = np.zeros(size1)
F = np.zeros((size1, 1))
d_F = np.zeros((size1, size1))


# loop
iteration_count = 0
done = False
while not done:
#for _ in range(iteration):
	iteration_count += 1
    old_u = np.array(list(u))
    
    # compute- f
    #f = 0.5*((1+x+u)**3)
    f = eval(f_expr)
    
    # compute- d_f
    if a1 != 0 or b1 != 0:
        #d_f = 1.5*((1+x+u)**2)
        d_f = eval(differentiate(f_expr,'u'))

    if a1 == 0 or b1 == 0:
        value = {}
        for i in range(size):
            value["u"+str(i)] = u[i]
            value["x"+str(i)] = x[i]
        
    # compute- F
    for i in range(size1):
        if a1 == 0 or b1 == 0:
            F[i,0] = f[i+1] - (u[i]-2*u[i+1]+u[i+2])*(1/(h**2))
            temp = "("
            j=0
            operators = ['+','-','*','(',')','.']
            num = ['0','1','2','3','4','5','6','7','8','9']
            while j < (len(f_expr)):
                if f_expr[j] in operators+num and f_expr[j] != ' ':
                    temp+=f_expr[j]
                    j+=1
                else:
                    temp1 = ''
                    while f_expr[j] not in operators+[' ']:
                        temp1 = temp1+f_expr[j]
                        j+=1
                    temp+=temp1+str(i+1)
            temp+=")"
            if i == 0:
                temp+="-("+str(u[i])+"-2*u"+str(i+1)+"+u"+str(i+2)+")*"+str(1/(h**2))
            elif i == size1-1:
                temp+="-(u"+str(i)+"-2*u"+str(i+1)+"+"+str(u[i+2])+")*"+str(1/(h**2))
            else:
                temp+="-(u"+str(i)+"-2*u"+str(i+1)+"+u"+str(i+2)+")*"+str(1/(h**2))
            #print(temp)
            for j in range(size1):
                d_F[i,j] = solve("".join(differentiate(temp,"u"+str(j+1)).split()),value)
                #d_F[i,j] = solve((differentiate(temp,"u"+str(i+1))),value)

        else:
            if i == 0:
                F[i,0] = (1+c)*u[0] - u[1] + ((h**2)/6)*(2*f[0]+f[1]) - (h*gamma1)/a1
            elif i == (size-1):
                F[i,0] = (-1)*u[i-1] + (1+alpha)*u[i] + ((h**2)/6)*(f[i-1]+2*f[i]) - (h*gamma2)/b1
            else:
                F[i,0] = (-1)*u[i-1] + 2*u[i] - u[i+1] + (h**2)*f[i] 
    
    # compute- d_F
    if a1 != 0 or b1 != 0:
        p = 0
        for i in range(size):
            if i == 0:
                d_F[i,0] = 1 + c + ((h**2)/3)*d_f[0]
                d_F[i,1] = (-1) + ((h**2)/6)*d_f[1]
            elif i == (size-1):
                d_F[i,size-2] =(-1) + ((h**2)/6)*d_f[i-1]
                d_F[i,size-1] = 1 + alpha + ((h**2)/3)*d_f[i]
            else:
                d_F[i,p+0] = -1
                d_F[i,p+1] = 2 + (h**2)*d_f[i]
                d_F[i,p+2] = -1
                p = p+1

    # matrix computation
    if a1 == 0 or b1 == 0:
        D = np.linalg.det(d_F)
        d_u = np.dot((1/D),(np.matmul(np.linalg.inv(d_F),np.dot(-1,F)))).reshape(size1)
        u += np.append(np.append(0,d_u),0)
    else:
        d_u = np.matmul(np.linalg.inv(d_F),np.dot(-1,F)).reshape(size1)
        u += d_u
    #print(u)
    
    # terminate loop
    #print((np.round(u,decimal_threshold)==np.round(old_u,decimal_threshold)).sum())
    if (np.round(u,decimal_threshold)==np.round(old_u,decimal_threshold)).sum()==size:
        done=True

if a1 == 0 or b1 == 0:
	print(u[1:(size1+1)])
else:
	print(u)
print(iteration_count)
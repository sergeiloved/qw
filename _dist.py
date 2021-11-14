""" Поиск ср.кв.отклонения через минимизацию
    расстояния между нормальным распределением
    и распределением, порождаемым решениями D-Wave """

""" Поиск бета через минимизацию
    расстояния JSD и L2 между распределением Больцмана
    и распределением, порождаемым решениями D-Wave """
import math
import time
import numpy as np
from numpy import random as rand
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import minimize as minim
from scipy import optimize as opt
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
from dwave.system import DWaveSampler, EmbeddingComposite
""" Функция распределения стандартного нормального распределения """
def Phi(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

""" Функция плотности стандартного нормального распределения """
def phi(x):
    return 1.0/math.sqrt(2*math.pi)*math.exp(-x*x/2.0)

a = 0.75
b = 0.25
expect = b/a
R = 5  # точность представления решений
M = 2**R    # количество возможных значений случайной величины
N = 100   # число семплов
c = 1
d = 0
delta = c/(2**(R-1))  # расстояние между ближайшими возможными значениями С.В.

""" Перевод бинарного массива в десятичное число из [-d,2c-d) """
def to_decimal_float(x):
    global c
    global d
    global R
    num = 0
    for i in range(R):
        num = num + x[i]/2**i
    return c * num - d

""" Перевод бинарного массива в десятичное натуральное число """
def to_decimal_int(x):
    global R
    num = 0
    for i in range(R):
        num = num + x[R-1-i] * (2**i)
    return num

print('a =',a,', b =',b)
print('b/a =',expect,', R =',R,', N =',N)
Q = {(0, 0): 0}

for i in range(R):
    temp1 = a*c/(2**i)
    for j in range(i,R):
        temp2 = a*c/(2**j)
        if j==i:
            Q[(i, j)] = temp1*(temp1 - 2*(a*d+b))
        else:
            Q[(i, j)] = 2*temp1*temp2

sampler_auto = EmbeddingComposite(
    DWaveSampler(solver={'topology__type': 'chimera'}))
start_time = time.time()
sampleset = sampler_auto.sample_qubo(Q, num_reads = N)
end_time = time.time()
print("--- %s seconds ---" % (end_time - start_time))

sam = sampleset.record # первый индекс - номер решения, второй: 0 - массив, 2 - количество появлений
print('Наилучшее решение =',to_decimal_float(sampleset.first[0]))

""" Преобразование выборки """

XiFull=[]  # множество возможных значений решений
for i in range(M):
    XiFull.append(c * i/2**(R-1) - d)

KiFull = [] # абсолютные частоты решений, индекс кодирует элемент выборки
for i in range(M):
    KiFull.append(0)

m = len(sam) # количество "различных" элементов в выборке (но массив не идеально скомпанован)
for i in range(m):  # вставка количества появлений решений
    KiFull[to_decimal_int(sam[i][0])] += sam[i][2]

WiFull=[]  # значения эмпирической ф.р. между каждыми соседними значениями 
tempint=KiFull[0]
for i in range(1,M):
    WiFull.append(tempint/N)
    tempint=tempint+KiFull[i]

Ki = [] # упорядоченный статистический ряд (без значений С.В., которые не появились)  
Xi = []
for i in range(M): 
    if KiFull[i] != 0:
        Ki.append(KiFull[i])
        Xi.append(c*i/2**(R-1)-d)

m = len(Xi) # теперь это истинное количество различных элементов в выборке
Wi=[]  # значения эмпирической функции распределения
tempint=0
for i in range(m):
    Wi.append(tempint/N)
    tempint=tempint+Ki[i]

Wi.append(1)

##########################################################################
## Функции для минимизации метрик L2 и C0 с нормальным распределением
##########################################################################
""" Поиск первого индекса, что Xi[l]>=expect """
el=math.floor(m/2)-1
if Xi[el]<expect:
    while Xi[el]<expect:
        el=el+1
        if el>m:
            el=el-1
            break
else:
    while Xi[el]>=expect:
        el=el-1
        if el<0:
            break
    el=el+1

""" Производная метрики L2 (с матожиданием) """
def metricL2Prime(s):
    global m
    global N
    global Xi
    global Ki
    global expect
    res=-N/math.sqrt(2)
    if s==0:
        return res
    else:
        for i in range(m):
            res=res+Ki[i]*math.exp(-(Xi[i]-expect)*(Xi[i]-expect)/2.0/s/s)
    return res

""" Производная метрики L2 (с выборочным средним) """
def metricL2PrimeWithSampleAverage(s):
    global m
    global N
    global Xi
    global Ki
    global expect_estim
    res=-N/math.sqrt(2)
    if s==0:
        return res
    else:
        for i in range(m):
            res=res+Ki[i]*math.exp(-(Xi[i]-expect_estim)*(Xi[i]-expect_estim)/2.0/s/s)
    return res

""" Метрика L2 (с матожиданием) """
def metricL2(s):
    global m
    global N
    global el
    global Xi
    global Wi
    global Ki
    global expect
    if s==0:
        res=0
        k=l
        for i in range(l):
            res=res-Ki[i]*(Xi[i]-expect)*(Wi[i+1]+Wi[i])
        if (Xi[el]-expect)==0:
            res+Ki[i]*(Xi[i]-expect)*(1-Wi[i+1]-Wi[i])
            k=l+1
        for i in range(k,m):
            res=res+Ki[i]*(Xi[i]-expect)*(2-Wi[i+1]-Wi[i])
        return res/N
    else:
        res=0
        for i in range(m):
            res=res+Ki[i]*phi((Xi[i]-expect)/s)
        res=s*(2*res/N-1.0/math.sqrt(math.pi))
        temp=0
        for i in range(m):
            temp=temp+Ki[i]*(Xi[i]-expect)*(2*Phi((Xi[i]-expect)/s)-Wi[i+1]-Wi[i])
        return res+temp/N

""" Метрика C0 """
def metricC0(s):
    global m
    global el
    global Xi
    global Wi
    global expect
    res=-1
    if s==0:
        if Xi[el]==expect:
            res=max(Wi[el],math.fabs(0.5-Wi[el]),math.fabs(0.5-Wi[el+1]), 1-Wi[el+1])
        else:
            res=max(Wi[el],1-Wi[el])
    else:
        for i in range(m):
            x=(Xi[i]-expect)/s
            temp1=math.fabs(Wi[i]-Phi(x))
            temp2=math.fabs(Wi[i+1]-Phi(x))
            if temp1>res:
                res=temp1
            if temp2>res:
                res=temp2
    return res

##########################################################################
## Функции для минимизации метрик L2 и JSD с распределением Больцмана
##########################################################################
Qi = []
for i in range(M):
    Qi.append(0)

def calculate_Qi_QM(beta):
    global M
    global a
    global b
    global XiFull
    global Qi
    
    for i in range(M):
        Qi[i] = math.exp((-1)*beta*(a*XiFull[i]-b)*(a*XiFull[i]-b))
    
    QM = sum(Qi)
    return QM

def metricJSD(beta):
    global N
    global M
    global a
    global b
    global XiFull
    global KiFull
    global Qi
    
    QM = calculate_Qi_QM(beta)
    res = math.log(2.0/math.sqrt(N*QM))
    temp = 0
    for i in range(M):
        if KiFull[i]!=0:
            temp += KiFull[i] * (math.log(KiFull[i]) - math.log(KiFull[i]/N+Qi[i]/QM))
    
    res += temp/2.0/N
    temp = 0
    for i in range(M):
        temp += (a*XiFull[i]-b)*(a*XiFull[i]-b)*Qi[i]
    
    res -= beta*temp/2.0/QM
    temp = 0
    for i in range(M):
        if KiFull[i] == 0:
            temp -= Qi[i]*(beta*(a*XiFull[i]-b)*(a*XiFull[i]-b) + math.log(QM))
        else:
            temp += Qi[i]*math.log(KiFull[i]/N+Qi[i]/QM)
    
    res -= temp/2.0/QM
    res = math.sqrt(res)
    return res

def metricL2(beta):
    global delta
    global WiFull
    global M
    global Qi
    
    Nui = 0
    QM = calculate_Qi_QM(beta)
    res = 0
    for i in range(M-1):
        Nui += Qi[i]
        res += (Nui/QM - WiFull[i]) * (Nui/QM - WiFull[i])
    
    res *= delta
    res = math.sqrt(res)
    return res

def calculateExpectAndVariance(beta):
    global M
    global XiFull
    global Qi
    
    QM = calculate_Qi_QM(beta)
    MathExpect = 0
    MathSqrExpect = 0
    for i in range(M):
        MathExpect += Qi[i]*XiFull[i]
        MathSqrExpect += Qi[i]*XiFull[i]*XiFull[i]
    
    MathExpect /= QM
    MathSqrExpect /= QM
    return [MathExpect, math.sqrt(MathSqrExpect - MathExpect*MathExpect)]


""" Выборочные оценки параметров распределения """
expect_estim=sum(np.multiply(Xi,Ki))/N  # выборочное среднее
sigma_estim=math.sqrt(sum(np.multiply(np.multiply(
    Xi-expect_estim,Xi-expect_estim),Ki))/(N-1)) # исправленное выборочное ско
if sigma_estim != 0:    # выборочное бета по предельной зависимости
    beta_estim = 0.5/a/a/sigma_estim/sigma_estim
else:
    beta_estim = 10.0**308

def LogFunOfMaxLikelihood(beta): # выборочное бета методом макс. правдоподобия
    global N
    global M
    global m
    global Qi
    global Xi
    global Ki
    global a
    global b
    
    QM=calculate_Qi_QM(beta)
    temp=0
    for i in range(m):
        temp+=Ki[i]*(a*Xi[i]-b)*(a*Xi[i]-b)
    
    return N*math.log(QM)+beta*temp

BetaML=opt.minimize_scalar(LogFunOfMaxLikelihood)
print('Оценка матожидания  =',expect_estim)
print('Оценка с.к.о.       =',sigma_estim)
print('Бета по оценочному значению сигма  =',beta_estim)
print('Бета по методу макс. правдоподобия =',BetaML.x)

##########################################################################
## Расчёт точки минимума метрики L2 и C0
##########################################################################
print('--- НОРМАЛЬНОЕ РАСПРЕДЕЛЕНИЕ ---')
RootL2expect=opt.root_scalar(metricL2Prime,
                             bracket=[0, 2*sigma_estim+2],
                             method='toms748',
                             xtol=8.881784197001252e-16,
                             rtol=8.881784197001252e-16)
print('--- Корень производной L2 с матожиданием ---')
print('Сигма =',RootL2expect.root)
if RootL2expect.root == 0:
    print('Бета  = +inf')
else:
    BetaL2expect=0.5/a/a/RootL2expect.root/RootL2expect.root
    print('Бета  =',BetaL2expect)

RootL2average=opt.root_scalar(metricL2PrimeWithSampleAverage,
                              bracket=[0, 2*sigma_estim+2],
                              method='toms748',
                              xtol=8.881784197001252e-16,
                              rtol=8.881784197001252e-16)
print('--- Корень производной L2 с выборочным средним ---')
print('Сигма =',RootL2average.root)
if RootL2average.root == 0:
    print('Бета  = +inf')
else:
    BetaL2average=0.5/a/a/RootL2average.root/RootL2average.root
    print('Бета  =',BetaL2average)

MinimumС0=opt.minimize_scalar(metricC0,
                              bracket=[0,2*sigma_estim+2],
                              method='golden',
                              tol=8.881784197001252e-16,
                              options={'xtol': 8.881784197001252e-16})
print('-- Минимизация метрики С0 --')
print('Сигма =',MinimumС0.x)
if MinimumС0.x == 0:
    print('Бета  = +inf')
else:
    BetaC0=0.5/a/a/MinimumС0.x/MinimumС0.x
    print('Бета  =',BetaC0)

###########################################
## Расчёт точки минимума метрики JSD и L2
###########################################
print('--- РАСПРЕДЕЛЕНИЕ БОЛЬЦМАНА ---')
MinimumJSD=opt.minimize_scalar(metricJSD,
                              method='golden',
                              tol=8.881784197001252e-16,
                              options={'xtol': 8.881784197001252e-16})

ExpectVarBolzJSD = calculateExpectAndVariance(MinimumJSD.x)
sigmaJSD = 1.0/a/math.sqrt(2*MinimumJSD.x)
print('-- Минимизация метрики JSD --')
print('Сигма =',sigmaJSD)
print('Бета  =',MinimumJSD.x)
print('Матожидание распр. Больцмана =',ExpectVarBolzJSD[0])
print('Ср.кв.откл. распр. Больцмана =',ExpectVarBolzJSD[1])

MinimumL2=opt.minimize_scalar(metricL2,
                              method='golden',
                              tol=8.881784197001252e-16,
                              options={'xtol': 8.881784197001252e-16})

ExpectVarBolzL2 = calculateExpectAndVariance(MinimumL2.x)
sigmaL2 = 1.0/a/math.sqrt(2*MinimumL2.x)
print('-- Минимизация метрики L2 --')
print('Сигма =',sigmaL2)
print('Бета  =',MinimumL2.x)
print('Матожидание распр. Больцмана =',ExpectVarBolzL2[0])
print('Ср.кв.откл. распр. Больцмана =',ExpectVarBolzL2[1])
print('')
print("%d \n& %.4f & %.4f & %.4f \n& %.4f & %.4f & %.4f & %.4f \\\\" % (R,
                                                                expect_estim,
                                                                sigma_estim,
                                                                beta_estim,
                                                                RootL2expect.root,
                                                                BetaL2expect,
                                                                RootL2average.root,
                                                                BetaL2average))
print("%d \n& %.4f & %.4f & %.4f & %.4f \n& %.4f & %.4f & %.4f \n& %.4f & %.4f & %.4f \\\\" % (R,
                                                                expect_estim,
                                                                sigma_estim,
                                                                beta_estim,
                                                                BetaML.x,
                                                                sigmaJSD,
                                                                MinimumJSD.x,
                                                                ExpectVarBolzJSD[1],
                                                                sigmaL2,
                                                                MinimumL2.x,
                                                                ExpectVarBolzL2[1]))

""" Построение гистограмм
    для сравнения эмпирической плотности выборки
    и плотности нормальных распределений с найденным сигма
    с истинным матожиданием и выборочным средним """

X=[] # формируем выборку из статистического ряда
for i in range(m):
    for j in range(Ki[i]):
        X.append(Xi[i])

density = gaussian_kde(X)
lag=0.01
width=3*sigma_estim
Left=min(Xi[0],expect-width)
Right=max(Xi[m-1],expect+width)
numberOfBins=math.floor(width*M)
x=np.arange(Left, Right, lag)
titles1=['L2 с матожиданием b/a',
         'L2 с выбор. сред.',
         'C0 с матожиданием b/a',
         'Выборочное сигма']
labels1=['Гистограмма частот','Эмпирич. плотность','Гипотетич. плотность']
y1=np.empty(len(x), dtype=float)
for i in range(len(x)):
    y1[i]=1.0/RootL2expect.root*phi((x[i]-expect)/RootL2expect.root)

y2=np.empty(len(x), dtype=float)
for i in range(len(x)):
    y2[i]=1.0/RootL2average.root*phi((x[i]-expect_estim)/RootL2average.root)

y3=np.empty(len(x), dtype=float)
for i in range(len(x)):
    y3[i]=1.0/MinimumС0.x*phi((x[i]-expect)/MinimumС0.x)

y4=np.empty(len(x), dtype=float)
for i in range(len(x)):
    y4[i]=1.0/sigma_estim*phi((x[i]-expect)/sigma_estim)

plt.figure(figsize=(11.5,6.5))
plt.suptitle('Сравнение эмпирических плотностей с нормальными, N='
             +str(N)+', '+'R='+str(R))
plt.subplot(2, 2, 1)
plt.title(titles1[0]+', sigma='+str(round(RootL2expect.root,3)))
plt.hist(X, density=True, bins=numberOfBins,
         range=(Left,Right),
         color='c',histtype='stepfilled',label=labels1[0])
plt.plot(x,density(x),label=labels1[1])
plt.plot(x,y1,color='r',label=labels1[2])
plt.legend(loc='best')
plt.grid(True)
plt.subplot(2, 2, 2)
plt.title(titles1[1]+', sigma='+str(round(RootL2average.root,3)))
plt.hist(X, density=True, bins=numberOfBins,
         range=(Left,Right),
         color='c',histtype='stepfilled',label=labels1[0])
plt.plot(x,density(x),label=labels1[1])
plt.plot(x,y2,color='r',label=labels1[2])
plt.grid(True)
plt.subplot(2, 2, 3)
plt.title(titles1[2]+', sigma='+str(round(MinimumС0.x,3)))
plt.hist(X, density=True, bins=numberOfBins,
         range=(Left,Right),
         color='c',histtype='stepfilled',label=labels1[0])
plt.plot(x,density(x),label=labels1[1])
plt.plot(x,y3,color='r',label=labels1[2])
plt.grid(True)
plt.subplot(2, 2, 4)
plt.title(titles1[3]+', sigma='+str(round(sigma_estim,3)))
plt.hist(X, density=True, bins=numberOfBins,
         range=(Left,Right),
         color='c',histtype='stepfilled',label=labels1[0])
plt.plot(x,density(x),label=labels1[1])
plt.plot(x,y4,color='r',label=labels1[2])
plt.grid(True)

""" Построение гистограмм
    для сравнения эмпирической плотности выборки
    и плотности распределения Больцмана с найденным beta """

LeftXi=0
RightXi=M
i=0
while XiFull[i]<Left:
    LeftXi+=1
    i+=1

i=M-1
while XiFull[i]>Right:
    RightXi-=1
    i-=1

titles2=['Метрика JSD',
         'Метрика L2',
         'Бета по ММП',
         'Бета по s^2']
labels2=['Гистограмма частот','Эмпирич. плотность','Гипотетич. плотность']
QM=calculate_Qi_QM(MinimumJSD.x)
y5=np.empty(M, dtype=float)
for i in range(M):
    y5[i]=Qi[i]/QM/delta

QM=calculate_Qi_QM(MinimumL2.x)
y6=np.empty(M, dtype=float)
for i in range(M):
    y6[i]=Qi[i]/QM/delta

QM=calculate_Qi_QM(BetaML.x)
y7=np.empty(M, dtype=float)
for i in range(M):
    y7[i]=Qi[i]/QM/delta

QM=calculate_Qi_QM(beta_estim)
y8=np.empty(M, dtype=float)
for i in range(M):
    y8[i]=Qi[i]/QM/delta

plt.figure(figsize=(11.5,6.5))
plt.suptitle('Сравнение эмпирических плотностей с распределением Больцмана, N='+str(N)+', '+'R='+str(R))
plt.subplot(2, 2, 1)
plt.title(titles2[0]+', beta='+str(round(MinimumJSD.x,2)))
plt.hist(X, density=True, bins=numberOfBins,
         range=(Left,Right),
         color='c',histtype='stepfilled',label=labels2[0])
plt.plot(x,density(x),label=labels2[1])
plt.plot(XiFull[LeftXi:RightXi+1],y5[LeftXi:RightXi+1],color='r',label=labels2[2])
plt.legend(loc='best')
plt.grid(True)
plt.subplot(2, 2, 2)
plt.title(titles2[1]+', beta='+str(round(MinimumL2.x,2)))
plt.hist(X, density=True, bins=numberOfBins,
         range=(Left,Right),
         color='c',histtype='stepfilled',label=labels2[0])
plt.plot(x,density(x),label=labels2[1])
plt.plot(XiFull[LeftXi:RightXi+1],y6[LeftXi:RightXi+1],color='r',label=labels2[2])
plt.grid(True)
plt.subplot(2, 2, 3)
plt.title(titles2[2]+', beta='+str(round(BetaML.x,2)))
plt.hist(X, density=True, bins=numberOfBins,
         range=(Left,Right),
         color='c',histtype='stepfilled',label=labels2[0])
plt.plot(x,density(x),label=labels2[1])
plt.plot(XiFull[LeftXi:RightXi+1],y7[LeftXi:RightXi+1],color='r',label=labels2[2])
plt.grid(True)
plt.subplot(2, 2, 4)
plt.title(titles2[3]+', beta='+str(round(beta_estim,2)))
plt.hist(X, density=True, bins=numberOfBins,
         range=(Left,Right),
         color='c',histtype='stepfilled',label=labels2[0])
plt.plot(x,density(x),label=labels2[1])
plt.plot(XiFull[LeftXi:RightXi+1],y8[LeftXi:RightXi+1],color='r',label=labels2[2])
plt.grid(True)
plt.show()

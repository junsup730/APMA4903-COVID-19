#coding:utf-8
import numpy as np
import pandas as pd
from plotly import graph_objs as go
import plotly.express as px
from IPython.display import Image
import datetime as dt
from scipy.integrate import solve_ivp,odeint
from scipy.optimize import minimize
import csv

######### read the data ###########
country = 'US'
#country = 'Italy'
#country = 'Germany'

# Confirmed cases
#df = pd.read_csv('E:/data analysis/1021/lts199811484/COVID-19-master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
df = pd.read_csv('time_series_covid_19_confirmed_global.csv')
df.head()
cols = df.columns[4:]
infected = df.loc[df['Country/Region']==country, cols].values.flatten()

# Deaths
#df = pd.read_csv('E:/data analysis/1021/lts199811484/COVID-19-master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
df = pd.read_csv('time_series_covid_19_deaths_global.csv')
deceased = df.loc[df['Country/Region']==country, cols].values.flatten()

# Recovered
#df = pd.read_csv('E:/data analysis/1021/lts199811484/COVID-19-master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
df = pd.read_csv('time_series_covid_19_recovered_global.csv')
recovered = df.loc[df['Country/Region']==country, cols].values.flatten()

############## time ################
dates = cols.values
x = [dt.datetime.strptime(d,'%m/%d/%y').date() for d in dates]


############# data selected #######################
infected_clean = infected[30:]
deceased_clean = deceased[30:]
recovered_clean = recovered[30:]

infected_clean =infected_clean - recovered_clean - deceased_clean 


###############actual data
def creatDataFrame():
	d={
	'date':x,
	'infected':infected,
	'deceased':deceased,
	'recovered':recovered
	}
	together=pd.DataFrame(d)
	print(together)
	
#together=creatDataFrame()

########################################

def fit_to_data(vec, t_q, N, test_size):
    beta, gamma, sigma, alpha,N = vec
    
    sol = solve_ivp(SEIR_q_stop, [0, t_f], y0, args=(beta, gamma, sigma, alpha, N,  t_q, t_stop), t_eval=t_eval)
    
    split = np.int((1-test_size) * infected_clean.shape[0])
    
    error = (
        np.sum(
            1.5*(deceased_clean[:split]+recovered_clean[:split]-sol.y[3][:split])**2) +
               
        np.sum(
            #(infected_clean[:split]-np.cumsum(sol.y[1][:split]+sol.y[2][:split]))**2)
            (infected_clean[:split]-sol.y[2][:split])**2)
            
    ) / split
    
    return error


############# quarantine and stop #############
def SEIR_q_stop(t, y, beta, gamma, sigma, alpha, N, t_quarantine, t_stop):
    """SEIR epidemic model.
        S: subsceptible
        E: exposed
        I: infected
        R: recovered
        
        N: total population (S+E+I+R)
        
        Social distancing is adopted when t>t_quarantine and t<=t_stop.
    """
    S = y[0]
    E = y[1]
    I = y[2]
    R = y[3]
    #gamma, sigma= [0.07, 0.2]
    tao3=125
    tao4=155
    tao5=218
    
    beta_t = beta
    if t<t_quarantine:
        beta_t = beta   
    elif  (t>=t_quarantine and t<=t_stop) :
        beta_t = beta*np.exp(-alpha*(t-t_quarantine))
    elif (t>t_quarantine and t<=tao3):
        beta_t=beta*np.exp(-alpha*47)
    elif (t>tao3 and t<=tao4):
        beta_t=beta*np.exp(-alpha*42)
    elif (t>tao4 and t<=tao5):
        beta_t=beta*np.exp(-alpha*50.5)
    else:
        beta_t = beta*np.exp(-alpha*48.5)
        #beta_t = beta
        
    dS = -beta_t*S*I/N
    dE = beta_t*S*I/N - sigma*E
    dI = sigma*E - gamma*I
    dR = gamma*I
    return [dS, dE, dI, dR]   


def SEIR(t, y, beta, gamma, sigma, alpha, N, t_quarantine, t_stop):
    """SEIR epidemic model.
        S: subsceptible
        E: exposed
        I: infected
        R: recovered
        
        N: total population (S+E+I+R)
        
        Social distancing is adopted when t>t_quarantine and t<=t_stop.
    """
    S = y[0]
    E = y[1]
    I = y[2]
    R = y[3]
    #gamma, sigma= [0.07, 0.2]
    tao3=125
    tao4=155
    tao5=218
    
    beta_t = beta

        
    dS = -beta_t*S*I/N
    dE = beta_t*S*I/N - sigma*E
    dI = sigma*E - gamma*I
    dR = gamma*I
    return [dS, dE, dI, dR]   

####################################################
############ minimize the loss function ############
####################################################
#N = 60e6 / (10/1.1) #italy
#N=8.2e7/5 #Germany
#N = 3.27e8 #USA
N=4.38206069e+07
#N = np.int(N)
t_q = 15# quarantine takes place
t_stop=70
t_f = infected_clean.shape[0]
y0 = [N-infected_clean[0], 0, infected_clean[0], 0]
t_eval = np.arange(0,t_f,1)
test_size = 0.1
#test_size = 0
#beta, gamma, sigma, alpha, N = [0.8, 0.07, 0.2, 0.0495, N]

#beta, gamma, sigma, alpha, N = [8.93649203e-01, 8.54313768e-03, 1.62382865e-01, 9.61380780e-02, 2.31195458e+07]
beta, gamma, sigma, alpha, N = [8.94930172e-01, 9.41360829e-03,1.69921176e-01, 8.39851335e-02,  3.49970672e+07]
opt = minimize(fit_to_data, [0.5, 0.07, 0.2, 0.08, N], method='Nelder-Mead', args=(t_q, t_stop,  test_size))

############### option method with bounds: L-BFGS-B, TNC, SLSQP and trust-constr
#opt = minimize(fit_to_data, [0.8, 0.07, 0.2, 0.0463, 2.24e9 ], bounds=((0.6,0.9),(0.06,0.08),(0.15,0.25),(0.04,0.06),(2e9,2.5e9)), method='trust-constr', args=(t_q, t_stop,  test_size))
##########
beta, gamma, sigma, alpha, N = opt.x

sol = solve_ivp(SEIR_q_stop, [0, t_f], y0, args=(beta, gamma, sigma, alpha, N, t_q ,t_stop), t_eval=t_eval)

###############
#fig = go.Figure(data=go.Scatter(x=x[30:], y=np.cumsum(sol.y[1]+sol.y[2]), name='E+I',
 #                              marker_color=px.colors.qualitative.Plotly[0]))
fig = go.Figure(data=go.Scatter(x=x[30:], y=sol.y[2], name='I',
                               marker_color=px.colors.qualitative.Plotly[0]))                               
                               
fig.add_trace(go.Scatter(x=x[30:], y=infected_clean, name='Infected', mode='markers', 
                         marker_color=px.colors.qualitative.Plotly[0]))
fig.add_trace(go.Scatter(x=x[30:], y=sol.y[3], name='R', mode='lines', 
                         marker_color=px.colors.qualitative.Plotly[1]))
fig.add_trace(go.Scatter(x=x[30:], y=deceased_clean+recovered_clean, name='Deceased+recovered', 
                         mode='markers', 
                         marker_color=px.colors.qualitative.Plotly[1]))
fig.add_trace(go.Scatter(x=[x[37], x[37]], y=[0, 100000], name='Quarantine', mode='lines',
                        marker_color='darkgrey'))
fig.update_layout(title='''Model's predictions vs historical data''',
                   xaxis_title='Days',
                   yaxis_title='Number of individuals')

fig.show()

####################### control ###############################
#N = 100
#beta, gamma, sigma, alpha, N = [2.24, 0.4, 0.1, 0.5, 100]
#t_q = 20
#t_stop = 60
#y0 = np.array([99, 0, 11/(3.49970672e+07)*100, 0])
sol = solve_ivp(SEIR_q_stop, [0, 1095], y0, t_eval=np.arange(0, 1095, 0.1), args=(beta, gamma, sigma, alpha, N, t_q, t_stop))

fig = go.Figure(data=go.Scatter(x=sol.t, y=sol.y[0], name='Susceptible, interrupted',
                               line=dict(color=px.colors.qualitative.Plotly[0])))
fig.add_trace(go.Scatter(x=sol.t, y=sol.y[1], name='Exposed, interrupted',
                        line=dict(color=px.colors.qualitative.Plotly[1])))
fig.add_trace(go.Scatter(x=sol.t, y=sol.y[2], name='Infected, interrupted',
                        line=dict(color=px.colors.qualitative.Plotly[2])))
fig.add_trace(go.Scatter(x=sol.t, y=sol.y[3], name='Recovered, interrupted',
                        line=dict(color=px.colors.qualitative.Plotly[3])))
#fig.show()

beta, gamma, sigma, alpha, N = opt.x

################# uncontrol ###############
sol = solve_ivp(SEIR, [0, 1095], y0, t_eval=np.arange(0, 1095, 0.1), args=(beta, gamma, sigma, alpha, N, t_q, t_stop))

fig.add_trace(go.Scatter(x=sol.t, y=sol.y[0], name='Susceptible, continuous',
                               line=dict(color=px.colors.qualitative.Plotly[0], dash='dash')))
fig.add_trace(go.Scatter(x=sol.t, y=sol.y[1], name='Exposed, continuous',
                        line=dict(color=px.colors.qualitative.Plotly[1], dash='dash')))
fig.add_trace(go.Scatter(x=sol.t, y=sol.y[2], name='Infected, continuous',
                        line=dict(color=px.colors.qualitative.Plotly[2], dash='dash')))
fig.add_trace(go.Scatter(x=sol.t, y=sol.y[3], name='Recovered, continuous',
                        line=dict(color=px.colors.qualitative.Plotly[3], dash='dash')))

fig.update_layout(title='SEIR epidemic model - effect of social-distancing',
                 xaxis_title='Days',
                 yaxis_title='Percentage of population')
fig.show()



################## thd max index ###################
lookformax=sol.y[2].tolist()
print("lookformax")


#print(max(lookformax),lookformax.index(max(lookformax))) 
max_num=max(lookformax)
max_index=lookformax.index(max_num)

print( "Max is %d,Index is %d"%(max_num,max_index))
print("The time at max:")
print(sol.t[max_index])
#print(max_num) 


###########################################################
N = 100
#beta, gamma, sigma, alpha, N = [2.24, 0.4, 0.1, 0.5, 100]
#t_q = 20
#t_stop = 60
y0 = np.array([99, 0, 11/(3.49970672e+07)*100, 0])
sol = solve_ivp(SEIR_q_stop, [0, 1095], y0, t_eval=np.arange(0, 1095, 0.1), args=(beta, gamma, sigma, alpha, N, t_q, t_stop))

fig = go.Figure(data=go.Scatter(x=sol.t, y=sol.y[0], name='Susceptible, interrupted',
                               line=dict(color=px.colors.qualitative.Plotly[0])))
fig.add_trace(go.Scatter(x=sol.t, y=sol.y[1], name='Exposed, interrupted',
                        line=dict(color=px.colors.qualitative.Plotly[1])))
fig.add_trace(go.Scatter(x=sol.t, y=sol.y[2], name='Infected, interrupted',
                        line=dict(color=px.colors.qualitative.Plotly[2])))
fig.add_trace(go.Scatter(x=sol.t, y=sol.y[3], name='Recovered, interrupted',
                        line=dict(color=px.colors.qualitative.Plotly[3])))
fig.show()

beta, gamma, sigma, alpha, N = opt.x

#########################################################
tao3=125
tao4=155
tao5=218
beta_t=[]
t_quarantine=t_q

for i in t_eval: 
	if i<t_quarantine:
		beta_t.append(beta) 
		#print(i)  
	elif  (i>=t_quarantine and i<=t_stop) :
		beta_t.append( beta*np.exp(-alpha*(i-t_quarantine)))
	elif (i>t_quarantine and i<=tao3):
		beta_t.append(beta*np.exp(-alpha*47))
	elif (i>tao3 and i<=tao4):
		beta_t.append(beta*np.exp(-alpha*42))
	elif (i>tao4 and i<=tao5):
		beta_t.append(beta*np.exp(-alpha*50.5))
	else:
		beta_t.append( beta*np.exp(-alpha*48.5))


fig = go.Figure(data=go.Scatter(x=t_eval, y=beta_t, name='beta',
                               line=dict(color=px.colors.qualitative.Plotly[0]))) 
fig.update_layout(title='SEIR epidemic model - effect of social-distancing',
                 xaxis_title='Days',
                 yaxis_title='beta')

fig.show()






#together=creatDataFrame()

d={
	'date':x,
	'infected':infected,
	'deceased':deceased,
	'recovered':recovered
	}
together=pd.DataFrame(d)

#together.to_csv('E:/data analysis/1021/lts199811484/together.csv', sep=',', header=True, index=True)
together.to_csv('together.csv', sep=',', header=True, index=True)

print("beta, gamma, sigma, alpha, N = opt.x")
print(opt.x)


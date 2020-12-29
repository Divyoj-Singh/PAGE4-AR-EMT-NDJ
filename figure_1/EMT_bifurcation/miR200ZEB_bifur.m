function [x,v,s,h,f] = miR200ZEB_bifur

curdir = pwd;
init;
cd(curdir);

opt = contset;
opt=contset(opt,'Singularities',1);
opt=contset(opt,'MaxNumPoints',20000);
opt=contset(opt,'MinStepsize',0.1);
opt=contset(opt,'MaxStepsize',100);
opt=contset(opt,'Eigenvalues',1);

% Degradation rate:
ku200 = 0.05;   kmz = 0.5;   kz = 0.1;  
% Transcription rate:
gu200 = 2100;   gmz = 11;   gz = 100;   
% Hills function threshold :
z0u200 = 220000;   z0mz = 25000;   s0u200 = 180000;   s0mz = 180000; u2000 = 10000;   
% Cooperativity:
nzu200 = 3;   nsu200 = 2;   nzmz = 2;   nsmz = 2;   nu200 = 6;...
% fold change
lamdazu200 =0.1;   lamdasu200 = 0.1;  lamdazmz = 7.5;   lamdasmz = 10;...
% Snail (external signal here)
s = 750000;

ap = 1; %describes the index of parameter for which the bifurcation is drawn using the init_EP_EP function. Currently, ap=1, thus bifurcation parameter is s (SNAIL levels)
handles = feval(@miR200ZEB);
tspan = 0:1:5000;

% initial condition
x_start = [33554.833280 56.500562 0];

%calculating steady state for given initial condition 
[t,x_time] = ode15s(@(t,kmrgd)handles{2}(t,kmrgd,s,ku200,kmz,kz,gu200,gmz,gz,z0u200,z0mz,s0u200,s0mz,u2000,nzu200,nsu200,nzmz,nsmz,nu200,lamdazu200,lamdasu200,lamdazmz,lamdasmz),tspan,x_start);
x_init = x_time(end,:)';

%drawing bifurcation using a continuation method
[x0,v0] = init_EP_EP(@miR200ZEB,x_init,[s,ku200,kmz,kz,gu200,gmz,gz,z0u200,z0mz,s0u200,s0mz,u2000,nzu200,nsu200,nzmz,nsmz,nu200,lamdazu200,lamdasu200,lamdazmz,lamdasmz],ap);
[x,v,s,h,f] = cont(@equilibrium, x0, v0,opt);

end



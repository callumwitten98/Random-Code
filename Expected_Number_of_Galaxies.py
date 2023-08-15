##############################################################################################################################

#				Calculate number of expected galaxies from the UV luminosity function
#							August 2023

##############################################################################################################################

##############################################################################################################################
#							REQUIRED PARAMETERS
##############################################################################################################################

area=144		#area to integrate over
m=26.0			#5-sigma magnitude of your observations

##############################################################################################################################
#					OPTIONAL  VARIABLES -- for standard use leave these unchanged
##############################################################################################################################

zmin,zmax=4,12		#range of redshifts that you are interested -- ordinarily z=4-12
dz=0.5			#redshift interval to integrate over -- ordinarily z-0.5 --> z+0.5
nstep=1000000		#number of steps used in each integration

##############################################################################################################################
#							    CONSTANTS
##############################################################################################################################

PI=3.141592654
h0=72 #km/s/Mpc
om=0.3
ov=0.7
c=2.99792458E5 #km/s

##############################################################################################################################
#							    PACKAGES
##############################################################################################################################

import numpy as np
import scipy.integrate as integrate

##############################################################################################################################
#							  LF parameters
##############################################################################################################################

#Source: Bouwens et al. 2015 (ApJ)
#bound: calculating the expected number ('mid'), the lower bound ('low') or the upper bound ('high') of galaxies

def M_P_a(za,bound):
    if bound=='mid':
        Mstar=-20.95+0.01*(za-6.0)
        Phistar=(0.47*10**(-0.27*(za-6.0)))*1E-3
        alpha=-1.87+(-0.10)*(za-6.0)
    
    if za>=12:
        Mstar=-19.92
        Phistar=10**(-5.09)
        alpha=-2.35

    if bound=='high':
        Mstar=(-20.95-0.10)+(0.01-0.06)*(za-6.0)
        Phistar=((0.47+0.11)*10**((-0.27+0.05)*(za-6.0)))*1E-3
        alpha=((-1.87-0.05))+(-0.10-0.03)*(za-6.0)
    
    if bound=='low':
        Mstar=(-20.95+0.10)+(0.01+0.06)*(za-6.0)
        Phistar=(0.47*10**(-0.27*(za-6.0)))*1E-3
        alpha=((-1.87+0.05))+(-0.10+0.03)*(za-6.0)
        
    return Mstar,Phistar,alpha

##############################################################################################################################
#							 Calculate Volume
##############################################################################################################################

def func_1(i,dz):
    a=(i-1)*dz
    b=i*dz
    s1=(1/np.sqrt(om*(1.+a)**3+(1.-om-ov)*(1.+a)**2+ov))
    s2=(1/np.sqrt(om*(1.+b)**3+(1.-om-ov)*(1.+b)**2+ov))
    s=dz*(s1+s2)/2.
    return s

def func_2(j,dz1,zmin):
    z=zmin+j*dz1
    dz=z/nstep
    s=0.
    d_l = integrate.quad(lambda i: func_1(i,dz), 1, nstep+2)[0]
    d_l=(1.+z)*d_l
    d_mod=5.*np.log10((c/h0)*d_l)+25.
    d_lum=10.*(10.0**(d_mod/5.) )   #pc 
    s1=(np.sqrt(om*(1.+z)**3+(1.-om-ov)*(1.+z)**2+ov)) 
    s2=(np.sqrt(om*(1.+z+dz1)**3+(1.-om-ov)*(1.+z+dz1)**2+ov))
    return (d_lum)**2/(((s1+s2)/2)*(1+z)**2)*dz1
    
def func_3(z_a,dz_):
    zmin=0
    zmax=z_a+dz_
    dz1=(zmax-zmin)/nstep
    dV = integrate.quad(lambda j: func_2(j,dz1,zmin), 1, nstep+1)[0]
    return (4*PI*c/(h0*1E-6)*dV)/1E18

def Vol(z_a,dz_,area):
    V1 = func_3(z_a,dz_)
    V2 = func_3(z_a,-dz_)
    dV_ = V1-V2
    V = (dV_)*area*8.46E-8/(4*PI)
    return V

##############################################################################################################################
#						Calculate luminosity distance
##############################################################################################################################

def s_2(i,dz):
    a=(i-1)*dz
    b=i*dz
    s1=(1/np.sqrt(om*(1.+a)**3+(1.-om-ov)*(1.+a)**2+ov))
    s2=(1/np.sqrt(om*(1.+b)**3+(1.-om-ov)*(1.+b)**2+ov))
    s=dz*(s1+s2)/2.
    return s

def lum_dist(z):
    dz=z/nstep
    s=0.
    d_l=0.
    d_l=integrate.quad(lambda i: s_2(i,dz), 1, nstep+2)[0]
    d_l=(1.+z)*d_l
    d_mod=5.*np.log10((c/h0)*d_l)+25.
    d_lum=10.*(10.0**(d_mod/5.))    #pc
    return d_lum

##############################################################################################################################
#						Calculate expected number
##############################################################################################################################

def s(i,za,bound,dm,mmin):
    Mstar,Phistar,alpha=M_P_a(za,bound)
    a=(i-1)*dm+mmin
    b=i*dm+mmin
    s1 = Phistar*np.log(10.)/2.5*(10**(-0.4*(a-Mstar)))**(alpha+1)*np.exp(-10**(-0.4*(a-Mstar)))
    s2 = Phistar*np.log(10.)/2.5*(10**(-0.4*(b-Mstar)))**(alpha+1)*np.exp(-10**(-0.4*(b-Mstar)))
    s=dm*(s1+s2)/2.
    return s

def No(za,dz,m,bound,area):
    d_lum = lum_dist(za)
    F=10**(-0.4*(m+48.6))*(3E18/(1500*1500*(1+za)**2))
    D=np.log10(4*PI*(1+za))+2*np.log10((d_lum*3E18))+np.log10(F)	#erg/s
    Mabs=-48.6-2.5*(D-np.log10(4*PI)-2*np.log10(10*3E18)+np.log10(1500*1500/3E18))
    mmin=-30.
    mmax=Mabs
    dm=(mmax-mmin)/nstep
    N = integrate.quad(lambda i: s(i,za,bound,dm,mmin), 1, nstep+1)[0]
    volume=Vol(za,dz,area)
    return N*volume
    
zs=np.arange(zmin,zmax+1,1)
for i in range(len(zs)):
    z=zs[i]
    print('z=',z,No(z,dz,m,'mid',area),No(z,dz,m,'high',area),No(z,dz,m,'low',area))

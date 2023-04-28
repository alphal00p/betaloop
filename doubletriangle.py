import vegas
import math
import numpy as np
import matplotlib.pyplot as plt

def E(q):
    return math.sqrt(q[0]**2+q[1]**2+q[2]**2)

def SP(q,p):
    return q[0]*p[0]-q[1]*p[1]-q[2]*p[2]-q[3]*p[3]

def SP_eucl(q,p):
    return q[0]*p[0]+q[1]*p[1]+q[2]*p[2]

def h(t):
    return math.exp(-t**2)/math.sqrt(math.pi)

def h2(t,r_star):
    return math.exp(-t**2-r_star**4/(t**2-r_star**2)**2+1)

def pinch_dampening(pinch, threshold):
    return pinch**4/(pinch**4+threshold**2)

def fake_num(k,l,m):
    return math.exp(-E(k)**2-E(l)**2-E(m)**2)


########################################
########## VIRTUAL DIAGRAMS ############
########################################

class virtuals:

    def __init__(self, q0, DEBUG=False):
        self.q0=q0
        self.debug=DEBUG

    def two_loop(self, k0, k, l, m):
        E0=E(l)
        E1=E(l-m)
        E2=E(m)
        E3=E(k-l)
        E4=E(l)
        E5=E(m)
        q0=self.q0

        #print("esurforig")
        #print(E0+E4-q0)

        result=-0.015625*(1/((E0 + E1 + E2)*(E1 + E4 + E5)*(E1 + E2 + E3 + k0)*(E1 + E2 + E4 - q0)) + 
            1/((E1 + E4 + E5)*(E0 + E3 + k0)*(E1 + E2 + E3 + k0)*(E1 + E2 + E4 - q0)) + 
            1/((E1 + E4 + E5)*(E0 + E3 + k0)*(E0 + E4 - q0)*(E1 + E2 + E4 - q0)) + 
            1/((E0 + E1 + E2)*(E0 + E3 + k0)*(E0 + E4 - q0)*(E0 + E1 + E5 - q0)) + 
            1/((E0 + E1 + E2)*(E1 + E2 + E3 + k0)*(E1 + E2 + E4 - q0)*(E2 + E5 - q0)) + 
            1/((E0 + E3 + k0)*(E1 + E2 + E3 + k0)*(E1 + E2 + E4 - q0)*(E2 + E5 - q0)) + 
            1/((E0 + E3 + k0)*(E0 + E4 - q0)*(E1 + E2 + E4 - q0)*(E2 + E5 - q0)) + 
            1/((E0 + E3 + k0)*(E0 + E4 - q0)*(E0 + E1 + E5 - q0)*(E2 + E5 - q0)) + 
            1/((E0 + E1 + E2)*(E1 + E4 + E5)*(E1 + E2 + E4 - q0)*(E3 + E4 - k0 - q0)) + 
            1/((E1 + E4 + E5)*(E0 + E4 - q0)*(E1 + E2 + E4 - q0)*(E3 + E4 - k0 - q0)) + 
            1/((E0 + E1 + E2)*(E0 + E4 - q0)*(E0 + E1 + E5 - q0)*(E3 + E4 - k0 - q0)) + 
            1/((E0 + E1 + E2)*(E0 + E3 - k0)*(E2 + E5 - q0)*(E3 + E4 - k0 - q0)) + 
            1/((E0 + E1 + E2)*(E1 + E2 + E4 - q0)*(E2 + E5 - q0)*(E3 + E4 - k0 - q0)) + 
            1/((E0 + E4 - q0)*(E1 + E2 + E4 - q0)*(E2 + E5 - q0)*(E3 + E4 - k0 - q0)) + 
            1/((E0 + E4 - q0)*(E0 + E1 + E5 - q0)*(E2 + E5 - q0)*(E3 + E4 - k0 - q0)) + 
            1/((E0 + E1 + E2)*(E1 + E4 + E5)*(E1 + E2 + E3 - k0)*(E1 + E3 + E5 - k0 - q0)) + 
            1/((E1 + E4 + E5)*(E0 + E3 - k0)*(E1 + E2 + E3 - k0)*(E1 + E3 + E5 - k0 - q0)) + 
            1/((E0 + E1 + E2)*(E1 + E4 + E5)*(E0 + E1 + E5 - q0)*(E1 + E3 + E5 - k0 - q0)) + 
            1/((E1 + E4 + E5)*(E0 + E3 - k0)*(E2 + E5 - q0)*(E1 + E3 + E5 - k0 - q0)) + 
            1/((E1 + E4 + E5)*(E0 + E1 + E5 - q0)*(E2 + E5 - q0)*(E1 + E3 + E5 - k0 - q0)) + 
            1/((E0 + E1 + E2)*(E1 + E2 + E3 - k0)*(E3 + E4 - k0 - q0)*(E1 + E3 + E5 - k0 - q0)) + 
            1/((E0 + E3 - k0)*(E1 + E2 + E3 - k0)*(E3 + E4 - k0 - q0)*(E1 + E3 + E5 - k0 - q0)) + 
            1/((E0 + E1 + E2)*(E0 + E1 + E5 - q0)*(E3 + E4 - k0 - q0)*(E1 + E3 + E5 - k0 - q0)) + 
            1/((E0 + E3 - k0)*(E2 + E5 - q0)*(E3 + E4 - k0 - q0)*(E1 + E3 + E5 - k0 - q0)) + 
            1/((E0 + E1 + E5 - q0)*(E2 + E5 - q0)*(E3 + E4 - k0 - q0)*(E1 + E3 + E5 - k0 - q0)) + 
            1/((E0 + E1 + E2)*(E0 + E3 - k0)*(E2 + E5 - q0)*(E0 + E4 + q0)) + 
            1/((E1 + E4 + E5)*(E0 + E3 - k0)*(E2 + E5 - q0)*(E0 + E4 + q0)) + 
            1/((E0 + E1 + E2)*(E1 + E4 + E5)*(E1 + E2 + E3 - k0)*(E1 + E2 + E4 + q0)) + 
            1/((E1 + E4 + E5)*(E0 + E3 - k0)*(E1 + E2 + E3 - k0)*(E1 + E2 + E4 + q0)) + 
            1/((E1 + E4 + E5)*(E0 + E3 - k0)*(E0 + E4 + q0)*(E1 + E2 + E4 + q0)) + 
            1/((E0 + E1 + E2)*(E1 + E4 + E5)*(E3 + E4 - k0 - q0)*(E0 + E1 + E5 + q0)) + 
            1/((E0 + E1 + E2)*(E0 + E3 - k0)*(E3 + E4 - k0 - q0)*(E0 + E1 + E5 + q0)) + 
            1/((E0 + E1 + E2)*(E0 + E3 - k0)*(E0 + E4 + q0)*(E0 + E1 + E5 + q0)) + 
            1/((E0 + E1 + E2)*(E0 + E3 + k0)*(E0 + E4 - q0)*(E2 + E5 + q0)) + 
            1/((E1 + E4 + E5)*(E0 + E3 + k0)*(E0 + E4 - q0)*(E2 + E5 + q0)) + 
            1/((E0 + E1 + E2)*(E1 + E2 + E3 - k0)*(E3 + E4 - k0 - q0)*(E2 + E5 + q0)) + 
            1/((E0 + E3 - k0)*(E1 + E2 + E3 - k0)*(E3 + E4 - k0 - q0)*(E2 + E5 + q0)) + 
            1/((E0 + E1 + E2)*(E0 + E4 - q0)*(E3 + E4 - k0 - q0)*(E2 + E5 + q0)) + 
            1/((E1 + E4 + E5)*(E0 + E4 - q0)*(E3 + E4 - k0 - q0)*(E2 + E5 + q0)) + 
            1/((E0 + E1 + E2)*(E1 + E2 + E3 - k0)*(E1 + E2 + E4 + q0)*(E2 + E5 + q0)) + 
            1/((E0 + E3 - k0)*(E1 + E2 + E3 - k0)*(E1 + E2 + E4 + q0)*(E2 + E5 + q0)) + 
            1/((E0 + E3 - k0)*(E0 + E4 + q0)*(E1 + E2 + E4 + q0)*(E2 + E5 + q0)) + 
            1/((E1 + E4 + E5)*(E3 + E4 - k0 - q0)*(E0 + E1 + E5 + q0)*(E2 + E5 + q0)) + 
            1/((E0 + E3 - k0)*(E3 + E4 - k0 - q0)*(E0 + E1 + E5 + q0)*(E2 + E5 + q0)) + 
            1/((E0 + E3 - k0)*(E0 + E4 + q0)*(E0 + E1 + E5 + q0)*(E2 + E5 + q0)) + 
            1/((E0 + E1 + E2)*(E1 + E4 + E5)*(E0 + E1 + E5 - q0)*(E3 + E4 + k0 + q0)) + 
            1/((E0 + E1 + E2)*(E0 + E3 + k0)*(E0 + E1 + E5 - q0)*(E3 + E4 + k0 + q0)) + 
            1/((E0 + E1 + E2)*(E1 + E2 + E3 + k0)*(E2 + E5 - q0)*(E3 + E4 + k0 + q0)) + 
            1/((E0 + E3 + k0)*(E1 + E2 + E3 + k0)*(E2 + E5 - q0)*(E3 + E4 + k0 + q0)) + 
            1/((E1 + E4 + E5)*(E0 + E1 + E5 - q0)*(E2 + E5 - q0)*(E3 + E4 + k0 + q0)) + 
            1/((E0 + E3 + k0)*(E0 + E1 + E5 - q0)*(E2 + E5 - q0)*(E3 + E4 + k0 + q0)) + 
            1/((E0 + E1 + E2)*(E2 + E5 - q0)*(E0 + E4 + q0)*(E3 + E4 + k0 + q0)) + 
            1/((E1 + E4 + E5)*(E2 + E5 - q0)*(E0 + E4 + q0)*(E3 + E4 + k0 + q0)) + 
            1/((E0 + E1 + E2)*(E1 + E4 + E5)*(E1 + E2 + E4 + q0)*(E3 + E4 + k0 + q0)) + 
            1/((E1 + E4 + E5)*(E0 + E4 + q0)*(E1 + E2 + E4 + q0)*(E3 + E4 + k0 + q0)) + 
            1/((E0 + E1 + E2)*(E0 + E4 + q0)*(E0 + E1 + E5 + q0)*(E3 + E4 + k0 + q0)) + 
            1/((E0 + E1 + E2)*(E0 + E3 + k0)*(E2 + E5 + q0)*(E3 + E4 + k0 + q0)) + 
            1/((E0 + E1 + E2)*(E1 + E2 + E4 + q0)*(E2 + E5 + q0)*(E3 + E4 + k0 + q0)) + 
            1/((E0 + E4 + q0)*(E1 + E2 + E4 + q0)*(E2 + E5 + q0)*(E3 + E4 + k0 + q0)) + 
            1/((E0 + E4 + q0)*(E0 + E1 + E5 + q0)*(E2 + E5 + q0)*(E3 + E4 + k0 + q0)) + 
            1/((E0 + E1 + E2)*(E1 + E4 + E5)*(E1 + E2 + E3 + k0)*(E1 + E3 + E5 + k0 + q0)) + 
            1/((E1 + E4 + E5)*(E0 + E3 + k0)*(E1 + E2 + E3 + k0)*(E1 + E3 + E5 + k0 + q0)) + 
            1/((E0 + E1 + E2)*(E1 + E4 + E5)*(E0 + E1 + E5 + q0)*(E1 + E3 + E5 + k0 + q0)) + 
            1/((E1 + E4 + E5)*(E0 + E3 + k0)*(E2 + E5 + q0)*(E1 + E3 + E5 + k0 + q0)) + 
            1/((E1 + E4 + E5)*(E0 + E1 + E5 + q0)*(E2 + E5 + q0)*(E1 + E3 + E5 + k0 + q0)) + 
            1/((E0 + E1 + E2)*(E1 + E2 + E3 + k0)*(E3 + E4 + k0 + q0)*(E1 + E3 + E5 + k0 + q0)) + 
            1/((E0 + E3 + k0)*(E1 + E2 + E3 + k0)*(E3 + E4 + k0 + q0)*(E1 + E3 + E5 + k0 + q0)) + 
            1/((E0 + E1 + E2)*(E0 + E1 + E5 + q0)*(E3 + E4 + k0 + q0)*(E1 + E3 + E5 + k0 + q0)) + 
            1/((E0 + E3 + k0)*(E2 + E5 + q0)*(E3 + E4 + k0 + q0)*(E1 + E3 + E5 + k0 + q0)) + 
            1/((E0 + E1 + E5 + q0)*(E2 + E5 + q0)*(E3 + E4 + k0 + q0)*(E1 + E3 + E5 + k0 + q0)))/(E0*E1*E2*E3*E4*E5)

        return result

    def two_loop_CT1_E0E4(self, k0, k, l, m):

        q0=self.q0

        r=math.sqrt(E(l)**2+E(m)**2)
        r_starp=(q0/2)*(r/E(l))*np.sign(m[2]) 
        r_starm=-(q0/2)*(r/E(l))*np.sign(m[2]) 

        lp=(r_starp/r)*l
        mp=(r_starp/r)*m

        lm=(r_starm/r)*l
        mm=(r_starm/r)*m

        jac_measure=abs(r_starp**5/r**5)
        jac_surf=abs(2*E(l)/r)

        E0=E(lp)
        E1=E(lp-mp)
        E2=E(mp)
        E3=E(k-lp)
        E4=E(lp)
        E5=E(mp)

        #print("damp1")
        #print((E1 + E2 + E3 + k0))
        #print("damp2")
        #print((E0 + E3 + k0))
        #print("damp3")
        #print((E3 + E4 - k0 - q0))
        #print("damp4")
        #print((E1 + E3 + E5 - k0 - q0))
        #print("damp5")
        #print((E2 + E5 - q0))
        #print("damp6")
        #print((E1 + E2 + E4 - q0))


        resultp=-0.015625*( 
            1/((E1 + E4 + E5)*(E0 + E3 + k0)*(E1 + E2 + E4 - q0)) + 
            1/((E0 + E1 + E2)*(E0 + E3 + k0)*(E0 + E1 + E5 - q0)) + 
            1/((E0 + E3 + k0)*(E1 + E2 + E4 - q0)*(E2 + E5 - q0)) + 
            1/((E0 + E3 + k0)*(E0 + E1 + E5 - q0)*(E2 + E5 - q0)) + 
            1/((E1 + E4 + E5)*(E1 + E2 + E4 - q0)*(E3 + E4 - k0 - q0)) + 
            1/((E0 + E1 + E2)*(E0 + E1 + E5 - q0)*(E3 + E4 - k0 - q0)) + 
            1/((E1 + E2 + E4 - q0)*(E2 + E5 - q0)*(E3 + E4 - k0 - q0)) + 
            1/((E0 + E1 + E5 - q0)*(E2 + E5 - q0)*(E3 + E4 - k0 - q0)) + 
            1/((E0 + E1 + E2)*(E0 + E3 + k0)*(E2 + E5 + q0)) + 
            1/((E1 + E4 + E5)*(E0 + E3 + k0)*(E2 + E5 + q0)) + 
            1/((E0 + E1 + E2)*(E3 + E4 - k0 - q0)*(E2 + E5 + q0)) + 
            1/((E1 + E4 + E5)*(E3 + E4 - k0 - q0)*(E2 + E5 + q0)))/(E0*E1*E2*E3*E4*E5)

        pinch_dampening_p=pinch_dampening((E1 + E2 + E3 + k0)*(E0 + E3 + k0)*(E3 + E4 - k0 - q0)*(E1 + E3 + E5 - k0 - q0),r-r_starp)#min(pinch_dampening((E1 + E2 + E3 + k0),r-r_starm),pinch_dampening((E0 + E3 + k0),r-r_starm),pinch_dampening((E3 + E4 - k0 - q0),r-r_starm),pinch_dampening((E1 + E3 + E5 - k0 - q0),r-r_starm))

        diagp=jac_measure*1/(jac_surf*(r-r_starp))*h2(r-r_starp,r_starp)*pinch_dampening_p*resultp


        if(self.debug):
            print("SURF1")
            print(pinch_dampening((E1 + E2 + E3 + k0)*(E0 + E3 + k0)*(E3 + E4 - k0 - q0)*(E1 + E3 + E5 - k0 - q0),r-r_starp))
            print((E1 + E2 + E3 + k0))
            print((E0 + E3 + k0))
            print((E3 + E4 - k0 - q0))
            print((E1 + E3 + E5 - k0 - q0))
            print((E2 + E5 - q0))
            print((E1 + E2 + E4 - q0))
            print(h2(r-r_starp,r_starp))
            print(r_starp)
            print(r-r_starp)
            print(jac_measure)
            print(jac_surf*(r-r_starp))
            print(resultp)

        E0=E(lm)
        E1=E(lm-mm)
        E2=E(mm)
        E3=E(k-lm)
        E4=E(lm)
        E5=E(mm)

        #print("damp5")
        #print((E2 + E5 - q0))
        
        resultm=-0.015625*( 
            1/((E1 + E4 + E5)*(E0 + E3 + k0)*(E1 + E2 + E4 - q0)) + 
            1/((E0 + E1 + E2)*(E0 + E3 + k0)*(E0 + E1 + E5 - q0)) +            
            1/((E0 + E3 + k0)*(E1 + E2 + E4 - q0)*(E2 + E5 - q0)) + 
            1/((E0 + E3 + k0)*(E0 + E1 + E5 - q0)*(E2 + E5 - q0)) +            
            1/((E1 + E4 + E5)*(E1 + E2 + E4 - q0)*(E3 + E4 - k0 - q0)) + 
            1/((E0 + E1 + E2)*(E0 + E1 + E5 - q0)*(E3 + E4 - k0 - q0)) +         
            1/((E1 + E2 + E4 - q0)*(E2 + E5 - q0)*(E3 + E4 - k0 - q0)) + 
            1/((E0 + E1 + E5 - q0)*(E2 + E5 - q0)*(E3 + E4 - k0 - q0)) +             
            1/((E0 + E1 + E2)*(E0 + E3 + k0)*(E2 + E5 + q0)) + 
            1/((E1 + E4 + E5)*(E0 + E3 + k0)*(E2 + E5 + q0)) +             
            1/((E0 + E1 + E2)*(E3 + E4 - k0 - q0)*(E2 + E5 + q0)) + 
            1/((E1 + E4 + E5)*(E3 + E4 - k0 - q0)*(E2 + E5 + q0)))/(E0*E1*E2*E3*E4*E5)

        pinch_dampening_m=pinch_dampening((E1 + E2 + E3 + k0)*(E0 + E3 + k0)*(E3 + E4 - k0 - q0)*(E1 + E3 + E5 - k0 - q0),r-r_starp)#pinch_dampening((E1 + E2 + E3 + k0),r-r_starm)*pinch_dampening((E0 + E3 + k0),r-r_starm)*pinch_dampening((E3 + E4 - k0 - q0),r-r_starm)*pinch_dampening((E1 + E3 + E5 - k0 - q0),r-r_starm)

        diagm=jac_measure*1/(jac_surf*(r-r_starm))*h2(r-r_starm,r_starm)*pinch_dampening_m*resultm

        #print("CTM")
        #print(resultp)
        #print(resultm)
        #print(r-r_starm)
        #print(pinch_dampening((E0 + E3 + k0)*(E3 + E4 - k0 - q0),r-r_starm))
        #print((E0 + E1 + E5 - q0))
        #print((E2 + E5 + q0))
        #print(E0*E1*E2*E3*E4*E5)
        #print(resultm)

        return diagp+diagm


    def two_loop_CT2_E2E5(self, k0, k, l, m):

        q0=self.q0

        r=math.sqrt(E(l)**2+E(m)**2)
        r_starp=(q0/2)*(r/E(m))*np.sign(m[2]) 
        r_starm=-(q0/2)*(r/E(m))*np.sign(m[2]) 

        lp=(r_starp/r)*l
        mp=(r_starp/r)*m

        lm=(r_starm/r)*l
        mm=(r_starm/r)*m

        jac_measure=abs(r_starp**5/r**5)
        jac_surf=abs(2*E(m)/r)

        E0=E(lp)
        E1=E(lp-mp)
        E2=E(mp)
        E3=E(k-lp)
        E4=E(lp)
        E5=E(mp)


        

        #print("damp1")
        #print((E1 + E2 + E3 + k0))
        #print("damp2")
        #print((E0 + E3 + k0))
        #print("damp3")
        #print((E3 + E4 - k0 - q0))
        #print("damp4")
        #print((E1 + E3 + E5 - k0 - q0))
        #print("damp6")
        #print((E1 + E2 + E4 - q0))



        resultp=-0.015625*(
            1/((E0 + E1 + E2)*(E1 + E2 + E3 + k0)*(E1 + E2 + E4 - q0)) + 
            1/((E0 + E3 + k0)*(E1 + E2 + E3 + k0)*(E1 + E2 + E4 - q0)) + 
            1/((E0 + E3 + k0)*(E0 + E4 - q0)*(E1 + E2 + E4 - q0)) + 
            1/((E0 + E3 + k0)*(E0 + E4 - q0)*(E0 + E1 + E5 - q0)) + 
            1/((E0 + E1 + E2)*(E0 + E3 - k0)*(E3 + E4 - k0 - q0)) + 
            1/((E0 + E1 + E2)*(E1 + E2 + E4 - q0)*(E3 + E4 - k0 - q0)) + 
            1/((E0 + E4 - q0)*(E1 + E2 + E4 - q0)*(E3 + E4 - k0 - q0)) + 
            1/((E0 + E4 - q0)*(E0 + E1 + E5 - q0)*(E3 + E4 - k0 - q0)) +            
            1/((E1 + E4 + E5)*(E0 + E3 - k0)*(E1 + E3 + E5 - k0 - q0)) + 
            1/((E1 + E4 + E5)*(E0 + E1 + E5 - q0)*(E1 + E3 + E5 - k0 - q0)) +           
            1/((E0 + E3 - k0)*(E3 + E4 - k0 - q0)*(E1 + E3 + E5 - k0 - q0)) + 
            1/((E0 + E1 + E5 - q0)*(E3 + E4 - k0 - q0)*(E1 + E3 + E5 - k0 - q0)) + 
            1/((E0 + E1 + E2)*(E0 + E3 - k0)*(E0 + E4 + q0)) + 
            1/((E1 + E4 + E5)*(E0 + E3 - k0)*(E0 + E4 + q0)) +
            1/((E0 + E1 + E2)*(E1 + E2 + E3 + k0)*(E3 + E4 + k0 + q0)) + 
            1/((E0 + E3 + k0)*(E1 + E2 + E3 + k0)*(E3 + E4 + k0 + q0)) + 
            1/((E1 + E4 + E5)*(E0 + E1 + E5 - q0)*(E3 + E4 + k0 + q0)) + 
            1/((E0 + E3 + k0)*(E0 + E1 + E5 - q0)*(E3 + E4 + k0 + q0)) + 
            1/((E0 + E1 + E2)*(E0 + E4 + q0)*(E3 + E4 + k0 + q0)) + 
            1/((E1 + E4 + E5)*(E0 + E4 + q0)*(E3 + E4 + k0 + q0)))/(E0*E1*E2*E3*E4*E5)


        if(self.debug):
            print("SURF2")
            print(pinch_dampening((E1 + E2 + E3 + k0)*(E0 + E3 + k0)*(E3 + E4 - k0 - q0)*(E1 + E3 + E5 - k0 - q0),r-r_starp))
            print((E1 + E2 + E3 + k0))
            print((E0 + E3 + k0))
            print((E3 + E4 - k0 - q0))
            print((E1 + E3 + E5 - k0 - q0))
            print((E0 + E4 - q0))
            print((E1 + E2 + E4 - q0))
            print(h2(r-r_starp,r_starp))
            print(r_starp)
            print(r-r_starp)
            print(jac_measure)
            print(jac_surf*(r-r_starp))
            print(resultp)


        diagp=jac_measure*1/(jac_surf*(r-r_starp))*h2(r-r_starp,r_starp)*pinch_dampening((E1 + E2 + E3 + k0)*(E0 + E3 + k0)*(E3 + E4 - k0 - q0)*(E1 + E3 + E5 - k0 - q0),r-r_starp)*resultp

        E0=E(lm)
        E1=E(lm-mm)
        E2=E(mm)
        E3=E(k-lm)
        E4=E(lm)
        E5=E(mm)


        resultm=-0.015625*(
            1/((E0 + E1 + E2)*(E1 + E2 + E3 + k0)*(E1 + E2 + E4 - q0)) + 
            1/((E0 + E3 + k0)*(E1 + E2 + E3 + k0)*(E1 + E2 + E4 - q0)) + 
            1/((E0 + E3 + k0)*(E0 + E4 - q0)*(E1 + E2 + E4 - q0)) + 
            1/((E0 + E3 + k0)*(E0 + E4 - q0)*(E0 + E1 + E5 - q0)) + 
            1/((E0 + E1 + E2)*(E0 + E3 - k0)*(E3 + E4 - k0 - q0)) + 
            1/((E0 + E1 + E2)*(E1 + E2 + E4 - q0)*(E3 + E4 - k0 - q0)) + 
            1/((E0 + E4 - q0)*(E1 + E2 + E4 - q0)*(E3 + E4 - k0 - q0)) + 
            1/((E0 + E4 - q0)*(E0 + E1 + E5 - q0)*(E3 + E4 - k0 - q0)) +            
            1/((E1 + E4 + E5)*(E0 + E3 - k0)*(E1 + E3 + E5 - k0 - q0)) + 
            1/((E1 + E4 + E5)*(E0 + E1 + E5 - q0)*(E1 + E3 + E5 - k0 - q0)) +           
            1/((E0 + E3 - k0)*(E3 + E4 - k0 - q0)*(E1 + E3 + E5 - k0 - q0)) + 
            1/((E0 + E1 + E5 - q0)*(E3 + E4 - k0 - q0)*(E1 + E3 + E5 - k0 - q0)) + 
            1/((E0 + E1 + E2)*(E0 + E3 - k0)*(E0 + E4 + q0)) + 
            1/((E1 + E4 + E5)*(E0 + E3 - k0)*(E0 + E4 + q0)) +
            1/((E0 + E1 + E2)*(E1 + E2 + E3 + k0)*(E3 + E4 + k0 + q0)) + 
            1/((E0 + E3 + k0)*(E1 + E2 + E3 + k0)*(E3 + E4 + k0 + q0)) + 
            1/((E1 + E4 + E5)*(E0 + E1 + E5 - q0)*(E3 + E4 + k0 + q0)) + 
            1/((E0 + E3 + k0)*(E0 + E1 + E5 - q0)*(E3 + E4 + k0 + q0)) + 
            1/((E0 + E1 + E2)*(E0 + E4 + q0)*(E3 + E4 + k0 + q0)) + 
            1/((E1 + E4 + E5)*(E0 + E4 + q0)*(E3 + E4 + k0 + q0)))/(E0*E1*E2*E3*E4*E5)

        diagm=jac_measure*1/(jac_surf*(r-r_starm))*h2(r-r_starm,r_starm)*pinch_dampening((E1 + E2 + E3 + k0)*(E0 + E3 + k0)*(E3 + E4 - k0 - q0)*(E1 + E3 + E5 - k0 - q0),r-r_starm)*resultm


        #print("-CT2")
        #print(resultp)
        #print(resultm)
        #print(pinch_dampening((E1 + E2 + E3 + k0)*(E0 + E3 + k0)*(E3 + E4 - k0 - q0)*(E1 + E3 + E5 - k0 - q0),r-r_starm))

        return diagp+diagm

    def two_loop_CT3(self, k0, k, l, m):

        q0=self.q0

        r=math.sqrt(E(l)**2+E(m)**2)
        r_starp=q0*(r/(E(m)+E(l)+E(l-m)))*np.sign(m[2]) 
        r_starm=-q0*(r/(E(m)+E(l)+E(l-m)))*np.sign(m[2]) 

        lp=(r_starp/r)*l
        mp=(r_starp/r)*m

        lm=(r_starm/r)*l
        mm=(r_starm/r)*m

        jac_measure=abs(r_starp**5/r**5)
        jac_surf=abs((E(m)+E(l)+E(l-m))/r) #abs((E(m)+E(l)+E(l-m))/r_starp)



        E0=E(lp)
        E1=E(lp-mp)
        E2=E(mp)
        E3=E(k-lp)
        E4=E(lp)
        E5=E(mp)


        

        #LOOK ESURFACES HERE

        resultp=-0.015625*(1/((E0 + E1 + E2)*(E1 + E4 + E5)*(E1 + E2 + E3 + k0)) + 
            1/((E1 + E4 + E5)*(E0 + E3 + k0)*(E1 + E2 + E3 + k0)) + 
            1/((E1 + E4 + E5)*(E0 + E3 + k0)*(E0 + E4 - q0)) + 
            1/((E0 + E1 + E2)*(E0 + E3 + k0)*(E0 + E4 - q0)) + 
            1/((E0 + E1 + E2)*(E1 + E2 + E3 + k0)*(E2 + E5 - q0)) + 
            1/((E0 + E3 + k0)*(E1 + E2 + E3 + k0)*(E2 + E5 - q0)) + 
            1/((E0 + E3 + k0)*(E0 + E4 - q0)*(E2 + E5 - q0)) + 
            1/((E0 + E3 + k0)*(E0 + E4 - q0)*(E2 + E5 - q0)) + 
            1/((E0 + E1 + E2)*(E1 + E4 + E5)*(E3 + E4 - k0 - q0)) + 
            1/((E1 + E4 + E5)*(E0 + E4 - q0)*(E3 + E4 - k0 - q0)) + 
            1/((E0 + E1 + E2)*(E0 + E4 - q0)*(E3 + E4 - k0 - q0)) + 
            1/((E0 + E1 + E2)*(E2 + E5 - q0)*(E3 + E4 - k0 - q0)) + 
            1/((E0 + E4 - q0)*(E2 + E5 - q0)*(E3 + E4 - k0 - q0)) + 
            1/((E0 + E4 - q0)*(E2 + E5 - q0)*(E3 + E4 - k0 - q0)) +     
            1/((E0 + E1 + E2)*(E1 + E4 + E5)*(E1 + E3 + E5 - k0 - q0)) + 
            1/((E1 + E4 + E5)*(E2 + E5 - q0)*(E1 + E3 + E5 - k0 - q0)) + 
            1/((E0 + E1 + E2)*(E3 + E4 - k0 - q0)*(E1 + E3 + E5 - k0 - q0)) + 
            1/((E2 + E5 - q0)*(E3 + E4 - k0 - q0)*(E1 + E3 + E5 - k0 - q0)) + 
            1/((E0 + E1 + E2)*(E1 + E4 + E5)*(E3 + E4 + k0 + q0)) + 
            1/((E0 + E1 + E2)*(E0 + E3 + k0)*(E3 + E4 + k0 + q0)) +      
            1/((E1 + E4 + E5)*(E2 + E5 - q0)*(E3 + E4 + k0 + q0)) + 
            1/((E0 + E3 + k0)*(E2 + E5 - q0)*(E3 + E4 + k0 + q0)))/(E0*E1*E2*E3*E4*E5)



        if(self.debug):
            print("SURF3")
            print(pinch_dampening((E1 + E2 + E3 + k0)*(E0 + E3 + k0)*(E3 + E4 - k0 - q0)*(E1 + E3 + E5 - k0 - q0),r-r_starp))
            print((E1 + E2 + E3 + k0))
            print((E0 + E3 + k0))
            print((E3 + E4 - k0 - q0))
            print((E1 + E3 + E5 - k0 - q0))
            print((E2 + E5 - q0))
            print((E0 + E4 - q0))
            print(h2(r-r_starp,r_starp))
            print(r_starp)
            print(r-r_starp)
            print(jac_measure)
            print(jac_surf*(r-r_starp))
            print(resultp)



        """resultp=-0.015625*(1/((E0 + E1 + E2)*(E1 + E4 + E5)*(E1 + E2 + E3 + k0)) + 
            1/((E1 + E4 + E5)*(E0 + E3 + k0)*(E1 + E2 + E3 + k0)) + 
            1/((E1 + E4 + E5)*(E0 + E3 + k0)*(E0 + E4 - q0)) + 
            1/((E0 + E1 + E2)*(E1 + E2 + E3 + k0)*(E2 + E5 - q0)) + 
            1/((E0 + E3 + k0)*(E1 + E2 + E3 + k0)*(E2 + E5 - q0)) + 
            1/((E0 + E3 + k0)*(E0 + E4 - q0)*(E2 + E5 - q0)) + 
            1/((E0 + E1 + E2)*(E1 + E4 + E5)*(E3 + E4 - k0 - q0)) + 
            1/((E1 + E4 + E5)*(E0 + E4 - q0)*(E3 + E4 - k0 - q0)) + 
            1/((E0 + E1 + E2)*(E2 + E5 - q0)*(E3 + E4 - k0 - q0)) + 
            1/((E0 + E4 - q0)*(E2 + E5 - q0)*(E3 + E4 - k0 - q0)))/(E0*E1*E2*E3*E4*E5)"""

        diagp=jac_measure*1/(jac_surf*(r-r_starp))*h2(r-r_starp,r_starp)*pinch_dampening((E1 + E2 + E3 + k0)*(E0 + E3 + k0)*(E3 + E4 - k0 - q0)*(E1 + E3 + E5 - k0 - q0),r-r_starp)*resultp


        E0=E(lm)
        E1=E(lm-mm)
        E2=E(mm)
        E3=E(k-lm)
        E4=E(lm)
        E5=E(mm)


        resultm=-0.015625*(1/((E0 + E1 + E2)*(E1 + E4 + E5)*(E1 + E2 + E3 + k0)) + 
            1/((E1 + E4 + E5)*(E0 + E3 + k0)*(E1 + E2 + E3 + k0)) + 
            1/((E1 + E4 + E5)*(E0 + E3 + k0)*(E0 + E4 - q0)) + 
            1/((E0 + E1 + E2)*(E0 + E3 + k0)*(E0 + E4 - q0)) + 
            1/((E0 + E1 + E2)*(E1 + E2 + E3 + k0)*(E2 + E5 - q0)) + 
            1/((E0 + E3 + k0)*(E1 + E2 + E3 + k0)*(E2 + E5 - q0)) + 
            1/((E0 + E3 + k0)*(E0 + E4 - q0)*(E2 + E5 - q0)) + 
            1/((E0 + E3 + k0)*(E0 + E4 - q0)*(E2 + E5 - q0)) + 
            1/((E0 + E1 + E2)*(E1 + E4 + E5)*(E3 + E4 - k0 - q0)) + 
            1/((E1 + E4 + E5)*(E0 + E4 - q0)*(E3 + E4 - k0 - q0)) + 
            1/((E0 + E1 + E2)*(E0 + E4 - q0)*(E3 + E4 - k0 - q0)) + 
            1/((E0 + E1 + E2)*(E2 + E5 - q0)*(E3 + E4 - k0 - q0)) + 
            1/((E0 + E4 - q0)*(E2 + E5 - q0)*(E3 + E4 - k0 - q0)) + 
            1/((E0 + E4 - q0)*(E2 + E5 - q0)*(E3 + E4 - k0 - q0)) +     
            1/((E0 + E1 + E2)*(E1 + E4 + E5)*(E1 + E3 + E5 - k0 - q0)) + 
            1/((E1 + E4 + E5)*(E2 + E5 - q0)*(E1 + E3 + E5 - k0 - q0)) + 
            1/((E0 + E1 + E2)*(E3 + E4 - k0 - q0)*(E1 + E3 + E5 - k0 - q0)) + 
            1/((E2 + E5 - q0)*(E3 + E4 - k0 - q0)*(E1 + E3 + E5 - k0 - q0)) + 
            1/((E0 + E1 + E2)*(E1 + E4 + E5)*(E3 + E4 + k0 + q0)) + 
            1/((E0 + E1 + E2)*(E0 + E3 + k0)*(E3 + E4 + k0 + q0)) +      
            1/((E1 + E4 + E5)*(E2 + E5 - q0)*(E3 + E4 + k0 + q0)) + 
            1/((E0 + E3 + k0)*(E2 + E5 - q0)*(E3 + E4 + k0 + q0)))/(E0*E1*E2*E3*E4*E5)


        diagm=jac_measure*1/(jac_surf*(r-r_starm))*h2(r-r_starm,r_starm)*pinch_dampening((E1 + E2 + E3 + k0)*(E0 + E3 + k0)*(E3 + E4 - k0 - q0)*(E1 + E3 + E5 - k0 - q0),r-r_starm)*resultm   

        return (diagp+diagm)



    def one_loop(self, l0, k, l, m):
        E0=E(m)
        E1=E(l-m)
        E2=E(m)
        q0=self.q0

        #print("surf1")
        #print((E0 + E1 + l0))
        #print("surf2")
        #print((E0 + E2 - q0))
        print((E1 + E2 - l0 - q0))

        result=(-0.125*(-(1/((E0 + E1 + l0)*(E0 + E2 - q0))) - 1/((E0 + E1 - l0)*(E1 + E2 - l0 - q0)) - 
            1/((E0 + E2 - q0)*(E1 + E2 - l0 - q0)) - 1/((E0 + E1 - l0)*(E0 + E2 + q0)) - 
            1/((E0 + E1 + l0)*(E1 + E2 + l0 + q0)) - 1/((E0 + E2 + q0)*(E1 + E2 + l0 + q0))))/(E0*E1*E2)

        return result

    def one_loop_CT1(self, l0, k, l, m):

        q0=self.q0

        r=E(m)
        r_starp=(q0/2)*(r/E(m))*np.sign(m[2]) 
        r_starm=-(q0/2)*(r/E(m))*np.sign(m[2]) 

        mp=(r_starp/r)*m
        mm=(r_starm/r)*m

        jac_measure=abs(r_starp**2/r**2)
        jac_surf=2#2*abs(E(m)/r_starp)

        E0=E(mp)
        E1=E(l-mp)
        E2=E(mp)

        E3=E(l)
        E4=E(k)
        E5=E(k-l)

        resultp=(-0.125*(-(1/((E0 + E1 + l0))) - 
            1/((E1 + E2 - l0 - q0))))/(E0*E1*E2)




        diagp=jac_measure*1/(jac_surf*(r-r_starp))*h2(r-r_starp,r_starp)*pinch_dampening((E0 + E1 + l0)*(E1 + E2 - l0 - q0),r-r_starp)*resultp

        #print(h2(r-r_starp,r_starp))

        E0=E(mm)
        E1=E(l-mm)
        E2=E(mm)

        resultm=(-0.125*(-(1/((E0 + E1 + l0))) - 
            1/((E1 + E2 - l0 - q0))))/(E0*E1*E2)


        diagm=jac_measure*1/(jac_surf*(r-r_starm))*h2(r-r_starm,r_starm)*pinch_dampening((E0 + E1 + l0)*(E1 + E2 - l0 - q0),r-r_starm)*resultm        

        return diagp+diagm



    def one_loop_VL_CT1(self, l0, k, l, m):

        q0=self.q0

        r=E(m)
        r_starp=(q0/2)*(r/E(m))*np.sign(m[2]) 
        r_starm=-(q0/2)*(r/E(m))*np.sign(m[2]) 

        mp=(r_starp/r)*m
        mm=(r_starm/r)*m

        jac_measure=abs(r_starp**2/r**2)
        jac_surf=2#2*abs(E(m)/r_starp)

        E0=E(mp)
        E1=E(l-mp)
        E2=E(mp)

        E3=E(l)
        E4=E(k)
        E5=E(k-l)

        resultp=(-0.125*(-(1/((E0 + E1 + l0))) - 
            1/((E1 + E2 - l0 - q0))))/(E0*E1*E2)


        diagp=jac_measure*1/(jac_surf*(r-r_starp))*h2(r-r_starp,r_starp)*pinch_dampening((E0 + E1 + l0)*(E1 + E2 - l0 - q0)*(2*E3-q0)*(2*E4-q0),r-r_starp)*resultp

        #print(h2(r-r_starp,r_starp))

        E0=E(mm)
        E1=E(l-mm)
        E2=E(mm)

        resultm=(-0.125*(-(1/((E0 + E1 + l0))) - 
            1/((E1 + E2 - l0 - q0))))/(E0*E1*E2)


        diagm=jac_measure*1/(jac_surf*(r-r_starm))*h2(r-r_starm,r_starm)*pinch_dampening((E0 + E1 + l0)*(E1 + E2 - l0 - q0)*(2*E3-q0)*(2*E4-q0),r-r_starm)*resultm        

        return diagp+diagm


    def one_loop_VL_CT2(self, l0, k, l, m):

        q0=self.q0



        r=E(m)

        r_starp=((l0+q0)**2-E(l)**2)/(2*((l0+q0)-SP_eucl(l,m)/r))
        r_starm=((l0+q0)**2-E(l)**2)/(2*(-(l0+q0)-SP_eucl(l,m)/r))
        mp=(r_starp/r)*m
        mm=(r_starm/r)*m

        jac_measure_p=abs(r_starp**2/r**2)
        jac_measure_m=abs(r_starm**2/r**2)
        jac_surf_p=abs(1+(r_starp-SP_eucl(l,m)/r)/E(mp-l))
        jac_surf_m=abs(1+(r_starm+SP_eucl(l,m)/r)/E(mm-l))

        E0=E(mp)
        E1=E(l-mp)
        E2=E(mp)

        E3=E(l)
        E4=E(k)
        E5=E(k-l)
        
        """
        print("---")
        print(r)
        print(r_starp)
        print((q0**2-2*q0*E(l))/(2*(q0-E(l)-SP_eucl(l,m)/r))*np.sign(m[2]) )
        print(q0)
        print((E0 + E1 - l0))
        print((E0 + E2 - q0))
        """
        
        
        
        

        resultp=(-0.125*(- 1/(E0 + E1 - l0) - 1/(E0 + E2 - q0)))/(E0*E1*E2)


        diagp=jac_measure_p*1/(jac_surf_p*(r-r_starp))*h2(r-r_starp,r_starp)*pinch_dampening((E0 + E1 + l0)*(2*E3-q0)*(2*E4-q0),r-r_starp)*resultp
        """
        print("p")
        print(r_starp)
        print(jac_surf_p)
        print(jac_surf_p*(r-r_starp))
        print(pinch_dampening((E0 + E1 + l0)*(2*E3-q0)*(2*E4-q0),r-r_starp))
        print(h2(r-r_starp,r_starp))
        print(jac_measure_p)

        print("m")
        print(r_starm)
        print(jac_surf_m)
        print(jac_surf_m*(r-r_starm))
        print(pinch_dampening((E0 + E1 + l0)*(2*E3-q0)*(2*E4-q0),r-r_starm))
        print(h2(r-r_starm,r_starm))
        print(jac_measure_m)
        """

        E0=E(mm)
        E1=E(l-mm)
        E2=E(mm)

        resultm=(-0.125*(- 1/(E0 + E1 - l0) - 1/(E0 + E2 - q0)))/(E0*E1*E2)


        diagm=jac_measure_m*1/(jac_surf_m*(r-r_starm))*h2(r-r_starm,r_starm)*pinch_dampening((E0 + E1 + l0)*(E1 + E2 - l0 - q0)*(2*E3-q0)*(2*E4-q0),r-r_starm)*resultm        

        return diagp+diagm




    
        



########################################
############ CROSS SECTION #############
########################################


class x_section:

    def __init__(self, q0, ct, debug=False):
        self.q0=q0
        self.ct=ct
        if(debug):
            self.loops=virtuals(q0, debug)
        else:
            self.loops=virtuals(q0, debug)
        self.debug=debug

    ######################################
    ########## DOUBLE VIRTUALS ###########
    ######################################

    def VVL(self,k,l,m):
        k0=-E(k)
        diag=self.loops.two_loop(k0,k,l,m)
        diagCT=self.loops.two_loop_CT1_E0E4(k0,k,l,m)+self.loops.two_loop_CT2_E2E5(k0,k,l,m)+self.loops.two_loop_CT3(k0,k,l,m)
        if(self.debug):
            print("ORIG")
            print(diag)
            print("CT1")
            print(self.loops.two_loop_CT1_E0E4(k0,k,l,m))
            print("CT2")
            print(self.loops.two_loop_CT2_E2E5(k0,k,l,m))
            print("CT3")
            print(self.loops.two_loop_CT3(k0,k,l,m))
            print("TOT DIAG CT")
            print(diagCT)
            print("----")
        return 1/(-2*k0*2*(k0+self.q0))*(diag-self.ct*diagCT)

    def VVL_xSec(self,k,l,m):
        k0=-E(k)
        t=self.q0/(2*E(k))
        jac=t**9/(2*E(k))
        diag=self.VVL(t*k,t*l,t*m)
        return h(t)*jac*diag*fake_num(t*k,t*l,t*m)

    def VVR_xSec(self,k,l,m):
        return self.VVL_xSec(-m,-l,-k)

    def VVM(self,k,l,m):
        l0=-E(l)
        diag=self.loops.one_loop(l0,k,l,m)*self.loops.one_loop(l0,m,l,k)
        diagCT=-self.loops.one_loop(l0,k,l,m)*self.loops.one_loop_CT1(l0,m,l,k)-self.loops.one_loop_CT1(l0,k,l,m)*self.loops.one_loop(l0,m,l,k)+self.loops.one_loop_CT1(l0,k,l,m)*self.loops.one_loop_CT1(l0,m,l,k)
        if(self.debug):
            print("ORIG")
            print(diag)
            print("CT1")
            print(self.loops.one_loop_CT1(k0,k,l,m))

        return -1/(-2*l0*2*(l0+self.q0))*(diag+self.ct*diagCT)
        

    def VVM_xSec(self,k,l,m):
        t=self.q0/(2*E(l))
        jac=t**9/(2*E(l))
        diag=self.VVM(t*k,t*l,t*m)
        return h(t)*jac*diag*fake_num(t*k,t*l,t*m)


    ######################################
    ########## SINGLE VIRTUALS ###########
    ######################################

    def VL(self,k,l,m):
        k0=-E(k)
        l0=-self.q0+E(l)
        l4=np.array([l0,l[0],l[1],l[2]])
        kq4=np.array([k0+self.q0,k[0],k[1],k[2]])
        diag=self.loops.one_loop(l0,k,l,m)
        diagCT=self.loops.one_loop_VL_CT1(l0,k,l,m)
        

        #DIAGCT INHERITS TREE SINGULARITY
        return -1/(-2*k0*2*(k0-l0)*2*(l0+self.q0))*1/SP(l4,l4)*1/SP(kq4,kq4)*(diag-self.ct*diagCT)

    def VL_xSec(self,k,l,m):
        t=self.q0/(E(k)+E(k-l)+E(l))
        jac=t**9/(E(k)+E(k-l)+E(l))
        diag=self.VL(t*k,t*l,t*m)
        return 2*h(t)*jac*diag*fake_num(t*k,t*l,t*m)  

    def VR_xSec(self,k,l,m):
        return self.VL_xSec(-m,-l,-k)

    ######################################
    ############## REALS #################
    ######################################

    def R(self,k,l,m):
        k0=-E(k)
        m0=-self.q0+E(m)
        l0=m0+E(l-m)

        l4=np.array([l0,l[0],l[1],l[2]])
        m4=np.array([m0,m[0],m[1],m[2]])
        kq4=np.array([k0+self.q0,k[0],k[1],k[2]])
        lq4=np.array([l0+self.q0,l[0],l[1],l[2]])

        return -1/(-2*k0*2*(k0-l0)*2*(l0-m0)*2*(m0+self.q0))*1/SP(l4,l4)*1/SP(m4,m4)*1/SP(kq4,kq4)*1/SP(lq4,lq4)

    def R_xSec(self,k,l,m):
        t=self.q0/(E(k)+E(k-l)+E(l-m)+E(m))
        jac=t**9/(E(k)+E(k-l)+E(l-m)+E(m))
        diag=self.R(t*k,t*l,t*m)
        return 2*h(t)*jac*diag*fake_num(t*k,t*l,t*m)

    def eval(self,k,l,m):
        return self.VVL_xSec(k,l,m)+self.VVR_xSec(k,l,m)+self.VVM_xSec(k,l,m)+self.VL_xSec(k,l,m)+self.VR_xSec(k,l,m)+self.R_xSec(k,l,m)
        

        

class integrand:

    def __init__(self,q0,ct):
        self.my_x_section=x_section(q0,ct)
        self.max_eval=0
        self.max_p=[]
        self.max_jac=0

    def test_function(self, k, l, m):
        return 1/(math.sqrt(math.pi)**9)*math.exp(-E(k)**2-E(l)**2-E(m)**2)

    def x_parametrise(self, x):
        r1=np.float128(x[0]/(1-x[0]))
        th1=np.float128(2*math.pi*x[1])
        ph1=np.float128(math.pi*x[2])
        r2=np.float128(x[3]/(1-x[3]))
        th2=np.float128(2*math.pi*x[4])
        ph2=np.float128(math.pi*x[5])
        r3=np.float128(x[6]/(1-x[6]))
        th3=np.float128(2*math.pi*x[7])
        ph3=np.float128(math.pi*x[8])

        kl=np.array([r1*math.cos(th1)*math.sin(ph1),r1*math.sin(th1)*math.sin(ph1),r1*math.cos(ph1)])
        lm=np.array([r2*math.cos(th2)*math.sin(ph2),r2*math.sin(th2)*math.sin(ph2),r2*math.cos(ph2)])
        m=np.array([r3*math.cos(th3)*math.sin(ph3),r3*math.sin(th3)*math.sin(ph3),r3*math.cos(ph3)])

        k=kl+lm+m
        l=lm+m

        jacobian_spherical=pow(r1,2)*math.sin(ph1)*pow(r2,2)*math.sin(ph2)*pow(r3,2)*math.sin(ph3)
        jacobian_x=2*pow(math.pi,2)/pow(1-x[0],2)*2*pow(math.pi,2)/pow(1-x[3],2)*2*pow(math.pi,2)/pow(1-x[6],2)

        if abs(np.dot(k,l))>E(k)*E(l)-0.00001 or abs(np.dot(m,l))>E(m)*E(l)-0.00001 or E(k)<0.00001 or E(l)<0.00001 or E(m)<0.00001:
            return 0

        if abs(jacobian_x*jacobian_spherical*self.my_x_section.eval(k,l,m))>abs(self.max_eval):
            self.max_eval=jacobian_x*jacobian_spherical*self.my_x_section.eval(k,l,m)
            self.max_p=[k,l,m]
            self.max_jac=jacobian_x*jacobian_spherical

        return jacobian_x*jacobian_spherical*self.my_x_section.eval(k,l,m)



if __name__=="__main__":

    my_virtuals=virtuals(1)
    my_xsec=x_section(1,1)
    ex_integrand=integrand(1,1)


    l0=1
    k0=1
    k=np.array([1,2,3])
    l=np.array([0,0,1])
    m=np.array([1/math.sqrt(2),1/2,1/2])
    #m=np.array([3,-1,2])

    v=np.array([1,1,1])

    for i in range(0,6):
        print(my_virtuals.one_loop(l0,k,l,m+10**(-i)*v))
    print("------")
    for i in range(0,6):
        print(my_virtuals.one_loop_VL_CT1(l0,k,l,m+10**(-i)*v))
    print("------")
    for i in range(0,6):
        print(my_virtuals.one_loop_VL_CT2(l0,k,l,m+10**(-i)*v))
    print("------")


    print("************")


    l0=1
    k0=1
    k=np.array([1,2,3])
    l=np.array([0,0,-1])
    m=np.array([1/math.sqrt(2),1/2,-1/2])
    #m=np.array([3,-1,2])

    v=np.array([1,1,1])

    for i in range(0,6):
        print(my_virtuals.one_loop(l0,k,l,m+10**(-i)*v))
    print("------")
    for i in range(0,6):
        print(my_virtuals.one_loop_VL_CT1(l0,k,l,m+10**(-i)*v))
    print("------")
    for i in range(0,6):
        print(my_virtuals.one_loop_VL_CT2(l0,k,l,m+10**(-i)*v))
    print("------")


    """ 

    l0=1
    k0=1
    k=np.array([1/math.sqrt(8),0,1/math.sqrt(8)])
    l=np.array([1/2,-3/4,1/4])
    m=np.array([0,1/math.sqrt(8),1/math.sqrt(8)])

    v=np.array([1,1,1])


    for i in range(0,6):
        print(my_xsec.eval(k,l,m+10**(-i)*v)) 
    print("------")


    k=np.array([1/math.sqrt(8),0,1/math.sqrt(8)])
    m=np.array([1/2,-3/2,1/4])
    l=np.array([0,1/math.sqrt(8),-1/math.sqrt(8)])
    """
    """
    for i in range(0,6):
        print(my_xsec.eval(k,l+10**(-i)*v,m)) 
    print("------")

    for i in range(0,6):
        print(my_xsec.VVL_xSec(k,l+10**(-i)*v,m)) 
    print("------")
    for i in range(0,6):
        print(my_xsec.VVR_xSec(k,l+10**(-i)*v,m)) 
    print("------")
    for i in range(0,6):
        print(my_xsec.VVM_xSec(k,l+10**(-i)*v,m)) 
    print("------")
    """
    #for i in range(0,6):
    #    print(my_xsec.R_xSec(k,l+10**(-i)*v,m)) 
    #print("------")
    #for i in range(0,6):
    #    print(my_xsec.VR_xSec(k,l+10**(-i)*v,m)) 
    #print("------")
    #for i in range(0,6):
    #    print(my_xsec.VL_xSec(k,l+10**(-i)*v,m)) 

        #print(my_xsec.eval(k,l+10**(-i)*v,m)) 
    
    #for i in range(0,6):    
    #    print(my_virtuals.two_loop_CT2_E2E5(k0,k,l,m+10**(-i)*v)) 



    #for i in range(0,6):
    #    print(my_virtuals.two_loop(k0,k,l,m+10**(-i)*v)) 
    #for i in range(0,6):    
    #    print(my_virtuals.two_loop_CT2_E2E5(k0,k,l,m+10**(-i)*v)) 

    print("------")

    #for i in range(0,6):
    #    print(my_virtuals.one_loop(l0,k,l,m+10**(-i)*v)) 
    #for i in range(0,6):    
    #    print(my_virtuals.one_loop_CT1(l0,k,l,m+10**(-i)*v)) 

    
    """
    print("------")

    l0=1
    k0=1
    k=np.array([1,2,3])
    m=np.array([1/2,-3/2,1/4])
    l=np.array([0,1/math.sqrt(8),-1/math.sqrt(8)])

    v=np.array([1,1,0])

    for i in range(0,6):
        print(my_xsec.eval(k,l+10**(-i)*v,m)) 
    print("------")

    #for i in range(0,6):
    #    print(my_virtuals.two_loop(l0,k,l+10**(-i)*v,m)) 
    #for i in range(0,6):    
    #    print(my_virtuals.two_loop_CT1_E0E4(l0,k,l+10**(-i)*v,m)) 

    print("------")

    l0=1
    k0=1
    k=np.array([1,2,3])
    m=np.array([0,0,-1/4])
    l=np.array([0,1/3,0])

    v=np.array([1,1,0])

    for i in range(0,6):
        print(my_xsec.eval(k,l+10**(-i)*v,m)) 
    print("------")

    #for i in range(0,6):
    #    print(my_virtuals.two_loop(l0,k,l+10**(-i)*v,m)) 
    #for i in range(0,6):    
    #    print(my_virtuals.two_loop_CT3(l0,k,l+10**(-i)*v,m)) 

    
    #print(my_virtuals.two_loop(k0,k,l,m))


    k=np.array([0,1/math.sqrt(8),1/math.sqrt(8)])
    l=np.array([1,0,1/4])
    m=1./2*m

    v=np.array([1,1,1])

    for i in range(0,10):
        print(my_xsec.eval(k+10**(-i/2)*v,l+10**(-i/2)*v,m)) 
    print("------")


    m=np.array([0,1/math.sqrt(8),1/math.sqrt(8)])
    l=np.array([1,0,1/4])
    k=1./2*m

    v=np.array([1,1,1])

    for i in range(0,5):
        #print(my_xsec.VVL_xSec(k,l+10**(-i)*v,m)) 
        #print("VRR")
        #print(my_xsec.VVR_xSec(k,l+10**(-i)*v,m))
        #print(my_xsec.VVM_xSec(k,l+10**(-i)*v,m)) 
        #print(my_xsec.VL_xSec(k,l+10**(-i)*v,m))
        #print("VR")
        #print(my_xsec.VR_xSec(k,l+10**(-i)*v,m))
        #print(my_xsec.R_xSec(k,l+10**(-i)*v,m))
        print(my_xsec.eval(k,l+10**(-i)*v,m)) 
    print("------")

    m=np.array([0,1/math.sqrt(8),1/math.sqrt(8)])
    l=1/2*m
    k=np.array([1/2,1,1/8])

    v=np.array([1,1,1])

    for i in range(0,5):
        #print(my_xsec.VVL_xSec(k,l+10**(-i)*v,m)) 
        #print("VRR")
        #print(my_xsec.VVR_xSec(k,l+10**(-i)*v,m))
        #print(my_xsec.VVM_xSec(k,l+10**(-i)*v,m)) 
        #print(my_xsec.VL_xSec(k,l+10**(-i)*v,m))
        #print("VR")
        #print(my_xsec.VR_xSec(k,l+10**(-i)*v,m))
        #print(my_xsec.R_xSec(k,l+10**(-i)*v,m))
        print(my_xsec.eval(k,l+10**(-i)*v,m)) 
    print("------")

    m=np.array([0,1/math.sqrt(8),1/math.sqrt(8)])
    l=1/2*m
    k=1/3*m

    v=np.array([1,0,0])

    for i in range(0,10):
        #print(my_xsec.VVL_xSec(k,l+10**(-i)*v,m)) 
        #print("VRR")
        #print(my_xsec.VVR_xSec(k,l+10**(-i)*v,m))
        #print(my_xsec.VVM_xSec(k,l+10**(-i)*v,m)) 
        #print(my_xsec.VL_xSec(k,l+10**(-i)*v,m))
        #print("VR")
        #print(my_xsec.VR_xSec(k,l+10**(-i)*v,m))
        #print(my_xsec.R_xSec(k,l+10**(-i)*v,m))
        print(my_xsec.eval(k,l+10**(-i/2)*v,m)) 
        #print("------")

    
    my_xsec=x_section(1,1,debug=0)
    m=np.array([1/2,1/math.sqrt(8),1/math.sqrt(8)])
    l=np.array([1,-1,1])
    k=l

    v=np.array([1,1,0])

    for i in range(0,5):
        #print(my_xsec.VVL_xSec(k,l+10**(-i)*v,m)) 
        #print("VRR")
        #print(my_xsec.VVR_xSec(k,l+10**(-i)*v,m))
        #print(my_xsec.VVM_xSec(k,l+10**(-i)*v,m)) 
        #print(my_xsec.VL_xSec(k,l+10**(-i)*v,m))
        #print("VR")
        #print(my_xsec.VR_xSec(k,l+10**(-i)*v,m))
        #print(my_xsec.R_xSec(k,l+10**(-i)*v,m))
        print(my_xsec.eval(k,l+10**(-i)*v+0*0.0000001,m)) 
        #print("------")
    
    my_xsec=x_section(1,1)
    print("----")

    print("----")

    k=np.array([2.75111464,  0.497482  , -0.93424451])
    l=np.array([1.17464843, -2.68572088, -0.35007014])
    m=np.array([1.57681865, -2.4912711 , -0.15011487])


    print(my_xsec.eval(k,l,m)) 
    print("VVL")
    print(my_xsec.VVL_xSec(k,l,m)*E(k-l)**2)
    print("VVR")
    print(my_xsec.VVR_xSec(k,l,m))  
    print("VVM")
    print(my_xsec.VVM_xSec(k,l,m)*E(k-l)**2)
    print("VL")
    print(my_xsec.VL_xSec(k,l,m)) 
    print("VR")
    print(my_xsec.VR_xSec(k,l,m))
    print("R")
    print(my_xsec.R_xSec(k,l,m))



    my_xsec=x_section(1,0,debug=False)
    print("----")

    print("----")

    k=np.array([2.75111464,  0.497482  , -0.93424451])
    l=np.array([1.17464843, -2.68572088, -0.35007014])
    m=np.array([1.57681865, -2.4912711 , -0.15011487])


    print(my_xsec.eval(k,l,m)) 
    print("VVL")
    print(my_xsec.VVL_xSec(k,l,m)*E(k-l)**2)
    print("VVR")
    print(my_xsec.VVR_xSec(k,l,m))  
    print("VVM")
    print(my_xsec.VVM_xSec(k,l,m)*E(k-l)**2)
    print("VL")
    print(my_xsec.VL_xSec(k,l,m)) 
    print("VR")
    print(my_xsec.VR_xSec(k,l,m))
    print("R")
    print(my_xsec.R_xSec(k,l,m))
    """

    """
    l0=1
    k0=1
    k=np.array([1,2,3])
    m=np.array([1/2,-3/2,1/4])
    l=np.array([0,1/math.sqrt(8),-1/math.sqrt(8)])

    v=np.array([1,1,0])

    for i in range(0,6):
        print("----")
        print(my_xsec.eval(k,l+10**(-i)*v,m)) 
        print(my_xsec.VVM_xSec(k,l+10**(-i)*v,m)) 
        print(my_xsec.VVL_xSec(k,l+10**(-i)*v,m)) 
    print("------")
    """

    #for i in range(0,6):
    #    print(my_virtuals.two_loop(l0,k,l+10**(-i)*v,m)) 
    #for i in range(0,6):    
    #    print(my_virtuals.two_loop_CT1_E0E4(l0,k,l+10**(-i)*v,m)) 


    """
    print("----")
    for i in range(1,5):
        print(ex_integrand.x_parametrise([0+10**(-i),0.1,0.2,0.3,0.023,0.47,0.12,0.78,0.11])) 
    print("----")
    for i in range(1,5):
        print(ex_integrand.x_parametrise([0.1,0.1,0.2,0+10**(-i),0.023,0.47,0.12,0.78,0.11])) 
    print("----")
    for i in range(1,5):
        print(ex_integrand.x_parametrise([0+10**(-i),0.1,0.2,0+10**(-i),0.023,0.47,0.12,0.78,0.11]))
    """ 


    """

    
    my_integrand=integrand(1,1)

    def f(x):
        return my_integrand.x_parametrise(x)

    integ = vegas.Integrator([[0.0001,1],[0,1],[0,1],[0.0001,1],[0,1],[0,1],[0.0001,1],[0,1],[0,1]])

    n_iterations_learning=10
    n_points_iteration_learning=10000
    n_iterations=1
    n_points_iteration=100000

    integ(f, nitn=n_iterations_learning, neval=n_points_iteration_learning)
    result = integ(f, nitn=n_iterations, neval=n_points_iteration)
    #if integ.mpi_rank == 0:
    print(result.summary())
    print('result = %s    Q = %.2f' % (result, result.Q))

    print(my_integrand.max_eval)
    print(my_integrand.max_p)
    print(my_integrand.max_jac)
    """
    
    
    
    
    
    
    



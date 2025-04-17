"""
Python prototype implementation of the vectorized PF algorithm
described in "Parallelizable Feynman-Kac models for universal probabilistic 
programming", Section 6.

In this script we consider the non-i.i.d. loops model described in [1]:


The implementation is based on TensorFlow and autograph:
https://github.com/tensorflow/tensorflow/blob/master/
tensorflow/python/autograph/g3doc/reference/index.md.
"""
import time
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions



@tf.function (input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)]*8) 
def non_iid_loops(h1,h2,hp1,hp2,n,SS,W,ZZ):  #with resampling
    print('Warm up')
    ones=1.0+ZZ
    ONE = 1.0+ZZ
    TWO = 2.0+ZZ
    THREE = 3.0+ZZ
    def fbody(h1,h2,hp1,hp2,n,SS,W):
        
        
        EW=tf.math.exp(W)
        P=EW/tf.reduce_sum(EW)
        cum_dist = tf.math.cumsum(P[0])
        cum_dist /= cum_dist[-1]  # to account for floating point ehp2ors
        #unif_samp = tf.random.uniform((N,), 0, 1)
        unif_samp = tfd.Uniform(ZZ, ONE).sample()[0]#tf.random.uniform((N,), 0, 1)
        rs = tf.searchsorted(cum_dist, unif_samp)  # indices for resampling
        state_t = tf.concat([h1,h2,hp1,hp2,n,SS],axis=0)    # state tensor  
        state_t = tf.gather(state_t,rs,axis=1) # resampled state tensor
        h1,h2,hp1,hp2,n,SS= tuple([state_t[tf.newaxis,j] for j in range(6)])  # sliced state_t
        #W=ZZ
        #### END RESAMPLING 1 ##########
                 
    
        # MASKS DEFINITIONS
        mask0= (SS==0.0)
        mask1a=(SS==1.0)& tf.math.logical_or(h1!=0, h2!=0)
        mask1b=(SS==1.0)& tf.math.logical_and(h1==0, h2==0)    
        mask2 =(SS==2.0)
        mask3 =(SS==3.0)
               

        # STATE 0
        h1=tf.where(mask0,ones,h1)
        h2=tf.where(mask0,ones,h2)
        n=tf.where(mask0,ZZ,n)
        hp1=tf.where(mask0,ones,hp1)
        hp2=tf.where(mask0,ones,hp2) 
        SS=tf.where(mask0,ONE,SS) 
        

        # STATE 1
        h1=tf.where(mask1a,tf.cast(tfd.Bernoulli(probs=1/2*ones).sample(),tf.float32),h1)
        h2=tf.where(mask1a,tf.cast(tfd.Bernoulli(probs=1/2*ones).sample(),tf.float32),h2)
        #W=tf.where(SS==1,tf.math.log(tf.cast(tf.math.logical_or(h1==hp1, h2==hp2),tf.float32)),W)
        SS=tf.where(mask1a,TWO,SS)
        SS=tf.where(mask1b,THREE,SS)
        
        # STATE 2
        hp1=tf.where(mask2,h1,hp1)
        hp2=tf.where(mask2,h2,hp2)
        n=tf.where(mask2,n+1,n)
        SS=tf.where(mask2,ONE,SS)
        
        # STATE 3
        SS=tf.where(mask3,THREE,SS) # self-loop of final state   
        

        # W update        
        W=tf.where( (SS==ONE)|(SS==TWO)|(SS==THREE),tf.math.log(tf.cast(tf.math.logical_or(h1==hp1, h2==hp2),tf.float32)),W) #tf.math.log(tf.cast(tf.math.logical_or(h1==hp1, h2==hp2),tf.float32))#tf.where( True,tf.math.log(tf.cast(tf.math.logical_or(h1==hp1, h2==hp2),tf.float32)),W) 
                     
        return  (h1,h2,hp1,hp2,n,SS,W) 

    def fcond(h1,h2,hp1,hp2,n,ss,W):
        return True 
    
    h1,h2,hp1,hp2,n,SS,W=tf.while_loop(fcond, fbody, (h1,h2,hp1,hp2,n,SS,W), maximum_iterations=70,parallel_iterations=10,back_prop=True)  #maximum_iterations = fiter !
    return h1,h2,hp1,hp2,n,SS,W




# warm up
N=1

ZZ=tf.zeros((1,N))
start_time=time.time()
res=non_iid_loops(ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ)
final_time=(time.time()-start_time)
print("TOTAL elapsed time  %s seconds -------        " % final_time)



#number of particles
N=10**6

ZZ=tf.zeros((1,N))
start_time=time.time()
res=non_iid_loops(ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ)
final_time=(time.time()-start_time)
print("TOTAL elapsed time  %s seconds -------        " % final_time)


#weights
W=res[-1]
#final state
S=res[-2]
#output function
R=res[-3]   


# lower bound computation
EW=tf.math.exp(W)
P=EW/tf.reduce_sum(EW)
Rt=tf.where(S==3.0,R, 0.0)  
l_L = tf.reduce_sum(Rt*P)     


# effective sample size for EW
ess = tf.reduce_sum(EW)**2/(tf.reduce_sum(EW**2)) 


nil_n = 3.0  
M=2
# upper bound computation
Term = tf.where(S==nil_n,1.0, 0.0)      
alpha = 1/tf.reduce_sum(Term*P)
l_U = l_L*alpha + M*(alpha-1)      



#normalizing constant
norm = EW.numpy().sum()/len(EW.numpy()[0])

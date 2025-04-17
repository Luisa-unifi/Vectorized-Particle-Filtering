include "math.mc"

let model: () -> Float = lam.  
  let r = assume (Uniform 0. 1.) in
  let y = 0. in
  let n0 = 0 in
  
   
  recursive let fact: Int -> Float -> Float -> Float =
    lam n0: Int. lam r: Float. lam y: Float.  

    
    let n1 = (addi n0 1) in
    let var= mulf 2. r in
    let y1 = assume (Gaussian y var) in
    

    if or (gtf (absf y1) 1. )  (gti n1 100) then 
        let answer = if (gti n1 2) then 1. else 0. in
        observe true (Bernoulli answer);
        resample;
        mulf r 1.0
    else   
        fact n1 r y1 
      in 


  fact n0 r y 
    
   

mexpr
model ()


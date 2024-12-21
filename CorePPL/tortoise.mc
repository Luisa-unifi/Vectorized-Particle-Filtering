include "math.mc"

let model: () -> Float = lam.  
  let tortoise = assume (Uniform 0. 10.) in
  let hare = 0. in
  let n0 = 0 in
  
   
  recursive let fact: Int -> Float -> Float -> Float =
    lam n0: Int. lam tortoise: Float. lam hare: Float.  

    
    let n1 = (addi n0 1) in
    let tortoise1 = (addf tortoise 1.) in
    let a1 = assume (Bernoulli 0.4) in
    let a2 = assume (Gaussian 4. 2.) in
    let hare1 = if (a1) then (addf hare a2) else hare in
    
    
    let answer1 = if (gtf 10. (absf (subf hare tortoise))) then 1. else 0. in
    observe true (Bernoulli answer1);
    resample;
 
    
    
    let flag = if ((gti 100 n1)) then 1. else 0. in
      
    if or (gtf hare1 tortoise1)  (gti n1 100) then 
        
    let answer = if (gti n1 20) then 1. else 0. in
    observe true (Bernoulli answer);
    resample;
 
    mulf hare1 flag

     else 
      fact n1 tortoise1 hare1 
      in 

  fact n0 tortoise hare
    
   

mexpr
model ()


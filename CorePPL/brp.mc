include "math.mc"



let model: () -> Int = lam.

  
  let s0 = 100 in
  let f0= 0 in
  let n0 = 0 in
  let t0 = 0 in

  recursive let fact: Int -> Int -> Int -> Int-> Int =
    lam s: Int. lam f: Int. lam n: Int. lam t: Int.     

   let t1 = addi t 1 in
   
   let ber = assume (Bernoulli 0.2) in
   let f1 = if (ber) then (addi f 1) else 0 in   
   let n1 = if (ber) then (addi n 1) else n in
   let s1 = if (ber) then s else subi s 1 in  
   
   let answer = if (and ber (gti 80 s1)) then 1. 
   else if (and ber (gti s1 80)) then 0. 
   else 1. in     
   observe true (Bernoulli answer);                           
   resample;
        
   let flag = if ((gti 280 t1)) then 1 else 0 in    
   let flag1 = if ((gti s1 0)) then 1 else 0 in              
   if or (or (eqi s1 0) (gti f1 4)) (gti t1 280) then muli flag1 flag
     else fact s1 f1 n1 t1
   in 

    
  fact s0 f0 n0 t0
    



mexpr
model ()


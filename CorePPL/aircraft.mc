---------------------------------------------------
-- A state-space model for aircraft localization --
---------------------------------------------------

include "math.mc"

-- Noisy satellite observations of position (accuracy is improved
-- at higher altitude)


let model: () -> Float = lam.
    let rad_x =  [0.0, 3.0, 1.5, 5.0, 6.0, 5.6] in 
    let rad_y =  [0.0, 0.0, -1.5, 1.3, -4.0,-3.0] in
    let radius = [2.0, 2.0, 2.0, 3.0, 4.0, 2.0] in
    
    
    let obssq0= [4.0,      2.25,   2.7396, 4.0,    4.0,        4.0,     4.0,      4.0] in
    let obssq1 = [3.2485,  2.25,  2.7396,  1.0,    0.499997,   3.196,   4.0,      4.0] in
    let obssq2 = [0.25,    2.25,  4.0,     4.0,    3.196,      4.0,     4.0,      4.0] in
    let obssq3 = [9.0,     9.0,   9.0,     8.3043, 7.6004,     9.0,     8.0961,   7.6004] in
    let obssq4 = [16.0,    16.0,  16.0,    16.0,   16.0,       10.567,  6.4995,   12.558 ] in
    let obssq5 = [4.0,     4.0,   4.0,     4.0,    4.0,         4.0,    2.2600,    4.0 ] in
 
    

  recursive let simulate: Int -> Float -> Float -> Float = lam t: Int. lam position: Float. lam altitude: Float.
    
    let position= if (eqi 0 t) then assume (Gaussian 2. 1.) else assume (Gaussian position 2.) in    
    let altitude = if (eqi 0 t) then assume (Gaussian -1.5 1.) else assume (Gaussian altitude 2.) in
    

    
    let d0=  addf  (mulf (subf position (get rad_x 0)) (subf position (get rad_x 0))) (mulf (subf position (get rad_x 0)) (subf position (get rad_x 0))) in
    let flag = assume (Bernoulli 0.999) in
    let app =  if (flag) then (mulf (get radius 0) (get radius 0)) else assume (Gaussian (mulf (get radius 0) (get radius 0)) 0.001) in  
    let obs_dist0 = if (gtf d0 (mulf (get radius 0) (get radius 0))) then app else assume (Gaussian d0 0.1) in   
    observe (get obssq0 t) (Gaussian obs_dist0 0.01);    
    resample; 
     
    let d1=  addf  (mulf (subf position (get rad_x 1)) (subf position (get rad_x 1))) (mulf (subf position (get rad_x 1)) (subf position (get rad_x 1))) in
    let flag = assume (Bernoulli 0.999) in
    let app =  if (flag) then (mulf (get radius 1) (get radius 1)) else assume (Gaussian (mulf (get radius 1) (get radius 1)) 0.001) in  
    let obs_dist1 = if (gtf d1 (mulf (get radius 1) (get radius 1))) then app else assume (Gaussian d1 0.1) in   
    observe (get obssq1 t) (Gaussian obs_dist1 0.01);    
    resample; 
    
    let d2=  addf  (mulf (subf position (get rad_x 2)) (subf position (get rad_x 2))) (mulf (subf position (get rad_x 2)) (subf position (get rad_x 2))) in
    let flag = assume (Bernoulli 0.999) in
    let app =  if (flag) then (mulf (get radius 2) (get radius 2)) else assume (Gaussian (mulf (get radius 2) (get radius 2)) 0.001) in  
    let obs_dist2 = if (gtf d2 (mulf (get radius 2) (get radius 2))) then app else assume (Gaussian d2 0.1) in   
    observe (get obssq2 t) (Gaussian obs_dist2 0.01);    
    resample; 
    
    let d3=  addf  (mulf (subf position (get rad_x 3)) (subf position (get rad_x 3))) (mulf (subf position (get rad_x 3)) (subf position (get rad_x 3))) in
    let flag = assume (Bernoulli 0.999) in
    let app =  if (flag) then (mulf (get radius 3) (get radius 3)) else assume (Gaussian (mulf (get radius 3) (get radius 3)) 0.001) in  
    let obs_dist3 = if (gtf d3 (mulf (get radius 3) (get radius 3))) then app else assume (Gaussian d3 0.1) in   
    observe (get obssq3 t) (Gaussian obs_dist3 0.01);    
    resample; 
    
    let d4=  addf  (mulf (subf position (get rad_x 4)) (subf position (get rad_x 4))) (mulf (subf position (get rad_x 4)) (subf position (get rad_x 4))) in
    let flag = assume (Bernoulli 0.999) in
    let app =  if (flag) then (mulf (get radius 4) (get radius 4)) else assume (Gaussian (mulf (get radius 4) (get radius 4)) 0.001) in  
    let obs_dist4 = if (gtf d4 (mulf (get radius 4) (get radius 4))) then app else assume (Gaussian d4 0.1) in   
    observe (get obssq4 t) (Gaussian obs_dist4 0.01);    
    resample; 
     
    let d5=  addf  (mulf (subf position (get rad_x 5)) (subf position (get rad_x 5))) (mulf (subf position (get rad_x 5)) (subf position (get rad_x 5))) in
    let flag = assume (Bernoulli 0.999) in
    let app =  if (flag) then (mulf (get radius 5) (get radius 5)) else assume (Gaussian (mulf (get radius 5) (get radius 5)) 0.001) in  
    let obs_dist5 = if (gtf d5 (mulf (get radius 5) (get radius 5))) then app else assume (Gaussian d5 0.1) in   
    observe (get obssq5 t) (Gaussian obs_dist5 0.01);    
    
    resample; 
    let t = addi t 1 in
    
    if eqi 8 t then position
    else simulate t position altitude
    in

  simulate 0 0. 0.




mexpr
model ()

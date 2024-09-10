#title: "Data-Level Transfer Learning for Degradation Modeling and Prognosis"
#author: 
#  date: "Mar 1st, 2022"  

# Read Me: This document provides a comprehensive demonstration of the Algorithm 1 in the manuscript.
# For the code clearness, we demonstrate the case for 'source is higher than the target', which covers all the steps in the Algorithm 1.
# We also include the two benchmarks, i.e., MGP and Non-transfer, in this code.
# Running this code 100 times can present the results similar to the Fig. 4 in the manuscript (subject to the randomness of data generation).
# The annotations along with the code can provide links to the steps or terminologies in the manuscript.


rm(list = ls())  # clearing Memory


  
  #Loading required packages:
  library(Matrix)
  library(nloptr)
  library(minqa)
  library(optimx)
  library(rootSolve)
  library(nlme)
  library(mvtnorm)
  library(MASS)
  library(LaplacesDemon)#IW
  
  library(lme4)
  
  
  #######target unit
  simul_num = 5 #Number_of_simulations
  matrix_same_result = matrix(NA , nrow = simul_num , ncol = 9) #Matrix_for results
  for (num_num in 1:simul_num) {
  tryCatch({
  res_mvn = 1
  while(res_mvn == 1){
  
  unit_4 = 5 #number of sample in target process
  x = seq(0,15,length.out = 35)
  y_4 =data.frame(unit_number= sort(rep(1,length(x))))
  fun3 = function(a,b,x){
    return(a + b*x )
  }
  p=2 #dimension of target process
  #generating target data
  y_4$observation = c(1:length(x))
  scale_matrix=10000*diag(p)
  parameter4=rnorminvwishart(1,c(20,20),2,scale_matrix,5)#(number,mean,positive scalar,scale matrix,df)
  parameter4$mu = c(19,19)#specified mean for target
  parameter4$Sigma = matrix(c(20,12,12,8),2,2) #specified covariance for target
  b_4=rmvnorm(1,parameter4$mu,parameter4$Sigma)
  y_4$intercept = rep(b_4[1,1],length(x))
  y_4$slope = rep(b_4[1,2],length(x))
  for (i in 2:unit_4) {
    new_unit =data.frame(unit_number= sort(rep(i,length(x))))
    new_unit$observation = c(1:length(x))
    b_4=rmvnorm(1,parameter4$mu,parameter4$Sigma)
    new_unit$intercept = rep(b_4[1,1],length(x))
    new_unit$slope = rep(b_4[1,2],length(x))
    y_4 = rbind(y_4,new_unit)
  }
  
  y_4$points = rep(x,unit_4)
  y_4$noise = rnorm(unit_4*length(x),0,1)
  y_4$degredation = fun3(y_4$intercept , y_4$slope,y_4$points) + y_4$noise# generating degradation data
  
  
  model4 = lmer(degredation ~1+points +(1+points|unit_number),data = y_4,REML=T) #fitting mixed-effects models
  
  mean4_estimated = fixef(model4)
  vc4 = VarCorr(model4)
  cov4_estimated = as.data.frame(vc4)
  summary(model4)
  parameter4
  #generating samples from intercepted parameters to conduct MVN-Test
  cov4 = matrix(NA , nrow = length(mean4_estimated), length(mean4_estimated)) 
  cov4[1,1] = cov4_estimated$vcov[1]
  cov4[2,2] = cov4_estimated$vcov[2]
  cov4[1,2] = cov4_estimated$vcov[3]
  cov4[2,1] = cov4_estimated$vcov[3]
  y4_m = rmvnorm(10 , mean4_estimated , cov4)
  
  
  #########transfer unit
  
  
  
  
  
  unit_3 = 300 #number of samples in source process
  #generating source data
  x = seq(0,15,length.out = 35)
  y_3 =data.frame(unit_number= sort(rep(1,length(x))))
  fun3 = function(a,b,x){
    return(a + b*x )
  }
  y_3$observation = c(1:length(x))
  
  parameter3=rnorminvwishart(1,c(20,20),2,scale_matrix,5)#(number,mean,positive scalar,scale matrix,df)
  #Note that when source process has higher dimension, we need marginalizing
  parameter3$mu = c(19,19) + 2 
  parameter3$Sigma = parameter4$Sigma +0
  b_3=rmvnorm(1,parameter3$mu,parameter3$Sigma)
  y_3$intercept = rep(b_3[1,1],length(x))
  y_3$slope = rep(b_3[1,2],length(x))
  for (i in 2:unit_3) {
    new_unit =data.frame(unit_number= sort(rep(i,length(x))))
    new_unit$observation = c(1:length(x))
    b_3=rmvnorm(1,parameter3$mu,parameter3$Sigma)
    new_unit$intercept = rep(b_3[1,1],length(x))
    new_unit$slope = rep(b_3[1,2],length(x))
    y_3 = rbind(y_3,new_unit)
  }
  
  y_3$points = rep(x,unit_3)
  y_3$noise = rnorm(unit_3*length(x),0,1)
  y_3$degredation = fun3(y_3$intercept , y_3$slope,y_3$points) + y_3$noise
  
  model3 = lmer(degredation ~1+points +(1+points|unit_number),data = y_3,REML=T)
  
  mean3_estimated = fixef(model3)
  vc3 = VarCorr(model3)
  cov3_estimated = as.data.frame(vc3)
  summary(model3)
  parameter3
  #generating samples from intercepted parameters to conduct MVN-Test
  
  cov3 = matrix(NA , nrow = length(mean3_estimated), length(mean3_estimated))
  cov3[1,1] = cov3_estimated$vcov[1]
  cov3[2,2] = cov3_estimated$vcov[2]
  cov3[1,2] = cov3_estimated$vcov[3]
  cov3[2,1] = cov3_estimated$vcov[3]
  y3_m = rmvnorm(50 , mean3_estimated , cov3)
  
  ########## MVN-test
  
  
  alpha=0.99#alpha for MVN-test
  mvn = function(y1,y2){
    
    p=ncol(y1)
    n_1= nrow(y1)
    n_2= nrow(y2)
    n = n_1+n_2
    y1_bar= colMeans(y1)
    y2_bar= colMeans(y2)
    y_bar= (y1_bar*nrow(y1)+y2_bar*nrow(y2))/n
    A= n_1*(y1_bar- y_bar)%*%t(y1_bar- y_bar)+ n_2*(y2_bar- y_bar)%*%t(y2_bar- y_bar)
    B_1= matrix(0,p, p)
    for (i in 1:n_1) {
      B_1=B_1+(y1[i,]-y1_bar)%*%t(y1[i,]-y1_bar)
    }
    B_2= matrix(0,p, p)
    for (i in 1:n_2) {
      B_2=B_2+(y2[i,]-y2_bar)%*%t(y2[i,]-y2_bar)
    }
    B= B_1+ B_2
    num_1= (det(B_1)^(n_1/2))*(det(B_2)^(n_2/2))
    denum_1= det(A+B)^(n/2)
    frac_1=num_1/denum_1
    num_2= n^(p*n/2)
    denum_2=((n_1)^(p*n_1/2))*((n_2)^(p*n_2/2))
    frac_2 = num_2/denum_2
    test.s= frac_1*frac_2
    r= 1-((2*p^2+9*p+11)/(6*(1)*(p+3)*n))*((n/n_1-1)+(n/n_2-1))
    f= 0.5*p*1*(p+3)
    q= -2*r*log(test.s)
    q_0= qchisq(alpha, f)
    result=0
    if (q>q_0) {
      result=1
    }
    
    
    return(result)
  }
  
  res_mvn = mvn(y4_m,y3_m)
  }
  ############# integration
    
    
    
    prior= function(x){ #mvn
      dmvnorm(x,mean_vector, Covariance_matrix)
    }
    
    f= function(x,p,v, mu,s){ #t-dist
      fun2=gamma((v+p)/2)/(gamma(v/2)*v^(p/2)*pi^(p/2)*det(s)^0.5)*(1+(1/v)*(x-mu)%*%solve(s)%*%t(x-mu))^(-(v+p)/2)
      
      return(fun2)
    }
    
    
  #Gaussian-Hermite integration  
    
    hermite <- function (points, z) {
      p1 <- 1/pi^0.4
      p2 <- 0
      for (j in 1:points) {
        p3 <- p2
        p2 <- p1
        p1 <- z * sqrt(2/j) * p2 - sqrt((j - 1)/j) * p3
      }
      pp <- sqrt(2 * points) * p2
      c(p1, pp)
    }
    
    gauss.hermite <- function (points, iterlim = 50) {
      x <- w <- rep(0, points)
      m <- (points + 1)/2
      for (i in 1:m) {
        z <- if (i == 1) 
          sqrt(2 * points + 1) - 2 * (2 * points + 1)^(-1/6)
        else if (i == 2) 
          z - sqrt(points)/z
        else if (i == 3 || i == 4) 
          1.9 * z - 0.9 * x[i - 2]
        else 2 * z - x[i - 2]
        for (j in 1:iterlim) {
          z1 <- z
          p <- hermite(points, z)
          z <- z1 - p[1]/p[2]
          if (abs(z - z1) <= 1e-15) 
            break
        }
        if (j == iterlim) 
          warning("iteration limit exceeded")
        x[points + 1 - i] <- -(x[i] <- z)
        w[i] <- w[points + 1 - i] <- 2/p[2]^2
      }
      r <- cbind(x * sqrt(2), w/sum(w))
      colnames(r) <- c("Points", "Weights")
      r
    }
    
    
    mgauss.hermite <- function(n, mu, sigma) {
      if(!all(dim(sigma) == length(mu)))
        stop("mu and sigma have nonconformable dimensions")
      
      dm  <- length(mu)
      gh  <- gauss.hermite(n)
      
      idx <- as.matrix(expand.grid(rep(list(1:n),dm)))
      pts <- matrix(gh[idx,1],nrow(idx),dm)
      wts <- apply(matrix(gh[idx,2],nrow(idx),dm), 1, prod)
      
      
      eig <- eigen(sigma) 
      rot <- eig$vectors %*% diag(sqrt(eig$values))
      pts <- t(rot %*% t(pts) + mu)
      return(list(points=pts, weights=wts))
    }
    
    
    
    
    x3=seq(13,25,length=50)# variables to find maximum of Gaussian Hermite
    y3=seq(13,25,length=50)#variables to find maximum of Gaussian Hermite
    b_plus_f=c()
    mean3_vector= as.vector(t(mean3_estimated))
    hermite3_res <-mgauss.hermite(4,mean3_vector,cov3)
    sample3 = hermite3_res$points#points for source
    weights3 = hermite3_res$weights#weights for source
    mean4_vector= as.vector(t(mean4_estimated))
    hermite4_res <-mgauss.hermite(4,mean4_vector,cov4)
    sample4 = hermite4_res$points#points for target 
    weights4 = hermite4_res$weights#weights for source
    result=c()
    gaussian_quad2=matrix(NA,length(x3),length(y3))
    result=c()
    observation=c()
    v_0=5# df
    k_0=10#we assume mean and covariance were estimated from 10 observations
    for (pog in 1:length(x3)) {
      for (gorp in 1:length(y3)) {
        for (i in 1:nrow(sample3)) {
          
          b_1 = sample3[i,]
          b_2 = sample4[i,]
          b_bar=(colMeans(sample3)+colMeans(sample4))/2
          v_n = v_0+1
          k_n= k_0+1
          mu= t((k_0*c(20,20)+2*nrow(sample3)*b_bar)/k_n) #updated mean
          
          p= 2
          new_scale_matrix = scale_matrix+(b_1-b_bar)%*%t(b_1-b_bar)+ (b_2-b_bar)%*%t(b_2-b_bar)+ (k_0*2/(k_n))*(b_bar -c(20,20)) %*%t(b_bar -c(20,20))
          s= new_scale_matrix*((k_n+1)/(k_n*(v_n-p+1)))
          result[i]= f(cbind(x3[pog],y3[gorp]),p,v_n,mu,s)
          
          
          
        }
        gaussian_quad2[pog,gorp] = sum(weights3*result)
      }
    }
    
    c=which(gaussian_quad2==max(gaussian_quad2), arr.ind = TRUE) #maximum value corresponding to Gaussian-Hermite quadrature
    c
    c(x3[c[1,1]], y3[c[1,2]])
    
    
    (parameter4$mu - c(x3[c[1,1]], y3[c[1,2]]))[1]
    (parameter4$mu - c(x3[c[1,1]], y3[c[1,2]]))[2]
    (parameter4$mu - mean4_estimated)[1]
    (parameter4$mu - mean4_estimated)[2]
    parameter4$mu
    mean4_estimated[1]
    mean4_estimated[2]
    mean3_estimated[1]
    mean3_estimated[2]
    
    
    
    #########Generating online signal
    
    y_online =data.frame(unit_number= sort(rep(1,length(x))))
    
    fun3 = function(a,b,x){
      return(a + b*x )
    }
    
    
    y_online$points = x
    b_online_observed = rmvnorm(1,parameter4$mu,parameter4$Sigma)
    y_online$intercept = b_online_observed[1,1]
    y_online$slope =  b_online_observed[1,2]
    y_online$noise = rnorm(1*length(x),0,1)
    y_online$degredation = fun3(y_online$intercept , y_online$slope,y_online$points) + y_online$noise
    y_online
    
    ######## observed value
    
    number_observed_points = 3 # Number of observed points
    y_online_observed = y_online[1:number_observed_points,]
    y_online_matrix = matrix(y_online_observed$degredation,nrow = 1, ncol = length(y_online_observed$points),byrow=T)
    y_online_x = y_online$points[1:number_observed_points]
    ############## non transfer update
    z_star = matrix(1,number_observed_points, length(mean4_estimated))
    z_star[,2] = y_online_x
    
    sigma_p = solve(solve(cov4)+ (t(z_star)%*%z_star)/cov4_estimated$vcov[4])
    mean_p = sigma_p%*%(((t(z_star)%*%t(y_online_matrix))/cov4_estimated$vcov[4])+solve(cov4)%*%as.matrix(mean4_estimated))
    
    #####transfer update
    #likelihood
    maxim = c(x3[c[1,1]], y3[c[1,2]])
    likelihood = c()
    like = function(sgima,k_0,k_1,k_2,k_3){
      return((1/(sqrt(2*pi*(sgima))))*exp(-((k_3-k_0-k_1*k_2)^2)/(2*sgima)))
    }
    
    for (i in 1:number_observed_points) {
      likelihood[i]= like(cov4_estimated$vcov[4],x3[c[1,1]], y3[c[1,2]]
                          , y_online_observed$points[i], y_online_observed$degredation[i])
      
    }
    likelihood = sum(likelihood)
    
    #posterior
    
    posterior = likelihood * max(gaussian_quad2)
    posterior
    
    
    ####metropolice hasting
    
    
    
    run = 10000 #number of samples 
    chosen = matrix(NA , run , 2)
    for (j in 1:run) {
      candidate = rmvnorm(1 , mean = c(x3[c[1,1]], y3[c[1,2]]), sigma = cov4/8)
      prior_candidate = 0
      for (i in 1:nrow(sample3)) {
        
        b_1 = sample3[i,]
        b_2 = sample4[i,]
        b_bar=(colMeans(sample3)+colMeans(sample4))/2
        v_n = v_0+1
        k_n= k_0+1
        mu= t((k_0*c(20,20)+2*nrow(sample3)*b_bar)/k_n) #updated mean
        
        p= 2
        new_scale_matrix = scale_matrix+(b_1-b_bar)%*%t(b_1-b_bar)+ (b_2-b_bar)%*%t(b_2-b_bar)+ (k_0*2/(k_n))*(b_bar -c(20,20)) %*%t(b_bar -c(20,20))
        s= new_scale_matrix*((k_n+1)/(k_n*(v_n-p+1)))
        result[i]= f(candidate,p,v_n,mu,s)
        
        
        
      }
      prior_candidate = sum(weights3*result)
      likelihood_candidate=c()
      for (i in 1:number_observed_points) {
        likelihood_candidate[i]= like(cov4_estimated$vcov[4],candidate[1], candidate[2]
                                      , y_online_observed$points[i], y_online_observed$degredation[i])
        
      }
      likelihood_candidate = sum(likelihood_candidate)
      posterior_candidate= likelihood_candidate*prior_candidate
      condition = posterior_candidate/posterior
      alpha_MH = runif(1)
      if (condition >=alpha_MH) {
        posterior = posterior_candidate
        chosen[j,]=candidate
        maxim = candidate
      }
      if (condition<alpha_MH) {
        chosen[j,]= maxim
      }
      
    }
    
    chosen[9990:10000,]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    b_online_transfer= colMeans(chosen[9500:10000,])
    
    
    y_transfer = c()
    for (i in 1:length(x)) {
      y_transfer[i] = b_online_transfer[1] + b_online_transfer[2]*x[i]
    }
    
    ##########MGP
    num=c(max(y_3$unit_number),max(y_4$unit_number),1)
    n=3
    
    trains <- x
    y_hist_MGP = y_3
    y_hist_MGP$quad = parameter3$mu[1] 
    funq = function(a,b,d,x){
      return(a + b*x + d*(x^2))
    }
    y_hist_MGP$degredation2 = funq(y_hist_MGP$intercept,y_hist_MGP$slope,y_hist_MGP$quad,y_hist_MGP$points) + y_hist_MGP$noise
    y_4_matrix = matrix(y_4$degredation,nrow = num[2], ncol = length(x),byrow=T)
    y_3_matrix = matrix(y_hist_MGP$degredation2,nrow = num[1], ncol = length(x),byrow=T)
    y_4_matrix = t(as.matrix(y_4_matrix[1,]))
    y_3_matrix = t(as.matrix(y_3_matrix[1,]))
    
    
    
    trainy1 = list(y_3_matrix,y_4_matrix)
    trainy2=matrix(, nrow=3,ncol=length(x)) #Mean matrix
    for(i in 1:(n-1)){
      trainy2[i,]=colMeans(trainy1[[i]])
    }
    
    rep_factor =c() # this vector is used to transform off diagonal element denominators from sqrt(m_i*m_n) to max(m_i,m_n)(refer to equation 13)
    for (i in 1:n-1) {
      rep_factor[i] = sqrt(num[i]*num[n])/max(num[i] , num[n])
    }
    trainy31 = y_online_matrix
    trainy3 = colMeans(trainy31)
    
    trains=lapply(1:(n-1),function(i){trains})
    trainy=lapply(1:(n-1),function(i){trainy2[i,]})
    tests=y_online_observed$points;testy=trainy3;m1=length(tests)
    
    
    trains1 = unlist(trains[[1]])
    
    
    xstar = x
    
    
    
    n=3 #number of profiles
    t= proc.time()
    index=function(n,len,m) #creating index for sparse matrix elements
    {
      p1=c();p2=c();p3=c();p4=c();p5=c();p6=c()
      pp=sum(len)
      for(j in 1:(n-1))
      {
        i1=1 + sum(len[0:(j-1)])
        for(i in i1:(i1+len[j]-1))
        {
          p1=c(p1,i1:i)
          p2=c(p2,rep(i,length(i1:i)))
        }
      }
      p3=rep(1:pp,m)
      for(i in 1:m)
      {
        p4=c(p4,rep(pp+i,pp))
      }
      i2=pp+1
      for(i in i2:(i2+m-1))
      {
        p5=c(p5,i2:i)
        p6=c(p6,rep(i,length(i2:i)))
      }
      
      return(list(pfi=c(p1,p3,p5),pfj=c(p2,p4,p6)))
    }
    pf=index(n,lengths(trains),m1)
    pfi=pf$pfi;pfj=pf$pfj
    
    cyii=function(a,b,L) #main diagonal elements of covariance matrix
    {
      d=outer(a,b,`-`);I=outer(a,b,`==`)
      d=d[upper.tri(d,diag=T)];I=I[upper.tri(I,diag=T)]
      L[1]^2*exp(-0.25*d^2/L[2]^2) +  I*L[3]^2
    }
    cyip=function(a,b,L) #5 #off diagonal elements of covariance matrix
    {
      d=outer(a,b,`-`)
      L[1]*L[3]*sqrt(2*abs(L[2]*L[4])/(L[2]^2+L[4]^2))*exp(-0.5*d^2/(L[2]^2+L[4]^2))
    }
    
    y=c(unlist(trainy),c(testy)) #list of trainning data
    D=outer(tests,tests,`-`);P=outer(tests,tests,`==`)
    D=D[upper.tri(D,diag=T)];P=P[upper.tri(P,diag=T)]
    leny=length(y)
    
    
    
    C=function(strain,H) #covariance matrix
    {
      zii=list();zip=list();zpp=c()
      zii = lapply(1:(n-1), function(i){cyii(strain[[i]],strain[[i]],H[c(2*i-1,2*i,4*n-1)])})
      zip = lapply(1:(n-1), function(i){cyip(strain[[i]],tests,H[c(2*i-1,2*i,2*n+2*i-1,2*n+2*i)])})
      #zip[[1]] = zip[[1]]* rep_factor[1] # if number of n is increased, this part need modification
      #zip[[2]] = zip[[2]]* rep_factor[2]
      # zip[[]] = zip[[2]]* rep_factor[3] if n=4
      K=H[(2*n-1):(4*n-1)]
      zpp=Reduce("+",lapply(1:n, function(i){K[2*i-1]^2*exp(-0.25*D^2/K[2*i]^2)}))+ (P*K[length(K)]^2)#/num[n]
      # /num[n] is used to trasnform the predicted variance from individual to mean
      b1=unlist(zii);b2=as.vector(do.call("rbind",zip));
      return(sparseMatrix(i=pfi,j=pfj,x=c(b1,b2,zpp),symmetric=T))
      
    }
    
    
    
    logL=function(H,fn) #loglikelihood 
    {
      B=C(trains,H)
      deter=det(B)
      if(deter>0) {a=0.5*(log(deter)+t(y)%*%solve(B,y)+log(2*pi)*leny)
      } else {
        ch=chol(B)
        logdeter=2*(sum(log(diag(ch))))
        a=0.5*(logdeter+t(y)%*%solve(B,y)+log(2*pi)*leny)
      }
      
      return(as.numeric(a))
    }
    logL_grad=function(H,fn)
    {
      return(nl.grad(H,fn))
    }
    
    x0=c(rep(2,4*n-2),5) # starting points for the optimizer
    
    opts <- list( "algorithm" = "NLOPT_LD_MMA","maxeval" = 2000) 
    
    one=tryCatch(nloptr(x0=x0,eval_f= logL,eval_grad_f = logL_grad,opts= opts,fn= logL ), error = function(e) e)
    
    H1=one$solution
    
    H0=H1
    H0
    time_4 = proc.time() - t #time to learn
    
    
    t= proc.time()
    zip_pred=list()
    zip_pred =lapply(1:(n-1), function(i){cyip(trains[[i]],xstar,H0[c(2*i-1,2*i,2*n+2*i-1,2*n+2*i)])})
    #zip_pred[[1]] = zip_pred[[1]]* rep_factor[1]
    #zip_pred[[2]] = zip_pred[[2]]* rep_factor[2]
    #zip[[]] = zip[[2]]* rep_factor[3] if n=4
    D1=outer(xstar,tests,`-`)
    K1=H0[(2*n-1):(4*n-1)]
    zip_pred[[n]]=t(Reduce("+",lapply(1:n, function(i){K1[(2*i-1)]^2*exp(-0.25*D1^2/K1[(2*i)]^2)})))
    Pk=t(do.call("rbind",zip_pred))
    
    D2=outer(xstar,xstar,`-`);P2=outer(xstar,xstar,`==`)
    sk=Reduce("+",lapply(1:n, function(i){K1[(2*i-1)]^2*exp(-0.25*D2^2/K1[(2*i)]^2)}))+(P2*K1[length(K1)]^2)#/num[n]
    # /num[n] is used to trasnform the predicted variance from individual to mean
    covM=C(trains,H0)
    raed=solve(covM,y)
    y_MGP=as.matrix(Pk%*%raed) # predicted meand
    
    
    ###########
    
    y_not_transfer = c()
    for (i in 1:length(x)) {
      y_not_transfer[i] = mean4_estimated[1] + mean4_estimated[2]*x[i]
    }
    
    
    ############
    
    treshhold = 150 #predefined treshold
    
    t_non_transfer = (treshhold - mean_p[1,1])/mean_p[2,1] #estimating time for hit for non-transfer method
    
    t_transfer = (treshhold - b_online_transfer[1])/ b_online_transfer[2] #estimating time for hit for transfer method
    
    t_real = (treshhold - b_online_observed[1,1])/b_online_observed[1,2]
    
    er = y_MGP - treshhold
    er_n = sort(er[which(er<0)])
    
    ind= which(er == max(er))
   
    t_MGP = xstar[ind] #estimating time for hit for MGP
    
    non_transfer1=t_real - t_non_transfer
    transfer1=t_real - t_transfer
    MGP1=t_real - t_MGP
    
    #
    
    
    
    ######## observed value
    
    number_observed_points = 6
    y_online_observed = y_online[1:number_observed_points,]
    y_online_matrix = matrix(y_online_observed$degredation,nrow = num[3], ncol = length(y_online_observed$points),byrow=T)
    y_online_x = y_online$points[1:number_observed_points]
    ############## non transfer update
    z_star = matrix(1,number_observed_points, length(mean4_estimated))
    z_star[,2] = y_online_x
    sigma_p = solve(solve(cov4)+ (t(z_star)%*%z_star)/cov4_estimated$vcov[4])
    mean_p = sigma_p%*%(((t(z_star)%*%t(y_online_matrix))/cov4_estimated$vcov[4])+solve(cov4)%*%as.matrix(mean4_estimated))
    
    
    #likelihood
    maxim = c(x3[c[1,1]], y3[c[1,2]])
    likelihood = c()
    like = function(sgima,k_0,k_1,k_2,k_3){
      return((1/(sqrt(2*pi*(sgima))))*exp(-((k_3-k_0-k_1*k_2)^2)/(2*sgima)))
    }
    
    for (i in 1:number_observed_points) {
      likelihood[i]= like(cov4_estimated$vcov[4],x3[c[1,1]], y3[c[1,2]]
                          , y_online_observed$points[i], y_online_observed$degredation[i])
      
    }
    likelihood = sum(likelihood)
    
    #posterior
    
    posterior = likelihood * max(gaussian_quad2)
    posterior
    
    
    ####metropolice hasting
    
    
    
    run = 10000
    chosen = matrix(NA , run , 2)
    for (j in 1:run) {
      candidate = rmvnorm(1 , mean = c(x3[c[1,1]], y3[c[1,2]]), sigma = cov4/8)
      prior_candidate = 0
      for (i in 1:nrow(sample3)) {
        
        b_1 = sample3[i,]
        b_2 = sample4[i,]
        b_bar=(colMeans(sample3)+colMeans(sample4))/2
        v_n = v_0+1
        k_n= k_0+1
        mu= t((k_0*c(20,20)+2*nrow(sample3)*b_bar)/k_n) #updated mean
        
        p= 2
        new_scale_matrix = scale_matrix+(b_1-b_bar)%*%t(b_1-b_bar)+ (b_2-b_bar)%*%t(b_2-b_bar)+ (k_0*2/(k_n))*(b_bar -c(20,20)) %*%t(b_bar -c(20,20))
        s= new_scale_matrix*((k_n+1)/(k_n*(v_n-p+1)))
        result[i]= f(candidate,p,v_n,mu,s)
        
        
        
      }
      prior_candidate = sum(weights3*result)
      likelihood_candidate=c()
      for (i in 1:number_observed_points) {
        likelihood_candidate[i]= like(cov4_estimated$vcov[4],candidate[1], candidate[2]
                                      , y_online_observed$points[i], y_online_observed$degredation[i])
        
      }
      likelihood_candidate = sum(likelihood_candidate)
      posterior_candidate= likelihood_candidate*prior_candidate
      condition = posterior_candidate/posterior
      alpha_MH = runif(1)
      if (condition >=alpha_MH) {
        posterior = posterior_candidate
        chosen[j,]=candidate
        maxim = candidate
      }
      if (condition<alpha_MH) {
        chosen[j,]= maxim
      }
      
    }
    
    chosen[9990:10000,]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    b_online_transfer= colMeans(chosen[9500:10000,])
    
    
    y_transfer = c()
    for (i in 1:length(x)) {
      y_transfer[i] = b_online_transfer[1] + b_online_transfer[2]*x[i]
    }
    
    ##########MGP
    num=c(max(y_3$unit_number),max(y_4$unit_number),1)
    n=3
    
    trains <- x
    
    
    
    
    trainy1 = list(y_3_matrix,y_4_matrix)
    trainy2=matrix(, nrow=3,ncol=length(x)) #Mean matrix
    for(i in 1:(n-1)){
      trainy2[i,]=colMeans(trainy1[[i]])
    }
    
    rep_factor =c() # this vector is used to transform off diagonal element denominators from sqrt(m_i*m_n) to max(m_i,m_n)(refer to equation 13)
    for (i in 1:n-1) {
      rep_factor[i] = sqrt(num[i]*num[n])/max(num[i] , num[n])
    }
    trainy31 = y_online_matrix
    trainy3 = colMeans(trainy31)
    
    trains=lapply(1:(n-1),function(i){trains})
    trainy=lapply(1:(n-1),function(i){trainy2[i,]})
    tests=y_online_observed$points;testy=trainy3;m1=length(tests)
    
    
    trains1 = unlist(trains[[1]])
    
    
   
    
    
    
    n=3 #number of profiles
    t= proc.time()
    index=function(n,len,m) #creating index for sparse matrix elements
    {
      p1=c();p2=c();p3=c();p4=c();p5=c();p6=c()
      pp=sum(len)
      for(j in 1:(n-1))
      {
        i1=1 + sum(len[0:(j-1)])
        for(i in i1:(i1+len[j]-1))
        {
          p1=c(p1,i1:i)
          p2=c(p2,rep(i,length(i1:i)))
        }
      }
      p3=rep(1:pp,m)
      for(i in 1:m)
      {
        p4=c(p4,rep(pp+i,pp))
      }
      i2=pp+1
      for(i in i2:(i2+m-1))
      {
        p5=c(p5,i2:i)
        p6=c(p6,rep(i,length(i2:i)))
      }
      
      return(list(pfi=c(p1,p3,p5),pfj=c(p2,p4,p6)))
    }
    pf=index(n,lengths(trains),m1)
    pfi=pf$pfi;pfj=pf$pfj
    
    cyii=function(a,b,L) #main diagonal elements of covariance matrix
    {
      d=outer(a,b,`-`);I=outer(a,b,`==`)
      d=d[upper.tri(d,diag=T)];I=I[upper.tri(I,diag=T)]
      L[1]^2*exp(-0.25*d^2/L[2]^2) +  I*L[3]^2
    }
    cyip=function(a,b,L) #5 #off diagonal elements of covariance matrix
    {
      d=outer(a,b,`-`)
      L[1]*L[3]*sqrt(2*abs(L[2]*L[4])/(L[2]^2+L[4]^2))*exp(-0.5*d^2/(L[2]^2+L[4]^2))
    }
    
    y=c(unlist(trainy),c(testy)) #list of trainning data
    D=outer(tests,tests,`-`);P=outer(tests,tests,`==`)
    D=D[upper.tri(D,diag=T)];P=P[upper.tri(P,diag=T)]
    leny=length(y)
    
    
    
    C=function(strain,H) #covariance matrix
    {
      zii=list();zip=list();zpp=c()
      zii = lapply(1:(n-1), function(i){cyii(strain[[i]],strain[[i]],H[c(2*i-1,2*i,4*n-1)])})
      zip = lapply(1:(n-1), function(i){cyip(strain[[i]],tests,H[c(2*i-1,2*i,2*n+2*i-1,2*n+2*i)])})
      #zip[[1]] = zip[[1]]* rep_factor[1] # if number of n is increased, this part need modification
      #zip[[2]] = zip[[2]]* rep_factor[2]
      # zip[[]] = zip[[2]]* rep_factor[3] if n=4
      K=H[(2*n-1):(4*n-1)]
      zpp=Reduce("+",lapply(1:n, function(i){K[2*i-1]^2*exp(-0.25*D^2/K[2*i]^2)}))+ (P*K[length(K)]^2)#/num[n]
      # /num[n] is used to trasnform the predicted variance from individual to mean
      b1=unlist(zii);b2=as.vector(do.call("rbind",zip));
      return(sparseMatrix(i=pfi,j=pfj,x=c(b1,b2,zpp),symmetric=T))
      
    }
    
    
    
    logL=function(H,fn) #loglikelihood 
    {
      B=C(trains,H)
      deter=det(B)
      if(deter>0) {a=0.5*(log(deter)+t(y)%*%solve(B,y)+log(2*pi)*leny)
      } else {
        ch=chol(B)
        logdeter=2*(sum(log(diag(ch))))
        a=0.5*(logdeter+t(y)%*%solve(B,y)+log(2*pi)*leny)
      }
      
      return(as.numeric(a))
    }
    logL_grad=function(H,fn)
    {
      return(nl.grad(H,fn))
    }
    
    x0=c(rep(2,4*n-2),5) # starting points for the optimizer
    
    opts <- list( "algorithm" = "NLOPT_LD_MMA","maxeval" = 2000) 
    
    one=tryCatch(nloptr(x0=x0,eval_f= logL,eval_grad_f = logL_grad,opts= opts,fn= logL ), error = function(e) e)
    
    H1=one$solution
    
    H0=H1
    H0
    time_4 = proc.time() - t #time to learn
    
    
    t= proc.time()
    zip_pred=list()
    zip_pred =lapply(1:(n-1), function(i){cyip(trains[[i]],xstar,H0[c(2*i-1,2*i,2*n+2*i-1,2*n+2*i)])})
    #zip_pred[[1]] = zip_pred[[1]]* rep_factor[1]
    #zip_pred[[2]] = zip_pred[[2]]* rep_factor[2]
    #zip[[]] = zip[[2]]* rep_factor[3] if n=4
    D1=outer(xstar,tests,`-`)
    K1=H0[(2*n-1):(4*n-1)]
    zip_pred[[n]]=t(Reduce("+",lapply(1:n, function(i){K1[(2*i-1)]^2*exp(-0.25*D1^2/K1[(2*i)]^2)})))
    Pk=t(do.call("rbind",zip_pred))
    
    D2=outer(xstar,xstar,`-`);P2=outer(xstar,xstar,`==`)
    sk=Reduce("+",lapply(1:n, function(i){K1[(2*i-1)]^2*exp(-0.25*D2^2/K1[(2*i)]^2)}))+(P2*K1[length(K1)]^2)#/num[n]
    # /num[n] is used to trasnform the predicted variance from individual to mean
    covM=C(trains,H0)
    raed=solve(covM,y)
    y_MGP=as.matrix(Pk%*%raed) # predicted meand
    
    
    ###########
    
    y_not_transfer = c()
    for (i in 1:length(x)) {
      y_not_transfer[i] = mean4_estimated[1] + mean4_estimated[2]*x[i]
    }
    
    
    ############
    
    
    
    t_non_transfer = (treshhold - mean_p[1,1])/mean_p[2,1]
    
    t_transfer = (treshhold - b_online_transfer[1])/ b_online_transfer[2]
    
    t_real = (treshhold - b_online_observed[1,1])/b_online_observed[1,2]
    
    er = y_MGP - treshhold
    er_n = sort(er[which(er<0)])
    
    ind= which(er == max(er))
    
    t_MGP = xstar[ind] #estimating time for hit for MGP
    
    
    non_transfer2=t_real - t_non_transfer
    transfer2=t_real - t_transfer
    MGP2=t_real - t_MGP
    
    #
    
    
    
    
    
    
    
    ######## observed value
    
    number_observed_points = 9
    y_online_observed = y_online[1:number_observed_points,]
    y_online_matrix = matrix(y_online_observed$degredation,nrow = num[3], ncol = length(y_online_observed$points),byrow=T)
    y_online_x = y_online$points[1:number_observed_points]
    ############## non transfer update
    z_star = matrix(1,number_observed_points, length(mean4_estimated))
    z_star[,2] = y_online_x
    sigma_p = solve(solve(cov4)+ (t(z_star)%*%z_star)/cov4_estimated$vcov[4])
    mean_p = sigma_p%*%(((t(z_star)%*%t(y_online_matrix))/cov4_estimated$vcov[4])+solve(cov4)%*%as.matrix(mean4_estimated))
    #likelihood
    maxim = c(x3[c[1,1]], y3[c[1,2]])
    likelihood = c()
    like = function(sgima,k_0,k_1,k_2,k_3){
      return((1/(sqrt(2*pi*(sgima))))*exp(-((k_3-k_0-k_1*k_2)^2)/(2*sgima)))
    }
    
    for (i in 1:number_observed_points) {
      likelihood[i]= like(cov4_estimated$vcov[4],x3[c[1,1]], y3[c[1,2]]
                          , y_online_observed$points[i], y_online_observed$degredation[i])
      
    }
    likelihood = sum(likelihood)
    
    #posterior
    
    posterior = likelihood * max(gaussian_quad2)
    posterior
    
    
    ####metropolice hasting
    
    
    
    run = 10000
    chosen = matrix(NA , run , 2)
    for (j in 1:run) {
      candidate = rmvnorm(1 , mean = c(x3[c[1,1]], y3[c[1,2]]), sigma = cov4/8)
      prior_candidate = 0
      for (i in 1:nrow(sample3)) {
        
        b_1 = sample3[i,]
        b_2 = sample4[i,]
        b_bar=(colMeans(sample3)+colMeans(sample4))/2
        v_n = v_0+1
        k_n= k_0+1
        mu= t((k_0*c(20,20)+2*nrow(sample3)*b_bar)/k_n) #updated mean
        
        p= 2
        new_scale_matrix = scale_matrix+(b_1-b_bar)%*%t(b_1-b_bar)+ (b_2-b_bar)%*%t(b_2-b_bar)+ (k_0*2/(k_n))*(b_bar -c(20,20)) %*%t(b_bar -c(20,20))
        s= new_scale_matrix*((k_n+1)/(k_n*(v_n-p+1)))
        result[i]= f(candidate,p,v_n,mu,s)
        
        
        
      }
      prior_candidate = sum(weights3*result)
      likelihood_candidate=c()
      for (i in 1:number_observed_points) {
        likelihood_candidate[i]= like(cov4_estimated$vcov[4],candidate[1], candidate[2]
                                      , y_online_observed$points[i], y_online_observed$degredation[i])
        
      }
      likelihood_candidate = sum(likelihood_candidate)
      posterior_candidate= likelihood_candidate*prior_candidate
      condition = posterior_candidate/posterior
      alpha_MH = runif(1)
      if (condition >=alpha_MH) {
        posterior = posterior_candidate
        chosen[j,]=candidate
        maxim = candidate
      }
      if (condition<alpha_MH) {
        chosen[j,]= maxim
      }
      
    }
    
    chosen[9990:10000,]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    b_online_transfer= colMeans(chosen[9500:10000,])
    
    
    y_transfer = c()
    for (i in 1:length(x)) {
      y_transfer[i] = b_online_transfer[1] + b_online_transfer[2]*x[i]
    }
    
    ##########MGP
    num=c(max(y_3$unit_number),max(y_4$unit_number),1)
    n=3
    
    trains <- x
    
    
    
    
    trainy1 = list(y_3_matrix,y_4_matrix)
    trainy2=matrix(, nrow=3,ncol=length(x)) #Mean matrix
    for(i in 1:(n-1)){
      trainy2[i,]=colMeans(trainy1[[i]])
    }
    
    rep_factor =c() # this vector is used to transform off diagonal element denominators from sqrt(m_i*m_n) to max(m_i,m_n)(refer to equation 13)
    for (i in 1:n-1) {
      rep_factor[i] = sqrt(num[i]*num[n])/max(num[i] , num[n])
    }
    trainy31 = y_online_matrix
    trainy3 = colMeans(trainy31)
    
    trains=lapply(1:(n-1),function(i){trains})
    trainy=lapply(1:(n-1),function(i){trainy2[i,]})
    tests=y_online_observed$points;testy=trainy3;m1=length(tests)
    
    
    trains1 = unlist(trains[[1]])
    
    

    
    
    n=3 #number of profiles
    t= proc.time()
    index=function(n,len,m) #creating index for sparse matrix elements
    {
      p1=c();p2=c();p3=c();p4=c();p5=c();p6=c()
      pp=sum(len)
      for(j in 1:(n-1))
      {
        i1=1 + sum(len[0:(j-1)])
        for(i in i1:(i1+len[j]-1))
        {
          p1=c(p1,i1:i)
          p2=c(p2,rep(i,length(i1:i)))
        }
      }
      p3=rep(1:pp,m)
      for(i in 1:m)
      {
        p4=c(p4,rep(pp+i,pp))
      }
      i2=pp+1
      for(i in i2:(i2+m-1))
      {
        p5=c(p5,i2:i)
        p6=c(p6,rep(i,length(i2:i)))
      }
      
      return(list(pfi=c(p1,p3,p5),pfj=c(p2,p4,p6)))
    }
    pf=index(n,lengths(trains),m1)
    pfi=pf$pfi;pfj=pf$pfj
    
    cyii=function(a,b,L) #main diagonal elements of covariance matrix
    {
      d=outer(a,b,`-`);I=outer(a,b,`==`)
      d=d[upper.tri(d,diag=T)];I=I[upper.tri(I,diag=T)]
      L[1]^2*exp(-0.25*d^2/L[2]^2) +  I*L[3]^2
    }
    cyip=function(a,b,L) #5 #off diagonal elements of covariance matrix
    {
      d=outer(a,b,`-`)
      L[1]*L[3]*sqrt(2*abs(L[2]*L[4])/(L[2]^2+L[4]^2))*exp(-0.5*d^2/(L[2]^2+L[4]^2))
    }
    
    y=c(unlist(trainy),c(testy)) #list of trainning data
    D=outer(tests,tests,`-`);P=outer(tests,tests,`==`)
    D=D[upper.tri(D,diag=T)];P=P[upper.tri(P,diag=T)]
    leny=length(y)
    
    
    
    C=function(strain,H) #covariance matrix
    {
      zii=list();zip=list();zpp=c()
      zii = lapply(1:(n-1), function(i){cyii(strain[[i]],strain[[i]],H[c(2*i-1,2*i,4*n-1)])})
      zip = lapply(1:(n-1), function(i){cyip(strain[[i]],tests,H[c(2*i-1,2*i,2*n+2*i-1,2*n+2*i)])})
      #zip[[1]] = zip[[1]]* rep_factor[1] # if number of n is increased, this part need modification
      #zip[[2]] = zip[[2]]* rep_factor[2]
      # zip[[]] = zip[[2]]* rep_factor[3] if n=4
      K=H[(2*n-1):(4*n-1)]
      zpp=Reduce("+",lapply(1:n, function(i){K[2*i-1]^2*exp(-0.25*D^2/K[2*i]^2)}))+ (P*K[length(K)]^2)#/num[n]
      # /num[n] is used to trasnform the predicted variance from individual to mean
      b1=unlist(zii);b2=as.vector(do.call("rbind",zip));
      return(sparseMatrix(i=pfi,j=pfj,x=c(b1,b2,zpp),symmetric=T))
      
    }
    
    
    
    logL=function(H,fn) #loglikelihood 
    {
      B=C(trains,H)
      deter=det(B)
      if(deter>0) {a=0.5*(log(deter)+t(y)%*%solve(B,y)+log(2*pi)*leny)
      } else {
        ch=chol(B)
        logdeter=2*(sum(log(diag(ch))))
        a=0.5*(logdeter+t(y)%*%solve(B,y)+log(2*pi)*leny)
      }
      
      return(as.numeric(a))
    }
    logL_grad=function(H,fn)
    {
      return(nl.grad(H,fn))
    }
    
    x0=c(rep(2,4*n-2),5) # starting points for the optimizer
    
    opts <- list( "algorithm" = "NLOPT_LD_MMA","maxeval" = 2000) 
    
    one=tryCatch(nloptr(x0=x0,eval_f= logL,eval_grad_f = logL_grad,opts= opts,fn= logL ), error = function(e) e)
    
    H1=one$solution
    
    H0=H1
    H0
    time_4 = proc.time() - t #time to learn
    
    
    t= proc.time()
    zip_pred=list()
    zip_pred =lapply(1:(n-1), function(i){cyip(trains[[i]],xstar,H0[c(2*i-1,2*i,2*n+2*i-1,2*n+2*i)])})
    #zip_pred[[1]] = zip_pred[[1]]* rep_factor[1]
    #zip_pred[[2]] = zip_pred[[2]]* rep_factor[2]
    #zip[[]] = zip[[2]]* rep_factor[3] if n=4
    D1=outer(xstar,tests,`-`)
    K1=H0[(2*n-1):(4*n-1)]
    zip_pred[[n]]=t(Reduce("+",lapply(1:n, function(i){K1[(2*i-1)]^2*exp(-0.25*D1^2/K1[(2*i)]^2)})))
    Pk=t(do.call("rbind",zip_pred))
    
    D2=outer(xstar,xstar,`-`);P2=outer(xstar,xstar,`==`)
    sk=Reduce("+",lapply(1:n, function(i){K1[(2*i-1)]^2*exp(-0.25*D2^2/K1[(2*i)]^2)}))+(P2*K1[length(K1)]^2)#/num[n]
    # /num[n] is used to trasnform the predicted variance from individual to mean
    covM=C(trains,H0)
    raed=solve(covM,y)
    y_MGP=as.matrix(Pk%*%raed) # predicted meand
    
    
    ###########
    
    y_not_transfer = c()
    for (i in 1:length(x)) {
      y_not_transfer[i] = mean4_estimated[1] + mean4_estimated[2]*x[i]
    }
    
    
    ############
    
    
    t_non_transfer = (treshhold - mean_p[1,1])/mean_p[2,1]
    
    t_transfer = (treshhold - b_online_transfer[1])/ b_online_transfer[2]
    
    t_real = (treshhold - b_online_observed[1,1])/b_online_observed[1,2]
    
    er = y_MGP - treshhold
    er_n = sort(er[which(er<0)])
    
    ind= which(er == max(er))
    
    t_MGP = xstar[ind] #estimating time for hit for MGP
    
    
    non_transfer3=t_real - t_non_transfer
    transfer3=t_real - t_transfer
    MGP3=t_real - t_MGP
    
    
    
    c(non_transfer1,transfer1,MGP1)
    c(non_transfer2,transfer2,MGP2)
    c(non_transfer3,transfer3,MGP3)
    matrix_same_result[num_num,1] = non_transfer1
    matrix_same_result[num_num,4] = non_transfer2
    matrix_same_result[num_num,7] = non_transfer3
    
    matrix_same_result[num_num,2] = transfer1
    matrix_same_result[num_num,5] = transfer2
    matrix_same_result[num_num,8] = transfer3
    
    matrix_same_result[num_num,3] = MGP1
    matrix_same_result[num_num,6] = MGP2
    matrix_same_result[num_num,9] = MGP3
  },error = function(e){})
    }

  
  
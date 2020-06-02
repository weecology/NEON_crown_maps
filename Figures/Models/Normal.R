sink("model/occurrence_repulsion.jags")
cat("
    model {
    
    for (x in 1:Nobs){
    #Observation of a flowering plant
    Y[x] ~ dbern(p[x])
    logit(p[x]) <-  e[Plant[x],Month[x]] + beta[Plant[x]] * elevation[x]
    
    #Residuals
    discrepancy[x] <- abs(Y[x] - p[x])
    
    #Assess Model Fit
    Ynew[x] ~ dbern(p[x])
    discrepancy.new[x]<-abs(Ynew[x] - p[x])
    }
    
    #Sum discrepancy
    fit <- sum(discrepancy)/Nobs
    fitnew <- sum(discrepancy.new)/Nobs
    
    #Prediction
    
    for(x in 1:Npreds){
    #predict value
    
    #Observation - probability of flowering
    prediction[x] ~ dbern(p_new[x])
    logit(p_new[x]) <-  e[NewPlant[x],NewMonth[x]] + beta[NewPlant[x]] * new_elevation[x]
    
    #predictive error
    pred_error[x] <- abs(Ypred[x] - p_new[x])
    }
    
    #Predictive Error
    fitpred<-sum(pred_error)/Npreds
    
    #########################
    #autocorrelation in error
    #########################
    
    #Error covariance among flowering plants
    for(k in 1:Months){
      e[1:Plants,k] ~ dmnorm(zeros,tauC[,])
    }
    
    ##covariance among similar species
    for(i in 1:Plants){
    for(j in 1:Plants){
    C[i,j] = exp(-lambda_cov*D[i,j])
    }
    }
    
    ## Covert variance to precision for each parameter, allow omega to shrink to identity matrix
    vCov = omega*C[,] + (1-omega) * I
    tauC = vCov*gamma^2
    
    #Autocorrelation priors
    gamma  ~ dunif(0,20)
    
    #Strength of covariance decay
    lambda_cov ~ dunif(0.1,2)
    omega ~ dbeta(1,1)
    
    #Priors
    #Species level priors
    for (j in 1:Plants){
      #effect of elevation
      beta[j] ~ dnorm(0,0.386)
      alpha[j] ~ dnorm(0,0.386)
    }
    
    }
    ",fill=TRUE)

sink()

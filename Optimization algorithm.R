


## SGD
SGD = setRefClass("SGD",
    fields = list(eta="numeric", t="numeric"),
    methods = list(
        step = function(gradient) {
            inc  =  - eta * gradient / t
            t <<- t + 1
            return(inc)
        },
        initialize = function(eta, t=1) {
            .self$eta = eta
            .self$t = t
        }
    )
)






##Momentum
momentum = setRefClass("momentum", #setRefClass() returns a generator function suitable for creating objects from the class, invisibly
    fields = list(eta="numeric", alpha="numeric", w="numeric", t="numeric"), #named list of the fields
    methods = list( #a named list of function definitions that can be invoked on objects from this class
        step = function(gradient) { #The "step" method take an estimated gradient as argument ,  function selects a model
            w <<- w * alpha - eta * gradient / t #update rule
		inc = w # increment it
            t <<- t + 1
            return(inc) # returns an increment to apply to the current parameter values
        },
        initialize = function(eta, alpha=0.5, w=0, t=1) { # best hyper-parameters inital values found from several papers
            .self$eta = eta #we set-up every inital values
            .self$alpha = alpha
            .self$w = w
            .self$t = t
        }
    )
)




##AdaGrad
adaGrad = setRefClass("adaGrad",
    fields = list(eta="numeric", epsilon="numeric", alpha="numeric", grad_squared="numeric", w="numeric"),
    methods = list(
        step = function(gradient) {
            grad_squared <<- grad_squared + c(gradient)^2
            w <<- w * alpha - eta * c(gradient) / (epsilon + sqrt(grad_squared))
            return(w)
        },
        initialize = function(eta, alpha=0, epsilon=1E-7) {
            .self$eta = eta
            .self$epsilon = epsilon
            .self$alpha = alpha
            .self$grad_squared = 0 # grad_squared and w start off as scalars but become vectors of the right length on their first update
            .self$w = 0
        }
    )
)





##ADAM
ADAM = setRefClass("ADAM",
    fields = list(eta="numeric", beta1="numeric", beta2="numeric", epsilon="numeric", m="numeric", v="numeric", t="numeric"),
    methods = list(
        step = function(gradient) {
            t <<- t + 1
            m <<- beta1*m + (1-beta1)*c(gradient)
            v <<- beta2*v + (1-beta2)*(c(gradient)^2)
            m_hat = m / (1-beta1^t)
            v_hat = v / (1-beta2^t)
            inc = - eta * m_hat / (sqrt(v_hat) + epsilon)
            return(inc)
        },
        initialize = function(eta=0.001, beta1=0.9, beta2=0.999, epsilon=1E-8) {
            .self$eta = eta
            .self$beta1 = beta1
            .self$beta2 = beta2
            .self$epsilon = epsilon
            .self$m = 0 # m and v start off as scalars but become vectors of the right length on their first update
            .self$v = 0
            .self$t = 0
        }
    )
)



##Nadam
Nadam = setRefClass("Nadam",
    fields = list(eta="numeric", beta1="numeric", beta2="numeric", epsilon="numeric", m="numeric", v="numeric", t="numeric"),
    methods = list(
        step = function(gradient) {
            t <<- t + 1
            m <<- beta1*m + (1-beta1)*c(gradient)
            v <<- beta2*v + (1-beta2)*(c(gradient)^2)
            m_hat = m / (1-beta1^t) + (1 - beta1) * gradient / (1-beta1^t)
            v_hat = v / (1-beta2^t)
            inc =  - eta * m_hat / (sqrt(v_hat) + epsilon)
            return(inc)
        },
        initialize = function(eta=0.001, beta1=0.9, beta2=0.999, epsilon=1E-8) {
            .self$eta = eta
            .self$beta1 = beta1
            .self$beta2 = beta2
            .self$epsilon = epsilon
            .self$m = 0 # m and v start off as scalars but become vectors of the right length on their first update
            .self$v = 0
            .self$t = 0
        }
    )
)







##RMSprop
rmsProp = setRefClass("rmsProp",
    fields = list(eta="numeric", grad_squared="numeric", epsilon="numeric", acc_grad="numeric"),
    methods = list(
        step = function(gradient) {
            acc_grad <<- grad_squared * acc_grad + (1-grad_squared)*c(gradient)^2
            inc =  - (eta  / sqrt(acc_grad+epsilon)) *  gradient
            return(inc)
        },
        initialize = function(eta=1E-3, grad_squared=0.9, epsilon=1E-6) { ##Defaults taken from a mix of places
            .self$eta = eta
            .self$grad_squared = grad_squared
            .self$epsilon = epsilon
            .self$acc_grad = 0 ##acc_grad start off as a scalar but becomes a vector of the right length on its first update
        }
    )
)







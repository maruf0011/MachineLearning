dr = function()
{
    setwd("/home/phenix/Documents/MLPractice/kaggleDigitInR/")
    data = read.csv("train.csv")
    #data = data[1:100, ]
    # create formula string
    strfor = paste(names(data)[1] , paste(names(data)[2:785] , collapse = " + ") , sep = " ~ ")
    
    library(NeuralNetTools)
    library(neuralnet)
    
    mod = neuralnet(as.formula(strfor) , data , hidden = c(100,50) , learningrate = .1 ,  algorithm = 'backprop' ,rep  = 10 , err.fct = 'sse' , linear.output = FALSE)
    #mod = neuralnet(form ,bdata, hidden = c(6,12,18) , rep = 10 , err.fct = 'ce' , linear.output = FALSE)
    test = read.csv("test.csv")
    #test = test[1:100 , ]
    res = compute(mod , test )$net.result
    res = sapply(res , function(x){
        ret =x 
        if(ceiling(x)-x<=.5) ret = ceiling(x)
        else ret = floor(x)
    })
    write.csv(x = res , file = "output.csv")
}
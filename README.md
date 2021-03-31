# learned
#### Machine Learning library for Python (very soon C and JavaScript)

## Table of Contents
- ### .neural_network
- #### [Sequential Class](#sequential-class-1)
  * ##### [parameters - hyperparameters](#sequentialparams-1)
  * ##### [methods](#sequentialmethods-1)
- #### [Layer Class](#layer-class-1)
  * ##### [parameters - hyperparameters](#layerparams-1)
  * ##### [methods](#layermethods-1)
  * ##### [example](#dnnexample-1)
- ### .models
- #### [KNN class](#knn-class-1)
  * ##### [parameters - hyperparameters](#knnparams-1)
  * ##### [methods](#knnmethods-1)
  * ##### [example](#knnexample-1)
- #### [LinReg class](#linreg-class-1)
  * ##### [parameters](#linregparams-1)
  * ##### [methods](#linregmethods-1)
  * ##### [example](#linregexample-1)
- #### [LogReg class](#logreg-class-1)
  * ##### [parameters - hyperparameters](#logregparams-1)
  * ##### [methods](#logregmethods-1)
  * ##### [example](#logregexample-1)
- #### [GradientDescent class](#gradientdescent-class-1)
  * ##### [parameters - hyperparameters](#graddescparams-1)
  * ##### [methods](#graddescmethods-1)
  * ##### [example](#graddescexample-1)
- ### .preprocessing
- #### [OneHotEncoder class](#onehotencoder-class-1)
  * ##### [parameters](#oheparams-1)
  * ##### [methods](#ohemethods-1)
  * ##### [example](#ohexample-1)
- #### [normalizer function](#normalizer-1)
  * ##### [parameters](#normalizerparams-1)
  * ##### [example](#normalizerexample-1)
- #### [get_split_data function](#get_split_data-1)
  * ##### [parameters - hyperparameters](#getsplitdataparams-1)
  * ##### [example](#getsplitdataexample-1)
- #### [polynomial_features function](#polynomial_features-1)
  * ##### [parameters - hyperparameters](#polynomialfeaturesparams-1)
  * ##### [example](#polynomialfeaturesparams-1)
- ### .metrics
- #### [confusion_matrix function](#confusionmatrix-1)
  * ##### [parameters](#confusionmatrixparams-1)
  * ##### [example](#confusionmatrixexample-1)
- #### [accuracy](#accuracy-1)
  * ##### [parameters](#accuracyparams-1)
  * ##### [example](#accuracyexample-1)

## LinReg class
    Explanation: 
            LinReg is a class that allows simple or multiple linear regressions and returns trained parameters.

    Parameters: 
            data: Unfragmented structure that contains inputs and outputs.
    Usage:
    '''
    // The "full_dataset" is an unfragmented structure that contains inputs and outputs.

    lin_reg = Learn.LinReg(data=full_dataset)  # or lin_reg = LinReg(full_dataset)
    '''

    Output:
            '''
            <Learn.LinReg at 0x1fdbd6b6220>
            ''' 

  ### LinReg.train
    Explanation: 
            It applies the training process for the dataset entered while creating the class.

    Parameters: 
            This method does not take parameter!
    Usage:
    '''
    lin_reg.train()
    '''

    Output:
            (An example simple linear regression output)
            '''
            Completed in 0.0 seconds.
            Training R2-Score: % 97.0552464372771
            Intercept: 10349.456288746507, Coefficients: [[812.87723722]]
            '''

  ### LinReg.test
    Explanation: 
            Applies the created model to a different input and gives the r2 score result.

    Parameters: 
            t_data: Unfragmented structure that contains inputs and outputs.
    Usage:
    '''
    lin_reg.test(t_data=test_dataset) # or lin_reg.test(test_dataset)
    '''

    Output:
            (An example simple linear regression output)
            '''
            Testing R2-Score: % 91.953582170654
            '''

    Note: 
            Returns an error message if applied for a model that has not been previously trained.
            '''
            Exception: Model not trained!
            '''
  ### LinReg.predict
    Explanation: 
            Applies the created model to the input data, which it takes as a parameter, and returns the estimated results.

    Parameters: 
            x: Input dataset consisting of arguments
    Usage:
    '''
    predicts = lin_reg.predict(x=x_set) # or predicts = lin_reg.predict(x_set)
    '''

    Output:
            Predicted values list

    Note: 
            Returns an error message if applied for a model that has not been previously trained.
            '''
            Exception: Model not trained!
            '''
  ### LinReg.r2_score
    Explanation: 
            It takes actual results and predicted results for the same inputs as parameters and returns the value of r2 score.

    Parameters: 
            y_true: Real results
            y_predict: Estimated results
    Usage:
    '''
    lin_reg.r2_score(y_true=real_results, y_predict=predicted_results) # or lin_reg.r2_score(real_results, predicted_results)
    '''

    Output:
            '''
            0.970552
            dtype: float64
            '''

  ### LinReg.intercept
    Explanation: 
            Returns the trained intercept value

    Parameters: 
            @property (Does not take parameter)
    Usage:
    '''
    intercept = lin_reg.intercept
    '''

    Output:
            '''
            10349.456288746507
            '''

  ### LinReg.coefficients 
    Explanation: 
            Returns the trained coefficients

    Parameters: 
            @property (Does not take parameter)
    Usage:
    '''
    coefficients = lin_reg.coefficients
    '''

    Output:
            '''
            array([[812.87723722]])
            '''
## LogReg class

### Parameters
### LogReg.train
### LogReg.predict

## GradientDescent class

### Parameters
### GradientDescent.optimizer
### GradientDescent.predict
### GradientDescent.get_parameters

## Preprocessing class

### Preprocessing.get_split_data

### TODO
- cross validation
- p-value
- Other algorithms
- Detailed documentation

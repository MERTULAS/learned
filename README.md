# learn
#### Machine Learning library for Python (very soon C and JavaScript)

## Table of Contents
- ### [LinReg class](#linreg-class-1)
  * #### [LinReg.train](#linregtrain-1)
  * #### [LinReg.test](#linregtest-1)
  * #### [LinReg.predict](#linregpredict-1)
  * #### [LinReg.r2_score](#linregr2_score-1)
  * #### [LinReg.intercept](#linregintercept-1)
  * #### [LinReg.coefficients](#linregcoefficients-1)
- ### [LogReg class](#logreg-class-1)
  * #### [LinReg.train](#logregtrain)
  * #### [LinReg.predict](#logregpredict)
- ### [GradientDescent class](#gradientdescent-class-1)
  * #### [Parameters](#parameters-1)
  * #### [GradientDescent.optimizer](#gradientdescentoptimizer-1)
  * #### [GradientDescent.predict](#gradientdescentpredict-1)
  * #### [GradientDescent.get_parameters](#gradientdescentget_parameters-1)
- ### [Preprocessing class](#preprocesssing-class-1)
  * #### [Preprocessing.get_split_data](#preprocessingget_split_data-1)

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

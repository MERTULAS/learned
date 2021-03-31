# learned
#### Machine Learning library for Python (very soon C and JavaScript)

# Table of Contents
- ## [.neural_network](#neural_network-1)
- ### [Sequential Class](#sequential-class-1)
  * #### [parameters - hyperparameters](#sequentialparams-1)
  * #### [methods](#sequentialmethods-1)
- ### [DNNModel Class](#dnnmodel-class-1)
  * #### [parameters](#dnnmodelparams-1)
  * #### [methods](#dnnmodelmethods-1)
  * #### [example](#dnnmodelexample)
- ### [Layer Class](#layer-class-1)
  * #### [parameters - hyperparameters](#layerparams-1)
  * #### [methods](#layermethods-1)
  * #### [example](#dnnexample-1)
- ## [.models](#models-1)
- ### [KNN class](#knn-class-1)
  * #### [parameters - hyperparameters](#knnparams-1)
  * #### [methods](#knnmethods-1)
  * #### [example](#knnexample-1)
- ### [LinReg class](#linreg-class-1)
  * #### [parameters](#linregparams-1)
  * #### [methods](#linregmethods-1)
  * #### [example](#linregexample-1)
- ### [LogReg class](#logreg-class-1)
  * #### [parameters - hyperparameters](#logregparams-1)
  * #### [methods](#logregmethods-1)
  * #### [example](#logregexample-1)
- ### [GradientDescent class](#gradientdescent-class-1)
  * #### [parameters - hyperparameters](#graddescparams-1)
  * #### [methods](#graddescmethods-1)
  * #### [example](#graddescexample-1)
- ## [.preprocessing](#preprocessing-1)
- ### [OneHotEncoder class](#onehotencoder-class-1)
  * #### [parameters](#oheparams-1)
  * #### [methods](#ohemethods-1)
  * #### [example](#ohexample-1)
- ### [normalizer function](#normalizer-1)
  * #### [parameters](#normalizerparams-1)
  * #### [example](#normalizerexample-1)
- ### [get_split_data function](#get_split_data-1)
  * #### [parameters - hyperparameters](#getsplitdataparams-1)
  * #### [example](#getsplitdataexample-1)
- ### [polynomial_features function](#polynomial_features-1)
  * #### [parameters - hyperparameters](#polynomialfeaturesparams-1)
  * #### [example](#polynomialfeaturesparams-1)
- ## [.metrics](#metrics-1)
- ### [confusion_matrix function](#confusionmatrix-1)
  * #### [parameters](#confusionmatrixparams-1)
  * #### [example](#confusionmatrixexample-1)
- ### [accuracy](#accuracy-1)
  * #### [parameters](#accuracyparams-1)
  * #### [example](#accuracyexample-1)



## .neural_network

Explanation: It contains the classes required for the deep neural network. These classes can be customized with various functions. The trained model can be saved as a folder, then call this folder and used to predict other entries

### Sequential class

	Explanation: 
			This class is used to create a sequential deep learning structure.
   
   	Parameters:
			x: input values, as type below
			For example, if you select images for input values and the input data contains 30 sample images in 28x28 size, the images should be flattened to (pixel x N_sample), converted to (784, 30) and then entered into the model.
			y: data which size of (1 x N_samples) for regression or (class_number x N_samples) for classification 
			Note that: It can be (1 x N_samples) for binary classification. (if output layer contains sigmoid function)
	
	Hyperparameters:
			learning_rate: it can be changed in case of exploding or vanishing of gradients. (Default value is 0.01)
			iteration: iteration number. (Default value is 1000)
			loss: the loss function to be applied to the model is specified. (Default value is "binary_cross_entropy")
				Speciable loss functions:
					For classifications:
						"binary_cross_entropy" : 
						![image](https://user-images.githubusercontent.com/67822910/113151329-a4c58300-923d-11eb-83d5-ca39e1dcc836.png)
						
						"cross_entropy" :
						![image](https://user-images.githubusercontent.com/67822910/113151263-94ada380-923d-11eb-8ad7-2747fb725d3f.png)
						
					For regressions:
						"mean_square_error" : 
						![image](https://user-images.githubusercontent.com/67822910/113151197-82cc0080-923d-11eb-8672-67b0cec52c12.png)
						
						"mean_absolute_error" :
						![image](https://user-images.githubusercontent.com/67822910/113151629-f4a44a00-923d-11eb-8ca2-bad33064df74.png)
			
      
   	methods: 
			Sequential.add(x): adds a layer to the model structure. (x is a object which includes "Layer" class (very soon also "Convolution" class))
			Sequential.train(): it starts the learning process and does not take parameters.
			Sequential.test(x, y): it gives the accuracy value for the test inputs and test outputs
			Sequential.predict(x): returns the predicted value / category for x value
			Sequential.save_model("model_name"): saves the trained model as a folder as specified in the parameter name. (To the same directory)
			Sequential.cost_list: it gives the costs for visualisation
			Sequential.accuracy_list: it gives the accuracies for visualisation


### DNNModel class

	Explanation:
			This class loads saved models
	
	Parameters:
			model_folder: it takes saved model's folder name
	
	Methods:
			DNNModel.predict(x): returns the predicted value / category for x value
			

### Layer class

	Explanation:
			ANN model's hidden layers are defined by this layer
			
	Hyperparameters:
			neurons: indicates how many neurons the layer has
			weights_initializer: determines how layer weights are started (default value is "uniform")
					"he_uniform":
							suitable_size_uniform_values * sqrt(6 / prev_layers_output_size)
					
					"he_normal":
							suitable_size_uniform_values * sqrt(2 / prev_layers_output_size)
							
					"xavier_uniform":
							suitable_size_uniform_values * sqrt(6 / (prev_layers_output_size + layer_neurons_size))
							
					"xavier_normal":
							suitable_size_uniform_values * sqrt(2 / (prev_layers_output_size + layer_neurons_size))
					
					"uniform":
							suitable_size_uniform_values * 0.1
							
					Note that: "he" initializers better for relu / leaky_relu activation functions
					
			activation: determines with which function the layer will be activated. (default values is "tanh")
					"sigmoid": 0 - 1
							![image](https://user-images.githubusercontent.com/67822910/113162450-fd9a1900-9247-11eb-9845-a9db231ff7d3.png)
					
					"tanh":  -1 - 1
							![image](https://user-images.githubusercontent.com/67822910/113162775-3df99700-9248-11eb-8301-6b014d3fb57a.png)
					
					"relu":  it makes all negative values to zero
							![image](https://user-images.githubusercontent.com/67822910/113163132-8b760400-9248-11eb-9e59-8f72ea9471dc.png)
					
					"softmax": it is a probability function, it return values which sums of values equal 1
							![image](https://user-images.githubusercontent.com/67822910/113164891-33d89800-924a-11eb-9b5f-8aab6b94af0c.png)
					
					"leaky_relu": it don't makes all negative values to zero but makes too close to zero
							![image](https://user-images.githubusercontent.com/67822910/113166087-1d7f0c00-924b-11eb-8a38-870a4149a081.png)
							
   		Example for neural network structure:
					from learned.neural_network.models import Sequential, DNNModel
					from learned.neural_network.layers import Layer
					from learned.preprocessing import get_split_data, normalizer, OneHotEncoder
					
					mnist = pd.read_csv("train.csv")
					mnist.head()
					train, test = get_split_data(mnist, test_percentage=0.33)
					print(train.shape)
					>>> (28140, 785)
					y_labels_tr = train[:, :1]
					y_labels_te = test[:, :1]
					pixels_tr = train[:, 1:]
					pixels_te = test[:, 1:]
					pixels_tr = normalizer(pixels_tr)
					pixels_te = normalizer(pixels_te)
					pixels_tr = pixels_tr.T
					pixels_te = pixels_te.T
					print(pixels_tr.shape)
					>>> (784, 28140)
					ohe_tr = OneHotEncoder(y_labels_tr).transform()
					ohe_te = OneHotEncoder(y_labels_te).transform()
					
					Model = Sequential(pixels_tr, ohe_tr, learning_rate=0.01, loss="cross_entropy", iteration=100)

					Model.add(Layer(neurons=150, activation="relu", weights_initializer="he_normal"))
					Model.add(Layer(neurons=150, activation="relu", weights_initializer="he_normal"))
					Model.add(Layer(neurons=150, activation="relu", weights_initializer="he_normal"))
					Model.add(Layer(neurons=10, activation="softmax", weights_initializer="xavier_normal"))
					
					Model.train()
					>>>![image](https://user-images.githubusercontent.com/67822910/113175015-a7cb6e00-9253-11eb-8963-ba9e52d8f9f3.png)
					pred = Model.predict(pixels_tr)
					Model.save_model("mnist_predicter")
					loaded_model = DNNModel("mnist_predicter")
					pred2 = loaded_model.predict(pixels_tr)
					
					>>> pred2 == pred


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

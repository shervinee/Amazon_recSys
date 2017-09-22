# Evaluations Implementation

The implementation for this project has been coded on top of scikit library known as Surprise. http://surprise.readthedocs.io/en/stable/

For running the implmentations surprise module should be downloaded on your system.
If you have the pip package manager for python you can directly install surprise. using the following command in your terminal. (given that your system already has the numpy and pandas modules from python)

    $ pip install scikit-surprise

It is recommended to install this module on an virtual environment of python, since after installing it directly on some systems surprise would not work.
After this all should be ready. The only thing needs to be done is simply running the code of the already written file. runner.py in the package with the given dataset which is a very small random sample of used dataset for the project.



## How implementations coded
In the case of SVD and SVD++ along with an unfinished version of AFM. The implemented algorithm, based on surprise documentation, would need to have an estimate function for do the estimation. Since we are dealing with model-based algorithms and not direct estimation we would need to train our approaches. In all classes there is a train function for training the model and estimation after would be used to test the model. All algorithms have been coded in a way that user can change all initial parameters including number of factors, learning rate, etc.
Other components of project include:
  - There are four files regarding the generation of the plots based on found values of evaluations. Results are being saved into corresponding png files.
  - The last two sampled dataset and their results is included in the package. As the results are shown the standard deviation of found RMSE values are little low.
  - Based on experience with surprise, many new implemented algorithms would generate a pyc file for havinh faster execution this is the case for our SVD and SVD++ file implementation in SVD_A.py
  - the default number of factors used based on the characteristics of dataset were 35 and 15 for SVD and SVD++ implementation respectively. SVD++ based on its computation time and also adding the implicit feedback would learn parameters faster than SVD and also is better for performance to have lower number of iterations.




**Thanks for using**

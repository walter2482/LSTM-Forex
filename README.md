# LSTM-Forex
This GitHub repository hosts a codebase that enables the efficient generation of over 150 technical indicators and statistical properties using modules such as pandas_ta and talib. These indicators and properties are utilized to feed an LSTM-GRU architecture capable of predicting the behavior of the EURUSD currency pair in the forex market, utilizing multiple time frames.

A notable feature of this project is its ability to automatically download the necessary data for model training. Additionally, the code takes care of evolving the hyperparameter network of the architecture and fine-tuning a base architecture to subsequently train an optimized model based on that architecture.

This repository provides a comprehensive solution for developing prediction models in the forex market, leveraging a wide variety of technical indicators and employing advanced machine learning techniques. The automated approach and model architecture optimization contribute to enhancing prediction accuracy, offering a valuable resource for individuals interested in the analysis and forecasting of EURUSD currency pair movements.

1. Obtaining the current date and setting the start and end date of the data to be used.
2. Creating a destination folder to store the data files.
3. Downloading data sets for the specified financial instrument pairs.
4. Reading of the downloaded data files and creation of a single DataFrame concatenating all DataFrames.
5. Calculation of technical indicators and market characteristics, such as momentum, trends, volatility, volume, among others.
6. Preparation of data for modeling in Keras, including normalization and division into training, validation and test sets.
7. Creation of a model in Keras to predict the closing prices using the processed data.
8. Training of the model using the training data and evaluation of performance using the validation and test sets.


## Creating a Virtual Environment

To create a virtual environment from the "requirements.txt" file and ensure that dependencies are installed correctly, follow these steps:

1. Open a terminal or command line in the directory where the "requirements.txt" file is located.

2. Create a new virtual environment using the tool of your choice, such as virtualenv or conda. For example, with virtualenv, you can run the following command:
virtualenv myenv

This will create a new virtual environment named "myenv" in the current directory.

3. Activate the virtual environment. The activation process may vary depending on the operating system and the tool used. For virtualenv, you can use the following commands:
- On Windows:
  ```
  myenv\Scripts\activate
  ```
- On Linux or macOS:
  ```
  source myenv/bin/activate
  ```

4. Once the virtual environment is activated, you can install the dependencies using the "requirements.txt" file. Run the following command:
pip install -r requirements.txt

This will install all the dependencies specified in the "requirements.txt" file into your virtual environment.

## GitHub Repository Structure

To structure the publication on GitHub, you can follow these steps:

1. Create a new repository on GitHub by clicking on the "New" button on the repository page.

2. Give your repository a name and provide an optional description.

3. Choose the visibility of your repository (public or private).

4. Optionally, you can initialize the repository with a README file. You can choose to create the README file now or add it later.

5. Once the repository is created, you can go to the repository's main page.

6. Click on the "Add file" button and select "Upload files" to upload your project files, including the "requirements.txt" file.

7. After uploading the files, you can provide a commit message and click on the "Commit changes" button to save the files to your repository.

8. Now, other users can access your repository, view the code, and download the "requirements.txt" file if needed.

Remember to include any additional documentation or instructions in your repository's README file to guide users on how to set up the environment and use your project.

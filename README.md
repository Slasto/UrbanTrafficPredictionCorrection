# Time Prediction of Urban Traffic from Real Data with spatial/temporal Imputation for Missing Gaps
 
This project is started as bachelor degree thesis of Agostino Vallefuoco and Giuseppe di Lorenzo in [Electronic and Computer Engineering](https://www.ingegneria.unicampania.it/didattica/corsi-di-studio/ingegneria-elettronica-e-informatica), and is connected to a paper (references will be added as soon as possible).

## Short Description

This study employs artificial neural network (ANN) architectures, with a particular focus on Stacked Long Short-Term Memory (LSTM) models aimed at providing reliable short-term traffic prediction.

To validate the proposed approach, we conducted a case study on the Adelaide city dataset.
Integrating contextual variables, such as weather conditions and holidays, has been shown to further enhance the predictive performance of the model.

The Stacked model was also tested on n-step-ahead traffic predictions.

Another aspect addressed in this work concerns recognizing the challenges associated with the accuracy of real-world data, often afflicted by recording errors due to sensor malfunctions, through temporal and spatial imputation.

## How to Run the Code

To execute the code for this project, you need to have Docker installed on your system.
Docker allows creating an isolated and reproducible environment for running the code, ensuring all requirements are met without interfering with other applications or libraries on the system.

### Project Initialization

1. Open a shell or terminal.
2. Navigate to the main directory of the project where the `docker-compose.yml` file is located.
3. Execute the following command to start the Dockerized environment: `docker compose up --build`
4. Enter the Dockerized environment by accessing the container (e.g., use the *Dev Containers* extension with VScode)
5. Inside the container, run the script `/data/Traffic_Intersection/Download_data.py` to download the traffic dataset.
6. Next, run `/data/Weather/Download_data.py` to download the weather dataset.
7. Finally, execute the notebooks in the `/src/` folder named `0_Fix_Traffic_Dataset` and `1_Merge_and_normalize_data` to obtain the final dataset
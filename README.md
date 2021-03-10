[](https://thornwaterbalance.com/index.html#)

## WaterbalANce
The Thornthwaite-Mather water balance (Thornthwaite & Mather, 1955; Thornthwaite & Mather, 1957) uses an accounting procedure to analyze the allocation of water among various components of the hydrologic system. 

We recently published a paper. Here is the link to our work; if you use the app / code, please cite our work.

> Mammoliti, E.; Fronzi, D.; Mancini, A.; Valigi, D.; Tazioli, A.
> WaterbalANce, a WebApp for Thornthwaite–Mather Water Balance
> Computation: Comparison of Applications in Two European Watersheds.
> _Hydrology_  **2021**, _8_, 34. https://doi.org/10.3390/hydrology8010034

## WebApp
We developed a web-app that is able to simplify the interaction with users that want to get results without installing any specific software.
The web-app is available at the following address:
https://thornwaterbalance.com/index.html
For more details visit the  [Methods](https://thornwaterbalance.com/methods.html)  

## Thornthwaite-Mather Python code

We also developed a python version of the code that could be easily run also online using Google Colab resources.
The demo.py provides a basic way to use the developed code.
Required inputs are described [here](https://thornwaterbalance.com/methods.html)  
An example of input data is stored at the following [address](https://thornwaterbalance.com/example.xlsx) while the template to provide data could be download [here](https://thornwaterbalance.com/example.xlsx).
To run the code just cole the repo, adjust the CONFIG_ALG in demo.py and then run it:

    python demo.py
Results are:

 - Excel data file
 - Plot in png format.

Dependencies could be installed using the following command:

    pip install -r requirements.txt

The code is licensed under the GNU GENERAL PUBLIC LICENSE umbrella.
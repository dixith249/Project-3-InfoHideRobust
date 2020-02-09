The first thing to do is install the requeriments. Go to your terminal, go to this folder and paste or type the command below:

pip install -r requirements.txt

After finish the install of requirements, in the same folder have the 3 components:

- main.py(the script)
- Input/ (the folder where you will place all images to be processed)
- Output/ (the folder wher will be saved all processed images, where each run generate a timestamp folder containing the processed images)

In the folder of the script, paste or type the command below to run:

python3 main.py 

In menu you will have the following options:

Attacks to image
0 - Rotation
1 - Translation
2 - Scaling
3 - Cropping
4 - Histogram equalization
5 - Sharpening
6 - Edge detection 1
7 - Edge detection 2
8 - Edge detection 3
9 - Noise - Salt & Pepper
10 - Noise - Gaussian
11 - Noise - Uniform
12 - Noise - Rayleigh
13 - Noise - Poisson
14 - Noise - Erlang
15 - Noise - Exponential
16 - Blur - Average
17 - Blur- Gaussian
18 - Blur - Median
19 - Blur - Bilateral
20 - All edge detection
21 - All noise
22 - All blur
23 - All attacks

You can select one or various options, for example, if you want just Rotation, Translation and Cropping, just type on terminal:

Please enter the number of options of the attack spaced:0 1 3

And hit enter, the process will start and make the choiced methods on the images of the Input Folder, just set parameters for each method choiced.

Any questions you can ask me.

Thank you!

-Dixith Pinjari

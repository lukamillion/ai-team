# Other - Dataset

## Dataset

We used the *Data archive of experimental data from studies about pedestrian dynamics* from the [Forschungszentrum Jülich](https://ped.fz-juelich.de/database/doku.php). We chose the two datasets for our further experiments: a corridor situation (UG) and a bottleneck (AO). A detailed explenation of the experimental situations can be found in the [description](VersuchsdokumentationHERMES.pdf)

In the Folers UG and AO one findes the pretracked data in the folder ug-vid and ao-vid the raw video footage is provided. 

### Corridor - UG

![Corridor scematics](https://github.com/lukamillion/ai-team/blob/main/other/UG/ug.png)

We used the preprocessed data given in the folder [UG-roh_nachkorrigiert](UG-roh_nachkorrigiert). We used the corridor with a width of 180cm and a person number between 15 and 230. The files are labeled: ug-[width]-[#people].txt 


### Bottleneck - AO

![Bottleneck scematics](https://github.com/lukamillion/ai-team/blob/main/other/AO/ao.jpg)

The datasets were converted to a fixed with file format in order to macht our dataloader needs.

In all datasets 400 people are walking thrugh a bottleneck with a width of 240-500 cm. We used the [ao-360-400_combine.txt](ao-360-400_combine.txt) for the final tests.
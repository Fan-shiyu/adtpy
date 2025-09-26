# Multivariate Cosine Similarity: A Method for Comparing Dataset Similarities

[![R package](https://img.shields.io/badge/R_package-GitHub-276DC3?logo=R&logoColor=white)](https://github.com/Fan-shiyu/Multivariate-Cosine-Similarity)
[![Python package](https://img.shields.io/badge/Python_Package-GitHub-3776AB?logo=python&logoColor=white)](https://github.com/Fan-shiyu/adtpy)
[![Web App](https://img.shields.io/badge/Web_App-Live-orange?logo=google-chrome&logoColor=white)](https://5683wm-shiyu-fan.shinyapps.io/Multivariate_Cosine_Similarity/)

For the theoretical background of this method, please refer to: [Multivariate Cosine Similarity Research](https://researchspace.auckland.ac.nz/items/7c97b5ae-985c-43c7-9c65-705caa7ac853) 

## Tutorial

This is one Python package which can be used to compare the similarity of two data sets. The method is based on projection, and it starts with the results of PCA (Principal Component Analysis). 

To install this package, please run:


```python
!pip install git+https://github.com/Fan-shiyu/adtpy.git
```

### Data Simulation

Firstly, two similar data sets are simulated by adding some noise to a reference data set. Here the reference data set is [mtcars](https://www.rdocumentation.org/packages/datasets/versions/3.6.2/topics/mtcars).


```python
from adtpy import *
import pandas as pd
```


```python
url = "https://raw.githubusercontent.com/selva86/datasets/master/mtcars.csv"
df1 = pd.read_csv(url).iloc[:, :11]
df2 = simulation(df1, sd=0.6, seed=123)
```

### PCA

To run a PCA method, here the number of principle components requires to be defined, these PCs (principle components) will be used in the following analysis.

Function `get_loadings()` can be implemented to choose how many principle components(PCs) you need. One way is to through the cumulative proportion of variance explained. This function provides argument `explain` to determine the lower bound on the cumulative proportion of variance explained of PCs. 

In addition, you can set the number of PCs that you want, but it should be less than 3. Such as `n_loadings=3`, that means you want to reserve three PCs. By setting `scree_plot_flag=TRUE`, a scree plot is drawn, and you can see the cumulative proportion of variance explained for each PC.


```python
d1_PCs = get_loadings(df1, n_loadings=3, scree_plot_flag=True)
d2_PCs = get_loadings(df2, n_loadings=3, scree_plot_flag=True)
```


<img width="545" height="393" alt="output_9_0" src="https://github.com/user-attachments/assets/f441e47d-6bc4-46ca-a287-841319a860be" />


    
<img width="536" height="393" alt="output_9_1" src="https://github.com/user-attachments/assets/85097fba-a276-4d78-97e7-37d2dc526076" />


### Projection

Class 'proj' contains the information of projection. Function `proj()` can transform PCs into the class 'proj'. To run function `proj_compute()`, you can get the results of projection..



```python
proj3 = proj(d1_PCs, d2_PCs, n_loadings=3)
proj3 = proj_compute(proj3)
proj3
```




    Proj(similarity=0.9578911120583546, n_loadings=3, proj_coords=
              x         y         z
    0  0.911463  0.341286  0.229693
    1 -0.358030  0.931916 -0.057862
    2  0.216733  0.167027  0.961836)



Venn Diagram can be plotted to show data similarity.


```python
venn_plot(proj3, label=True, title=True, anno=True)
```

<img width="617" height="566" alt="output_13_0" src="https://github.com/user-attachments/assets/4e3c8e11-a336-442e-b122-fc2000d9762a" />
 


### Some Plots for Data Exploration

Here are some functions to visualize the relationships between PCs.

Run `pair_density_plot()` to plot paired density plot.



```python
pair_density_plot(proj3,
                  cols=('red', 'blue'),
                  lwd=2,
                  xlim=(-1, 1),
                  ylim=(0, 2),
                  legend=True,
                  title='Density Comparison of PCs')
```


    
<img width="1489" height="413" alt="output_16_0" src="https://github.com/user-attachments/assets/a141d919-0630-4353-b646-5adfbadbee46" />


Run `pair_correlation_plot()` to plot regression scatter plot.


```python
pair_correlation_plot(proj3,
                      legend=True,
                      figsize=(12, 4),
                      title='Correlation of PCs')
```


    
    
<img width="1189" height="425" alt="output_18_0" src="https://github.com/user-attachments/assets/9b30e95b-d09d-473b-8381-9bcd46816ca7" />


Run `pair_boxplot()` to plot paired boxplot.


```python
pair_boxplot(proj3,
             cols=('#fbb4ae','#b3cde3'),  
             title='Box Plot of PCs',
             legend_text=('Data 1', 'Data 2'))
```


    
<img width="889" height="490" alt="output_20_0" src="https://github.com/user-attachments/assets/59d04084-6b99-44f7-9a7b-28c1dc6493de" />


Run `pair_vioplot()` to plot paired violin plot.


```python
pair_vioplot(proj3,
             title='Violin Plot of PCs',
             legend_text=('Data 1', 'Data 2'))
```

<img width="989" height="589" alt="output_22_0" src="https://github.com/user-attachments/assets/db62bcaf-ca8d-4ccc-bec6-c607a00ff2da" />

    
### Summary  

The **`adtpy`** package offers a complete workflow for comparing the similarity of two datasets using a projection-based approach. It allows users to extract principal components, project them between datasets, and compute similarity scores through cosine-based metrics. In addition to numerical outputs such as similarity values, vector lengths, and angles, the package provides a variety of visualization tools including scree plots, Venn diagrams, and exploratory plots such as density, correlation, box, and violin plots, making it straightforward to assess dataset similarity both quantitatively and visually.  

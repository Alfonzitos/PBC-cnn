# PBC-cnn
A Convolutional Neural Network(CNN) written in C++ capable of classifying Peripheral Blood Cells(PBC) from microscopy images at 90% validation accuracy.

This project does not rely on any High-level Machine Learning libraries(Pytorch, Tensorflow). Instead, the underlying mathematics have been implemented primarily using the library Eigen for fast, efficient linear algebra computations.   

The network consists of the following components
  - Convolutional layers
  - Max Pooling Layers
  - Global Average Pooling layers
  - Dense layers

The models specific design can be found in src/ as well as learned parameters in training/. The model was trained on ~15.000 images of PBC found at https://www.kaggle.com/datasets/kylewang1999/pbc-dataset. The performance was then tested on ~2500 images that was not part of the training data to correct for memorisation.

## Results:
```
Training cost: 0.256077, validation cost: 0.269354, validation accuracy: 91.3315% (2339/2561)
Per-class validation accuracy:
  basophil: 87.3626% (159/182)
  eosinophil: 98.5011% (460/467)
  erythroblast: 83.1897% (193/232)
  ig: 83.6405% (363/434)
  lymphocyte: 91.2088% (166/182)
  monocyte: 87.3239% (186/213)
  neutrophil: 93.3868% (466/499)
  platelet: 98.2955% (346/352)
```
## Error analysis
The confusion matrix below gives insights into what the model thinks each actual value(rows) should be(columns). E.g monocytes are often mistaken for IG.
| actual/predicted | basophil | eosinophil | erythroblast | ig  | lymphocyte | monocyte | neutrophil | platelet |
| :--------------- | :------- | :--------- | :----------- | :-- | :--------- | :------- | :--------- | :------- |
| basophil         | 159      | 2          | 0            | 14  | 4          | 1        | 2          | 0        |
| eosinophil       | 4        | 460        | 0            | 0   | 0          | 0        | 3          | 0        |
| erythroblast     | 2        | 1          | 193          | 13  | 7          | 3        | 9          | 4        |
| ig               | 16       | 2          | 6            | 363 | 3          | 16       | 28         | 0        |
| lymphocyte       | 1        | 0          | 4            | 9   | 166        | 2        | 0          | 0        |
| monocyte         | 3        | 0          | 0            | 24  | 0          | 186      | 0          | 0        |
| neutrophil       | 1        | 1          | 2            | 26  | 3          | 0        | 466        | 0        |
| platelet         | 0        | 0          | 6            | 0   | 0          | 0        | 0          | 346      |

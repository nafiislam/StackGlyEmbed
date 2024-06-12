# EMNGLY

The EMNGly approach utilizes pretrained protein language model (Evolutionary Scale Modeling) and pretrained protein structure model (Inverse Folding Model) for features extraction and Support Vector Machine(SVM) for classification. Ten-fold cross-validation and independent tests show that this approach has outperformed existing techniques. 

How to run
```
git clone xxx
cd EMNgly
python test.py
```


```
├── README.md
├── data           
│   └── N-GlycositeAtlas-test.csv
├── model
│    └──  EMNgly.pickle
│
├──  test.py  
│
└──  train.py
```    

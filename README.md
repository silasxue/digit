###Digit Recognition via CNN
![](http://7xn7wz.com1.z0.glb.clouddn.com/digit.jpg)  
####to train:
* download [dataset](http://7xocv2.dl1.z0.glb.clouddn.com/digit.tar.gz)
* at `examples/digit/` run `creat_digit.sh` to create *LMDB* database for training  
* run solve.sh to train *digital model*  

####to test:
* download pre-trained [model](http://7xocv2.dl1.z0.glb.clouddn.com/digit_iter_5000.caffemodel) or train your own model as metioned above.
* run `demo.m`(need Matlab and matcaffe)
* python bindings are on the go.

Code base on [*caffe*](http://caffe.berkeleyvision.org/)  

***  

Mail: <zhaok.shu@gmail.com> Page: <http://zhaok.xyz>
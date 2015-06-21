Theano是深度学习的利器。虽然现在有很多基于Theano的Python包，简化了构造深度神经网络的过程，如Lasagne，blocks等。但是使用了Lasagne之后，觉得还是不够灵活。直接使用Theano才是研究者应该有的态度。

从Theano给出的基本Logistic Regression和MLP的例子出发，一步一步地实现adaptive Learning rate, RMPprop, drop out等算法。本文的代码可以在GitHub下载(https://github.com/huangzhengsjtu/MLP)。使用的测试集是MNIST。

##1 先运行一下deeplearning.net给出的例子

mlp_ver1.py基于Theano tutorial给出的例子，里面增加了输出Classification的概率。
```
python mlp_ver1.py
```

输出(verr: validation error, merr: model err)：

```
...
e 80, verr 11.8 % , merr 12.5 % , pp 1059, n 1100 ,  pp 96.3 %
e 81, verr 11.8 % , merr 12.5 % , pp 1059, n 1100 ,  pp 96.3 %
e 82, verr 11.8 % , merr 12.5 % , pp 1060, n 1101 ,  pp 96.3 %
e 83, verr 11.7 % , merr 12.4 % , pp 1062, n 1103 ,  pp 96.3 %
e 84, verr 11.7 % , merr 12.4 % , pp 1062, n 1103 ,  pp 96.3 %
e 85, verr 11.6 % , merr 12.4 % , pp 1062, n 1103 ,  pp 96.3 %
e 86, verr 11.6 % , merr 12.3 % , pp 1062, n 1103 ,  pp 96.3 %
e 87, verr 11.6 % , merr 12.3 % , pp 1062, n 1103 ,  pp 96.3 %
e 88, verr 11.6 % , merr 12.3 % , pp 1063, n 1104 ,  pp 96.3 %
e 89, verr 11.5 % , merr 12.2 % , pp 1063, n 1104 ,  pp 96.3 %
e 90, verr 11.5 % , merr 12.2 % , pp 1063, n 1104 ,  pp 96.3 %
e 91, verr 11.5 % , merr 12.2 % , pp 1063, n 1104 ,  pp 96.3 %
e 92, verr 11.4 % , merr 12.2 % , pp 1063, n 1104 ,  pp 96.3 %
e 93, verr 11.4 % , merr 12.1 % , pp 1063, n 1103 ,  pp 96.4 %
e 94, verr 11.4 % , merr 12.1 % , pp 1063, n 1103 ,  pp 96.4 %
e 95, verr 11.4 % , merr 12.1 % , pp 1063, n 1103 ,  pp 96.4 %
e 96, verr 11.3 % , merr 12.0 % , pp 1063, n 1103 ,  pp 96.4 %
e 97, verr 11.3 % , merr 12.0 % , pp 1063, n 1103 ,  pp 96.4 %
e 98, verr 11.3 % , merr 12.0 % , pp 1063, n 1102 ,  pp 96.5 %
e 99, verr 11.3 % , merr 12.0 % , pp 1063, n 1103 ,  pp 96.4 %
e 100, verr 11.2 % , merr 11.9 % , pp 1063, n 1103 ,  pp 96.4 %
The code for file mlp.py ran for 26.98m

```


##2 改变Learning rate. 

改变Learning  Rate的关键是将learning_rate声明为一个Symbolic Variable，并且在Training  Function调用的时候，改变learning_rate的值。

```
lrlist=numpy.arange(learning_rate_start,learning_rate_end,(learning_rate_end-learning_rate_start)/n_epochs)
learning_rate = T.scalar('lr')  # learning rate to use

...

train_model = theano.function(inputs=[index , learning_rate], outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size],
                is_train: numpy.cast['int32'](1)})
```

```
python mlp_ver2.py
```

输出：


```
e 80, verr 9.6 % , merr 10.0 % , pp 1079, n 1118 ,  pp 96.5 %
e 81, verr 9.6 % , merr 10.0 % , pp 1079, n 1119 ,  pp 96.4 %
e 82, verr 9.6 % , merr 10.0 % , pp 1079, n 1119 ,  pp 96.4 %
e 83, verr 9.6 % , merr 10.0 % , pp 1079, n 1119 ,  pp 96.4 %
e 84, verr 9.6 % , merr 10.0 % , pp 1079, n 1119 ,  pp 96.4 %
e 85, verr 9.6 % , merr 10.0 % , pp 1079, n 1119 ,  pp 96.4 %
e 86, verr 9.6 % , merr 10.0 % , pp 1079, n 1119 ,  pp 96.4 %
e 87, verr 9.6 % , merr 10.0 % , pp 1079, n 1119 ,  pp 96.4 %
e 88, verr 9.6 % , merr 10.0 % , pp 1079, n 1119 ,  pp 96.4 %
e 89, verr 9.6 % , merr 10.0 % , pp 1079, n 1119 ,  pp 96.4 %
e 90, verr 9.6 % , merr 10.0 % , pp 1079, n 1119 ,  pp 96.4 %
e 91, verr 9.6 % , merr 10.0 % , pp 1079, n 1119 ,  pp 96.4 %
e 92, verr 9.6 % , merr 10.0 % , pp 1079, n 1120 ,  pp 96.3 %
e 93, verr 9.6 % , merr 10.0 % , pp 1079, n 1120 ,  pp 96.3 %
e 94, verr 9.6 % , merr 10.0 % , pp 1079, n 1120 ,  pp 96.3 %
e 95, verr 9.6 % , merr 10.0 % , pp 1079, n 1120 ,  pp 96.3 %
e 96, verr 9.6 % , merr 10.0 % , pp 1079, n 1120 ,  pp 96.3 %
e 97, verr 9.6 % , merr 10.0 % , pp 1079, n 1120 ,  pp 96.3 %
e 98, verr 9.6 % , merr 10.0 % , pp 1079, n 1120 ,  pp 96.3 %
e 99, verr 9.6 % , merr 10.0 % , pp 1079, n 1120 ,  pp 96.3 %

The code for file mlp.py ran for 27.70m
```

##3 使用sklearn来分析一下输出结果. 

添加用pyplot和sklearn来分析一下输出的结果。

```
python mlp_ver3.py
```


```
e 89, verr 9.6 % , merr 10.0 % , pp 1079, n 1119 ,  pp 96.4 %
e 90, verr 9.6 % , merr 10.0 % , pp 1079, n 1119 ,  pp 96.4 %
e 91, verr 9.6 % , merr 10.0 % , pp 1079, n 1119 ,  pp 96.4 %
e 92, verr 9.6 % , merr 10.0 % , pp 1079, n 1120 ,  pp 96.3 %
e 93, verr 9.6 % , merr 10.0 % , pp 1079, n 1120 ,  pp 96.3 %
e 94, verr 9.6 % , merr 10.0 % , pp 1079, n 1120 ,  pp 96.3 %
e 95, verr 9.6 % , merr 10.0 % , pp 1079, n 1120 ,  pp 96.3 %
e 96, verr 9.6 % , merr 10.0 % , pp 1079, n 1120 ,  pp 96.3 %
e 97, verr 9.6 % , merr 10.0 % , pp 1079, n 1120 ,  pp 96.3 %
e 98, verr 9.6 % , merr 10.0 % , pp 1079, n 1120 ,  pp 96.3 %
e 99, verr 9.6 % , merr 10.0 % , pp 1079, n 1120 ,  pp 96.3 %

Classification report:
             precision    recall  f1-score   support

          0       0.93      0.97      0.95      1001
          1       0.90      0.98      0.94      1127
          2       0.92      0.86      0.89       991
          3       0.89      0.87      0.88      1032
          4       0.90      0.92      0.91       980
          5       0.88      0.83      0.86       863
          6       0.92      0.95      0.93      1014
          7       0.93      0.91      0.92      1070
          8       0.90      0.83      0.86       944
          9       0.86      0.90      0.88       978
```



##4 改变Update的方法 

直接用Stocastic Desendent还是比较慢的，RMSProp，AdaDelta等方法据说是更快的。实现的原理就改变updates，根据Theano的设计，每次调用Theano function的时候，顺便会做一个update。update是一个List（或者dictionary），看了就明白是如何update。Lasagne中有个updates.py，里面实现了众多的update方法，我就直接用了，例如使用RMSprop来update参数：

```
python mlp_ver4.py
```

结果：可以看出RMSprop收敛的确快很多。 学习的模型比上一个模型好。

```
e 0, verr 10.1 % , merr 10.6 % , pp 1086, n 1145 ,  pp 94.8 %
e 1, verr 9.4 % , merr 9.7 % , pp 1091, n 1145 ,  pp 95.3 %
e 2, verr 9.0 % , merr 9.3 % , pp 1091, n 1141 ,  pp 95.6 %
e 3, verr 8.6 % , merr 8.8 % , pp 1092, n 1137 ,  pp 96.0 %
e 4, verr 8.2 % , merr 8.3 % , pp 1092, n 1135 ,  pp 96.2 %
e 5, verr 7.7 % , merr 7.9 % , pp 1093, n 1129 ,  pp 96.8 %
e 6, verr 7.3 % , merr 7.4 % , pp 1092, n 1126 ,  pp 97.0 %
e 7, verr 6.9 % , merr 7.0 % , pp 1093, n 1124 ,  pp 97.2 %
e 8, verr 6.6 % , merr 6.6 % , pp 1093, n 1124 ,  pp 97.2 %
e 9, verr 6.4 % , merr 6.3 % , pp 1092, n 1124 ,  pp 97.2 %
e 10, verr 6.2 % , merr 6.0 % , pp 1093, n 1124 ,  pp 97.2 %
e 11, verr 5.9 % , merr 5.7 % , pp 1093, n 1124 ,  pp 97.2 %
e 12, verr 5.8 % , merr 5.5 % , pp 1092, n 1122 ,  pp 97.3 %
e 13, verr 5.6 % , merr 5.3 % , pp 1092, n 1121 ,  pp 97.4 %
e 14, verr 5.4 % , merr 5.1 % , pp 1093, n 1121 ,  pp 97.5 %
e 15, verr 5.3 % , merr 4.9 % , pp 1093, n 1121 ,  pp 97.5 %
e 16, verr 5.0 % , merr 4.8 % , pp 1094, n 1122 ,  pp 97.5 %
...
e 88, verr 3.5 % , merr 2.8 % , pp 1100, n 1126 ,  pp 97.7 %
e 89, verr 3.5 % , merr 2.8 % , pp 1100, n 1126 ,  pp 97.7 %
e 90, verr 3.5 % , merr 2.8 % , pp 1100, n 1126 ,  pp 97.7 %
e 91, verr 3.5 % , merr 2.8 % , pp 1100, n 1126 ,  pp 97.7 %
e 92, verr 3.5 % , merr 2.8 % , pp 1100, n 1126 ,  pp 97.7 %
e 93, verr 3.5 % , merr 2.8 % , pp 1100, n 1126 ,  pp 97.7 %
e 94, verr 3.5 % , merr 2.8 % , pp 1100, n 1126 ,  pp 97.7 %
e 95, verr 3.5 % , merr 2.8 % , pp 1100, n 1126 ,  pp 97.7 %
e 96, verr 3.5 % , merr 2.8 % , pp 1100, n 1126 ,  pp 97.7 %
e 97, verr 3.5 % , merr 2.8 % , pp 1100, n 1126 ,  pp 97.7 %
e 98, verr 3.5 % , merr 2.8 % , pp 1100, n 1126 ,  pp 97.7 %
e 99, verr 3.4 % , merr 2.8 % , pp 1100, n 1126 ,  pp 97.7 %
Classification report:
             precision    recall  f1-score   support

          0       0.98      0.98      0.98      1001
          1       0.97      0.98      0.97      1127
          2       0.97      0.95      0.96       991
          3       0.95      0.96      0.95      1032
          4       0.97      0.96      0.97       980
          5       0.96      0.96      0.96       863
          6       0.97      0.98      0.97      1014
          7       0.97      0.97      0.97      1070
          8       0.95      0.95      0.95       944
          9       0.96      0.96      0.96       978

avg / total       0.97      0.97      0.97     10000


```


![运行结果](http://imageshack.com/a/img909/7594/o1E5XT.png)




##5 Dropout

从上一个模型来看，在超过50个epoch后，明显Validation error大于Model error，这就是过拟合了，overfit。dropout是解决过拟合的方法，继续实现dropout。（其实我是先实现了dropout，最后才想起写。所以最初的模型中实际已经有dropout了）在我的模型中，我没有对最后的Logistic Regression Layer进行Dropout。

```
python mlp_ver5.py
```

```
e 80, verr 6.1 % , merr 5.8 % , pp 1093, n 1131 ,  pp 96.6 %
e 81, verr 6.0 % , merr 5.8 % , pp 1093, n 1134 ,  pp 96.4 %
e 82, verr 5.9 % , merr 5.8 % , pp 1095, n 1135 ,  pp 96.5 %
e 83, verr 5.9 % , merr 5.7 % , pp 1093, n 1132 ,  pp 96.6 %
e 84, verr 5.8 % , merr 5.7 % , pp 1091, n 1129 ,  pp 96.6 %
e 85, verr 6.0 % , merr 5.7 % , pp 1088, n 1120 ,  pp 97.1 %
e 86, verr 5.8 % , merr 5.7 % , pp 1090, n 1124 ,  pp 97.0 %
e 87, verr 6.0 % , merr 5.7 % , pp 1089, n 1123 ,  pp 97.0 %
e 88, verr 5.9 % , merr 5.7 % , pp 1091, n 1127 ,  pp 96.8 %
e 89, verr 5.9 % , merr 5.7 % , pp 1094, n 1132 ,  pp 96.6 %
e 90, verr 5.9 % , merr 5.7 % , pp 1091, n 1128 ,  pp 96.7 %
e 91, verr 5.9 % , merr 5.7 % , pp 1093, n 1130 ,  pp 96.7 %
e 92, verr 5.9 % , merr 5.6 % , pp 1091, n 1126 ,  pp 96.9 %
e 93, verr 5.8 % , merr 5.7 % , pp 1093, n 1132 ,  pp 96.6 %
e 94, verr 5.9 % , merr 5.7 % , pp 1091, n 1127 ,  pp 96.8 %
e 95, verr 5.8 % , merr 5.7 % , pp 1093, n 1130 ,  pp 96.7 %
e 96, verr 5.8 % , merr 5.6 % , pp 1093, n 1129 ,  pp 96.8 %
e 97, verr 5.8 % , merr 5.6 % , pp 1090, n 1126 ,  pp 96.8 %
e 98, verr 5.7 % , merr 5.6 % , pp 1094, n 1132 ,  pp 96.6 %
e 99, verr 5.8 % , merr 5.6 % , pp 1092, n 1128 ,  pp 96.8 %
Classification report:
             precision    recall  f1-score   support

          0       0.96      0.98      0.97      1001
          1       0.95      0.97      0.96      1127
          2       0.95      0.92      0.94       991
          3       0.94      0.92      0.93      1032
          4       0.94      0.95      0.94       980
          5       0.93      0.93      0.93       863
          6       0.95      0.97      0.96      1014
          7       0.96      0.94      0.95      1070
          8       0.93      0.91      0.92       944
          9       0.92      0.93      0.92       978

avg / total       0.94      0.94      0.94     10000


```

![dropout](http://imageshack.com/a/img901/7320/Gz3mjy.png)

可以看出，dropout得到的模型不一定更好，dropout的抖动比较大，但是validation  error和model error两者一直是纠缠在一起的，没有明显的Overfit。

##小节

本文实现了one hidden layer的MLP，完善了参数更新方法。在此基础上做一个多层的MLP也是相当容易的。

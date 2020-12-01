# text-glyph-in-NLU

通常我们做NLP任务都使用基于tokens（字或词）的ID序列，那么基于纯字形的方法是否有效？


这里尝试验证基于纯字形在NLU中是否有效，即验证神经网络是否“看”懂了文本。


`model_glyph_flatten.py`中，THUCNews新闻标题分类任务，batch_size=32, epochs=10的效果，

```txt
- loss: 1.0370 - accuracy: 0.6737 
- val_loss: 0.7873 - val_accuracy: 0.7496
- test_loss: 0.7999 - test_accuracy: 0.7476
```

可以看到，纯字形效果还是不错的。

鉴于算力限制，后期再展开更多的实验~

其他ideas：

- 利用预训练思想，随机mask字形的局部，然后预测mask部分

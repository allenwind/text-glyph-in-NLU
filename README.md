# text-glyph-in-NLU

通常我们做NLP任务都使用基于tokens（字或词）的ID序列，那么基于纯字形的方法是否有效？


这里尝试验证基于纯字形在NLU中是否有效，即验证神经网络是否“看”懂了文本。

基于纯字形的方法，数据结构组织有两种：

- `(batch_size, seq_len, image_width, image_depth, 1)`
- `(batch_size, image_width, seq_len * image_depth, 1)`

`model_glyph_flatten.py`中使用第二种数据结构组织，THUCNews新闻标题分类任务，batch_size=32, epochs=10的效果，

```txt
- loss: 1.0370 - accuracy: 0.6737 
- val_loss: 0.7873 - val_accuracy: 0.7496
- test_loss: 0.7999 - test_accuracy: 0.7476
```

`model_glyph.py`中使用第一种数据结构组织，THUCNews新闻标题分类任务，batch_size=32, epochs=1的效果，

```txt
- loss: 1.3591 - accuracy: 0.5769 
- val_loss: 1.0656 - val_accuracy: 0.6703
- loss: 1.0660 - accuracy: 0.6705
```

由于算力问题，这种方法的模型目前还没有充分训练。



整体上，可以看到，纯字形效果还是不错的。


下面是基于纯字形预测中，文本中字重要性的可视化：

![](./asset/glyph_view_1.png)


![](./asset/glyph_view_2.png)


![](./asset/glyph_view_3.png)



鉴于算力限制，后期再展开更多的实验~

其他ideas：

- 利用预训练思想，随机mask字形的局部，然后预测mask部分
- Embedding结合字形信息

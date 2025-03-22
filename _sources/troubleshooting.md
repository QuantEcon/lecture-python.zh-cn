---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(troubleshooting)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# 故障排除

```{contents} 目录
:depth: 2
```

本页面面向在运行讲座代码时遇到错误的读者。

## 修复本地环境

讲座的基本假设是，当满足以下条件时，讲座中的代码应该能够执行：

1. 代码在Jupyter笔记本中执行
1. 笔记本运行在安装了最新版本Anaconda Python的机器上。

按照[本讲座](https://python-programming.quantecon.org/getting_started.html)的说明，你已经安装了Anaconda，对吧？

假设你已经安装了，我们的读者最常见的问题是他们的Anaconda发行版不是最新的。

[这里有一篇有用的文章](https://www.anaconda.com/blog/keeping-anaconda-date)介绍如何更新Anaconda。

另一个选择是直接删除Anaconda然后重新安装。

你还需要保持外部代码库（如[QuantEcon.py](https://quantecon.org/quantecon-py)）的更新。

对于这个任务，你可以：

* 在命令行中使用`pip install --upgrade quantecon`，或者
* 在Jupyter notebook中执行`!pip install --upgrade quantecon`

如果你的本地环境仍然不能正常工作，你可以做两件事。

首先，你可以使用远程机器，只需点击每个讲座提供的Launch Notebook图标

```{image} _static/lecture_specific/troubleshooting/launch.png

```

第二，您可以报告问题，这样我们就可以尝试修复您的本地设置。

我们很希望收到关于讲座的反馈，所以请不要犹豫与我们联系。

## 报告问题

提供反馈的一种方式是通过我们的[问题追踪器](https://github.com/QuantEcon/lecture-python/issues)提出问题。

请尽可能具体。告诉我们问题出在哪里，并提供尽可能多的关于您本地设置的详细信息。

另一个反馈选项是使用我们的[讨论论坛](https://discourse.quantecon.org/)。

最后，您可以直接向[contact@quantecon.org](mailto:contact@quantecon.org)提供反馈。


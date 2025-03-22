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

# 执行统计

此表格包含最新的执行统计数据。

```{nb-exec-table}
```

(status:machine-details)=

这些讲座是通过`github actions`在`linux`实例上构建的。

这些讲座使用以下Python版本

```{code-cell} ipython
!python --version
```

以及以下软件包版本

```{code-cell} ipython
:tags: [hide-output]
!conda list
```

本讲座系列还可以使用以下GPU

```{code-cell} ipython
!nvidia-smi
```


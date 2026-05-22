huai_forwork 项目文档
=====================

欢迎使用 huai_forwork 项目文档！

huai_forwork 是一个专为工作流程自动化设计的Python工具库，帮助你快速完成日常开发中的重复性任务。

快速开始
--------

复制下面的代码，运行你的第一个示例：

.. code-block:: python
   :linenos:

   # 导入核心模块
   from huai_forwork import Hello

   # 调用函数
   result = Hello.say("世界")
   print(result)

运行后你会看到：
.. code-block:: text

   Hello, 世界!

文档目录
--------

.. toctree::
   :maxdepth: 2
   :caption: 用户指南:

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: 开发文档:

   api
   changelog

索引和搜索
----------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
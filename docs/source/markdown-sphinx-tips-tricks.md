---
orphan: true
---
# Markdown in Sphinx Tips and Tricks

In this repository, you can use both markdown and reSTructuredText to author
your content. This section lists most common examples of how you can use
Sphinx directives in your markdown files to expand your contributions.
For more information, see
[MyST Parser Documentation](https://myst-parser.readthedocs.io/en/v0.17.1/sphinx/intro.html)
and [reSTructuredText to Markdown mapping](https://myst-parser.readthedocs.io/en/v0.17.1/syntax/syntax.html#syntax-directives).

## Admonitions

Here is an example of how you can add a note. Similarly, you can add
`{tip}` and `{warning}`.

::::{tab-set}

:::{tab-item} Example
```{image} /_static/img/s_demo_note_render.png
:alt: note
:class: bg-primary
:width: 210px
:align: center
```
:::

:::{tab-item} Source
```{image} /_static/img/s_demo_note_source.png
:alt: note
:class: bg-primary
:width: 170px
:align: center
```
:::

::::

## Images

[This page](https://myst-parser.readthedocs.io/en/latest/syntax/images_and_figures.html)
has extensive reference on how to add an image. You can use the standard markdown
syntax as well as an extended one that allows you to modify width, alignment, and
other parameters of an image.

::::{tab-set}

:::{tab-item} Standard syntax
```{code-block}
![image example][/_static/img/example-image.png]
```
:::

:::{tab-item} Extended Syntax
````{code-block}
```{image} img/s_demo_note_source.png
:alt: example
:class: bg-primary
:width: 150px
:align: center
```
````
:::

::::

## Code Block

You can use standard code blocks as well as the extended syntax and
include the code from other files as. More information can be
found on [this page](https://myst-parser.readthedocs.io/en/latest/syntax/code_and_apis.html).
Examples:

::::{tab-set}

:::{tab-item} Standard syntax
````{code-block}
```python
a = 1
b = 2
c = a + b
print(c)
```
````
:::

:::{tab-item} Output
```python
a = 1
b = 2
c = a + b
print(c)
```
:::

::::

::::{tab-set}

:::{tab-item} Extended Syntax
````{code-block}
```{code-block} python
:caption: My example code
:emphasize-lines: 4
:lineno-start: 1

a = 1
b = 2
c = a + b
print(c)
```
````
:::

:::{tab-item} Output
```{code-block} python
:caption: My example code
:emphasize-lines: 4
:lineno-start: 1

a = 1
b = 2
c = a + b
print(c)
```
:::

::::

::::{tab-set}

:::{tab-item} Include from other files
Here is how you can include the code from another file.
In this example, we will only include the code between
the `start-after` and `end-before` markers.

````{code-block}
```{literalinclude} _static/example.py
:start-after: start
:end-before: end
```
````
The `example.py` file looks like this:
```{code-block} python
:emphasize-lines: 10, 16
"""
A sample python file
"""

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    # start

    def introduce(self):
        print("Hello, my name is", self.name)
        print("I am", self.age, "years old")

    # end

person = Person("Alice", 25)
person.introduce()
:::

:::{tab-item} Output
```{literalinclude} _static/example.py
:start-after: start
:end-before: end
```
:::
::::

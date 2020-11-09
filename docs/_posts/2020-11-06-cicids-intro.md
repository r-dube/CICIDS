---
title: Introduction to the CICIDS analysis project
---
The CICIDS analysis project seeks to analyze the CICIDS2017 dataset from the University of New Brunswick (UNB). The CICIDS2017 dataset contains information on network traffic flows. The traffic flows are tagged as benign or one of several attacks. The analysis project attempts to understand the characteristics of various techniques that separate benign traffic flows from attack traffic flows.

The analysis is presented as "blog" posts and interprets the classification techniques implemented in Python.

### Useful URLs for this project
* [The Github repository for the project](https://github.com/r-dube/CICIDS)
* [The top page for this blog](https://r-dube.github.io/CICIDS/)
* [The raw data from UNB](https://www.unb.ca/cic/datasets/ids-2017.html)

### File and directory organization
1. MachineLearningCVE/ 
   * This directory contains processed data used by the classifiers. It also has data used to debug the Python scripts for combining and cleaning the raw data files.
1. docs/
   * This directory contains the blog posts. It also has the configuration files used by Github pages to render the posts.
1. notes/
   * This directory contains lab notes compiled during the analysis. These notes are used as raw material for blog posts.
1. scripts/
   * The Python code for this project was initially developed as scripts that run on a local machine. This directory contains those scripts. The scripts will continue to be maintained.
1. admin/
   * Scripts and guidelines for administering the Github repository.
1. README.md
   * This is the top-level markdown file that describes the project. Since all documentation is being maintained as blog posts, the file points to the blog.
1. *.ipynb files
   * These are Google Colab Jupyter notebooks. The first few notebooks were the Python scripts from scripts/ massaged to work on Colab. New notebooks will be added as the analysis develops. The notebooks (on Colab) will be periodically committed to the project's Github repository.

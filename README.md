# language-technology-humanities
Python code to reproduce some experiments published in Language Technology and Humanities Research by Barbara McGillivray and Gabor M. Toth (Palgrave, 2020)

Repository not maintained (issues disable); it is meant to help readers to see some research code underlying experiments in the book.

## Requirements

- Python3 (repository tested on 3.7.2)
- Python3 has to be the default python in your system (i.e by running any script with the 'python' command, the script should run automatically in python3)


## Getting Started

1. Git clone this library
2. Install python requirements

Use either pip:

```
pip3 install -r requirements_pip.txt
```

or conda:

```
conda install --file requirements_conda.txt
```

3. Install the Brown Corpus through NLTK

```
python install_brown_corpus.py
```

## Reproduce experiments

### Measure relative frequency of 'never' in two poems by Emily Dickinson (chapter 3)

The goal of this simple experiment is to demonstrate the concepts of raw frequency (count) and relative frequency. By running the following bash script (calling various python scripts), you can reproduce the details discussed in the book.

```
sh process_poems_dickinson.sh
```

If you want to save the output of the script above, run:

```
sh process_poems_dickinson.sh > report_dickinson_poems.txt
```

Open report_dickinson_poems.txt to see the output of the script. 

### Identify the characteristic vocabulary of the Moonstone by Wilkie Collins (chapter 3)

The goal of this experiment is to detect those terms in the novel that have an impact on the overall mood of the novel. By running the following bash script (calling various python scripts), you can reproduce the details discussed in the book. The script takes a few seconds.

```
sh process_moonstone.sh
```

If you want to save the output of the script above, run:

```
sh process_moonstone.sh > report_moonstone.txt
```

Open report_moonstone.txt to see the output of the script.

### Study the Anglo-Saxon Chronicle through a feature space representation

The goal of this experiment is to demonstrate the use of feature space representation of texts through the study of the Anglo-Saxon Chronicle. By running the following bash script (calling various python scripts), you can reproduce the details discussed in the book (chapter 6.)

```
sh process_anglo_saxon_chronicle.sh
```

If you want to save the output of the script above, run:

```
sh process_anglo_saxon_chronicle.sh > report_anglo_saxon_chronicle.txt
```

Open report_anglo_saxon_chronicle.txt to see the output of the script.

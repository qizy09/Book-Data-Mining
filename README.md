# Book-Data-Mining
This repository is used for backup of the SVM part source codes, of the PSU CSE 597A(2015 Winter) course final project .
---------------------

# Introduction
This project proposed approaches to construct feature sets in order to
predict a book’s quality based on the data collected from the online bookstore.
Review and rating data of books in different categories are collected from the Barnes
& Noble website. Various regression and classification algorithms are performed on
the constructed feature sets, while the rating data is used as the test set to verify the
correctness of the model.

# Team
[Yu-Hsuan Kuo](http://www.personal.psu.edu/yzk5145/), [Yu-San Lin](http://yusanlin.com/), [Ziyang Qi](qizy09.github.io/), [Yang Zheng](https://www.linkedin.com/in/yang-zheng-15889a34?authType=NAME_SEARCH&authToken=qNow&locale=en_US&srchid=662713351454704663627&srchindex=3&srchtotal=1809&trk=vsrp_people_res_name&trkInfo=VSRPsearchId%3A662713351454704663627%2CVSRPtargetId%3A121634132%2CVSRPcmpt%3Aprimary%2CVSRPnm%3Atrue%2CauthType%3ANAME_SEARCH)

# Data Source
[Barnes & Noble, Inc,](www.barnesandnobleinc.com/) contains detailed information of
books and authors, so it is one of the best options for our experiments.

We fetch the webpages of books, parse the webpages so that we have fields including
Title, Author, Price, Nook, Audio, Hardcover, Subject, Publisher, Published date, Pages,
Number of reviews and Rating. These data will be used for feature extraction.

# Summary
We collected data of 1224 books from one of the largest online bookstores, Barnes and
Noble. Various experiments are run on different feature sets to see whether we can
predict/capture books’ ratings by looking at data other than the content of the books.

We found that it is possible to predict a book’s rating on the online store without
knowing the content. Four combinations of feature sets are tested: When considering the
basic profile, the title, and the subjects of a book, Random Forest gives the best prediction
of the book’s rating.

# License

Copyright 2016.
For any questions, feel free to let me know.

# Information Retrieval - Search-Engine
## Summery:
As part of our bachelor degree in "Software and Information System Engineering" we took the "Introduction to Information Retrieval" course, This project is the final project in the course 👷.

In that course, we acquired knowledge regarding different indexing, retrieval, crawling, and "IR-Engines" evaluation techniques.

In this project, we developed a search engine over [Wikipedia](https://www.wikipedia.org/) data.
We have built the Indexes for out engine using the [Google Cloud Platform](https://cloud.google.com/) computing power.


## Retrival Methodes:
in this search-engine we have combine several retrival methodes include:
* Inverted Index
* TF-IDF
* cosine simularity
* BM-25
* Page View
* Page Rank

## Indexes:
* Wikipedia title
* Wikipedia body text
* Wikipedia anchor text

## Endpoints
The engine use 6 different endpoints, each endpoint retrival data in a diffarent way:
- **/search?query=??? [Get]-** ??? for your wuery. Use of BM25 method (title and body) for retrival, Use BM-25 score, page-rank and page-view for rating the retrivaled documents.
- **/search_body?query=??? [Get]-** ??? for your query. Use tf-idf and cosine-similarity score to select the best resaults.
- **/search_title?query=??? [Get]-** ??? for your query. Use a terms binary ranking existing in the title.
- **/search_anchor?query=??? [Get]-** ??? for your query. Use a terms binary ranking existing in the anchor text.
- **/get_pageview [POST]-** Insert wiki id's in the request body using the 'json' parameter. Retrive the wiki doc page views (Augoust 2021).
- **/get_pagerank [POST]-** Insert wiki id's in the request body using the 'json' parameter. Retrive the wiki doc page rank.

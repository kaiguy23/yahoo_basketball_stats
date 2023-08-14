# Yahoo Fantasy Basketball Stat Analysis

Have you ever lost a week in fantasy and wondered if you had a bad week, or if your opponent got lucky? This repository has been created to analyze matchups in yahoo fantasy basketball to make you feel better (or worse).

## Installation

You will need to install the following packages or just clone the environment file

### Pip Dependencies

* __https://github.com/josuebrunel/yahoo-oauth__
* __https://github.com/spilchen/yahoo_fantasy_api__
* __https://pypi.org/project/dataframe-image/__

```python
pip install yahoo_oauth
pip install yahoo_fantasy_api
pip install dataframe-image
```

### Create Conda Environment 

```python
conda env create --file=environment.yaml
```

## Setup OAUTH

In order to access the yahoo api, you will need to create yahoo developer account.
https://developer.yahoo.com/oauth2/guide/openid_connect/getting_started.html

You probably already have a yahoo account, so you just need to fill out this form:
https://developer.yahoo.com/apps/create/

For our purposes, the Homepage URL and Redirect URI(s) do not matter, but Redirect URI(s) is required. You can simply put your local host as the Redirect URI (https://localhost:8001).

Select confidential client.

When you create the app, there will be a Client ID  (Consumer Key) and Client Secret (Consumer Secret) listed. DO NOT copy the App ID. Create a json file titled yahoo_oauth.json:

```json
{
    "consumer_key": "my_very_long_and_weird_consumer_key",
    "consumer_secret": "my_not_that_long_consumer_secret"
}
```

Congrats, you can now start running the script!

### Note:

If you haven't connected to your developer account before, the command line will prompt you for a key. The yahoo_oauth package will automatically launch an internet browser window with your key. Simply paste the key into the terminal and you should be good to go!


### Description of Prediction Algorithm

To do the predictions, we assume that the statistics of player performance are represented by Poisson random variables. This represents a distribution of integers greater than or equal to zero, and is used to represent discrete processes that happen integer number of times in physics (e.x. nuclear decay). It's also widely used in sports statistial analysis (or so Google tells me). It also makes the statistics easy to work with, as Poisson processes have the useful property that the variance is equal to the mean, and the sum of multiple Poisson random variables itself a Poisson process with $\mu = \mu_1 + \mu_2$.

To calculate the probability of victory in any one category, we take the difference of two Poisson distributions, known as a [Skellam distribution](https://en.wikipedia.org/wiki/Skellam_distribution), and sum the total probabilities to be negative (distribution 2 wins), zero (tie), and positive (distribution 1 wins).

Free throw and fg percentage is more difficult, as it's the ratio of two non-independant Poisson distributions.


### Notes on Database:

NBA and Yahoo sometimes have different names for players with abbreviations, like O.G. Aununoby vs. OG Anunoby. By default I have the "name" (info from yahoo_api) or "PLAYER_NAME" (info from nba api) set to be the NBA api name. In the fantasy rosters table I kept the yahoo names in another column titled "yahoo_name"

### Possible Improvements:

- Acurrately assigning rosters for future days. Currently takes the 10 highest by fantasy points, even if their positions would be incompatible.
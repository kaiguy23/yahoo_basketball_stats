# Yahoo Fantasy Basketball Stat Analysis

Have you ever lost a week in fantasy and wondered if you had a bad week, or if your opponent got lucky? This repository has been created to analyze matchups in yahoo fantasy basketball to make you feel better (or worse).

## Installation

You will need to install the following packages or just clone the environment file

### Pip Dependencies

* __https://github.com/josuebrunel/yahoo-oauth__
* __https://github.com/spilchen/yahoo_fantasy_api__

```python
pip install yahoo_oauth
pip install yahoo_fantasy_api
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

When you create the app, there will be a Client ID  (Consumer Key) and Client Secret (Consumer Secret) listed. Create a json or yaml file with these values.

```json
{
    "consumer_key": "my_very_long_and_weird_consumer_key",
    "consumer_secret": "my_not_that_long_consumer_secret"
}
```

Congrats, you can now start running the script!

### Note:

If you haven't connected to your developer account before, the command line will prompt you for a key. The yahoo_oauth package will automatically launch an internet browser window with your key. Simply paste the key into the terminal and you should be good to go!

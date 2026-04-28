# DS 4320 Project 2: Magic the Gathering Deck Evaluation Tool

**Executive Summary**: This readme documents a Magic the Gathering secondary dataset for DS 4320 Data By Design. This project includes completed project metadata such as a name, computing ID, DOI, analysis code links, and a license. A summary table is included additionally. A problem statement, rationale, references, terminology table, and code are also included.

**Name**: Aidan Mayhue

**NetID**: xdw9vp

**DOI**: https://doi.org/10.5281/zenodo.19668390

**Press Release**: https://github.com/AidanMayhue/DBD_P2/blob/main/press_release.md

**Pipeline**: https://github.com/AidanMayhue/DBD_P2/blob/main/mtg_deck_scorer.py

**License**: https://github.com/AidanMayhue/DBD_P2/blob/main/LICENSE

## Problem Definition

**Initial General Problem**

The general problem is predicting the outcome of a sports game. The specific problem is to problem is to identify the outcome of a trading card game match given a decklist

**motivation**

I am a big fan of trading card games and enjoy theorycrafting decklists. Magic the Gathering is my favorite TCG, it furthermore has a wealth of data and clear confined card pools. Since I play somewhat competitively, I am interested in identifying the best decks in the metagame along with the best ratio of cards in those decks. This is intended as a tool for competitive players to identify powerful cards and their win rates within the format. This project if successful offers significant utility to me, as I primarily come from a casual background and cannot build a high tier deck on my own.

**rationale**

The general problem is describing some kind of method to predict the outcome of a game. While the sport qualifier is a little specific, I would argue its vague enough to include card games. Since the data analytics skills needed to predict the outcome of a card game are nearly identical to those needed to predict the outcome of a sport, I would claim the specific problem matches the general problem appropriately. Furthermore, since individual decks fill a similar role to athletes in a sport, predicting the probability of a deck winning a match is equivalent to predicting the outcome of an actual sport. Some nuance arises where the quality of deck is independent of the quality of the player. If you give a highly complex deck to a beginner player, even if it is stronger in theory, it is not difficult to imagine them losing due to misunderstanding a combo.


**headline**

New Magic The Gathering Win Probability Calculator

https://github.com/AidanMayhue/DBD_P2/blob/main/press_release.md

## Domain Exposition

**Terminology**

| term | summary |
|------|---------|
|trading card game | a card game that emphasizes a unqiue combination of cards (known as a deck) that operate within a system of rules|
| game in hand win rate | the probability of winning a game given that you drew a unique card throughout the game |
|mana | resource in magic the gathering, used as a currency |
|mana cost | the amount of mana needed to play a card|
|land | cards that generate mana |
|color | category that a card falls into, there are five (white, blue, black, red, green), each color plays differently |
|card pool | the list of cards that you are allowed to use within a given format |
|decking | when you have no cards left in your deck, this causes you to lose the game |
|combo | a playstyle that emphasizes playing a short sequence of cards that directly causes a winning game state |
|control | a playstyle that wants to slow the game down and win by preventing key plays from your opponent |
|aggro| a playstyle that wants to win the game as quick as possible, typically through damage |
|midrange | a playstyle that employs some combination of combo, aggro, and control |

**Domain**

This project lives in the domain of sports analytics. While the game may be a bit of a stretch to refer to as a sport, many of the trappings of the field. The project is interested in identifying an ideal combination of resources within the game that will develop a winning strategy. This project is designed explicitly to create success in competitive play and has no bearing on casual play. Sports analytics is highly useful in this instance and appropriately explains the domain because the field allows players to identify synergy between cards and exploit them.

**Background Reading**

https://myuva-my.sharepoint.com/:f:/g/personal/xdw9vp_virginia_edu/IgBcc1_M6ojZQpapVXxXlbsRAcC_bkv7Lzynx5-oIxFasrQ?e=Q5lz7u

**Summary of Readings**

|title | description | link|
|------|-------------|-----|
|How to Play Magic the Gathering |A comprehensive introductory explanation of magic the gathering rules| https://myuva-my.sharepoint.com/:b:/g/personal/xdw9vp_virginia_edu/IQAl1HCKs8XNQ5dZdDyNwT88AQKApUUL2Sbl61nHQUcOzv8?e=rrQxfQ|
|A short Introduction of Game Theory|A brief explanation of the key principles of game theory, the mathematical study of strategic decision making|https://myuva-my.sharepoint.com/:b:/g/personal/xdw9vp_virginia_edu/IQAVtBPfmtQrQq2RgHTigGyfAWVNVfCLZXZJFR1aKeuqyPU?e=Z6Zz9Z|
|The Hypergeometric Probability Distribution|Explanation of a hypergemoetric distribution, the probability of y successes when sampling without replacement|https://myuva-my.sharepoint.com/:b:/g/personal/xdw9vp_virginia_edu/IQCpRwvzdyHFRoO4GOIN63LlARPwuJ7zbkJQMVn5M0r6RHc?e=JE9XfV|
|Magic The Gathering is Turing Complete|A paper highlighting the importance of synergistic plays within MTG|https://myuva-my.sharepoint.com/:b:/g/personal/xdw9vp_virginia_edu/IQDKnFd88ZQIS4zOEy5nMVbOAZn5kSvVcWkegyz3ceL-HIM?e=M5D6qC|
|Final Fantasy Standard Metagame Breakdown|A competitive breakdown of top performing decks within one format, showing some of the most powerful current decks|https://myuva-my.sharepoint.com/:b:/g/personal/xdw9vp_virginia_edu/IQA2XiL6FjIKRI_UzxSyMtAaAXcLHPXvRr-29FXuQzjJY04?e=onelUE|

## Data Creation

**Raw Data Acquisition**

For this dataset the main source of data that I needed was a comprehensive list of all currently legal cards in the standard format of Magic the Gathering. This needs a variety of information about a card such as its mana cost, however information, such as the artist, is not relevant. To collect this data I utilized an API called the scryfall API. Scryfall is a card database often used by players who are looking for cards that match a set of criteria. Essentially, an already fully functioning database that utilizes a query system. I immediately noticed a concern upon downloading, the number of records I was anticipating was quadrupled due to scryfall counting cards with different art as unique cards.

**code**

|title|description|link|
|---|----|---|
|card_importer.py|a file that collects card data from the scryfall API and places it within MongoDB | https://github.com/AidanMayhue/DBD_P2/blob/main/card-importer.py|

**rationale**

Due to concerns about price inflating the perceived power of some cards, I am making the decision to only include the cheapest version of all cards. Since there are multiple versions of cards, present (900 versions of essential cards that are in every deck for example ) This will drastically shrink the card pool, but shrink the number of unique cards. Since competitive viability is not dependent on the art and only the text, this decision should not have any impact

**Bias identification**

A potential source of bias found in this dataset is alternative card variants. Many popular and powerful magic cards have special art variants that tend to be priced significantly higher. Since scryfall considered alternative arts as separate cards, I will need to consider how differing prices may affect a deck. However, not all powerful cards are expensive, this may create a bias towards more expensive cards in the model, since a 90 dollar alternative art badgermole cub will be worth more than a $10 stock up. This poses issues since stock up is arguably just as powerful in the meta as badgermole cub.

**Bias Mitigation**

A way to mitigate this is to exclude these alternative art cards. By selecting the lowest cost version of cards with the same name, this should prune the dataset down to the originally anticipated 4000 cards. Price is generally correlated to power in this game, so I will need to find a way to appropriately model price without skewing the data by completely ignoring the relationship between price and power or overcorrecting and listing a weaker but rare card as meta defining.

## Metadata

**Implicit Schema**

<img width="1534" height="1296" alt="image" src="https://github.com/user-attachments/assets/ab14da5e-b93f-4a2a-97ac-742278f78c56" />


**Data Summary**

This database contains information about individual cards. Each entry represents a card.

Cards are placed into a nested file structure like so. This database uses an embedding strategy for its data.

<img width="2712" height="1022" alt="image" src="https://github.com/user-attachments/assets/3a656675-6db3-44fe-bb08-1aa17406538e" />


**Data Dictionary**

|Name| Data Type | description | example |
|----|-----------|-------------|---------|
|scryfall_id|string|a unique identifier for the card within scryfall|0000419-0bba-4488-8f7a-6194544ce91|
|card_faces| string | This describes if a card is double sided or not, most will be null | array (2), [Object] |
|cmc | int | the total cost of the card ignoring color | 5 |
|collector number | string| the specific card printing in the context of the set code | "63" |
| color_identity | Array | the colors that make up the card, does not have to include the cost | Array (1)"U"|
|colors | Array | colors present in the mana cost | Array (1)"G", "U" |
|image_uris | string | A link to the card image | https://cards.scryfall.io/small/front/0/0/0000419b-0bba-4488-8f7a-6194544ce91e.jpg?1721427487 |
|keywords| Array | a list of keywords present on the card (flying, vigilance, etc ) | Array (3) |
|last_synced | String | a note of when the database was last synced with the server, in the form of a date | "2026-04-15T14:55:00.727478+00:00" |
|legalities| Dictionary | A list of key value pairs for each format with each value stating if it is legal in that format | standard : "legal" |
|loyalty | int | The amount of counters a card type known as a planeswalker enters with | 5 |
|mana cost | string | The amount of mana needed to cast the card including color | "{2}{R}"|
|Name | string | The name of the card | "Elspeth, Stormslayer"|
|oracle id| string | the id of the card in the official MTG database | "f8d7bcbd-9a1d-4fc6-abdd-88a7e01b9411"|
|oracle text| string | The text on the card, sometimes includes erratas for playability | "At the beginning of combat on your turn, put a +1/+1 counter on target…" |
|Power | string | The power of the creature card, not applicable to non creature cards | 3|
|Prices | dictionary | the monetary value of the card | usd "0.29"|
| raw | dictionary | a dictionary containing all previous values | object *every other value* |
|released at | string | the release date for the card | 2024-09-14|
|scryfall_uri|string | the link for the card api within the scryfall database |"https://scryfall.com/card/otj/171/ornery-tumblewagg?utm_source=api"|
|set_code| string | the 3 letter code for the set the card is from | "neo"|
|set_name| string| the name of the set the card is from | "Kamigawa Neon Dynasty" |
|toughness | string | the amount of damage a creature can take | "3"|
|type_line|string|information about what type of creature the card is, along with what card type it is | "Legendary Creature - Human Warrior"|


**Quantification of Uncertainty**

|Feature|Count|Mean|Std|Min|25%|Median|75%|95%|Max|Nulls|
|---|----|----|---|---|----|---|----|---|---|---|
|CMC|17,251|2.14|2.13|0|0|2|4|6|12|0|
|Price USD|\$13,291|\$3.05|\$33.13|\$0.01|\$0.15|\$0.30|\$1.01|\$10.79|\$2,801.98|3,960|
|Price USD foil|11,855|\$10.92|\$108.81|\$0.02|\$0.27|\$0.70|\$3.81|\$29.01|\$6,175.00|5,396|
|Power (numeric)|6,380|3.03|1.88|0|2|3|4|6|20|10,871|
|Toughness (numeric)|6,414|3.24|2.03|0|2|3|4|6|30|10,837|

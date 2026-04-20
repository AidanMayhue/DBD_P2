# DS 4320 Project 2: Magic the Gathering Deck Evaluation Tool

**Executive Summary**:

**Name**: Aidan Mayhue

**NetID**: xdw9vp

**DOI**:

**Press Release**:

**Pipeline**: https://github.com/AidanMayhue/DBD_P2/blob/main/press_release.md

**License**:

## Problem Definition

**Initial General Problem**

The general problem is predicting the outcome of a sports game. The specific problem is to problem is to identify the outcome of a trading card game match given a decklist

**motivation**

I am a big fan of trading card games and enjoy theorycrafting decklists. Magic the Gathering is my favorite TCG, it furthermore has a wealth of data and clear confined card pools. Since I play somewhat competitively, I am interested in identifying the best decks in the metagame along with the best ratio of cards in those decks. This is intended as a tool for competitive players to identify powerful cards and their win rates within the format. This project if successful offers significant utility to me, as I primarily come from a casual background and cannot build a high tier deck on my own.

**rationale**

The general problem is describing some kind of method to predict the outcome of a game. While the sport qualifier is a little specific, I would argue its vague enough to include card games. Since the data analytics skills needed to predict the outcome of a card game are nearly identical to those needed to predict the outcome of a sport, I would claim the specific problem matches the general problem appropriately. Furthermore, since individual decks fill a similar role to athletes in a sport, predicting the probability of a deck winning a match is equivalent to predicting the outcome of an actual sport. Some nuance arises where the quality of deck is independent of the quality of the player. If you give a highly complex deck to a beginner player, even if it is stronger in theory, it is not difficult to imagine them losing due to misunderstanding a combo.


**headline**

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

